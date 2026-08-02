"""Train the two-tower model with in-batch softmax negatives + logQ correction.

Batch of B (user, positive_item) pairs -> B x B score matrix -> cross-entropy
with the diagonal as the label. Each user's negatives are the other B-1 items
in the batch. Large batch (4096) = more and harder negatives.

Early stopping on val Recall@100 (full ranking over a sampled val-user subset).

Usage:
    python -m src.models.train_retrieval                 # full run
    python -m src.models.train_retrieval --no-logq       # ablation (artifacts get _nologq suffix)
    python -m src.models.train_retrieval --overfit 1000  # debug: loss must go ~0 on 1k rows
"""

from __future__ import annotations

import argparse
import copy
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common import load_config, load_split, resolve, set_seed, user_positives
from src.data.genres import build_genre_matrix
from src.eval.run_eval import EmbeddingScorer, evaluate_scorer
from src.models.two_tower import TwoTower


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-logq", action="store_true", help="ablation: disable logQ correction")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overfit", type=int, default=None,
                        help="debug: train on N interactions only, skip val eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    r = cfg["retrieval"]
    set_seed(cfg["seed"])
    device = get_device(r["device"])
    use_logq = r["logq_correction"] and not args.no_logq
    suffix = "" if use_logq else "_nologq"
    epochs = args.epochs or r["epochs"]
    print(f"device={device} logq={use_logq} epochs={epochs}")

    split = load_split(cfg)
    train = split["train"]
    n_users, n_items = split["n_users"], split["n_items"]
    if args.overfit:
        train = train.head(args.overfit)
        print(f"OVERFIT MODE: {len(train)} interactions")

    genre_matrix, genre_vocab = build_genre_matrix(split["items"], n_items)
    print(f"{n_users:,} users | {n_items:,} items | {len(genre_vocab)} genres | "
          f"{len(train):,} train pairs")

    model = TwoTower(
        n_users=n_users,
        n_items=n_items,
        genre_matrix=torch.from_numpy(genre_matrix),
        emb_dim=r["embedding_dim"],
        user_mlp_dims=r["user_mlp_dims"],
        item_mlp_dims=r["item_mlp_dims"],
        genre_proj_dim=r["genre_proj_dim"],
        temperature=r["temperature"],
        learnable_temperature=r["learnable_temperature"],
    ).to(device)

    # empirical item sampling probability for the logQ correction
    counts = np.bincount(train["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)
    probs = np.clip(counts / max(counts.sum(), 1), 1e-12, None)
    log_q = torch.from_numpy(np.log(probs)).float().to(device)

    dataset = TensorDataset(
        torch.from_numpy(train["user_idx"].to_numpy().astype(np.int64)),
        torch.from_numpy(train["item_idx"].to_numpy().astype(np.int64)),
    )
    batch_size = min(r["batch_size"], len(dataset))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(dataset) > batch_size, num_workers=r["num_workers"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])

    val_gt = user_positives(split["val"])
    train_hist = user_positives(split["train"])
    rng = np.random.default_rng(cfg["seed"])
    val_users = np.array(sorted(val_gt), dtype=np.int64)
    n_sample = min(cfg["eval"]["val_users_sample"], len(val_users))
    if n_sample and n_sample < len(val_users):
        val_users = rng.choice(val_users, size=n_sample, replace=False)

    art = resolve(cfg, "artifacts_dir")
    log_rows: list[dict] = []
    best_recall, best_state, patience_left = -1.0, None, r["early_stopping_patience"]

    for epoch in range(1, epochs + 1):
        model.train()
        t0, total_loss, n_batches = time.time(), 0.0, 0
        for user_ids, item_ids in loader:
            user_ids, item_ids = user_ids.to(device), item_ids.to(device)
            user_vecs, item_vecs = model(user_ids, item_ids)
            logits = model.in_batch_logits(
                user_vecs, item_vecs, log_q[item_ids] if use_logq else None
            )
            loss = F.cross_entropy(logits, torch.arange(len(user_ids), device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        if args.overfit:
            print(f"epoch {epoch}: loss={avg_loss:.4f} ({time.time() - t0:.1f}s)")
            continue

        model.eval()
        scorer = EmbeddingScorer(
            model.all_user_embeddings().cpu().numpy(),
            model.all_item_embeddings().cpu().numpy(),
        )
        metrics = evaluate_scorer(
            scorer, val_gt, train_hist,
            k_values=cfg["eval"]["k_values"],
            batch_size=cfg["eval"]["user_batch_size"],
            users=val_users, show_progress=False,
        )
        recall100 = metrics[f"Recall@{max(cfg['eval']['k_values'])}"]
        print(f"epoch {epoch}: loss={avg_loss:.4f} val={metrics} ({time.time() - t0:.1f}s)")
        log_rows.append({"epoch": epoch, "loss": avg_loss, **metrics})

        if recall100 > best_recall:
            best_recall = recall100
            best_state = copy.deepcopy(model.state_dict())
            patience_left = r["early_stopping_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch} (best val Recall = {best_recall:.4f})")
                break

    if args.overfit:
        return

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(model.state_dict(), art / f"two_tower{suffix}.pt")
    np.save(art / f"user_embeddings{suffix}.npy", model.all_user_embeddings().cpu().numpy())
    np.save(art / f"item_embeddings{suffix}.npy", model.all_item_embeddings().cpu().numpy())
    if log_rows:
        with open(art / f"training_log{suffix}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
    print(f"saved artifacts with suffix '{suffix}' | best val Recall@100 = {best_recall:.4f}")


if __name__ == "__main__":
    main()
