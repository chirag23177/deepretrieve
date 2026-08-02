"""Full-ranking offline evaluation. Every system goes through this one harness.

Protocol (fixed in week 1, never changed):
- For each test user, score ALL items EXCEPT that user's train-window history.
- Ground truth = that user's test-window positives.
- Metrics: Recall@{10,50,100}, NDCG@{10,50,100}, averaged over test users.
- No sampled negatives: sampled-negative evaluation (1 pos vs 100 random negs)
  is a known-biased shortcut; we deliberately do full ranking.

Usage:
    python -m src.eval.run_eval --systems popularity item_knn --out baselines.md
    python -m src.eval.run_eval --systems popularity item_knn two_tower two_tower_nologq two_tower_lgbm --out final_eval.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import load_config, load_split, resolve, set_seed, user_positives
from src.eval.metrics import (
    ndcg_at_k_from_topk,
    recall_at_k_from_topk,
    topk_from_scores,
)

NEG_INF = np.float32(-1e9)


def evaluate_scorer(
    scorer,
    eval_gt: dict[int, np.ndarray],
    train_hist: dict[int, np.ndarray],
    k_values: list[int],
    batch_size: int,
    users: np.ndarray | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    """Full-ranking evaluation of any scorer exposing score_users(users) -> (B, n_items)."""
    if users is None:
        users = np.array(sorted(eval_gt), dtype=np.int64)
    max_k = max(k_values)
    per_user: dict[str, list[np.ndarray]] = {
        f"{m}@{k}": [] for m in ("Recall", "NDCG") for k in k_values
    }
    iterator = range(0, len(users), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"eval {scorer.name}")
    for start in iterator:
        batch = users[start : start + batch_size]
        scores = scorer.score_users(batch).astype(np.float32, copy=False)
        # Mask each user's train history — forgetting this is the #1 silent
        # inflation bug, so it lives here in the harness, not in the scorers.
        for row, u in enumerate(batch):
            hist = train_hist.get(int(u))
            if hist is not None:
                scores[row, hist] = NEG_INF
        topk = topk_from_scores(scores, max_k)
        gt = [eval_gt[int(u)] for u in batch]
        for k in k_values:
            per_user[f"Recall@{k}"].append(recall_at_k_from_topk(topk, gt, k))
            per_user[f"NDCG@{k}"].append(ndcg_at_k_from_topk(topk, gt, k))
    return {name: float(np.concatenate(vals).mean()) for name, vals in per_user.items()}


class EmbeddingScorer:
    """Scores = user_emb @ item_emb.T — used for the trained two-tower model."""

    def __init__(self, user_emb: np.ndarray, item_emb: np.ndarray, name: str = "Two-tower"):
        self.user_emb = user_emb.astype(np.float32)
        self.item_emb = item_emb.astype(np.float32)
        self.name = name

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self.user_emb[users] @ self.item_emb.T


class RerankScorer:
    """Two-stage system: two-tower top-K retrieval -> LightGBM rerank.

    Non-candidate items get -inf, so metrics reflect exactly what the served
    system would return. Recall@100 can only come from the retrieved top-K —
    the ranker reorders, it cannot recover items retrieval missed.
    """

    name = "Two-tower + LightGBM"

    def __init__(self, retrieval: EmbeddingScorer, feature_builder, booster,
                 train_hist: dict[int, np.ndarray], retrieve_k: int):
        self.retrieval = retrieval
        self.fb = feature_builder
        self.booster = booster
        self.train_hist = train_hist
        self.retrieve_k = retrieve_k

    def score_users(self, users: np.ndarray) -> np.ndarray:
        scores = self.retrieval.score_users(users)
        for row, u in enumerate(users):
            hist = self.train_hist.get(int(u))
            if hist is not None:
                scores[row, hist] = NEG_INF
        cand = topk_from_scores(scores, self.retrieve_k)  # (B, K)
        row_idx = np.arange(len(users))[:, None]
        retr = scores[row_idx, cand]
        feats = self.fb.build(np.repeat(users, cand.shape[1]), cand.ravel(), retr.ravel())
        ranked = self.booster.predict(feats).reshape(cand.shape).astype(np.float32)
        out = np.full_like(scores, NEG_INF)
        out[row_idx, cand] = ranked
        return out


def build_scorer(name: str, cfg: dict, split: dict, train_hist: dict[int, np.ndarray]):
    art = resolve(cfg, "artifacts_dir")
    train, n_users, n_items = split["train"], split["n_users"], split["n_items"]

    if name == "popularity":
        from src.baselines.popularity import PopularityScorer

        return PopularityScorer(train, n_items)

    if name == "item_knn":
        from src.baselines.item_knn import ItemKNNScorer

        knn_cfg = cfg["baselines"]["item_knn"]
        return ItemKNNScorer(
            train, n_users, n_items,
            top_neighbors=knn_cfg["top_neighbors"],
            chunk_size=knn_cfg["chunk_size"],
            cache_path=art / "item_knn_sim.npz",
        )

    if name in ("two_tower", "two_tower_nologq"):
        suffix = "_nologq" if name.endswith("nologq") else ""
        label = "Two-tower (no logQ) — ablation" if suffix else "Two-tower"
        return EmbeddingScorer(
            np.load(art / f"user_embeddings{suffix}.npy"),
            np.load(art / f"item_embeddings{suffix}.npy"),
            name=label,
        )

    if name == "two_tower_lgbm":
        import lightgbm as lgb

        from src.ranking.features import FeatureBuilder

        retrieval = EmbeddingScorer(
            np.load(art / "user_embeddings.npy"),
            np.load(art / "item_embeddings.npy"),
        )
        fb = FeatureBuilder.load(art / "feature_store.npz")
        booster = lgb.Booster(model_file=str(art / "ranker.txt"))
        return RerankScorer(
            retrieval, fb, booster, train_hist, cfg["serving"]["retrieve_k"]
        )

    raise ValueError(f"unknown system: {name}")


def write_results(results: dict[str, dict[str, float]], cfg: dict, out_name: str) -> None:
    res_dir = resolve(cfg, "results_dir")
    csv_path = res_dir / "eval.csv"

    df = pd.DataFrame(results).T
    df.index.name = "System"
    if csv_path.exists():
        old = pd.read_csv(csv_path, index_col="System")
        old = old.drop(index=[i for i in df.index if i in old.index])
        df = pd.concat([old, df])
    df.to_csv(csv_path)

    md_path = res_dir / out_name
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Offline evaluation — full-catalog ranking, temporal test split\n\n")
        f.write(df.round(4).to_markdown())
        f.write("\n\nProtocol: for each test user, score all items minus their train history; ")
        f.write("ground truth = test-window positives. No sampled negatives.\n")
    print(f"\nwrote {csv_path} and {md_path}\n")
    print(df.round(4).to_markdown())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--systems", nargs="+", required=True,
        choices=["popularity", "item_knn", "two_tower", "two_tower_nologq", "two_tower_lgbm"],
    )
    parser.add_argument("--out", default="eval.md", help="markdown filename in results/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    split = load_split(cfg)
    train_hist = user_positives(split["train"])
    test_gt = user_positives(split["test"])
    print(f"test users: {len(test_gt):,} | items: {split['n_items']:,}")

    results: dict[str, dict[str, float]] = {}
    for name in args.systems:
        scorer = build_scorer(name, cfg, split, train_hist)
        t0 = time.time()
        metrics = evaluate_scorer(
            scorer, test_gt, train_hist,
            k_values=cfg["eval"]["k_values"],
            batch_size=cfg["eval"]["user_batch_size"],
        )
        print(f"{scorer.name}: {metrics} ({time.time() - t0:.1f}s)")
        results[scorer.name] = metrics

    write_results(results, cfg, args.out)


if __name__ == "__main__":
    main()
