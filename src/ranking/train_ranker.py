"""Train the LightGBM LambdaRank reranker on retrieval's own candidates.

Label construction (mirrors production, and the #1 leakage trap — read carefully):
- The last `ranker_window_frac` of TRAIN interactions (by global timestamp) is
  carved out as the label window. Val/test are NEVER touched.
- For each user with positives in that window, retrieve their two-tower top-K
  candidates (masking base-train history), label a candidate 1 if it appears in
  the user's label window, else 0. The ranker learns to reorder what retrieval
  actually surfaces, not random negatives.
- Features for ranker TRAINING are computed from the base train window only
  (train minus label window), so no label information leaks into features.
  The feature store used at eval/serving time is rebuilt on the full train
  window and saved to artifacts/feature_store.npz.

Usage:  python -m src.ranking.train_ranker
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from src.common import load_config, load_split, resolve, set_seed, user_positives
from src.eval.metrics import topk_from_scores
from src.ranking.features import FeatureBuilder

NEG_INF = np.float32(-1e9)


def build_ranker_dataset(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    fb: FeatureBuilder,
    base_hist: dict[int, np.ndarray],
    label_gt: dict[int, np.ndarray],
    candidates_per_user: int,
    max_users: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (N, F), y (N,), group sizes, and the user id per group."""
    rng = np.random.default_rng(seed)
    users = np.array(sorted(label_gt), dtype=np.int64)
    if len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False)
        users.sort()

    X_parts, y_parts, groups, group_users = [], [], [], []
    skipped = 0
    batch = 512
    for start in tqdm(range(0, len(users), batch), desc="ranker dataset"):
        ub = users[start : start + batch]
        scores = (user_emb[ub] @ item_emb.T).astype(np.float32)
        for row, u in enumerate(ub):
            hist = base_hist.get(int(u))
            if hist is not None:
                scores[row, hist] = NEG_INF
        cand = topk_from_scores(scores, candidates_per_user)
        retr = scores[np.arange(len(ub))[:, None], cand]
        for row, u in enumerate(ub):
            labels = np.isin(cand[row], label_gt[int(u)]).astype(np.int32)
            if labels.sum() == 0:
                skipped += 1  # no retrievable positive -> group teaches nothing
                continue
            feats = fb.build(
                np.full(cand.shape[1], u, dtype=np.int64), cand[row], retr[row]
            )
            X_parts.append(feats)
            y_parts.append(labels)
            groups.append(cand.shape[1])
            group_users.append(u)

    print(f"kept {len(groups):,} users, skipped {skipped:,} with no retrieved positive")
    return (
        np.concatenate(X_parts),
        np.concatenate(y_parts),
        np.array(groups),
        np.array(group_users),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rc = cfg["ranker"]
    set_seed(cfg["seed"])

    split = load_split(cfg)
    train, items = split["train"], split["items"]
    n_users, n_items = split["n_users"], split["n_items"]
    art = resolve(cfg, "artifacts_dir")
    res = resolve(cfg, "results_dir")

    # carve the label window out of late train data
    cutoff = train["timestamp"].quantile(1.0 - rc["ranker_window_frac"])
    base = train[train["timestamp"] <= cutoff]
    label_window = train[train["timestamp"] > cutoff]
    print(f"base train {len(base):,} rows | ranker label window {len(label_window):,} rows")

    base_hist = user_positives(base)
    label_gt = user_positives(label_window)
    # users with no base history have no usable retrieval signal
    label_gt = {u: v for u, v in label_gt.items() if u in base_hist}

    user_emb = np.load(art / "user_embeddings.npy")
    item_emb = np.load(art / "item_embeddings.npy")

    fb_train = FeatureBuilder.fit(base, items, n_users, n_items, rc["top_genres"])
    X, y, groups, group_users = build_ranker_dataset(
        user_emb, item_emb, fb_train, base_hist, label_gt,
        rc["candidates_per_user"], rc["max_users"], cfg["seed"],
    )

    # user-level split for early stopping (never split within a user's group)
    rng = np.random.default_rng(cfg["seed"])
    val_mask_g = rng.random(len(groups)) < rc["ranker_val_frac"]
    row_mask = np.repeat(val_mask_g, groups)
    lgbm = rc["lightgbm"]
    dtrain = lgb.Dataset(
        X[~row_mask], label=y[~row_mask], group=groups[~val_mask_g],
        feature_name=fb_train.feature_names,
    )
    dval = lgb.Dataset(
        X[row_mask], label=y[row_mask], group=groups[val_mask_g], reference=dtrain,
    )
    params = {
        "objective": lgbm["objective"],
        "metric": lgbm["metric"],
        "eval_at": lgbm["eval_at"],
        "num_leaves": lgbm["num_leaves"],
        "max_depth": lgbm["max_depth"],
        "learning_rate": lgbm["learning_rate"],
        "min_child_samples": lgbm["min_child_samples"],
        "verbosity": -1,
        "seed": cfg["seed"],
    }
    booster = lgb.train(
        params, dtrain,
        num_boost_round=lgbm["n_estimators"],
        valid_sets=[dval], valid_names=["ranker_val"],
        callbacks=[
            lgb.early_stopping(lgbm["early_stopping_rounds"]),
            lgb.log_evaluation(25),
        ],
    )

    booster.save_model(str(art / "ranker.txt"))

    # feature store for eval/serving is rebuilt on the FULL train window
    FeatureBuilder.fit(train, items, n_users, n_items, rc["top_genres"]).save(
        art / "feature_store.npz"
    )

    # importance: instant interview material
    imp = sorted(
        zip(fb_train.feature_names, booster.feature_importance(importance_type="gain")),
        key=lambda t: -t[1],
    )
    with open(res / "feature_importance.csv", "w", encoding="utf-8") as f:
        f.write("feature,gain\n")
        for name, gain in imp:
            f.write(f"{name},{gain:.2f}\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = lgb.plot_importance(booster, importance_type="gain", max_num_features=15)
        ax.figure.tight_layout()
        ax.figure.savefig(res / "feature_importance.png", dpi=150)
    except Exception as e:  # plot is nice-to-have; csv above is the record
        print(f"skipping importance plot: {e}")

    print("top features by gain:")
    for name, gain in imp[:8]:
        print(f"  {name:>24}  {gain:12.1f}")
    print(f"saved {art / 'ranker.txt'} and {art / 'feature_store.npz'}")


if __name__ == "__main__":
    main()
