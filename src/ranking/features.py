"""Reranker features. ~15 deliberately interpretable features per (user, candidate).

All aggregates are computed from the TRAIN window only (the ranker label window
is carved out separately in train_ranker.py — features never see val/test).

Feature groups:
- retrieval_score            the two-tower dot product (the strongest signal)
- item popularity (log) + item mean rating + top-8 genre one-hot + item median timestamp age
- user activity (log) + user mean rating
- CROSS features (the reason the ranking stage exists — they cannot go into a
  two-tower model that must be servable with ANN):
    * user-genre affinity: dot(user's train genre distribution, item genre vector)
    * recency gap: user's last train timestamp - item's median interaction timestamp (days)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.genres import build_genre_matrix

SECONDS_PER_DAY = 86400.0


class FeatureBuilder:
    def __init__(self, arrays: dict[str, np.ndarray], feature_names: list[str]):
        self.a = arrays
        self.feature_names = feature_names

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        items: pd.DataFrame,
        n_users: int,
        n_items: int,
        top_genres: int = 8,
    ) -> "FeatureBuilder":
        item_idx = train["item_idx"].to_numpy()
        user_idx = train["user_idx"].to_numpy()

        item_pop = np.bincount(item_idx, minlength=n_items).astype(np.float64)
        item_pop_log = np.log1p(item_pop).astype(np.float32)

        global_mean = float(train["rating"].mean())
        item_mean = train.groupby("item_idx")["rating"].mean()
        item_mean_rating = np.full(n_items, global_mean, dtype=np.float32)
        item_mean_rating[item_mean.index.to_numpy()] = item_mean.to_numpy()

        item_median_ts = np.zeros(n_items, dtype=np.float64)
        med = train.groupby("item_idx")["timestamp"].median()
        item_median_ts[med.index.to_numpy()] = med.to_numpy()

        genre_full, genre_vocab = build_genre_matrix(items, n_items)
        top_idx = np.argsort(-genre_full.sum(axis=0))[:top_genres]
        top_genre_names = [genre_vocab[i] for i in top_idx]
        item_genre_top = genre_full[:, top_idx]  # (n_items, top_genres)

        user_activity = np.bincount(user_idx, minlength=n_users).astype(np.float64)
        user_activity_log = np.log1p(user_activity).astype(np.float32)

        u_mean = train.groupby("user_idx")["rating"].mean()
        user_mean_rating = np.full(n_users, global_mean, dtype=np.float32)
        user_mean_rating[u_mean.index.to_numpy()] = u_mean.to_numpy()

        user_last_ts = np.zeros(n_users, dtype=np.float64)
        last = train.groupby("user_idx")["timestamp"].max()
        user_last_ts[last.index.to_numpy()] = last.to_numpy()

        # user genre distribution over their train items, L1-normalized
        user_genre = np.zeros((n_users, genre_full.shape[1]), dtype=np.float32)
        np.add.at(user_genre, user_idx, genre_full[item_idx])
        row_sums = user_genre.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        user_genre /= row_sums

        arrays = {
            "item_pop_log": item_pop_log,
            "item_mean_rating": item_mean_rating,
            "item_median_ts": item_median_ts,
            "item_genre_top": item_genre_top,
            "item_genre_full": genre_full,
            "user_activity_log": user_activity_log,
            "user_mean_rating": user_mean_rating,
            "user_last_ts": user_last_ts,
            "user_genre": user_genre,
            "top_genre_names": np.array(top_genre_names),
        }
        feature_names = (
            ["retrieval_score", "item_pop_log", "item_mean_rating"]
            + [f"genre_{g}" for g in top_genre_names]
            + ["user_activity_log", "user_mean_rating", "user_genre_affinity", "recency_days"]
        )
        return cls(arrays, feature_names)

    def build(
        self, users: np.ndarray, items: np.ndarray, retrieval_scores: np.ndarray
    ) -> np.ndarray:
        """Feature matrix (N, F) for aligned arrays of user idx / item idx / score."""
        a = self.a
        affinity = np.einsum(
            "ng,ng->n", a["user_genre"][users], a["item_genre_full"][items]
        ).astype(np.float32)
        recency_days = (
            (a["user_last_ts"][users] - a["item_median_ts"][items]) / SECONDS_PER_DAY
        ).astype(np.float32)
        cols = [
            retrieval_scores.astype(np.float32),
            a["item_pop_log"][items],
            a["item_mean_rating"][items],
            *a["item_genre_top"][items].T,
            a["user_activity_log"][users],
            a["user_mean_rating"][users],
            affinity,
            recency_days,
        ]
        return np.column_stack(cols).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, feature_names=np.array(self.feature_names), **self.a
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureBuilder":
        data = np.load(path, allow_pickle=False)
        feature_names = [str(x) for x in data["feature_names"]]
        arrays = {k: data[k] for k in data.files if k != "feature_names"}
        return cls(arrays, feature_names)
