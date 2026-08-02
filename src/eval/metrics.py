"""Ranking metrics. Written once, used by every stage (baselines, retrieval, ranker).

All functions operate on either a dense score matrix (n_users, n_items) or on
precomputed top-K index arrays, plus per-user ground-truth item-index arrays.

Conventions (fixed for the whole project — see README "eval protocol"):
- Recall@K  = |top-K ∩ ground_truth| / |ground_truth|
- NDCG@K    = DCG@K / IDCG@K with binary relevance,
              DCG@K  = sum_{i in top-K hits} 1 / log2(rank_i + 2)   (rank 0-based)
              IDCG@K = sum_{i=0}^{min(|gt|, K)-1} 1 / log2(i + 2)
- Users with empty ground truth are excluded by the caller (our temporal split
  guarantees every evaluated user has >= 1 test positive).
"""

from __future__ import annotations

import numpy as np


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per row, sorted by descending score. scores: (U, I)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, part], axis=1, kind="stable")
    return part[row, order]


def recall_at_k_from_topk(topk: np.ndarray, ground_truth: list[np.ndarray], k: int) -> np.ndarray:
    """Per-user Recall@K. topk: (U, >=k) sorted item indices."""
    out = np.zeros(len(ground_truth), dtype=np.float64)
    for u, gt in enumerate(ground_truth):
        if len(gt) == 0:
            raise ValueError(f"user {u} has empty ground truth; filter before calling")
        hits = np.isin(topk[u, :k], gt).sum()
        out[u] = hits / len(gt)
    return out


def ndcg_at_k_from_topk(topk: np.ndarray, ground_truth: list[np.ndarray], k: int) -> np.ndarray:
    """Per-user NDCG@K with binary relevance."""
    discounts = 1.0 / np.log2(np.arange(k) + 2.0)
    out = np.zeros(len(ground_truth), dtype=np.float64)
    for u, gt in enumerate(ground_truth):
        if len(gt) == 0:
            raise ValueError(f"user {u} has empty ground truth; filter before calling")
        rel = np.isin(topk[u, :k], gt).astype(np.float64)
        dcg = float(rel @ discounts[: len(rel)])
        idcg = float(discounts[: min(len(gt), k)].sum())
        out[u] = dcg / idcg
    return out


def recall_at_k(scores: np.ndarray, ground_truth: list[np.ndarray], k: int) -> float:
    """Mean Recall@K over users from a dense score matrix."""
    return float(recall_at_k_from_topk(topk_from_scores(scores, k), ground_truth, k).mean())


def ndcg_at_k(scores: np.ndarray, ground_truth: list[np.ndarray], k: int) -> float:
    """Mean NDCG@K over users from a dense score matrix."""
    return float(ndcg_at_k_from_topk(topk_from_scores(scores, k), ground_truth, k).mean())
