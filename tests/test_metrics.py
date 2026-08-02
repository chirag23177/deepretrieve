"""Hand-computed test cases for eval/metrics.py.

Every expected value below was computed by hand (shown in comments). A bug in
these functions silently invalidates every number in the project, so these
tests are the first thing that must pass.
"""

import numpy as np
import pytest

from src.eval.metrics import (
    ndcg_at_k,
    ndcg_at_k_from_topk,
    recall_at_k,
    recall_at_k_from_topk,
    topk_from_scores,
)


def test_topk_from_scores_orders_by_descending_score():
    scores = np.array([[0.4, 0.9, 0.1, 0.7]])
    # 0.9 (item 1) > 0.7 (item 3) > 0.4 (item 0) > 0.1 (item 2)
    np.testing.assert_array_equal(topk_from_scores(scores, 4), [[1, 3, 0, 2]])
    np.testing.assert_array_equal(topk_from_scores(scores, 2), [[1, 3]])


def test_perfect_ranking_gives_1():
    scores = np.array([[0.9, 0.5, 0.1]])
    gt = [np.array([0])]
    assert recall_at_k(scores, gt, 1) == 1.0
    assert ndcg_at_k(scores, gt, 1) == 1.0


def test_ndcg_single_relevant_at_rank_2():
    # ranking = [0, 1, 2]; the only relevant item (2) lands at 0-based rank 2.
    # DCG  = 1 / log2(2 + 2) = 1 / 2 = 0.5
    # IDCG = 1 / log2(0 + 2) = 1
    # NDCG@3 = 0.5
    scores = np.array([[0.9, 0.5, 0.1]])
    gt = [np.array([2])]
    assert ndcg_at_k(scores, gt, 3) == pytest.approx(0.5)
    # the relevant item is outside the top-2 cutoff -> 0
    assert ndcg_at_k(scores, gt, 2) == 0.0
    assert recall_at_k(scores, gt, 2) == 0.0


def test_recall_counts_hits_over_ground_truth_size():
    # ranking = [3, 0, 1, 2]; gt = {0, 2}; top-2 = [3, 0] -> 1 hit of 2 -> 0.5
    scores = np.array([[0.6, 0.3, 0.2, 0.8]])
    gt = [np.array([0, 2])]
    assert recall_at_k(scores, gt, 2) == pytest.approx(0.5)
    # top-4 contains both -> 1.0
    assert recall_at_k(scores, gt, 4) == 1.0


def test_recall_when_ground_truth_larger_than_k():
    # 5 relevant items, k=2, both top-2 are hits -> 2/5 (denominator is |gt|, not k)
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.1]])
    gt = [np.array([0, 1, 2, 3, 4])]
    assert recall_at_k(scores, gt, 2) == pytest.approx(2 / 5)


def test_ndcg_two_relevant_hand_computed():
    # scores -> ranking [1, 3, 0, 2]; gt = {0, 1}
    # hits at 0-based ranks 0 (item 1) and 2 (item 0)
    # DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309297535714575 = 1.6309297535714575
    # NDCG@4 = 1.5 / 1.6309297535714575 = 0.9197207891481876
    scores = np.array([[0.4, 0.9, 0.1, 0.7]])
    gt = [np.array([0, 1])]
    assert ndcg_at_k(scores, gt, 4) == pytest.approx(0.9197207891481876)


def test_mean_over_users():
    # user 0: perfect (recall 1.0); user 1: miss (recall 0.0) -> mean 0.5
    scores = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.9, 0.1, 0.1],
        ]
    )
    gt = [np.array([0]), np.array([2])]
    assert recall_at_k(scores, gt, 1) == pytest.approx(0.5)
    # NDCG: user 0 = 1.0; user 1: hit at rank 2 -> 0.5 -> mean 0.75
    assert ndcg_at_k(scores, gt, 3) == pytest.approx(0.75)


def test_topk_variants_match_score_matrix_variants():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(8, 30))
    gt = [rng.choice(30, size=3, replace=False) for _ in range(8)]
    topk = topk_from_scores(scores, 10)
    assert recall_at_k(scores, gt, 10) == pytest.approx(
        recall_at_k_from_topk(topk, gt, 10).mean()
    )
    assert ndcg_at_k(scores, gt, 10) == pytest.approx(
        ndcg_at_k_from_topk(topk, gt, 10).mean()
    )


def test_empty_ground_truth_raises():
    scores = np.array([[0.9, 0.1]])
    with pytest.raises(ValueError):
        recall_at_k(scores, [np.array([], dtype=int)], 1)
    with pytest.raises(ValueError):
        ndcg_at_k(scores, [np.array([], dtype=int)], 1)
