"""Popularity baseline: everyone gets the same ranking by train interaction count."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityScorer:
    name = "Popularity"

    def __init__(self, train: pd.DataFrame, n_items: int):
        counts = np.bincount(train["item_idx"].to_numpy(), minlength=n_items)
        self.scores = counts.astype(np.float32)

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(users), 1))
