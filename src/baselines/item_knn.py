"""Item-kNN baseline: cosine similarity on the item-user co-occurrence matrix.

score(u, i) = sum over j in u's train items of cos(i, j)

The full 60K x 60K similarity matrix would be ~14 GB dense, so we compute it in
chunks and truncate to the top-N neighbors per item (config: baselines.item_knn).
The truncated sparse matrix is cached in artifacts/ because it takes a few
minutes to build on the full dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm


def build_user_item_matrix(train: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (train["user_idx"].to_numpy(), train["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )


def build_similarity(
    ui: sp.csr_matrix, top_neighbors: int, chunk_size: int
) -> sp.csr_matrix:
    """Truncated item-item cosine similarity (n_items, n_items), zero diagonal."""
    n_items = ui.shape[1]
    # L2-normalize item columns so that X^T X = cosine similarity
    norms = np.sqrt(ui.multiply(ui).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    ui_norm = ui @ sp.diags(1.0 / norms)
    ui_norm_csc = ui_norm.tocsc()

    rows, cols, vals = [], [], []
    for start in tqdm(range(0, n_items, chunk_size), desc="item-kNN similarity"):
        end = min(start + chunk_size, n_items)
        sim = (ui_norm_csc[:, start:end].T @ ui_norm).tocsr()  # (chunk, n_items)
        for local in range(end - start):
            row = sim.getrow(local)
            idx, val = row.indices, row.data
            self_pos = idx == (start + local)
            idx, val = idx[~self_pos], val[~self_pos]
            if len(val) > top_neighbors:
                keep = np.argpartition(-val, top_neighbors - 1)[:top_neighbors]
                idx, val = idx[keep], val[keep]
            rows.append(np.full(len(idx), start + local, dtype=np.int32))
            cols.append(idx.astype(np.int32))
            vals.append(val.astype(np.float32))

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items),
    )


class ItemKNNScorer:
    name = "Item-kNN"

    def __init__(
        self,
        train: pd.DataFrame,
        n_users: int,
        n_items: int,
        top_neighbors: int,
        chunk_size: int,
        cache_path: Path | None = None,
    ):
        self.ui = build_user_item_matrix(train, n_users, n_items)
        if cache_path is not None and cache_path.exists():
            print(f"loading cached item-kNN similarity from {cache_path}")
            self.sim = sp.load_npz(cache_path)
        else:
            self.sim = build_similarity(self.ui, top_neighbors, chunk_size)
            if cache_path is not None:
                sp.save_npz(cache_path, self.sim)

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return np.asarray((self.ui[users] @ self.sim).todense(), dtype=np.float32)
