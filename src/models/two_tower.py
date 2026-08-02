"""Two-tower retrieval model.

UserTower:  user_id -> Embedding(n_users, D) -> MLP -> L2-normalize
ItemTower:  item_id -> Embedding(n_items, D)
            genres  -> multi-hot -> Linear(n_genres -> genre_proj)
            concat  -> MLP -> L2-normalize
score(u, i) = dot(user_vec, item_vec) / temperature

Design notes (interview ammo, also in README):
- No cross features: item vectors must be precomputable so FAISS can serve them.
  Cross features live in the LightGBM reranker — that is WHY the system is two-stage.
- L2-normalize + dot = cosine: bounds scores, stabilizes the in-batch softmax, and
  makes FAISS inner-product search equivalent to cosine search.
- Genres only on the item side; the user side stays id-only (user metadata is a
  stated extension, not a gap).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    def __init__(self, n_users: int, emb_dim: int, mlp_dims: list[int]):
        super().__init__()
        self.embedding = nn.Embedding(n_users, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.mlp = _mlp([emb_dim, *mlp_dims])

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.embedding(user_ids)), dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        emb_dim: int,
        genre_matrix: torch.Tensor,  # (n_items, n_genres) multi-hot
        genre_proj_dim: int,
        mlp_dims: list[int],
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.register_buffer("genre_matrix", genre_matrix)
        self.genre_proj = nn.Linear(genre_matrix.shape[1], genre_proj_dim)
        self.mlp = _mlp([emb_dim + genre_proj_dim, *mlp_dims])

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.embedding(item_ids)
        genre_emb = self.genre_proj(self.genre_matrix[item_ids])
        return F.normalize(self.mlp(torch.cat([id_emb, genre_emb], dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        genre_matrix: torch.Tensor,
        emb_dim: int = 64,
        user_mlp_dims: list[int] = (128, 64),
        item_mlp_dims: list[int] = (128, 64),
        genre_proj_dim: int = 32,
        temperature: float = 0.05,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        self.user_tower = UserTower(n_users, emb_dim, list(user_mlp_dims))
        self.item_tower = ItemTower(
            n_items, emb_dim, genre_matrix, genre_proj_dim, list(item_mlp_dims)
        )
        log_t = torch.tensor(math.log(temperature), dtype=torch.float32)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temperature", log_t)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)

    def in_batch_logits(
        self,
        user_vecs: torch.Tensor,
        item_vecs: torch.Tensor,
        log_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, B) score matrix; diagonal = positives, off-diagonal = in-batch negatives.

        logQ correction: in-batch negatives are sampled proportional to item
        popularity, which over-penalizes popular items. Subtracting log(p_item)
        from each item's logits (sampled-softmax correction) fixes the bias.
        """
        logits = (user_vecs @ item_vecs.T) / self.temperature
        if log_q is not None:
            logits = logits - log_q.unsqueeze(0)
        return logits

    @torch.no_grad()
    def all_item_embeddings(self, batch_size: int = 8192) -> torch.Tensor:
        device = next(self.parameters()).device
        n = self.item_tower.embedding.num_embeddings
        out = []
        for start in range(0, n, batch_size):
            ids = torch.arange(start, min(start + batch_size, n), device=device)
            out.append(self.item_tower(ids))
        return torch.cat(out)

    @torch.no_grad()
    def all_user_embeddings(self, batch_size: int = 8192) -> torch.Tensor:
        device = next(self.parameters()).device
        n = self.user_tower.embedding.num_embeddings
        out = []
        for start in range(0, n, batch_size):
            ids = torch.arange(start, min(start + batch_size, n), device=device)
            out.append(self.user_tower(ids))
        return torch.cat(out)
