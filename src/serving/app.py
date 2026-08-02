"""FastAPI serving: user_id -> user vector -> FAISS top-K -> features -> LightGBM -> top-k.

Everything loads once at startup: user embeddings (precomputable because the user
tower is id-only), FAISS index, LightGBM booster, feature store, item titles,
and each user's train history (seen items are filtered out of recommendations).

Endpoints:
    GET /health
    GET /recommend/{user_id}?k=10            full two-stage pipeline
    GET /recommend_retrieval_only/{user_id}?k=10   retrieval stage only (to quote both latencies)

Run:    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
Set DEEPRETRIEVE_CONFIG to serve from an alternative config (e.g. the smoke setup).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.common import load_config, load_split, resolve
from src.ranking.features import FeatureBuilder

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(os.environ.get("DEEPRETRIEVE_CONFIG"))
    art = resolve(cfg, "artifacts_dir")
    split = load_split(cfg)

    state["cfg"] = cfg
    state["user_emb"] = np.load(art / "user_embeddings.npy").astype(np.float32)
    state["index"] = faiss.read_index(str(art / "index_flat.faiss"))
    state["booster"] = lgb.Booster(model_file=str(art / "ranker.txt"))
    state["fb"] = FeatureBuilder.load(art / "feature_store.npz")
    state["user_map"] = split["user_map"]  # raw userId -> user_idx
    state["n_items"] = split["n_items"]

    items = split["items"].set_index("item_idx")
    state["titles"] = items["title"].to_dict()
    state["movie_ids"] = items["movieId"].to_dict()

    train = split["train"]
    state["seen"] = {
        int(u): set(g.to_numpy()) for u, g in train.groupby("user_idx")["item_idx"]
    }
    print(f"loaded: {len(state['user_map']):,} users, {state['n_items']:,} items")
    yield
    state.clear()


app = FastAPI(title="deepretrieve", lifespan=lifespan)


def _retrieve(user_id: int, n_candidates: int) -> tuple[int, np.ndarray, np.ndarray]:
    user_idx = state["user_map"].get(user_id)
    if user_idx is None:
        raise HTTPException(404, f"unknown user_id {user_id} (not in train window)")
    seen = state["seen"].get(user_idx, set())
    fetch = min(n_candidates + len(seen), state["n_items"])
    vec = state["user_emb"][user_idx : user_idx + 1]
    scores, idx = state["index"].search(vec, fetch)
    keep = [i for i, item in enumerate(idx[0]) if item not in seen][:n_candidates]
    return user_idx, idx[0][keep], scores[0][keep]


def _payload(items: np.ndarray, scores: np.ndarray, k: int) -> list[dict]:
    return [
        {
            "movieId": int(state["movie_ids"][int(i)]),
            "title": state["titles"][int(i)],
            "score": round(float(s), 4),
        }
        for i, s in zip(items[:k], scores[:k])
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10) -> dict:
    cfg = state["cfg"]
    user_idx, cand, retr = _retrieve(user_id, cfg["serving"]["retrieve_k"])
    feats = state["fb"].build(
        np.full(len(cand), user_idx, dtype=np.int64), cand.astype(np.int64), retr
    )
    ranked_scores = state["booster"].predict(feats)
    order = np.argsort(-ranked_scores)
    return {
        "user_id": user_id,
        "k": k,
        "stage": "retrieval+rerank",
        "results": _payload(cand[order], ranked_scores[order], k),
    }


@app.get("/recommend_retrieval_only/{user_id}")
def recommend_retrieval_only(user_id: int, k: int = 10) -> dict:
    _, cand, retr = _retrieve(user_id, max(k, 1))
    return {
        "user_id": user_id,
        "k": k,
        "stage": "retrieval",
        "results": _payload(cand, retr, k),
    }
