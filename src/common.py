"""Shared helpers: config loading, path resolution, seeding, dataset loading."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a paths.* entry relative to the repo root and create it."""
    p = REPO_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split(cfg: dict) -> dict:
    """Load the processed temporal split and id maps produced by preprocess.py."""
    proc = resolve(cfg, "processed_dir")
    out = {
        name: pd.read_parquet(proc / f"{name}.parquet")
        for name in ("train", "val", "test")
    }
    with open(proc / "user_map.json", "r", encoding="utf-8") as f:
        out["user_map"] = {int(k): v for k, v in json.load(f).items()}
    with open(proc / "item_map.json", "r", encoding="utf-8") as f:
        out["item_map"] = {int(k): v for k, v in json.load(f).items()}
    out["n_users"] = len(out["user_map"])
    out["n_items"] = len(out["item_map"])
    items_path = proc / "items.parquet"
    if items_path.exists():
        out["items"] = pd.read_parquet(items_path)
    return out


def user_positives(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Map user index -> sorted array of item indices in this dataframe."""
    return {
        u: g.to_numpy()
        for u, g in df.groupby("user_idx")["item_idx"]
    }
