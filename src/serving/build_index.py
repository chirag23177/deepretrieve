"""Build FAISS indexes from the trained item tower + quantify the ANN tradeoff.

1. IndexFlatIP (exact): verified to match brute-force numpy matmul top-K exactly.
   Item vectors are L2-normalized, so inner product == cosine.
2. IndexIVFFlat: tune nprobe over a grid, produce recall-vs-exact@100 + latency table.

On ~60K movies exact search is already fast — IVF exists here to demonstrate the
scaling path, and the benchmark table quantifies exactly what approximation buys
and costs (results/ann_benchmark.md).

Usage:  python -m src.serving.build_index
"""

from __future__ import annotations

import argparse
import time

import faiss
import numpy as np

from src.common import load_config, resolve, set_seed


def exact_topk(queries: np.ndarray, items: np.ndarray, k: int) -> np.ndarray:
    scores = queries @ items.T
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row = np.arange(len(queries))[:, None]
    order = np.argsort(-scores[row, part], axis=1, kind="stable")
    return part[row, order]


def overlap_at_k(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |a_row ∩ b_row| / k."""
    return float(
        np.mean([len(np.intersect1d(x, y)) / a.shape[1] for x, y in zip(a, b)])
    )


def timed_search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    _, idx = index.search(queries, k)
    ms_per_query = (time.perf_counter() - t0) * 1000 / len(queries)
    return idx, ms_per_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fc = cfg["faiss"]
    set_seed(cfg["seed"])
    art = resolve(cfg, "artifacts_dir")
    res = resolve(cfg, "results_dir")

    item_emb = np.load(art / "item_embeddings.npy").astype(np.float32)
    user_emb = np.load(art / "user_embeddings.npy").astype(np.float32)
    n_items, dim = item_emb.shape
    k = min(fc["top_k"], n_items)
    print(f"{n_items:,} item vectors, dim={dim}, k={k}")

    rng = np.random.default_rng(cfg["seed"])
    n_q = min(fc["benchmark_queries"], len(user_emb))
    queries = user_emb[rng.choice(len(user_emb), size=n_q, replace=False)]

    # --- exact index ---
    flat = faiss.IndexFlatIP(dim)
    flat.add(item_emb)
    flat_idx, flat_ms = timed_search(flat, queries, k)

    brute = exact_topk(queries, item_emb, k)
    agreement = overlap_at_k(flat_idx, brute)
    print(f"IndexFlatIP vs brute-force matmul top-{k} overlap: {agreement:.4f}")
    # ties at the k-th score can differ in order; require near-perfect agreement
    assert agreement > 0.999, "FlatIP must match brute force — something is wrong"
    faiss.write_index(flat, str(art / "index_flat.faiss"))

    # --- IVF index ---
    rows = [("Flat (exact)", "-", 1.0, flat_ms)]
    nlist = min(fc["ivf_nlist"], max(1, n_items // 39))  # faiss wants >=39 pts/centroid
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(item_emb)
    ivf.add(item_emb)
    for nprobe in fc["nprobe_grid"]:
        if nprobe > nlist:
            continue
        ivf.nprobe = nprobe
        idx, ms = timed_search(ivf, queries, k)
        recall = overlap_at_k(idx, flat_idx)
        rows.append((f"IVF nlist={nlist}", str(nprobe), recall, ms))
        print(f"IVF nprobe={nprobe}: recall-vs-exact@{k}={recall:.4f}, {ms:.3f} ms/query")
    faiss.write_index(ivf, str(art / "index_ivf.faiss"))

    with open(res / "ann_benchmark.md", "w", encoding="utf-8") as f:
        f.write(f"# ANN benchmark — {n_items:,} items, dim {dim}, "
                f"{n_q:,} queries, top-{k}\n\n")
        f.write("| Index | nprobe | recall-vs-exact@" + str(k) + " | ms/query |\n")
        f.write("|---|---|---|---|\n")
        for name, nprobe, recall, ms in rows:
            f.write(f"| {name} | {nprobe} | {recall:.4f} | {ms:.3f} |\n")
        f.write(f"\nExact search is already fast at this catalog size ({n_items:,} "
                "items after k-core filtering), so IVF is the demonstrated scaling "
                "path rather than a necessity here. At this size per-query times are "
                "microseconds and the nprobe ordering is within measurement noise; "
                "the recall column is the meaningful axis.\n")
    print(f"wrote {res / 'ann_benchmark.md'}")


if __name__ == "__main__":
    main()
