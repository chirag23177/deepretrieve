"""Fire N sequential requests at the serving API, report p50/p95/p99.

Start the server first (uvicorn src.serving.app:app --port 8000), then:
    python scripts/benchmark_latency.py --n 1000
Writes results/latency.md.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent


def bench(
    session: requests.Session, base: str, endpoint: str, user_ids: list[int], n: int
) -> dict[str, float]:
    """Latency of `endpoint` over a KEPT-ALIVE connection.

    The connection reuse is not a detail — it is the measurement. With a fresh
    TCP connection per request (plain `requests.get`), Windows loopback adds a
    ~12 ms connect + delayed-ACK stall that swamps the handler: bare /health,
    which does no work, measured p50 13.9 ms, and the *cheaper* retrieval-only
    endpoint came out 2x SLOWER than the full rerank pipeline because the
    reranker's ~3 ms of CPU happened to mask the stall. Keep-alive removes the
    artifact and restores the expected ordering. Always benchmark the handler,
    not the client's connection setup.
    """
    # /health takes no path parameter; the recommend endpoints take a user id
    def url(uid: int) -> str:
        return f"{base}/{endpoint}" if endpoint == "health" else f"{base}/{endpoint}/{uid}"

    params = None if endpoint == "health" else {"k": 10}

    # warm up: first call on a new connection pays handshake + lazy imports
    session.get(url(user_ids[0]), params=params, timeout=30)
    lat_ms = []
    for i in range(n):
        uid = user_ids[i % len(user_ids)]
        t0 = time.perf_counter()
        r = session.get(url(uid), params=params, timeout=30)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    arr = np.array(lat_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(REPO_ROOT / args.processed_dir / "user_map.json", encoding="utf-8") as f:
        raw_ids = [int(k) for k in json.load(f)]
    random.seed(args.seed)
    user_ids = random.sample(raw_ids, min(500, len(raw_ids)))

    assert requests.get(f"{args.base}/health", timeout=10).status_code == 200

    rows = []
    with requests.Session() as session:
        # /health is the measurement floor: HTTP + JSON round trip with no model
        # work. Quote the endpoints net of it, or you are reporting your client.
        for endpoint in ("health", "recommend", "recommend_retrieval_only"):
            stats = bench(session, args.base, endpoint, user_ids, args.n)
            rows.append((endpoint, stats))
            print(f"{endpoint}: " + " ".join(f"{k}={v:.1f}ms" for k, v in stats.items()))

    out = REPO_ROOT / args.results_dir / "latency.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Serving latency — {args.n} sequential requests, k=10\n\n")
        f.write("| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |\n|---|---|---|---|---|\n")
        for endpoint, s in rows:
            f.write(f"| /{endpoint} | {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | {s['mean']:.1f} |\n")
        f.write(
            "\nSingle uvicorn worker, sequential requests over one kept-alive "
            "connection, measured on the host (not in Docker). `/health` is the "
            "HTTP floor — subtract it to get the model's own cost.\n"
            "\nConnection reuse is load-bearing: with a fresh TCP connection per "
            "request, Windows loopback adds a ~12 ms connect/delayed-ACK stall "
            "that dominates everything (bare `/health` measured p50 13.9 ms) and "
            "even inverts the ranking, making retrieval-only look 2x slower than "
            "the full rerank pipeline. See the README's \"What didn't work\".\n"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
