# Serving latency — 1000 sequential requests, k=10

| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|
| /health | 1.0 | 1.6 | 2.0 | 1.1 |
| /recommend | 5.5 | 6.2 | 6.8 | 5.5 |
| /recommend_retrieval_only | 1.8 | 2.6 | 2.9 | 1.9 |

Single uvicorn worker, sequential requests over one kept-alive connection, measured on the host (not in Docker). `/health` is the HTTP floor — subtract it to get the model's own cost.

Connection reuse is load-bearing: with a fresh TCP connection per request, Windows loopback adds a ~12 ms connect/delayed-ACK stall that dominates everything (bare `/health` measured p50 13.9 ms) and even inverts the ranking, making retrieval-only look 2x slower than the full rerank pipeline. See the README's "What didn't work".
