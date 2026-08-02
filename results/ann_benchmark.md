# ANN benchmark — 17,421 items, dim 64, 2,000 queries, top-100

| Index | nprobe | recall-vs-exact@100 | ms/query |
|---|---|---|---|
| Flat (exact) | - | 1.0000 | 0.016 |
| IVF nlist=446 | 8 | 0.8617 | 0.006 |
| IVF nlist=446 | 32 | 0.9935 | 0.005 |
| IVF nlist=446 | 128 | 0.9999 | 0.008 |

Exact search is already fast at this catalog size; IVF is the demonstrated scaling path, not a necessity at 60K items.
