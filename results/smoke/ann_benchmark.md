# ANN benchmark — 298 items, dim 32, 500 queries, top-100

| Index | nprobe | recall-vs-exact@100 | ms/query |
|---|---|---|---|
| Flat (exact) | - | 1.0000 | 0.007 |
| IVF nlist=7 | 1 | 0.2343 | 0.006 |
| IVF nlist=7 | 2 | 0.4104 | 0.005 |
| IVF nlist=7 | 4 | 0.7192 | 0.005 |

Exact search is already fast at this catalog size; IVF is the demonstrated scaling path, not a necessity at 60K items.
