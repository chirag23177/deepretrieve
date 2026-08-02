# Offline evaluation — full-catalog ranking, temporal test split

| System     |   Recall@10 |   Recall@50 |   Recall@100 |   NDCG@10 |   NDCG@50 |   NDCG@100 |
|:-----------|------------:|------------:|-------------:|----------:|----------:|-----------:|
| Popularity |      0.0265 |      0.0861 |       0.1333 |    0.054  |    0.0662 |     0.0799 |
| Item-kNN   |      0.0356 |      0.1116 |       0.1755 |    0.0691 |    0.0837 |     0.1026 |

Protocol: for each test user, score all items minus their train history; ground truth = test-window positives. No sampled negatives.
