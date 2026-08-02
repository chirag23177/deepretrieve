# Offline evaluation — full-catalog ranking, temporal test split

| System     |   Recall@10 |   Recall@50 |   Recall@100 |   NDCG@10 |   NDCG@50 |   NDCG@100 |
|:-----------|------------:|------------:|-------------:|----------:|----------:|-----------:|
| Popularity |      0.2249 |      0.7578 |       0.9211 |    0.1632 |    0.3314 |     0.3724 |
| Item-kNN   |      0.3719 |      0.7966 |       0.8934 |    0.2922 |    0.4283 |     0.4534 |

Protocol: for each test user, score all items minus their train history; ground truth = test-window positives. No sampled negatives.
