"""Generate a tiny synthetic MovieLens-shaped dataset for pipeline smoke tests.

Writes data/raw/ml-synthetic/{ratings.csv, movies.csv} with real preference
structure (users have latent genre tastes) so the two-tower model has signal
to learn and can beat popularity even on toy data.

Usage:  python scripts/make_synthetic_data.py
Then run the pipeline with --config configs/config.smoke.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

N_USERS = 2000
N_MOVIES = 400
RATINGS_PER_USER = 60
GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
START_TS = 1_500_000_000
SPAN = 3 * 365 * 86400  # 3 years


def main() -> None:
    rng = np.random.default_rng(7)
    out = REPO_ROOT / "data" / "raw" / "ml-synthetic"
    out.mkdir(parents=True, exist_ok=True)

    # movies: 1-3 genres each, popularity follows a power law
    movie_genres = []
    for _ in range(N_MOVIES):
        k = rng.integers(1, 4)
        movie_genres.append(rng.choice(len(GENRES), size=k, replace=False))
    movie_pop = rng.zipf(1.4, size=N_MOVIES).astype(np.float64)
    movie_pop /= movie_pop.sum()

    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, N_MOVIES + 1),
            "title": [f"Synthetic Movie {i} ({2000 + i % 20})" for i in range(1, N_MOVIES + 1)],
            "genres": ["|".join(GENRES[g] for g in gl) for gl in movie_genres],
        }
    )
    movies.to_csv(out / "movies.csv", index=False)

    # genre affinity matrix: users like 2-3 genres
    user_pref = np.full((N_USERS, len(GENRES)), 0.15)
    for u in range(N_USERS):
        liked = rng.choice(len(GENRES), size=rng.integers(2, 4), replace=False)
        user_pref[u, liked] = 1.0

    rows = []
    for u in range(N_USERS):
        n = max(10, int(rng.normal(RATINGS_PER_USER, 15)))
        seen = rng.choice(N_MOVIES, size=min(n, N_MOVIES), replace=False, p=movie_pop)
        # user's activity concentrated in a random window (temporal structure)
        t_center = rng.uniform(0.15, 0.85)
        for m in seen:
            affinity = user_pref[u, movie_genres[m]].mean()
            rating = np.clip(2.0 + 2.5 * affinity + rng.normal(0, 0.7), 0.5, 5.0)
            rating = round(rating * 2) / 2
            ts = int(START_TS + SPAN * np.clip(t_center + rng.normal(0, 0.12), 0, 1))
            rows.append((u + 1, m + 1, rating, ts))

    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    ratings.to_csv(out / "ratings.csv", index=False)
    print(f"wrote {len(ratings):,} ratings for {N_USERS} users x {N_MOVIES} movies to {out}")


if __name__ == "__main__":
    main()
