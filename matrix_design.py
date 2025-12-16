"""
Matrix design construction for the movie-rating problem.

Provides a single helper to build the user–movie design matrix A_design and
ratings vector b from the ratings CSV. This is shared by both MINRES and
Conjugate Gradient solvers.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_design_matrix_and_rhs(csv_path: str):
    """Build the user-movie design matrix A_design and ratings vector b.

    Parameters
    ----------
    csv_path : str
        Path to the ratings CSV file (e.g. "data/rating.csv").

    Returns
    -------
    A_design : csr_matrix, shape (n_ratings, n_users + n_movies)
        Sparse design matrix with one column per user and one column per movie.
    b : ndarray, shape (n_ratings,)
        Ratings vector.
    """
    ratings = pd.read_csv(csv_path)

    # Categorical codes for users and movies
    user_ids = ratings["userId"].astype("category").cat.codes.to_numpy(
        dtype=np.int64, copy=False
    )
    movie_ids = ratings["movieId"].astype("category").cat.codes.to_numpy(
        dtype=np.int64, copy=False
    )

    n_users = int(user_ids.max()) + 1
    n_movies = int(movie_ids.max()) + 1
    n_ratings = len(ratings)

    # Each rating contributes 2 entries: one for the user, one for the movie
    row_idx = np.repeat(np.arange(n_ratings, dtype=np.int64), 2)
    col_idx = np.empty(2 * n_ratings, dtype=np.int64)
    col_idx[0::2] = user_ids
    col_idx[1::2] = n_users + movie_ids

    data = np.ones(2 * n_ratings, dtype=np.float64)

    A_design = csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_ratings, n_users + n_movies),
    )

    b = ratings["rating"].astype(float).to_numpy()
    return A_design, b


