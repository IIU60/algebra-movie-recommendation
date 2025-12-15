"""
Run SciPy MINRES on the movie-rating design matrix for different lambda values
and plot residual convergence.

We solve the (regularized) normal equations

    (A^T A + λ^2 I) w = A^T b

for several λ values using the `minres_scipy` wrapper defined in
`run_scipy_solvers.py`, and plot the residual norm ||r_k||_2 vs iteration
for each λ on a semilogarithmic scale.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from run_scipy_solvers import build_normal_system, build_rhs, minres_scipy


def build_design_matrix_and_rhs(csv_path: str):
    """Build the user-movie design matrix A_design and ratings vector b."""
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


def run_minres_for_lambdas(csv_path: str, lambda_values=None, tol=1e-10, maxiter=500):
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 1.0]

    A_design, b = build_design_matrix_and_rhs(csv_path)
    n_params = A_design.shape[1]

    results = {}

    for lam in lambda_values:
        print(f"Running MINRES for λ={lam} ...")
        M_op = build_normal_system(A_design, lam)
        f = build_rhs(A_design, b)

        x0 = np.zeros(n_params, dtype=np.float64)

        start = time.perf_counter()
        w_mr, res_curve, iters, info = minres_scipy(
            M_op, f, x0=x0, tol=tol, maxiter=maxiter
        )
        runtime = time.perf_counter() - start

        print(
            f"  λ={lam} | iter={iters} | time={runtime:.4f}s | "
            f"final_res={res_curve[-1]:.3e} | info={info}"
        )

        results[lam] = {
            "w": w_mr,
            "residuals": res_curve,
            "iterations": iters,
            "time": runtime,
            "info": info,
        }

    return results


def plot_residual_convergence(results):
    plt.figure(figsize=(10, 6))

    for lam, info in results.items():
        plt.plot(info["residuals"], label=f"λ={lam}")

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    plt.title("MINRES Residual Convergence for Different λ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_runtime_and_iterations(results):
    """Bar plots of runtime and iteration count vs lambda, analogous to CG plots."""
    lams = list(results.keys())
    times = [results[lam]["time"] for lam in lams]
    iters = [results[lam]["iterations"] for lam in lams]

    x = np.arange(len(lams))

    plt.figure(figsize=(12, 5))

    # Runtime
    plt.subplot(1, 2, 1)
    plt.bar(x, times, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.xticks(x, [f"λ={lam}" for lam in lams])
    plt.ylabel("Runtime (seconds)")
    plt.title("MINRES Runtime Comparison")
    plt.grid(axis="y")

    # Iterations
    plt.subplot(1, 2, 2)
    plt.bar(x, iters, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.xticks(x, [f"λ={lam}" for lam in lams])
    plt.ylabel("Iterations")
    plt.title("MINRES Iteration Count Comparison")
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()


def main():
    csv_path = "data/rating.csv"
    lambda_values = [0.0, 0.1, 1.0]

    results = run_minres_for_lambdas(csv_path, lambda_values=lambda_values)
    plot_residual_convergence(results)
    plot_runtime_and_iterations(results)


if __name__ == "__main__":
    main()


