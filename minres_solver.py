"""
Standalone MINRES solver utilities for the movie-rating normal equations.

This module depends only on SciPy and NumPy and can be used independently
of the original demo script.
"""

import time
from typing import Dict, Iterable, Tuple, Any

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import LinearOperator, minres as sp_minres


def build_normal_system(A: csr_matrix, lam: float) -> LinearOperator:
    """Return matrix-free LinearOperator for M x = (A^T A + lam**2 I) x."""
    n = A.shape[1]
    lam2 = float(lam) ** 2

    def matvec(x: np.ndarray) -> np.ndarray:
        y = A @ x
        z = A.T @ y
        if lam2 != 0.0:
            z = z + lam2 * x
        return z

    return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)


def build_rhs(A: csr_matrix, b: np.ndarray) -> np.ndarray:
    """Compute f = A^T b."""
    return (A.T @ b).astype(np.float64)


def minres_scipy(
    M: LinearOperator,
    f: np.ndarray,
    x0=None,
    tol: float = 1e-6,
    maxiter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Wrapper around scipy.sparse.linalg.minres with residual tracking.

    Uses the SciPy 1.14+ MINRES API, which accepts ``rtol`` but not ``tol``/``atol``.
    """
    residuals = []

    iter_count = {"k": 0}

    def callback(xk):
        # Compute true residual norm
        r = f - M @ xk
        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)

        iter_count["k"] += 1
        k = iter_count["k"]
        # Light-weight progress logging
        if k % 10 == 0:
            print(f"    MINRES iteration {k}, ||r_k||_2 = {res_norm:.3e}")

    x, info = sp_minres(
        M,
        f,
        x0=x0,
        rtol=tol,
        maxiter=maxiter,
        callback=callback,
    )
    iters = len(residuals)
    if iters == 0:
        residuals = [np.linalg.norm(f - M @ x)]
        iters = 1
    return x, np.array(residuals, dtype=float), iters, info


def run_minres_for_lambdas(
    A: csr_matrix,
    b: np.ndarray,
    lambda_values: Iterable[float],
    tol: float = 1e-10,
    maxiter: int = 500,
) -> Dict[float, Dict[str, Any]]:
    """Run MINRES on the normal equations for several lambda values.

    Parameters
    ----------
    A : csr_matrix
        Design matrix.
    b : ndarray
        Ratings vector.
    lambda_values : iterable of float
        Regularization parameters.
    tol : float
        Relative tolerance passed to SciPy MINRES (rtol).
    maxiter : int
        Maximum number of MINRES iterations.

    Returns
    -------
    results : dict
        Mapping lambda -> dict with keys:
        ``w``, ``residuals``, ``iterations``, ``time``, ``info``.
    """
    n_params = A.shape[1]
    results: Dict[float, Dict[str, Any]] = {}

    for lam in lambda_values:
        print(f"[MINRES] Starting solve for λ={lam} with tol={tol}, maxiter={maxiter}")
        M_op = build_normal_system(A, lam)
        f = build_rhs(A, b)

        x0 = np.zeros(n_params, dtype=np.float64)

        start = time.perf_counter()
        w_mr, res_curve, iters, info = minres_scipy(
            M_op, f, x0=x0, tol=tol, maxiter=maxiter
        )
        runtime = time.perf_counter() - start

        final_res = float(res_curve[-1]) if len(res_curve) > 0 else float("nan")
        print(
            f"[MINRES] λ={lam} finished: iter={iters}, time={runtime:.4f}s, "
            f"final_res={final_res:.3e}, info={info}"
        )

        results[lam] = {
            "w": w_mr,
            "residuals": res_curve,
            "iterations": iters,
            "time": runtime,
            "info": info,
        }

    return results



