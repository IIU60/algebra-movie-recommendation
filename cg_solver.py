"""
Conjugate Gradient solver for the movie-rating normal equations.

This implements CG on the system

    (A^T A + lam^2 I) w = A^T b

using matrix-vector products with A and A^T rather than forming A^T A
explicitly. It is designed to work with the design matrix A_design built
from the ratings data.
"""

import time
from typing import Dict, Iterable, Tuple, Any

import numpy as np
from scipy.sparse import csr_matrix


def _normal_matvec(w: np.ndarray, A: csr_matrix, lam: float) -> np.ndarray:
    """Compute (A^T A + lam^2 I) w via matrix-vector products."""
    y = A @ w
    z = A.T @ y
    if lam != 0.0:
        z = z + (lam ** 2) * w
    return z


def conjugate_gradient(
    A: csr_matrix,
    b: np.ndarray,
    lam: float,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Run Conjugate Gradient on (A^T A + lam^2 I) w = A^T b.

    Parameters
    ----------
    A : csr_matrix
        Design matrix.
    b : ndarray
        Ratings vector.
    lam : float
        Regularization parameter.
    tol : float
        Relative tolerance on the residual norm, with respect to the initial
        residual norm.
    max_iter : int
        Maximum number of CG iterations.

    Returns
    -------
    w : ndarray
        Approximate solution.
    residuals : ndarray
        Residual norms per iteration (including iteration 0).
    iters : int
        Number of iterations performed.
    runtime : float
        Wall-clock runtime in seconds.
    """
    # Right-hand side f = A^T b
    r0 = A.T @ b

    n = A.shape[1]
    w = np.zeros(n, dtype=np.float64)

    # Initial residual r = f - M w
    r = r0 - _normal_matvec(w, A, lam)
    p = r.copy()

    residuals = [np.linalg.norm(r)]
    res0 = residuals[0] if residuals[0] > 0 else 1.0

    start = time.perf_counter()

    k = 0
    for k in range(max_iter):
        Ap = _normal_matvec(p, A, lam)

        r_dot = float(r.dot(r))
        pAp = float(p.dot(Ap))
        if pAp == 0.0:
            # Breakdown
            print("    [CG] Breakdown: p^T (A^T A + lam^2 I) p = 0")
            break

        alpha = r_dot / pAp
        w_new = w + alpha * p
        r_new = r - alpha * Ap

        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)

        # Progress logging
        if (k + 1) % 10 == 0:
            print(
                f"    [CG] iteration {k+1}, "
                f"λ={lam}, ||r_k||_2 = {residual_norm:.3e}, "
                f"rel = {residual_norm / res0:.3e}"
            )

        # Relative tolerance stopping rule
        if residual_norm / res0 < tol:
            k += 1  # account for the current iteration
            break

        beta = float(r_new.dot(r_new) / r_dot)
        p = r_new + beta * p
        w, r = w_new, r_new
    else:
        # Loop finished without break; last k is max_iter-1
        k = max_iter
        w = w_new
        r = r_new

    runtime = time.perf_counter() - start

    return w, np.array(residuals, dtype=float), k, runtime


def run_cg_for_lambdas(
    A: csr_matrix,
    b: np.ndarray,
    lambda_values: Iterable[float],
    tol: float = 1e-4,
    max_iter: int = 500,
) -> Dict[float, Dict[str, Any]]:
    """Run CG on the normal equations for several lambda values.

    Parameters
    ----------
    A : csr_matrix
        Design matrix.
    b : ndarray
        Ratings vector.
    lambda_values : iterable of float
        Regularization parameters.
    tol : float
        Relative tolerance on the residual norm.
    max_iter : int
        Maximum CG iterations.

    Returns
    -------
    results : dict
        Mapping lambda -> dict with keys:
        ``w``, ``residuals``, ``iterations``, ``time``.
    """
    results: Dict[float, Dict[str, Any]] = {}

    for lam in lambda_values:
        print(f"[CG] Starting solve for λ={lam} with tol={tol}, max_iter={max_iter}")

        w_cg, res_curve, iters, runtime = conjugate_gradient(
            A, b, lam, tol=tol, max_iter=max_iter
        )

        final_res = float(res_curve[-1]) if len(res_curve) > 0 else float("nan")
        print(
            f"[CG] λ={lam} finished: iter={iters}, time={runtime:.4f}s, "
            f"final_res={final_res:.3e}"
        )

        results[lam] = {
            "w": w_cg,
            "residuals": res_curve,
            "iterations": iters,
            "time": runtime,
        }

    return results



