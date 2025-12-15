"""
Quick script to test SciPy MINRES and GMRES on a sparse normal-equations system.

We build a small random sparse matrix A, form the normal equations

    (A^T A + λ^2 I) x = A^T b,

and solve them with SciPy's minres and gmres using LinearOperator.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sprand
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import (
    LinearOperator,
    minres as sp_minres,
    gmres as sp_gmres,
    spsolve,
)


def build_normal_system(A: csr_matrix, lam: float) -> LinearOperator:
    """Return matrix-free LinearOperator for M x = (A^T A + lam**2 I) x."""
    n = A.shape[1]
    lam2 = float(lam) ** 2

    def matvec(x):
        y = A @ x
        z = A.T @ y
        if lam2 != 0.0:
            z = z + lam2 * x
        return z

    return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)


def build_rhs(A: csr_matrix, b: np.ndarray) -> np.ndarray:
    """Compute f = A^T b."""
    return (A.T @ b).astype(np.float64)


def minres_scipy(M: LinearOperator, f: np.ndarray, x0=None, tol=1e-6, maxiter=500):
    """Wrapper around scipy.sparse.linalg.minres with residual tracking.

    Uses the SciPy 1.14+ MINRES API, which accepts ``rtol`` but not ``tol``/``atol``.
    """
    residuals = []

    def callback(xk):
        r = f - M @ xk
        residuals.append(np.linalg.norm(r))

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
    return x, residuals, iters, info


def gmres_scipy(M: LinearOperator, f: np.ndarray, x0=None, tol=1e-6, maxiter=500, restart=50):
    """Wrapper around scipy.sparse.linalg.gmres with residual tracking.

    Uses the SciPy 1.14+ GMRES API, which accepts ``rtol``/``atol`` but not ``tol``.
    """
    residuals = []

    def callback(rk):
        # SciPy's GMRES can call callback with the residual norm directly
        if np.isscalar(rk):
            residuals.append(abs(rk))
        else:
            residuals.append(np.linalg.norm(rk))

    x, info = sp_gmres(
        M,
        f,
        x0=x0,
        rtol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=callback,
        callback_type="pr_norm",
    )
    iters = len(residuals)
    if iters == 0:
        residuals = [np.linalg.norm(f - M @ x)]
    return x, residuals, iters, info


def main():
    # Small random sparse test problem
    m, n = 500, 200
    density = 0.02
    lam = 0.1

    rng = np.random.default_rng(42)
    A = sprand(m, n, density=density, format="csr", dtype=np.float64, random_state=rng)

    # True solution and rhs
    x_true = rng.standard_normal(n)
    b = A @ x_true

    M_op = build_normal_system(A, lam)
    f = build_rhs(A, b)

    # Direct solve on explicit normal equations for validation
    AtA = (A.T @ A).tocsr()
    AtA_reg = AtA + (lam ** 2) * eye(n, format="csr", dtype=np.float64)
    x_direct = spsolve(AtA_reg, f)

    x0 = np.zeros(n, dtype=np.float64)

    print("Running SciPy MINRES...")
    x_mr, res_mr, it_mr, info_mr = minres_scipy(M_op, f, x0=x0, tol=1e-8, maxiter=5000)
    rel_err_mr = np.linalg.norm(x_mr - x_direct) / np.linalg.norm(x_direct)
    print(f"  MINRES:  iters={it_mr}, final_res={res_mr[-1]:.3e}, info={info_mr}, rel_err={rel_err_mr:.3e}")

    print("Running SciPy GMRES...")
    x_gm, res_gm, it_gm, info_gm = gmres_scipy(M_op, f, x0=x0, tol=1e-4, maxiter=5000, restart=50)
    rel_err_gm = np.linalg.norm(x_gm - x_direct) / np.linalg.norm(x_direct)
    print(f"  GMRES:   iters={it_gm}, final_res={res_gm[-1]:.3e}, info={info_gm}, rel_err={rel_err_gm:.3e}")

    # Plot residual norms vs iteration on a semilog scale
    it_mr_arr = np.arange(1, len(res_mr) + 1)
    it_gm_arr = np.arange(1, len(res_gm) + 1)

    plt.figure(figsize=(6, 4))
    plt.semilogy(it_mr_arr, res_mr, marker="o", label="MINRES")
    plt.semilogy(it_gm_arr, res_gm, marker="s", label="GMRES")
    plt.xlabel("Iteration")
    # Plain-text label to avoid mathtext parsing issues
    plt.ylabel("||r_k||_2")
    plt.title("Residual norms per iteration")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


