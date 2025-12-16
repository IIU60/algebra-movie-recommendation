"""
Master script to run MINRES and CG experiments on the movie-rating design matrix.

This script:
  * Builds the design matrix A_design and ratings vector b.
  * Runs MINRES and Conjugate Gradient for a list of lambda values.
  * Prints verbose progress updates.
  * Saves plots and full results dictionaries into a unique experiments
    subdirectory for each run.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

from matrix_design import build_design_matrix_and_rhs
from minres_solver import run_minres_for_lambdas
from cg_solver import run_cg_for_lambdas
from plot_residuals import plot_method_residuals, plot_runtime_and_iterations


def create_experiment_directory(root: str = "experiments") -> str:
    """Create and return a unique experiment subdirectory path."""
    os.makedirs(root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(root, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_results_dict(
    results: Dict[float, Dict[str, Any]],
    path_npz: str,
    path_pkl: str,
) -> None:
    """Save results dictionary in both NPZ and pickle formats."""
    # For NPZ, we store arrays and scalars in a structured way.
    npz_payload: Dict[str, Any] = {}
    for lam, info in results.items():
        key_prefix = f"lam_{lam}"
        npz_payload[f"{key_prefix}_w"] = np.asarray(info["w"])
        npz_payload[f"{key_prefix}_residuals"] = np.asarray(info["residuals"])
        npz_payload[f"{key_prefix}_iterations"] = np.array(info["iterations"])
        npz_payload[f"{key_prefix}_time"] = np.array(info["time"])
        if "info" in info:
            npz_payload[f"{key_prefix}_info"] = np.array(info["info"])

    np.savez(path_npz, **npz_payload)

    # Full Python dict in pickle format
    import pickle

    with open(path_pkl, "wb") as f:
        pickle.dump(results, f)


def main() -> None:
    # Configuration
    csv_path = "data/rating.csv"
    lambda_values: List[float] = [0.0, 0.1, 1.0, 5]

    # External comparison tolerance: both methods are measured against this
    # relative residual level ||r_k|| / ||r_0|| <= target_rel.
    target_rel = 1e-4

    # Internal solver tolerances (set tighter than target_rel)
    minres_tol_internal = 1e-10
    minres_maxiter = 500

    cg_internal_tol = 1e-8
    cg_maxiter = 500

    print("Building design matrix and RHS from ratings data...")
    A_design, b = build_design_matrix_and_rhs(csv_path)
    print(f"  A_design shape = {A_design.shape}, nnz = {A_design.nnz}")

    exp_dir = create_experiment_directory()
    print(f"Experiment outputs will be saved under: {exp_dir}")

    # Run MINRES
    print("Running MINRES for all lambda values...")
    minres_results = run_minres_for_lambdas(
        A_design,
        b,
        lambda_values=lambda_values,
        tol=minres_tol_internal,
        maxiter=minres_maxiter,
        target_rel=target_rel,
    )

    # Run CG
    print("Running Conjugate Gradient for all lambda values...")
    cg_results = run_cg_for_lambdas(
        A_design,
        b,
        lambda_values=lambda_values,
        target_rel=target_rel,
        max_iter=cg_maxiter,
        internal_tol=cg_internal_tol,
    )

    # Plot residual convergence
    print("Generating residual convergence plots...")
    minres_residual_plot = os.path.join(exp_dir, "minres_residuals.png")
    cg_residual_plot = os.path.join(exp_dir, "cg_residuals.png")

    plot_method_residuals(
        minres_results,
        method_name="MINRES",
        save_path=minres_residual_plot,
    )
    plot_method_residuals(
        cg_results,
        method_name="Conjugate Gradient",
        save_path=cg_residual_plot,
    )

    # Plot runtime and iterations
    print("Generating runtime and iteration comparison plots...")
    minres_perf_plot = os.path.join(exp_dir, "minres_performance.png")
    cg_perf_plot = os.path.join(exp_dir, "cg_performance.png")

    plot_runtime_and_iterations(
        minres_results,
        method_name="MINRES",
        save_path=minres_perf_plot,
    )
    plot_runtime_and_iterations(
        cg_results,
        method_name="Conjugate Gradient",
        save_path=cg_perf_plot,
    )

    # Save full result dictionaries
    print("Saving full results dictionaries...")
    minres_npz = os.path.join(exp_dir, "minres_results.npz")
    minres_pkl = os.path.join(exp_dir, "minres_results.pkl")
    save_results_dict(minres_results, minres_npz, minres_pkl)

    cg_npz = os.path.join(exp_dir, "cg_results.npz")
    cg_pkl = os.path.join(exp_dir, "cg_results.pkl")
    save_results_dict(cg_results, cg_npz, cg_pkl)

    # Save metadata / configuration
    meta = {
        "csv_path": csv_path,
        "lambda_values": lambda_values,
        "target_rel": target_rel,
        "minres_tol_internal": minres_tol_internal,
        "minres_maxiter": minres_maxiter,
        "cg_internal_tol": cg_internal_tol,
        "cg_maxiter": cg_maxiter,
        "A_shape": A_design.shape,
        "A_nnz": int(A_design.nnz),
    }
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print("Experiment completed successfully.")
    print(f"Results and plots saved under: {exp_dir}")


if __name__ == "__main__":
    main()


