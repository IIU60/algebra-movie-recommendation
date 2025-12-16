"""
Plotting utilities for residual convergence and performance summaries.

These helpers are designed to work with the results dictionaries returned by
the MINRES and CG drivers.
"""

from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt


def _lambda_color_map(lams: List[float]) -> Dict[float, str]:
    """Deterministically assign a color to each lambda.

    This is shared between residual and performance plots so that the same
    lambda is always rendered with the same color in both figures.
    """
    if not lams:
        return {}

    # Use the current Matplotlib color cycle
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if prop_cycle is not None:
        colors = prop_cycle.by_key().get("color", [])
    else:
        colors = []

    if not colors:
        # Fallback to default named colors if needed
        colors = [f"C{i}" for i in range(len(lams))]

    mapping: Dict[float, str] = {}
    for i, lam in enumerate(sorted(lams)):
        mapping[lam] = colors[i % len(colors)]
    return mapping


def plot_method_residuals(
    results: Dict[float, Dict[str, Any]],
    method_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot residual convergence curves for all lambdas for a single method.

    Parameters
    ----------
    results : dict
        Mapping lambda -> info dict containing a ``residuals`` array.
    method_name : str
        Name of the method, used for the title and legend label prefix.
    save_path : str or None
        If not None, path to save the PNG figure. If None, the figure is only
        shown interactively.
    """
    plt.figure(figsize=(8, 5))

    lams = list(results.keys())
    color_map = _lambda_color_map(lams)

    for lam, info in results.items():
        res = np.asarray(info["residuals"], dtype=float)
        if res.size == 0:
            continue
        plt.plot(res, label=f"λ={lam}", color=color_map.get(lam))

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    plt.title(f"{method_name} Residual Convergence for Different λ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_runtime_and_iterations(
    results: Dict[float, Dict[str, Any]],
    method_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Bar plots of runtime and iteration count vs lambda for a given method."""
    lams = list(results.keys())
    times = [results[lam]["time"] for lam in lams]
    iters = [results[lam]["iterations"] for lam in lams]

    x = np.arange(len(lams))

    color_map = _lambda_color_map(lams)
    bar_colors = [color_map.get(lam) for lam in lams]

    plt.figure(figsize=(10, 4))

    # Runtime
    plt.subplot(1, 2, 1)
    plt.bar(x, times, color=bar_colors)
    plt.xticks(x, [f"λ={lam}" for lam in lams])
    plt.ylabel("Runtime (seconds)")
    plt.title(f"{method_name} Runtime Comparison")
    plt.grid(axis="y")

    # Iterations
    plt.subplot(1, 2, 2)
    plt.bar(x, iters, color=bar_colors)
    plt.xticks(x, [f"λ={lam}" for lam in lams])
    plt.ylabel("Iterations")
    plt.title(f"{method_name} Iteration Count Comparison")
    plt.grid(axis="y")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close()



