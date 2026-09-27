"""Case-level bootstrap confidence intervals: 10000 draws, seed 20260901.

Every reported median in this comparison is a **case-level** bootstrap: one value per case is
resampled (with replacement), never a pooled resample over individual triple lines or
interfaces within a case, so a case with many small features cannot outvote one with few.
"""

from __future__ import annotations

import numpy as np

SEED = 20260901
N_DRAWS = 10_000
TOLERANCE = 1e-9


def bootstrap_median_ci(values: list[float], *, seed: int = SEED, n_draws: int = N_DRAWS, alpha: float = 0.05) -> dict:
    """Case-level bootstrap CI of the median over `values` (one value per case).

    Deterministic given `seed`: two independent calls with the same `values` (in the same
    order) and the same `seed` reproduce the same CI to within `TOLERANCE` -- this is the
    reproducibility every reported CI in this comparison relies on (Safeguard 3's null
    control is about reconstruction determinism; this is the matching guarantee for the
    aggregation step itself).

    Returns:
        dict: `{"median", "ci_lo", "ci_hi", "n_cases", "n_draws", "seed"}`. `median` is
            `None` (not run) when `values` is empty.
    """
    if not values:
        return {"median": None, "ci_lo": None, "ci_hi": None, "n_cases": 0, "n_draws": n_draws, "seed": seed}
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    point = float(np.median(array))
    if n == 1:
        return {"median": point, "ci_lo": point, "ci_hi": point, "n_cases": 1, "n_draws": n_draws, "seed": seed}
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_draws, n))
    draws = np.median(array[resample_indices], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "median": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_cases": int(n),
        "n_draws": n_draws,
        "seed": seed,
    }


def verify_reproducible(values: list[float], *, tolerance: float = TOLERANCE) -> bool:
    """Run the bootstrap twice and check the two CIs agree within `tolerance`. Safeguard/self-test."""
    first = bootstrap_median_ci(values)
    second = bootstrap_median_ci(values)
    if first["median"] is None:
        return second["median"] is None
    return (
        abs(first["ci_lo"] - second["ci_lo"]) <= tolerance
        and abs(first["ci_hi"] - second["ci_hi"]) <= tolerance
        and first["median"] == second["median"]
    )
