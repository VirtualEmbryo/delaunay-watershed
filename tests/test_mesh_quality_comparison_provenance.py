"""Cross-repository provenance checks for `dw3d_benchmarks.mesh_quality_comparison`.

This package restates (rather than imports) two things from `foambryo`, specifically so it
never needs a path outside its own repository at runtime: the figure palette, and the dataset
cohort/equilibrium-split constants. These tests check the restated copies still agree with their
source of truth, in the foambryo checkout named by the environment variable
`FOAMBRYO_REPOSITORY`. They skip when it is not set, and fail when it is set but does not hold
the expected files.

The checkout is never guessed from a sibling path: a guess once found an older foambryo checkout
(a `v0.4.2` one, through a symlink) at the sibling location and compared against it without saying
so.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from dw3d_benchmarks.mesh_quality_comparison import bootstrap, case_metadata
from dw3d_benchmarks.mesh_quality_comparison.figures import _style

FOAMBRYO_REPOSITORY_VARIABLE = "FOAMBRYO_REPOSITORY"
FOAMBRYO_STYLE = Path("foambryo_benchmarks") / "figures" / "_style.py"
FOAMBRYO_CASE_GROUPS = Path("foambryo_benchmarks") / "case_groups.py"


def _foambryo_file(relative: Path) -> Path:
    """`relative` inside the foambryo checkout named by `FOAMBRYO_REPOSITORY`, or skip if it is unset."""
    value = os.environ.get(FOAMBRYO_REPOSITORY_VARIABLE)
    if not value:
        pytest.skip(
            f"{FOAMBRYO_REPOSITORY_VARIABLE} is not set: set it to the foambryo checkout to compare against "
            "(it is never guessed from a sibling path)",
        )
    path = Path(value) / relative
    if not path.is_file():
        pytest.fail(f"{FOAMBRYO_REPOSITORY_VARIABLE}={value} has no {relative}")
    return path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_palette_matches_foambryo() -> None:
    foambryo_style = _load_module(_foambryo_file(FOAMBRYO_STYLE), "_foambryo_figures_style")
    assert _style.CATEGORICAL == foambryo_style.CATEGORICAL
    assert _style.SEQUENTIAL_BLUE == foambryo_style.SEQUENTIAL_BLUE
    assert _style.DIVERGING == foambryo_style.DIVERGING
    assert _style.TEXT_PRIMARY == foambryo_style.TEXT_PRIMARY
    assert _style.TEXT_SECONDARY == foambryo_style.TEXT_SECONDARY
    assert _style.GRID == foambryo_style.GRID
    assert _style.SURFACE == foambryo_style.SURFACE


def test_case_metadata_matches_foambryo() -> None:
    foambryo_case_groups = _load_module(_foambryo_file(FOAMBRYO_CASE_GROUPS), "_foambryo_case_groups")
    assert case_metadata.NON_EQUILIBRIUM_CASE_IDS == foambryo_case_groups.NON_EQUILIBRIUM_CASE_IDS
    assert case_metadata.INTEGRITY_FLAGGED_CASE_IDS == foambryo_case_groups.INTEGRITY_FLAGGED_CASE_IDS


def test_median_of_determined_refuses_partial_by_default() -> None:
    with pytest.raises(ValueError, match="undetermined"):
        case_metadata.median_of_determined([1.0, float("nan"), 2.0])


def test_median_of_determined_allow_partial() -> None:
    result = case_metadata.median_of_determined([1.0, float("nan"), 2.0, 3.0], allow_partial=True)
    assert result == {"median": 2.0, "n_determined": 3, "n_total": 4}


def test_median_of_determined_all_nan_returns_none_not_nan() -> None:
    result = case_metadata.median_of_determined([float("nan"), float("nan")], allow_partial=True)
    assert result["median"] is None
    assert result["n_determined"] == 0
    assert result["n_total"] == 2


def test_require_single_group_raises_on_mixed_cohort() -> None:
    records = [{"case": "000"}, {"case": "005"}]  # 000 equilibrium, 005 non-equilibrium
    with pytest.raises(ValueError, match="refusing to aggregate"):
        case_metadata.require_single_group(records)


def test_require_single_group_accepts_single_group() -> None:
    records = [{"case": "000"}, {"case": "001"}]
    assert case_metadata.require_single_group(records) == case_metadata.EQUILIBRIUM


def test_damaged_masks_excluded_from_non_equilibrium_and_flagged_sets() -> None:
    # 031/038 are a data-corruption exclusion, independent of the equilibrium split.
    assert case_metadata.DAMAGED_MASK_CASE_IDS.isdisjoint(case_metadata.NON_EQUILIBRIUM_CASE_IDS)
    assert case_metadata.DAMAGED_MASK_CASE_IDS.isdisjoint(case_metadata.INTEGRITY_FLAGGED_CASE_IDS)


def test_bootstrap_median_ci_is_deterministic() -> None:
    values = [0.01, 0.03, -0.02, 0.04, 0.0, -0.01, 0.02]
    assert bootstrap.verify_reproducible(values)


def test_bootstrap_median_ci_single_case() -> None:
    result = bootstrap.bootstrap_median_ci([0.05])
    assert result["median"] == result["ci_lo"] == result["ci_hi"] == 0.05
    assert result["n_cases"] == 1
