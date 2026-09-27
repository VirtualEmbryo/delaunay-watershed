"""Cross-repository provenance checks for `dw3d_benchmarks.mesh_quality_comparison`.

This package restates (rather than imports) two things from a sibling `foambryo` checkout,
specifically so it never needs a path outside its own repository at runtime: the figure
palette, and the dataset cohort/equilibrium-split constants. These tests check the restated
copies still agree with their source of truth -- but only when a sibling `foambryo` checkout
happens to be present at the conventional workspace location, so a clean, standalone
`delaunay-watershed-3d` checkout is never required to have one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dw3d_benchmarks.mesh_quality_comparison import bootstrap, case_metadata
from dw3d_benchmarks.mesh_quality_comparison.figures import _style

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SIBLING_FOAMBRYO_STYLE = WORKSPACE_ROOT / "foambryo" / "foambryo_benchmarks" / "figures" / "_style.py"
SIBLING_CASE_GROUPS = WORKSPACE_ROOT / "foambryo" / "foambryo_benchmarks" / "case_groups.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SIBLING_FOAMBRYO_STYLE.exists(), reason="no sibling foambryo checkout in this workspace")
def test_palette_matches_foambryo() -> None:
    foambryo_style = _load_module(SIBLING_FOAMBRYO_STYLE, "_sibling_foambryo_figures_style")
    assert _style.CATEGORICAL == foambryo_style.CATEGORICAL
    assert _style.SEQUENTIAL_BLUE == foambryo_style.SEQUENTIAL_BLUE
    assert _style.DIVERGING == foambryo_style.DIVERGING
    assert _style.TEXT_PRIMARY == foambryo_style.TEXT_PRIMARY
    assert _style.TEXT_SECONDARY == foambryo_style.TEXT_SECONDARY
    assert _style.GRID == foambryo_style.GRID
    assert _style.SURFACE == foambryo_style.SURFACE


@pytest.mark.skipif(not SIBLING_CASE_GROUPS.exists(), reason="no sibling foambryo checkout in this workspace")
def test_case_metadata_matches_foambryo() -> None:
    sibling = _load_module(SIBLING_CASE_GROUPS, "_sibling_foambryo_case_groups")
    assert case_metadata.NON_EQUILIBRIUM_CASE_IDS == sibling.NON_EQUILIBRIUM_CASE_IDS
    assert case_metadata.INTEGRITY_FLAGGED_CASE_IDS == sibling.INTEGRITY_FLAGGED_CASE_IDS


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
