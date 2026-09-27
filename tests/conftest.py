"""Shared pytest fixtures for dw3d.

Golden-master fixtures load `data/Images/*.tif`. `data/` is gitignored (see
`.gitignore`), so these images exist locally (the same environment the original reference
numbers were measured in) but are not committed to git. There is no
CI configured for this repo, so this is an accepted limitation, not a regression risk;
tests skip with a clear reason if an image is missing rather than failing opaquely.
"""

from pathlib import Path

import pytest
import skimage.io as io

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "data" / "Images"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
# Stage-0 fixtures, kept verbatim; see tests/test_golden_master.py's module docstring.
DITHERED_GOLDEN_DIR = GOLDEN_DIR / "dithered"
# Deterministic fixtures: an earlier default, before the boundary layer and junction
# protection flipped it to the offset-included configuration.
DETERMINISTIC_GOLDEN_DIR = GOLDEN_DIR / "deterministic"
# Boundary-layer variant fixtures (opt-in method, not the default).
BOUNDARY_LAYER_GOLDEN_DIR = GOLDEN_DIR / "boundary_layer"
# Junction-protected variant fixtures (boundary layer + junction protection) -- a former
# default, and the configuration `get_offset_included_algorithm` names. These were
# byte-identical to `GOLDEN_DIR`'s before the flip to link-checked offset exclusion;
# `test_golden_master.py` now asserts they *differ* from it, and that
# `get_offset_included_algorithm` still reproduces them.
JUNCTION_PROTECTED_GOLDEN_DIR = GOLDEN_DIR / "junction_protected"
# Cubic-interpolation variant fixtures: boundary layer + junction protection with a cubic
# B-spline EDT interpolant.
CUBIC_SCORE_GOLDEN_DIR = GOLDEN_DIR / "cubic_score"
# Offset-excluded variant fixtures: boundary layer + junction protection with the
# boundary-layer offsets kept out of the extracted surface by an unconditional weld. Same
# tesselation and labelling, a different surface -- and a topology regression, which is why
# the link-condition-checked collapse and not the unconditional weld became the default.
OFFSET_EXCLUDED_GOLDEN_DIR = GOLDEN_DIR / "offset_excluded"
# Link-checked variant fixtures: the same exclusion as a link-condition-checked collapse.
# **This is the default**, so these must stay byte-identical to `GOLDEN_DIR`'s;
# `test_golden_master.py` asserts it.
LINK_CHECKED_GOLDEN_DIR = GOLDEN_DIR / "link_checked"

# Since 0.5.0 every configuration relocates junction vertices by default. The fixtures under
# `GOLDEN_DIR` are the relocated meshes; the byte-identical 0.4 fixtures they replaced live
# under `GOLDEN_WITHOUT_RELOCATION_DIR` with the same relative paths, and every golden-master
# test is run against both, the second with `relocate_junctions=False` set explicitly.
GOLDEN_WITHOUT_RELOCATION_DIR = Path(__file__).resolve().parent / "golden_without_relocation"


def golden_dir(directory: Path, relocate_junctions: bool) -> Path:
    """The fixture directory for `directory` (a path under `GOLDEN_DIR`) with or without relocation."""
    if relocate_junctions:
        return directory
    return GOLDEN_WITHOUT_RELOCATION_DIR / directory.relative_to(GOLDEN_DIR)


IMAGE_NAMES = ["1.tif", "2.tif", "3.tif", "4.tif"]


def load_image_or_skip(name: str):
    """Load a segmentation mask from `data/Images/`, skipping the test if it is absent."""
    path = IMAGES_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not found locally (data/ is gitignored, not committed)")
    return io.imread(path)


@pytest.fixture(params=IMAGE_NAMES)
def image_name(request) -> str:
    """Parametrized fixture over the 4 in-repo reference images."""
    return request.param


def dataset_masks(limit: int | None = None) -> list[Path]:
    """`benchmarking-dataset` masks, resolved in a way that also works from a git worktree.

    `benchmarks/run_baseline.py` uses `REPO_ROOT.parent / "benchmarking-dataset"`, which is
    correct for the normal checkout (`<workspace>/delaunay-watershed-3d`) but wrong for a
    worktree created outside it (`<workspace>/dw3d-worktree`), where the dataset is one level
    further in. Both layouts are tried, and an empty list is returned if neither resolves, so
    callers skip rather than fail -- same contract as `load_image_or_skip`.
    """
    candidates = [
        REPO_ROOT.parent / "benchmarking-dataset",
        REPO_ROOT.parent / "foambryo" / "benchmarking-dataset",
    ]
    for directory in candidates:
        masks = sorted(directory.glob("*_labels_filled.tif"))
        if masks:
            return masks[:limit] if limit is not None else masks
    return []
