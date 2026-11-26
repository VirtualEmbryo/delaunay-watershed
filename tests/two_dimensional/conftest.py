"""Test configuration of dw2d's suite: import dw2d from this checkout's ``src`` and locate the test data.

The golden-master tests must exercise the code of this checkout, never an installed copy, so
``src`` is put first on ``sys.path``. This directory is a package (``two_dimensional``) so that its
modules (``conftest``, ``exact_delaunay_checks``, ``test_golden_master``, ...) never collide with
dw3d's modules of the same names in ``tests/``; the test modules import from it relatively.
The shipped masks are under ``data/two_dimensional/Masks`` of the repository.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

TESTS = Path(__file__).resolve().parent
GOLDEN = TESTS / "golden"
DATA = TESTS / "data"
MASKS = REPOSITORY / "data" / "two_dimensional" / "Masks"


def golden_cases():
    """(case name, label image, min_dist, expansion_labels) of every golden case, in a fixed order."""
    import skimage.io as io

    cases = []
    for name, path, expansion in (("c_elegans", DATA / "c_elegans_label.png", 0),
                                  ("mask_Cells_1", MASKS / "mask_Cells_1.tif", 0),
                                  ("mask_Cells_3", MASKS / "mask_Cells_3.tif", 0),
                                  ("mask_Cells_2_expansion_6", MASKS / "mask_Cells_2.tif", 6)):
        lab = io.imread(path)
        if lab.ndim == 3:
            lab = lab[..., 0]
        for md in (3, 5):
            cases.append((f"{name}__md{md}", lab, md, expansion))
    doublets = np.load(DATA / "exact_doublet_masks.npz")
    for name in sorted(doublets.files):
        cases.append((f"{name}__md3", doublets[name].astype(np.int64), 3, 0))
    return cases


def pytest_collection_modifyitems(items):
    """The suite builds most meshes through dw2d 0.1.5's names, deprecated since 0.8.0: their own warning is expected
    there and is filtered, so that every other warning still shows. The new names are tested in
    `test_names_mirrored_on_dw3d.py`, which checks that the old ones warn."""
    here = Path(__file__).resolve().parent
    for item in items:
        if Path(str(item.fspath)).resolve().parent == here and item.module.__name__.split(".")[-1] != "test_names_mirrored_on_dw3d":
            item.add_marker(pytest.mark.filterwarnings("ignore:dw2d.geometry_reconstruction_2d is deprecated:DeprecationWarning"))


@pytest.fixture(scope="session")
def golden():
    import json

    return np.load(GOLDEN / "dw2d_0.1.5_on_scipy_1.13_outputs.npz"), json.load(open(GOLDEN / "dw2d_0.1.5_on_scipy_1.13_outputs.json"))
