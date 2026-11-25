"""Store Qhull's tetrahedralisations of the thin-margin cubes whose tests need exact values.

Writes `tests/golden/thin_margin_tetrahedralisations.npz`: for each case of `THIN_MARGIN_CASES` (default
configuration, `min_distance` 3), and for each tesselation the reconstruction asks for (the first mesh, then the padded
one when the mask is padded), the points (uint16) and Qhull's tetrahedra in canonical order (uint16). The points are
on the integer lattice and include cospherical sets (the cube's corners and faces), whose tetrahedralisation Qhull
chooses by floating-point rounding: on Linux the 12-voxel cube 4 voxels from every face, unpadded, meshes to 1.52
times its voxels instead of 2.98, and the 8-voxel cube 5 voxels from every face collapses to a 4-vertex mesh with no
vertex on the image box, so it is not padded. `tests/test_thin_medium_margins.py` reconstructs these cases from the
stored tetrahedra where it checks exact values, and checks that Qhull's tetrahedralisation on the machine running the
tests differs from the stored one only inside cospherical sets.

Run from the repository root, once, on the machine whose Qhull made the stored values (macOS arm64):

    python -m tests.make_thin_margin_tetrahedralisations

A regenerated file must leave every test of `tests/test_thin_medium_margins.py` passing; regenerating it is a
deliberate decision, as for the golden fixtures.
"""

import warnings

import numpy as np

from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.tesselation import canonical_tetrahedron_order, simple_delaunay_tesselation
from tests.conftest import GOLDEN_DIR

THIN_MARGIN_TETRAHEDRALISATIONS = GOLDEN_DIR / "thin_margin_tetrahedralisations.npz"

#: Case name -> (cube side, margin, padding switched on). `cube_12_margin_4_unpadded` is meshed with the padding off.
THIN_MARGIN_CASES = {
    "cube_12_margin_4_unpadded": (12, 4, False),
    **{f"cube_8_margin_{margin}": (8, margin, True) for margin in (2, 3, 4, 5)},
}


def cube(side: int, margin: int) -> np.ndarray:
    """One cell, a cube of `side` voxels, `margin` voxels of label 0 from every face of the array."""
    mask = np.zeros((side + 2 * margin,) * 3, dtype=np.uint16)
    mask[margin : margin + side, margin : margin + side, margin : margin + side] = 1
    return mask


def tesselations_of(case: str, tesselate=simple_delaunay_tesselation) -> list[tuple[np.ndarray, np.ndarray]]:
    """The points and tetrahedra of each tesselation the reconstruction of `case` asks for, with `tesselate`."""
    side, margin, pad = THIN_MARGIN_CASES[case]
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording(points, weights=None):
        points, tetrahedra = tesselate(points, weights)
        calls.append((np.asarray(points, dtype=np.float64), np.asarray(tetrahedra, dtype=np.int64)))
        return points, tetrahedra

    algorithm = get_default_mesh_reconstruction_algorithm(3)
    algorithm.pad_thin_medium_margins = pad
    algorithm.tesselation_creation_function = recording
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_from_segmentation_mask(cube(side, margin))
    return calls


def stored_tesselations(case: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The stored points (float64) and canonical tetrahedra (int64) of each tesselation of `case`, in order."""
    result = []
    with np.load(THIN_MARGIN_TETRAHEDRALISATIONS) as stored:
        run = 0
        while f"{case}_run_{run}_points" in stored.files:
            result.append(
                (
                    stored[f"{case}_run_{run}_points"].astype(np.float64),
                    stored[f"{case}_run_{run}_tetrahedra"].astype(np.int64),
                ),
            )
            run += 1
    assert result, f"no stored tetrahedralisation for {case}"
    return result


def stored_tesselation_function(case: str):
    """A tesselation function giving the stored tetrahedra of `case`, after checking the points are the stored ones."""
    stored = stored_tesselations(case)

    def tesselate(points, _weights=None):
        points = np.asarray(points, dtype=np.float64)
        for stored_points, stored_tetrahedra in stored:
            if np.array_equal(points, stored_points):
                return points, stored_tetrahedra.copy()
        message = f"{case}: the point placement no longer gives a stored point set"
        raise AssertionError(message)

    return tesselate


def main() -> None:
    arrays = {}
    for case in THIN_MARGIN_CASES:
        for run, (points, tetrahedra) in enumerate(tesselations_of(case)):
            assert np.array_equal(points, np.rint(points))
            assert points.min() >= 0
            assert points.max() < 2**16
            assert len(points) < 2**16
            arrays[f"{case}_run_{run}_points"] = points.astype(np.uint16)
            arrays[f"{case}_run_{run}_tetrahedra"] = canonical_tetrahedron_order(tetrahedra).astype(np.uint16)
            print(f"{case} run {run}: {len(points)} points, {len(tetrahedra)} tetrahedra")
    np.savez_compressed(THIN_MARGIN_TETRAHEDRALISATIONS, **arrays)
    print(f"wrote {THIN_MARGIN_TETRAHEDRALISATIONS} ({THIN_MARGIN_TETRAHEDRALISATIONS.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
