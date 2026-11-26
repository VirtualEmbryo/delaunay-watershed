"""`reconstruct_mask_from_dict` assigns every voxel centre to the region the mesh's triangles enclose, exactly.

The rebuild used to subdivide the mesh with trimesh, mark the voxels nearest the subdivided vertices and flood
the rest by a seeded watershed; the result depended on the trimesh release (509 to 819 voxels per test image
between trimesh 4.12.2 and 5.1.1). It now decides, in exact arithmetic, whether the segment between two
neighbouring voxel centres crosses a triangle, and names each connected region by the seed it holds. Pinned
here: the rule for a centre lying exactly on a surface, the box rule, exactness near a tie, independence of
triangle and vertex order, the regions holding several seeds or none, bitwise-stored rebuilds of the four test
images (so that every CI environment, NumPy 2.0 to 2.5 and SciPy 1.13 to 1.18, Linux and macOS, checks the same
values), and the agreement with the masks (measured 0.9957 to 0.9963; the trimesh route gave 0.9886 to 0.9916).
"""

import hashlib

import numpy as np
import pytest

from dw3d import reconstruct_mask_from_dict
from dw3d.exact_voxelisation import enclosed_regions_of_voxel_centres
from tests.conftest import GOLDEN_DIR, IMAGE_NAMES, load_image_or_skip

#: The cube's 12 triangles over its 8 corners `(i, j, k)`, `i, j, k` in {0, 1}, numbered `4 i + 2 j + k`.
CUBE_TRIANGLES = np.array(
    [
        (0, 1, 3),
        (0, 3, 2),
        (4, 6, 7),
        (4, 7, 5),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 7),
        (2, 7, 6),
        (0, 2, 6),
        (0, 6, 4),
        (1, 5, 7),
        (1, 7, 3),
    ],
)

#: SHA-256 of each test image's rebuilt mask (little-endian int64, C order), from the stored compressed
#: segmentation; measured on macOS arm64 and identical in the four CI environments (Python 3.12 and 3.13,
#: current and floor dependencies).
REBUILT_MASK_SHA256 = {
    "1": "7cba91b310c1a3fc15666daca5e2e5480c8f825effb7a207401a4ab60bd14a5f",
    "2": "5d390a148514c6ea1f6b6bb6437c5b3611216a8d9e6a298da5493d8c2764cb46",
    "3": "57a51005e66ff90c6af56aa339ce819d83c835a4dff6fc74e133f5ae2c7307ac",
    "4": "cc71800ac02fda79fcfb8c3992f0e1b6a638f20192010c8dd21747715665c853",
}


def cube(low: float, high: float) -> np.ndarray:
    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)
    return low + (high - low) * corners


def rebuild(points: np.ndarray, triangles: np.ndarray, seeds: list, shape: tuple) -> np.ndarray:
    compressed = {"points": points, "triangles": triangles, "seeds": np.array(seeds), "image_shape": shape}
    return reconstruct_mask_from_dict(compressed)


def box(low: int, high: int) -> set:
    return {(i, j, k) for i in range(low, high) for j in range(low, high) for k in range(low, high)}


def inside(labels: np.ndarray, label: int) -> set:
    return {tuple(int(v) for v in index) for index in np.argwhere(labels == label)}


def test_a_centre_on_a_surface_goes_where_the_stated_move_takes_it():
    """Cube faces through the centres 1 and 3: a centre on a face moves towards increasing index, so 1 is in, 3 out."""
    labels = rebuild(cube(1.0, 3.0), CUBE_TRIANGLES, [(0, 0, 0), (2, 2, 2)], (5, 5, 5))
    assert inside(labels, 1) == box(1, 3)
    assert inside(labels, 0) == box(0, 5) - box(1, 3)


def test_a_cell_on_the_image_faces_keeps_its_outermost_voxels():
    """Faces through the first and the last centres of every axis: the last ones move into the image, so are in."""
    labels = rebuild(cube(0.0, 4.0), CUBE_TRIANGLES, [(2, 2, 2)], (5, 5, 5))
    assert np.all(labels == 0)
    _, n_regions = enclosed_regions_of_voxel_centres(cube(0.0, 4.0), CUBE_TRIANGLES, (5, 5, 5))
    assert n_regions == 1


@pytest.mark.parametrize(("offset", "expected"), [(1e-12, box(2, 4)), (-1e-12, box(1, 3)), (2.0**-40, box(2, 4))])
def test_a_surface_a_hair_off_the_centres_is_decided_exactly(offset, expected):
    """1e-12 off the lattice: no tie, and the float64 bound hands the close calls to exact rationals."""
    labels = rebuild(cube(1.0 + offset, 3.0 + offset), CUBE_TRIANGLES, [(0, 0, 0), (2, 2, 2)], (5, 5, 5))
    assert inside(labels, 1) == expected


def test_triangle_order_vertex_order_and_orientation_do_not_change_the_rebuild():
    points = cube(1.0, 4.0) + np.array([0.0, 0.5, 0.0])
    reference = rebuild(points, CUBE_TRIANGLES, [(0, 0, 0), (2, 2, 2)], (6, 6, 6))
    rng = np.random.default_rng(3)
    for _ in range(5):
        triangles = CUBE_TRIANGLES[rng.permutation(len(CUBE_TRIANGLES))]
        triangles = np.array([np.roll(t, rng.integers(3)) for t in triangles])
        flipped = rng.random(len(triangles)) < 0.5
        triangles[flipped] = triangles[flipped][:, ::-1]
        assert np.array_equal(rebuild(points, triangles, [(0, 0, 0), (2, 2, 2)], (6, 6, 6)), reference)
    reversed_orientation = CUBE_TRIANGLES[:, ::-1]
    assert np.array_equal(rebuild(points, reversed_orientation, [(0, 0, 0), (2, 2, 2)], (6, 6, 6)), reference)


def test_a_region_holding_two_seeds_goes_to_the_deeper_one():
    """A cell absent from the mesh leaves its seed at the edge of a neighbour's region: it takes no voxel."""
    labels = rebuild(cube(1.0, 7.0), CUBE_TRIANGLES, [(0, 0, 0), (1, 1, 1), (4, 4, 4)], (9, 9, 9))
    assert inside(labels, 2) == box(1, 7)
    assert inside(labels, 1) == set()


def test_a_region_holding_no_seed_takes_the_label_it_shares_most_links_with():
    labels = rebuild(cube(1.0, 3.0), CUBE_TRIANGLES, [(0, 0, 0)], (5, 5, 5))
    assert np.all(labels == 0)


@pytest.fixture(scope="module")
def stored_compressed_segmentations() -> dict:
    stored = np.load(GOLDEN_DIR / "compressed_segmentations_of_the_test_images.npz")
    keys = ("points", "triangles", "seeds", "image_shape")
    stems = [name.rsplit(".", 1)[0] for name in IMAGE_NAMES]
    return {stem: {key: stored[f"{stem}_{key}"] for key in keys} for stem in stems}


@pytest.mark.parametrize("image", sorted(REBUILT_MASK_SHA256))
def test_the_rebuilt_masks_of_the_test_images_are_the_stored_ones_bitwise(stored_compressed_segmentations, image):
    labels = reconstruct_mask_from_dict(stored_compressed_segmentations[image])
    digest = hashlib.sha256(np.ascontiguousarray(labels, dtype="<i8").tobytes()).hexdigest()
    assert digest == REBUILT_MASK_SHA256[image]


@pytest.mark.parametrize("image", sorted(REBUILT_MASK_SHA256))
def test_the_rebuilt_masks_agree_with_the_test_images(stored_compressed_segmentations, image):
    """At least 0.995 of the voxels (measured 0.9957 to 0.9963; the trimesh route gave 0.9886 to 0.9916)."""
    mask = load_image_or_skip(f"{image}.tif")
    ranks = np.searchsorted(np.unique(mask), mask)
    labels = reconstruct_mask_from_dict(stored_compressed_segmentations[image])
    assert labels.dtype == np.int64
    assert np.mean(labels == ranks) >= 0.995
