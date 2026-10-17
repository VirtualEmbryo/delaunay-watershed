"""The vectorised distance-transform masks and tesselation graph against the loops they replaced.

`compute_edt_classical`, `compute_edt_boundary_bias`, `get_total_boundaries`, `_pad_mask` and
`TesselationGraph`'s face table, face pairing and networkx graph were rewritten without loops.
The previous implementations are kept below, verbatim apart from their names, as a test-only
reference, and every output is compared with theirs **bit for bit and dtype for dtype**: the
default configuration reads the distance transform's dtype (`test_edt_dtype_composition.py`) and
the benchmark fingerprints hash the graph's arrays as raw bytes.

The inputs are the edge cases of the rewrite: a single label (no boundary at all), a mask with
no background, labels touching the image box, an axis of length 3 (where the bounding-box shell
leaves a one-voxel interior), a mask resampled to cubic voxels, distance-transform ties on a
symmetric mask, a single tetrahedron, an empty tesselation, and points on a lattice (Qhull's
flat tetrahedra).

Five behaviours are new and are tested on their own: `_pad_mask` returns all ones with the mask's
shape on an axis no longer than `2 * pad_size` (the previous code returned a wrong shape on an
axis of length 1) and refuses `pad_size < 1`; an `ignore_index` that does not fit in the `uint8`
boundary image is refused; a triangle shared by three tetrahedrons raises; and a tesselation with
no shared face (a single tetrahedron) gives `triangle_faces` of shape `(0, 3)` and
`nodes_linked_by_faces` of shape `(0, 2)`, where the loop gave `(0,)` arrays. A reconstruction
never meets the last case: its tesselation always contains the eight corners of the image box.
"""

import networkx
import numpy as np
import pytest
from edt import edt as euclidean_dt
from scipy.spatial import Delaunay
from skimage.segmentation import find_boundaries

from dw3d import resample_to_isotropic
from dw3d.edt import (
    _pad_mask,
    _StandardLabelToBoundary,
    compute_edt_boundary_bias,
    compute_edt_classical,
    get_total_boundaries,
)
from dw3d.tesselation_graph import TesselationGraph

# ---------------------------------------------------------------------------------------
# Test-only reference: the previous implementations
# ---------------------------------------------------------------------------------------


def _reference_boundaries(m: np.ndarray) -> np.ndarray:
    """`_StandardLabelToBoundary()(m)[0]` as it was: the `int32` thick boundary image."""
    return np.stack([find_boundaries(m, connectivity=2, mode="thick").astype("int32")], axis=0)[0]


def _reference_pad_mask(mask: np.ndarray, pad_size: int = 1) -> np.ndarray:
    padded_mask = mask.copy()[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
    return np.pad(
        padded_mask,
        ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)),
        "constant",
        constant_values=1,
    )


def _reference_edt_classical(segmentation_mask: np.ndarray) -> np.ndarray:
    b = _reference_boundaries(segmentation_mask)
    mask_2 = b
    edt_2 = euclidean_dt(mask_2)
    b = _reference_pad_mask(b)
    mask_1 = 1 - b
    edt_1 = euclidean_dt(mask_1)
    inv = np.amax(edt_2) - edt_2
    return (edt_1 + np.amax(edt_2)) * mask_1 + inv * mask_2


def _reference_total_boundaries(segmented_mask: np.ndarray) -> np.ndarray:
    total_boundaries = np.zeros(segmented_mask.shape)
    for index in np.unique(segmented_mask):
        if index == 0:
            region_labels = np.where(segmented_mask == 0, 1, 0)
        else:
            region_labels = np.where(segmented_mask == index, segmented_mask, 0)
        total_boundaries += _reference_boundaries(region_labels)
    return np.amax(total_boundaries) - total_boundaries


def _reference_edt_boundary_bias(segmentation_mask: np.ndarray) -> np.ndarray:
    total_boundaries = _reference_total_boundaries(segmentation_mask)
    b = _reference_boundaries(segmentation_mask)
    mask_2 = b
    edt_2 = euclidean_dt(mask_2)
    b = _reference_pad_mask(b)
    mask_1 = 1 - b
    edt_1 = euclidean_dt(mask_1)
    inv = np.amax(edt_2) - edt_2 + total_boundaries
    return (edt_1 + np.amax(inv)) * mask_1 + inv * mask_2


def _reference_graph_arrays(nodes: np.ndarray) -> dict:
    """The face table, the pairing loop and the graph construction as they were."""
    tetrahedrons = np.sort(nodes, axis=1)
    sorted_tetrahedrons = tetrahedrons.copy()
    tetrahedrons += 1
    rows = []
    for i, (a, b, c, d) in enumerate(tetrahedrons):
        rows += [[a, b, c, i], [a, b, d, i], [a, c, d, i], [b, c, d, i]]
    faces_table = np.array(rows, dtype=np.int64)
    edges_table = faces_table[np.lexsort((faces_table[:, 3], faces_table[:, 2], faces_table[:, 1], faces_table[:, 0]))]

    index, n = 0, len(edges_table)
    triangle_faces, nodes_linked_by_faces, lone_faces, nodes_linked_by_lone_faces = [], [], [], []
    nodes_on_the_border = np.zeros(len(nodes))
    faces_of_nodes = {i: [] for i in range(len(nodes))}
    while index < n:
        if index + 1 < n and all(edges_table[index][k] == edges_table[index + 1][k] for k in range(3)):
            a, b = edges_table[index][3], edges_table[index + 1][3]
            triangle_faces.append(edges_table[index][:-1] - 1)
            nodes_linked_by_faces.append([a, b])
            faces_of_nodes[a].append(len(triangle_faces) - 1)
            faces_of_nodes[b].append(len(triangle_faces) - 1)
            index += 2
        else:
            nodes_on_the_border[edges_table[index][3]] = 1
            lone_faces.append(edges_table[index][:-1] - 1)
            nodes_linked_by_lone_faces.append(edges_table[index][3])
            index += 1
    return {
        "tetrahedrons": sorted_tetrahedrons,
        "triangle_faces": np.array(triangle_faces, dtype=np.uint),
        "nodes_linked_by_faces": np.array(nodes_linked_by_faces),
        "lone_faces": np.array(lone_faces),
        "nodes_linked_by_lone_faces": np.array(nodes_linked_by_lone_faces),
        "nodes_on_the_border": nodes_on_the_border,
        "faces_of_nodes": faces_of_nodes,
    }


def _reference_networkx_graph(graph: TesselationGraph, reference: dict) -> networkx.Graph:
    nx_graph = networkx.Graph()
    volumes = graph._compute_volumes()
    nx_graph.add_nodes_from(zip(np.arange(len(volumes)), [{"volume": x} for x in volumes], strict=False))
    areas = graph._compute_areas()
    pairs = reference["nodes_linked_by_faces"]
    nx_graph.add_edges_from(
        np.array(
            [
                (pairs[idx][0], pairs[idx][1], {"score": graph.scores[idx], "area": areas[idx]})
                for idx in np.arange(len(reference["triangle_faces"]))
            ],
        ),
    )
    return nx_graph


# ---------------------------------------------------------------------------------------
# Helpers and inputs
# ---------------------------------------------------------------------------------------


def _assert_bitwise(actual: np.ndarray, expected: np.ndarray) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _two_cells_in_medium() -> np.ndarray:
    mask = np.zeros((12, 14, 16), dtype=np.int32)
    mask[2:10, 2:7, 3:13] = 1
    mask[2:10, 7:12, 3:13] = 2
    return mask


def _cells_touching_the_image_box() -> np.ndarray:
    mask = np.zeros((10, 12, 9), dtype=np.int32)
    mask[:, :6, :] = 3
    mask[:, 6:, :5] = 7
    return mask


def _symmetric_four_cells() -> np.ndarray:
    """A mirror-symmetric mask: every distance transform has exact ties across the planes."""
    mask = np.zeros((11, 11, 11), dtype=np.int32)
    mask[1:10, 1:5, 1:5] = 1
    mask[1:10, 6:10, 1:5] = 2
    mask[1:10, 1:5, 6:10] = 3
    mask[1:10, 6:10, 6:10] = 4
    return mask


def _random_voronoi_labels(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 20, size=(9, 3))
    grid = np.stack(np.meshgrid(*[np.arange(20)] * 3, indexing="ij"), axis=-1)
    distances = np.linalg.norm(grid[..., None, :] - centres, axis=-1)
    labels = (np.argmin(distances, axis=-1) + 1).astype(np.int32)
    labels[np.linalg.norm(grid - 9.5, axis=-1) > 9] = 0
    return labels


MASKS = {
    "two_cells_in_medium": _two_cells_in_medium(),
    "one_label_everywhere": np.full((6, 7, 8), 5, dtype=np.int32),
    "background_only": np.zeros((6, 7, 8), dtype=np.int32),
    "no_background": _cells_touching_the_image_box(),
    "axis_of_length_three": _two_cells_in_medium()[3:6],
    "symmetric_ties": _symmetric_four_cells(),
    "random_voronoi": _random_voronoi_labels(0),
    "uint16_labels": _random_voronoi_labels(1).astype(np.uint16),
    "resampled_to_cubic_voxels": resample_to_isotropic(_random_voronoi_labels(2)[::2], (2.0, 1.0, 1.0)),
}


# ---------------------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MASKS))
def test_classical_edt_is_the_previous_field_bit_for_bit(name: str) -> None:
    mask = MASKS[name]
    _assert_bitwise(compute_edt_classical(mask), _reference_edt_classical(mask))


@pytest.mark.parametrize("name", sorted(MASKS))
def test_total_boundaries_are_the_previous_counts_bit_for_bit(name: str) -> None:
    mask = MASKS[name]
    _assert_bitwise(get_total_boundaries(mask), _reference_total_boundaries(mask))


@pytest.mark.parametrize("name", sorted(MASKS))
def test_boundary_bias_edt_is_the_previous_field_bit_for_bit(name: str) -> None:
    mask = MASKS[name]
    _assert_bitwise(compute_edt_boundary_bias(mask), _reference_edt_boundary_bias(mask))


def test_boundary_extractor_marks_the_same_voxels_in_uint8() -> None:
    mask = MASKS["random_voronoi"]
    boundaries = _StandardLabelToBoundary().get_boundaries(mask)
    assert boundaries.dtype == np.uint8
    np.testing.assert_array_equal(boundaries, _reference_boundaries(mask))
    stacked = _StandardLabelToBoundary(foreground=True)(mask)
    assert stacked.shape == (2, *mask.shape)
    np.testing.assert_array_equal(stacked[1], boundaries)


def test_ignore_index_is_written_back_and_must_fit_in_uint8() -> None:
    mask = MASKS["two_cells_in_medium"]
    boundaries = _StandardLabelToBoundary(ignore_index=2).get_boundaries(mask)
    assert np.all(boundaries[mask == 2] == 2)
    with pytest.raises(ValueError, match="ignore_index"):
        _StandardLabelToBoundary(ignore_index=256)
    with pytest.raises(ValueError, match="ignore_index"):
        _StandardLabelToBoundary(ignore_index=-1)


@pytest.mark.parametrize("shape", [(9, 8, 7), (3, 8, 8), (8, 3, 8), (2, 8, 8)])
def test_pad_mask_is_the_previous_shell_whenever_that_kept_the_shape(shape: tuple[int, int, int]) -> None:
    mask = np.random.default_rng(3).integers(0, 2, size=shape).astype(np.uint8)
    _assert_bitwise(_pad_mask(mask), _reference_pad_mask(mask))


def test_pad_mask_on_an_axis_of_length_one_is_all_ones_with_the_mask_shape() -> None:
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    assert _reference_pad_mask(mask).shape == (2, 8, 8)  # the previous code changed the shape
    padded = _pad_mask(mask)
    assert padded.shape == mask.shape
    assert np.all(padded == 1)


def test_pad_mask_refuses_a_pad_size_below_one() -> None:
    with pytest.raises(ValueError, match="pad_size"):
        _pad_mask(np.zeros((5, 5, 5), dtype=np.uint8), pad_size=0)


# ---------------------------------------------------------------------------------------
# Tesselation graph
# ---------------------------------------------------------------------------------------


def _scores(_edt_image, _vertices, triangle_faces):
    """Distinct scores with ties, so that the graph's edge data are checked value for value."""
    return np.round(np.linspace(0.0, 3.0, len(triangle_faces)), 1)


def _graph(points: np.ndarray, tetrahedrons: np.ndarray) -> TesselationGraph:
    return TesselationGraph(
        np.asarray(points, dtype=np.float64),
        tetrahedrons,
        indices_of_sorted_maxes=np.array([], dtype=np.uint),
        score_computation_function=_scores,
        edt_image=None,
    )


def _lattice_points() -> np.ndarray:
    return np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)


TESSELATIONS = {
    "two_tetrahedrons_sharing_a_face": (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64),
        np.array([[3, 1, 2, 0], [1, 2, 3, 4]], dtype=np.int64),
    ),
    "random_points": (pts := np.random.default_rng(4).uniform(0, 10, size=(300, 3)), Delaunay(pts).simplices),
    "lattice_with_flat_tetrahedra": (_lattice_points(), Delaunay(_lattice_points()).simplices),
}


@pytest.mark.parametrize("name", sorted(TESSELATIONS))
def test_graph_arrays_are_the_previous_loops_bit_for_bit(name: str) -> None:
    points, tetrahedrons = TESSELATIONS[name]
    graph = _graph(points, tetrahedrons)
    reference = _reference_graph_arrays(tetrahedrons)
    for attribute in (
        "tetrahedrons",
        "triangle_faces",
        "nodes_linked_by_faces",
        "lone_faces",
        "nodes_linked_by_lone_faces",
        "nodes_on_the_border",
    ):
        _assert_bitwise(getattr(graph, attribute), reference[attribute])
    assert isinstance(graph.faces_of_nodes, dict)
    assert graph.faces_of_nodes == reference["faces_of_nodes"]


@pytest.mark.parametrize("name", sorted(TESSELATIONS))
def test_networkx_graph_is_the_previous_graph(name: str) -> None:
    points, tetrahedrons = TESSELATIONS[name]
    graph = _graph(points, tetrahedrons)
    expected = _reference_networkx_graph(graph, _reference_graph_arrays(tetrahedrons))
    actual = graph.to_networkx_graph()
    assert list(actual.nodes(data=True)) == list(expected.nodes(data=True))
    assert list(actual.edges(data=True)) == list(expected.edges(data=True))
    # what the watershed reads: the edge table it sorts
    _assert_bitwise(np.array(list(actual.edges.data("score"))), np.array(list(expected.edges.data("score"))))


def test_a_single_tetrahedron_has_four_lone_faces_and_empty_two_dimensional_pairs() -> None:
    tetrahedrons = np.array([[0, 1, 2, 3]], dtype=np.int32)
    graph = _graph(np.eye(4, 3), tetrahedrons)
    reference = _reference_graph_arrays(tetrahedrons)
    for attribute in ("tetrahedrons", "lone_faces", "nodes_linked_by_lone_faces", "nodes_on_the_border"):
        _assert_bitwise(getattr(graph, attribute), reference[attribute])
    assert reference["triangle_faces"].shape == (0,)  # the loop's empty arrays had no columns
    assert graph.triangle_faces.shape == (0, 3)
    assert graph.triangle_faces.dtype == np.uint
    assert graph.nodes_linked_by_faces.shape == (0, 2)
    assert graph.faces_of_nodes == {0: []}
    assert graph.to_networkx_graph().number_of_edges() == 0


def test_an_empty_tesselation_gives_an_empty_graph() -> None:
    graph = _graph(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64))
    assert len(graph.triangle_faces) == 0
    assert len(graph.lone_faces) == 0
    assert graph.faces_of_nodes == {}
    assert graph.to_networkx_graph().number_of_nodes() == 0


def test_a_triangle_in_three_tetrahedrons_raises() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1]], dtype=np.float64)
    tetrahedrons = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], dtype=np.int64)
    with pytest.raises(ValueError, match="more than two tetrahedrons"):
        _graph(points, tetrahedrons)
