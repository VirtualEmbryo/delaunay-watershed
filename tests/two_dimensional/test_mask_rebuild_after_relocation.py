"""The mask rebuild ends after the junction relocation: `trace_line` rounds the ends of each edge to the pixel lattice.

Before 0.8.0, `trace_line` walked from one end of an edge to the other by unit steps until it met the far end exactly
(`while x != x1` on floats), so it never ended when the two ends differed by a non-integer. The junction relocation, on
by default since 0.8.0, moves junctions onto pixel corners (half-integers), so `reconstruct_mask_from_dict` hung on
every relocated mesh (measured: no end in 60 s on a 40 x 60 three-cell mask, and the 2-D example notebook's curvature
plot never ended). The test below calls only names that exist in both the old code and this one (the deprecated
`geometry_reconstruction_2d`, `export_segmentation`, `reconstruct_mask_from_dict` given a file name), under a 20 s
timer: on the old code it fails by timeout; here it ends at once. On integer ends nothing changes (pinned against the
mask rebuilt from the mesh without relocation, which the old code also rebuilt).
"""

import signal
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

import dw2d
import dw2d.functions  # the deprecated `geometry_reconstruction_2d`; `import dw2d` alone does not load this module
import dw2d.Mask_reconstruction
from dw2d.Mask_reconstruction import trace_line

TIMEOUT_SECONDS = 20


@contextmanager
def _at_most(seconds: float):
    """Raise TimeoutError in the main thread when the block runs longer than `seconds` (a POSIX interval timer)."""

    def _expired(signum, frame):  # noqa: ARG001 -- signal handler signature
        raise TimeoutError(f"did not end within {seconds} s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _three_cells() -> np.ndarray:
    labels = np.zeros((40, 60), dtype=np.int64)
    labels[5:35, 5:25] = 1
    labels[5:20, 25:55] = 2
    labels[20:35, 25:55] = 3
    return labels


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs a POSIX interval timer")
def test_the_rebuild_of_a_relocated_mesh_ends(tmp_path):
    labels = _three_cells()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        relocated = dw2d.functions.geometry_reconstruction_2d(labels, min_dist=3)
        relocated.export_segmentation(tmp_path / "relocated.npy")
        verts, _ = relocated.return_mesh()
        assert np.any(verts != np.round(verts)), "the relocation put no vertex off the lattice: the test tests nothing"
        with _at_most(TIMEOUT_SECONDS):
            rebuilt = dw2d.Mask_reconstruction.reconstruct_mask_from_dict(str(tmp_path / "relocated.npy"))
    assert rebuilt.shape == labels.shape
    assert np.mean(rebuilt == labels) > 0.95


def test_integer_ends_are_traced_as_before():
    # Bresenham's line between integer ends, as dw2d 0.1.5 traced it.
    assert trace_line((0, 0), (3, 1)).tolist() == [[0, 0], [1, 0], [2, 1], [3, 1]]
    assert trace_line((2.0, 5.0), (2.0, 2.0)).tolist() == [[2, 5], [2, 4], [2, 3], [2, 2]]


def test_half_integer_ends_are_rounded_to_the_nearest_pixel_centre_and_clipped():
    assert trace_line((0.5, 0.5), (3.5, 1.5)).tolist() == [[0, 0], [1, 0], [2, 1], [3, 1], [4, 2]]
    assert trace_line((-0.4, 3.0), (9.6, 3.0), image_shape=(10, 10))[[0, -1]].tolist() == [[0, 3], [9, 3]]


def test_a_mesh_without_relocation_is_rebuilt_as_before(tmp_path):
    labels = _three_cells()
    algorithm = dw2d.MeshReconstructionAlgorithmFactory().set_peak_local_points_placement_method(3)
    algorithm = algorithm.set_junction_relocation(False).make_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(labels)
    compressed = algorithm.last_compressed_segmentation
    assert np.all(compressed["points"] == np.round(compressed["points"]))
    rebuilt = dw2d.reconstruct_mask_from_dict(compressed)
    assert np.mean(rebuilt == labels) > 0.95


def test_the_rebuild_traces_each_edge_from_its_second_vertex():
    """The rebuild traces each edge from its second vertex, the direction dw2d stored its edges in before 0.8.0.

    Bresenham's walk breaks ties towards its start, so the drawn pixels depend on the direction of an edge. Since 0.8.0
    every edge is stored the other way round (one orientation rule in 2D and 3D), and tracing from the second vertex
    rebuilds a mesh to the same mask as before. On the three-cell mask below, the other direction draws different
    pixels, so this pins the direction.
    """
    from dw2d.Mask_reconstruction import create_mesh_semantic_masks

    labels = np.ones((40, 60), np.int64)
    labels[:, 30:] = 2
    r, c = np.indices((40, 60))
    labels[(r - 19.5) ** 2 + (c - 29.5) ** 2 < 64] = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
        points, segments, _ = algorithm.construct_mesh_from_segmentation_mask(labels)
    drawn = create_mesh_semantic_masks(points, segments, labels.shape)
    backwards, forwards = np.zeros(labels.shape), np.zeros(labels.shape)
    for v1, v2 in segments:
        backwards[tuple(trace_line(points[v2], points[v1], labels.shape).T)] = 1
        forwards[tuple(trace_line(points[v1], points[v2], labels.shape).T)] = 1
    assert np.array_equal(drawn, backwards)
    assert not np.array_equal(drawn, forwards)
