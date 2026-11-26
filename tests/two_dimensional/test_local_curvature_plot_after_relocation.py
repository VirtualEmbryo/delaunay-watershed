"""`plot_local_curvature` ends on a relocated mesh: its `trace_line` rounds the ends of each edge to the pixel lattice.

`dw2d.Dcel` had its own copy of Bresenham's walk, stepping by one pixel until it met the far end exactly, so it never
ended when the two ends differed by a non-integer and its list of points grew without bound (measured: 34 GB before
it was stopped, on a relocated mesh). The junction relocation, on by default since 0.8.0, puts junctions on pixel
corners (half-integers). The same defect was fixed for the mask rebuild in `dw2d.Mask_reconstruction.trace_line`;
`dw2d.Dcel.trace_line` is now that function. The test calls only names the old code has (the deprecated
`geometry_reconstruction_2d`, `return_dcel`, `plot_local_curvature`) under a 20 s timer: on the old code it fails by
timeout, here it ends at once. On integer ends the pixels are those of the old walk.
"""

import signal
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

import dw2d.functions  # the deprecated `geometry_reconstruction_2d`; `import dw2d` alone does not load this module
from dw2d.Dcel import trace_line

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
def test_the_local_curvature_plot_of_a_relocated_mesh_ends():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _three_cells()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        relocated = dw2d.functions.geometry_reconstruction_2d(labels, min_dist=3)
        verts, _ = relocated.return_mesh()
        assert np.any(verts != np.round(verts)), "the relocation put no vertex off the lattice: the test tests nothing"
        mesh = relocated.return_dcel()
        with _at_most(TIMEOUT_SECONDS):
            mesh.plot_local_curvature(labels)
    image = plt.gca().get_images()[0].get_array()
    assert image.shape == labels.shape
    assert np.count_nonzero(np.asarray(image)) > 0  # the interfaces were drawn
    plt.close("all")


def test_integer_ends_are_traced_as_the_old_walk():
    assert trace_line(0, 0, 3, 1).tolist() == [[0, 0], [1, 0], [2, 1], [3, 1]]
    assert trace_line(2.0, 5.0, 2.0, 2.0).tolist() == [[2, 5], [2, 4], [2, 3], [2, 2]]
    assert trace_line(7, 3, 4, 9).tolist() == [[7, 3], [7, 4], [6, 5], [6, 6], [5, 7], [5, 8], [4, 9]]


def test_half_integer_ends_are_rounded_to_the_nearest_pixel_centre():
    assert trace_line(0.5, 0.5, 3.5, 1.5).tolist() == [[0, 0], [1, 0], [2, 1], [3, 1], [4, 2]]
