"""The vectorised edge scores are bitwise the per-edge, per-sample loop of dw2d 0.1.5."""
import numpy as np
from .conftest import golden_cases


class PerPointOnly:
    """Wraps an interpolant and hides its vectorised ``at``, forcing the legacy loop."""

    def __init__(self, f):
        self._f = f

    def __call__(self, row, col):
        return self._f(row, col)


def test_vectorised_scores_equal_the_loop_on_a_tissue_mask():
    from dw2d import geometry_reconstruction_2d
    from dw2d.Graph_functions import Delaunay_Graph

    name, labels, md, expansion = [c for c in golden_cases() if c[0] == "mask_Cells_2_expansion_6__md3"][0]
    dw = geometry_reconstruction_2d(labels, min_dist=md, expansion_labels=expansion)
    g = dw.Delaunay_Graph
    loop = Delaunay_Graph(dw.tri, PerPointOnly(g.edt), PerPointOnly(g.labels), g.nx, g.ny, npoints=10)
    assert np.array_equal(loop.Scores, g.Scores)
    assert np.array_equal(loop.compute_zero_nodes(), g.compute_zero_nodes())
