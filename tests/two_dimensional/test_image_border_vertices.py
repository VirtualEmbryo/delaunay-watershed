"""The image border vertices: marked from the triangulation's structure, never moved, carried by the half-edge mesh.

dw2d triangulates its points with the four image corners, so the hull edges of the triangulation (one triangle each)
run along the image border. ``geometry_reconstruction_2d.image_border_vertices`` holds their endpoints, taken before
any vertex moves; the junction relocation and the projection never move them; ``return_dcel`` passes them on as
``DCEL_Data.image_border_vertices``, which foambryo's 2D inference reads to exclude the junctions on the border.
Until 0.8.0 the relocation could move a junction on the border onto the mask's junction half a pixel inside, where a
reader of the positions no longer saw it on the border: on ``mask_Cells_2`` (medium on its border) at ``min_dist=3``,
one junction.
"""
import contextlib
import io

import numpy as np
import pytest
from .conftest import MASKS, golden_cases

CASES = golden_cases()


def mask_cells_2():
    import skimage.io as skio

    lab = skio.imread(MASKS / "mask_Cells_2.tif")
    return lab[..., 0] if lab.ndim == 3 else lab


def build(labels, **options):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        return geometry_reconstruction_2d(labels, **options)


@pytest.mark.parametrize(("name", "labels", "min_dist", "expansion"), CASES, ids=[c[0] for c in CASES])
def test_the_mark_is_the_vertices_on_the_image_rectangle_before_anything_moves(name, labels, min_dist, expansion):
    seg = build(labels, min_dist=min_dist, expansion_labels=expansion, relocate_junctions=False)
    v = np.asarray(seg.Delaunay_Graph.Vertices, float)
    h, w = seg.labels.shape
    on_rectangle = (v[:, 0] == 0) | (v[:, 1] == 0) | (v[:, 0] == h - 1) | (v[:, 1] == w - 1)
    in_a_triangle = np.isin(np.arange(len(v)), seg.tri.simplices)
    assert seg.image_border_vertices.dtype == np.int64
    assert np.array_equal(seg.image_border_vertices, np.flatnonzero(on_rectangle & in_a_triangle))


@pytest.mark.parametrize("project", [False, True])
def test_no_marked_vertex_moves_and_the_dcel_carries_the_mark(project):
    from dw2d.Dcel import Clean_mesh

    seg = build(mask_cells_2(), min_dist=3, project_interface_vertices=project)
    with contextlib.redirect_stdout(io.StringIO()):
        plain, _ = Clean_mesh(seg)
        moved, _ = seg.return_mesh()
        dcel = seg.return_dcel()
    marked = seg.image_border_vertices
    assert np.array_equal(moved[marked], np.asarray(plain, float)[marked])
    assert np.array_equal(dcel.image_border_vertices, marked)
    assert seg.junction_relocation_report["fixed"] == 1
    assert seg.junction_relocation_report["moved"] > 0


def test_negative_control_without_the_mark_a_border_junction_moves_off_the_border():
    """The relocation alone, given no fixed vertex, moves one border junction of mask_Cells_2 half a pixel inside."""
    from dw2d.Dcel import Clean_mesh
    from dw2d.junction_relocation import _relocate_on_edges as relocate_junction_vertices

    seg = build(mask_cells_2(), min_dist=3, relocate_junctions=False)
    with contextlib.redirect_stdout(io.StringIO()):
        verts, edges = Clean_mesh(seg)
    free, report = relocate_junction_vertices(verts, edges, seg.labels)
    marked = seg.image_border_vertices
    off = marked[np.any(free[marked] != np.asarray(verts, float)[marked], axis=1)]
    assert len(off) == 1 and report["fixed"] == 0
    held, report = relocate_junction_vertices(verts, edges, seg.labels, fixed_vertices=marked)
    assert np.array_equal(held[marked], np.asarray(verts, float)[marked]) and report["fixed"] == 1


def test_a_dcel_built_without_the_mark_has_none():
    from dw2d.Dcel import DCEL_Data

    seg = build(mask_cells_2(), min_dist=3, expansion_labels=6)
    verts, edges = seg.return_mesh()
    with contextlib.redirect_stdout(io.StringIO()):
        assert DCEL_Data(verts, edges, allow_several_loops=True).image_border_vertices is None
