"""A triangle whose first-label tetrahedron is flat is wound by its second-label tetrahedron.

The winding rule turns each mesh triangle's normal away from the centroid of the tetrahedron that
carries its first label. Qhull emits flat tetrahedra (four coplanar lattice points) inside the
tesselation as well as on its hull, and a flat tetrahedron's centroid lies in the plane of each of
its faces: the dot product the rule reads is then a rounding residue of either sign, and about half
of these triangles came out wound the wrong way (10 of 19 on the 47 benchmark masks, 39 of 73 over
three populations), each giving its two cells a divergence-theorem volume different from their
tetrahedra's. The rule now decides flatness on the tetrahedron's own orientation determinant and,
for a flat one, turns the normal towards the second-label tetrahedron instead.
"""

import numpy as np
import pytest

from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.mesh_utilities import NO_TETRAHEDRON, _reorient_triangles, _tetrahedra_are_flat
from tests.conftest import dataset_masks

# Face (0, 1, 2) in the plane z = 0. Tetrahedron 0 is flat (the unit square, vertex 3 coplanar);
# tetrahedron 1 has its apex at z = +1. With the flat one on the first label's side, the second
# label is on +z, so the normal must point to +z whichever way the face was extracted.
VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, 1.0]])
TETRAHEDRA = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])


def _normal_z(wound: np.ndarray) -> float:
    p = VERTICES[wound[0].astype(np.int64)]
    return float(np.cross(p[1] - p[0], p[2] - p[0])[2])


def test_flatness_is_exact_on_the_lattice_and_certified_off_it():
    flat_on_lattice = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [7.0, 9.0, 0.0]])
    unit = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Coplanar in decimals (x + y + z = 1) but not exactly representable: the computed determinant
    # is a residue (-3.5e-18), within the rounding bound, so the orientation is not certain.
    coplanar_in_decimals = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.6, 0.1, 0.3], [0.2, 0.5, 0.3]])
    thin = coplanar_in_decimals.copy()
    thin[3, 2] += 1e-9  # thin, but its orientation is certain
    vertices = np.vstack([flat_on_lattice, unit, coplanar_in_decimals, thin])
    tetrahedra = np.arange(16).reshape(4, 4)
    assert _tetrahedra_are_flat(vertices, tetrahedra).tolist() == [True, False, True, False]


def test_a_flat_first_label_tetrahedron_defers_to_the_second():
    for triangle in ([0, 1, 2], [0, 2, 1]):
        wound = _reorient_triangles(np.array([triangle], dtype=np.uint), VERTICES, TETRAHEDRA, np.array([[0, 1]]))
        assert _normal_z(wound) > 0, triangle


def test_a_non_flat_first_label_tetrahedron_still_decides():
    """With the apex tetrahedron on the first side, the normal points away from it, to -z."""
    for triangle in ([0, 1, 2], [0, 2, 1]):
        wound = _reorient_triangles(np.array([triangle], dtype=np.uint), VERTICES, TETRAHEDRA, np.array([[1, 0]]))
        assert _normal_z(wound) < 0, triangle


def test_two_flat_tetrahedra_keep_the_extracted_order():
    """No side is readable; the historical rule's exact-zero dot product leaves the order alone."""
    vertices = np.vstack([VERTICES, [[0.5, -1.0, 0.0]]])
    tetrahedra = np.array([[0, 1, 2, 3], [0, 1, 2, 5]])
    for triangle in ([0, 1, 2], [0, 2, 1]):
        wound = _reorient_triangles(np.array([triangle], dtype=np.uint), vertices, tetrahedra, np.array([[0, 1]]))
        assert wound[0].tolist() == triangle


def test_a_border_triangle_is_unaffected():
    """No first-label tetrahedron: the normal points into the hull, as before."""
    for triangle in ([0, 1, 2], [0, 2, 1]):
        wound = _reorient_triangles(
            np.array([triangle], dtype=np.uint),
            VERTICES,
            TETRAHEDRA,
            np.array([[NO_TETRAHEDRON, 1]]),
        )
        assert _normal_z(wound) > 0, triangle


def test_every_cell_reads_its_tetrahedra_volume_on_a_benchmark_mask():
    """Benchmark case `034`: one triangle on a flat first-label tetrahedron was wound wrongly.

    Without relocation the mesh vertices are tesselation vertices, so each cell's
    divergence-theorem volume must equal the summed volume of the tetrahedra labelled with it.
    Before the fix one cell of this mask read 0.4 % off.
    """
    masks = [path for path in dataset_masks() if path.name.startswith("034_")]
    if not masks:
        pytest.skip("benchmarking-dataset case 034 not found")
    import tifffile

    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.relocate_junctions = False
    points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(tifffile.imread(masks[0]))
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    corners = points[triangles]
    det = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))
    tet_label = np.asarray(algorithm._map_node_id_to_label, dtype=np.int64)
    tet_volume = algorithm._tesselation_graph._compute_volumes()
    for cell in np.unique(labels[labels > 0]).tolist():
        mesh_volume = (det[labels[:, 0] == cell].sum() - det[labels[:, 1] == cell].sum()) / 6.0
        assert mesh_volume == pytest.approx(tet_volume[tet_label == cell].sum(), rel=1e-9), cell
