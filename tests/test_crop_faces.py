"""Crop faces and cut cells: what the reconstruction records where a mask touches the image box.

A cell that touches a face of the image box was cropped; the triangles that close it on that face
(the *border triangles*) are not contact with the medium. The reconstruction now marks them as
*crop faces*, and the cells they touch as *cut cells*, so that foambryo can keep them out of its
equations, unless the face is declared physical (a coverslip). These tests pin:

* nothing changes for a mask that touches no image face: no crop face, and a `.rec` file byte for
  byte what it was, even when the reconstruction closes a cell on the image box because the mask
  leaves only a voxel or two of medium between the cell and the image face;
* a border triangle is a crop face only on an image face its cell reaches in the mask;
* the mesh is the same whatever is declared physical: the declaration changes the mask only;
* every crop face is a `(0, c)` triangle on the image face the crop is on, and the cut cells are
  exactly the cells with a vertex on one;
* the `.rec` and `.vtk` files keep every crop face through a write and a read, and a reader that
  ignores the crop still reads a valid mesh.
"""

import warnings

import numpy as np
import pytest

from dw3d import IMAGE_FACES, MeshReconstructionAlgorithmFactory, get_default_mesh_reconstruction_algorithm
from dw3d.crop_faces import (
    NOT_A_BORDER_TRIANGLE,
    ON_NO_SINGLE_IMAGE_FACE,
    CropFaceReport,
    border_triangles_the_cell_reaches,
    cells_on_each_image_face,
    cells_touching_faces,
    normalise_physical_image_faces,
)
from dw3d.io import (
    crop_faces_from_appendix_range,
    load_meshio_mesh,
    load_meshio_mesh_with_crop_faces,
    load_rec,
    load_rec_appendix,
    load_rec_with_crop_faces,
    save_rec,
)


def _three_cells(cropped: bool, medium_below: int | None = None) -> np.ndarray:
    """A ball cut in three: cells 1 and 2 side by side, cell 3 below both along axis 0.

    Cropped, the image ends ten slices past the ball's centre on axis 0, through cells 1 and 2.
    With `medium_below`, the image starts that many slices of medium below the ball on axis 0
    (ten otherwise).
    """
    n, radius = 60, 20
    z, y, x = np.mgrid[:n, :n, :n]
    centre = n / 2
    inside = (z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 <= radius**2
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[inside & (y < centre) & (z >= centre - 8)] = 1
    mask[inside & (y >= centre) & (z >= centre - 8)] = 2
    mask[inside & (z < centre - 8)] = 3
    first = 0 if medium_below is None else int(centre) - radius - medium_below
    last = int(centre) + 10 if cropped else n
    return np.ascontiguousarray(mask[first:last])


def _reconstruct(mask: np.ndarray, physical_image_faces=()):
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.physical_image_faces = physical_image_faces
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    return algorithm, np.asarray(points), np.asarray(triangles, dtype=np.int64), np.asarray(labels, dtype=np.int64)


@pytest.fixture(scope="module")
def cropped():
    return _reconstruct(_three_cells(cropped=True))


@pytest.fixture(scope="module")
def complete():
    return _reconstruct(_three_cells(cropped=False))


def test_a_mask_that_touches_no_image_face_has_no_crop_face(complete, tmp_path):
    algorithm, points, triangles, labels = complete
    report = algorithm.crop_face_report
    assert report.n_border_triangles == 0
    assert not algorithm.crop_faces.any()
    assert algorithm.crop_faces.shape == (len(triangles),)
    assert algorithm.cut_cells == ()
    # The file is byte for byte what `save_rec` wrote before crop faces were recorded.
    algorithm.save_to_rec_mesh(tmp_path / "algorithm.rec")
    save_rec(tmp_path / "plain.rec", points, triangles, labels)
    assert (tmp_path / "algorithm.rec").read_bytes() == (tmp_path / "plain.rec").read_bytes()


@pytest.fixture(scope="module")
def one_voxel_short():
    """Complete, with a single slice of medium between cell 3 and the image face below it."""
    return _reconstruct(_three_cells(cropped=False, medium_below=1))


def test_a_cell_that_stops_short_of_the_image_face_is_not_cut(one_voxel_short, tmp_path):
    algorithm, points, triangles, labels = one_voxel_short
    report = algorithm.crop_face_report
    # The reconstruction closes cell 3 on the image face below it, although the mask leaves a
    # slice of medium between them: border triangles, but no cell on any image face.
    assert report.n_border_triangles > 0
    assert report.cells_on_image_faces == ((),) * 6
    assert report.n_border_triangles_the_cell_does_not_reach == report.n_border_triangles
    assert not report.meets_the_image_box
    assert not algorithm.crop_faces.any()
    assert algorithm.cut_cells == ()
    # Both files are written as before crop faces were recorded.
    algorithm.save_to_rec_mesh(tmp_path / "algorithm.rec")
    save_rec(tmp_path / "plain.rec", points, triangles, labels)
    assert (tmp_path / "algorithm.rec").read_bytes() == (tmp_path / "plain.rec").read_bytes()
    algorithm.save_to_vtk_mesh(tmp_path / "algorithm.vtk")
    assert load_meshio_mesh_with_crop_faces(tmp_path / "algorithm.vtk")[3] is None


def test_only_the_image_face_a_cell_reaches_holds_crop_faces():
    algorithm, _points, _triangles, labels = _reconstruct(_three_cells(cropped=True, medium_below=1))
    report = algorithm.crop_face_report
    low, high = IMAGE_FACES.index("axis0_low"), IMAGE_FACES.index("axis0_high")
    on_low = report.image_face_of_triangle == low
    on_high = report.image_face_of_triangle == high
    assert on_low.any()
    assert on_high.any()
    assert report.cells_on_image_faces[high] == (1, 2)
    assert report.cells_on_image_faces[low] == ()
    # Cell 3 is closed on the face below, which it does not reach: contact with the medium.
    assert not algorithm.crop_faces[on_low].any()
    assert set(labels[on_low].max(axis=1).tolist()) == {3}
    assert algorithm.crop_faces[on_high].all()
    assert algorithm.cut_cells == (1, 2)
    assert report.n_border_triangles_the_cell_does_not_reach == int(on_low.sum())
    assert report.cells_reaching_a_crop_without_a_crop_face == ()


def test_which_cells_reach_each_image_face():
    mask = np.zeros((4, 5, 6), dtype=np.uint16)
    mask[0, 1, 1] = 7
    mask[3, 2, 2] = 8
    mask[1:3, 0, 3] = 9
    mask[2, 2, 5] = 7
    assert cells_on_each_image_face(mask) == ((7,), (8,), (9,), (), (), (7,))


def test_a_border_triangle_on_no_single_image_face_needs_its_cell_on_one_of_its_faces():
    shape = (10, 10, 10)
    # One triangle with a vertex on axis0_low and one on axis1_high, none sharing a face.
    points = np.array([[0.0, 4.0, 4.0], [5.0, 9.0, 4.0], [5.0, 5.0, 4.0]])
    triangles = np.array([[0, 1, 2]])
    labels = np.array([[0, 4]])
    face_of = np.array([ON_NO_SINGLE_IMAGE_FACE])
    nowhere = ((),) * 6
    on_axis1_high = ((), (), (), (4,), (), ())
    on_axis2_low = ((), (), (), (), (4,), ())
    reaches = border_triangles_the_cell_reaches
    assert not reaches(points, triangles, labels, face_of, nowhere, shape).any()
    assert reaches(points, triangles, labels, face_of, on_axis1_high, shape).all()
    # The cell reaches an image face, but not one the triangle has a vertex on.
    assert not reaches(points, triangles, labels, face_of, on_axis2_low, shape).any()
    # Declared physical, axis1_high no longer counts.
    assert not reaches(points, triangles, labels, face_of, on_axis1_high, shape, faces=(0, 1, 2, 4, 5)).any()


def test_a_cell_reaching_a_crop_without_a_crop_face_is_reported():
    report = CropFaceReport(
        crop_faces=np.array([False, True]),
        image_face_of_triangle=np.array([NOT_A_BORDER_TRIANGLE, 1]),
        cell_reaches_image_face=np.array([False, True]),
        cut_cells=(2,),
        physical_image_faces=("axis2_low",),
        image_shape=(10, 10, 10),
        cells_on_image_faces=((), (2, 5), (), (), (6,), ()),
    )
    # Cell 5 reaches axis0_high without a crop face; cell 6 reaches only a physical face.
    assert report.cells_reaching_a_crop_without_a_crop_face == (5,)
    assert report.as_dict()["cells_reaching_a_crop_without_a_crop_face"] == [5]
    assert report.as_dict()["cells_on_image_faces"]["axis0_high"] == [2, 5]


def test_crop_faces_lie_on_the_cropped_image_face_and_carry_the_medium(cropped):
    algorithm, _points, triangles, labels = cropped
    report = algorithm.crop_face_report
    crop = algorithm.crop_faces
    assert crop.any()
    assert np.array_equal(crop, report.image_face_of_triangle != NOT_A_BORDER_TRIANGLE)
    assert set(report.image_face_of_triangle[crop].tolist()) == {IMAGE_FACES.index("axis0_high")}
    assert np.all(labels[crop, 0] == 0)
    assert set(labels[crop, 1].tolist()) == {1, 2}
    # Cell 3 is far from the crop and is not cut.
    assert algorithm.cut_cells == (1, 2)
    assert algorithm.cut_cells == cells_touching_faces(triangles, labels, crop)
    assert report.as_dict()["border_triangles_per_image_face"]["axis0_high"] == int(crop.sum())


def test_declaring_the_face_physical_changes_the_mask_and_not_the_mesh(cropped):
    algorithm, points, triangles, labels = cropped
    physical, p_points, p_triangles, p_labels = _reconstruct(_three_cells(cropped=True), ("axis0_high",))
    assert np.array_equal(points, p_points)
    assert np.array_equal(triangles, p_triangles)
    assert np.array_equal(labels, p_labels)
    assert not physical.crop_faces.any()
    assert physical.cut_cells == ()
    assert physical.crop_face_report.n_border_triangles == algorithm.crop_face_report.n_border_triangles
    # Another face declared physical leaves this crop a crop.
    other, *_ = _reconstruct(_three_cells(cropped=True), ("axis0_low", "axis2_high"))
    assert np.array_equal(other.crop_faces, algorithm.crop_faces)


def test_physical_image_faces_are_checked_and_ordered():
    assert normalise_physical_image_faces(None) == ()
    assert normalise_physical_image_faces("axis1_low") == ("axis1_low",)
    assert normalise_physical_image_faces(["axis2_high", "axis0_low", "axis2_high"]) == ("axis0_low", "axis2_high")
    with pytest.raises(ValueError, match="unknown image face"):
        normalise_physical_image_faces(["bottom"])
    factory = MeshReconstructionAlgorithmFactory().set_physical_image_faces("axis0_low")
    assert factory.make_algorithm().physical_image_faces == ("axis0_low",)
    with pytest.raises(ValueError, match="unknown image face"):
        MeshReconstructionAlgorithmFactory().set_physical_image_faces("coverslip")


@pytest.mark.parametrize("binary_mode", [False, True])
def test_rec_round_trip_keeps_every_crop_face(cropped, tmp_path, binary_mode):
    algorithm, points, triangles, labels = cropped
    path = tmp_path / "cropped.rec"
    algorithm.save_to_rec_mesh(path, binary_mode=binary_mode)
    r_points, r_triangles, r_labels, r_crop = load_rec_with_crop_faces(path)
    crop = algorithm.crop_faces
    n_crop = int(crop.sum())
    # The crop faces are the last block of rows, the other rows first, both in their own order.
    assert np.array_equal(r_crop, np.arange(len(triangles)) >= len(triangles) - n_crop)
    assert np.array_equal(r_triangles[~r_crop], triangles[~crop])
    assert np.array_equal(r_triangles[r_crop], triangles[crop])
    assert np.array_equal(r_labels[r_crop], labels[crop])
    assert np.array_equal(r_points, points)
    appendix = load_rec_appendix(path)
    assert appendix["crop_faces"] == (len(triangles) - n_crop, n_crop)
    assert appendix["physical_image_faces"] == "none"
    # A reader that ignores the appendix reads the same, valid, mesh.
    legacy = load_rec(path)
    assert np.array_equal(legacy[1], r_triangles)
    assert np.array_equal(legacy[2], r_labels)


def test_rec_records_an_empty_crop_when_every_border_face_is_physical(tmp_path):
    physical, _points, triangles, _labels = _reconstruct(_three_cells(cropped=True), ("axis0_high",))
    path = tmp_path / "physical.rec"
    physical.save_to_rec_mesh(path)
    *_geometry, crop = load_rec_with_crop_faces(path)
    assert not crop.any()
    appendix = load_rec_appendix(path)
    assert appendix["crop_faces"] == (len(triangles), 0)
    assert appendix["physical_image_faces"] == "axis0_high"


def test_a_file_without_the_key_reads_no_crop_information(cropped, tmp_path):
    _algorithm, points, triangles, labels = cropped
    path = tmp_path / "old_style.rec"
    save_rec(path, points, triangles, labels)
    *_geometry, crop = load_rec_with_crop_faces(path)
    assert crop is None


def test_an_inconsistent_crop_range_is_refused():
    labels = np.array([[1, 2], [0, 1], [0, 2]])
    assert np.array_equal(crop_faces_from_appendix_range((1, 2), labels), [False, True, True])
    with pytest.raises(ValueError, match="not the last block"):
        crop_faces_from_appendix_range((0, 2), labels)
    with pytest.raises(ValueError, match="medium label 0"):
        crop_faces_from_appendix_range((0, 3), labels)
    with pytest.raises(ValueError, match="first row"):
        crop_faces_from_appendix_range((3,), labels)
    with pytest.raises(ValueError, match="boolean array"):
        save_rec("unused.rec", np.zeros((3, 3)), np.array([[0, 1, 2]] * 3), labels, crop_faces=np.array([0, 1, 1]))


def test_vtk_round_trip_keeps_every_crop_face(cropped, tmp_path):
    algorithm, _points, triangles, labels = cropped
    path = tmp_path / "cropped.vtk"
    algorithm.save_to_vtk_mesh(path)
    _p, r_triangles, r_labels, r_crop = load_meshio_mesh_with_crop_faces(path)
    assert np.array_equal(r_crop, algorithm.crop_faces)
    assert np.array_equal(r_triangles, triangles)
    assert np.array_equal(r_labels, labels)
    assert np.array_equal(load_meshio_mesh(path)[1], triangles)


def test_vtk_without_border_triangles_has_no_crop_array(complete, tmp_path):
    algorithm, *_ = complete
    path = tmp_path / "complete.vtk"
    algorithm.save_to_vtk_mesh(path)
    assert load_meshio_mesh_with_crop_faces(path)[3] is None
