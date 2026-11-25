"""dw3d's algorithm hands a 2-D mask to dw2d, and checks every mask's dimension at its entry points.

Pinned: (1) the default (dithered) and the deterministic algorithms mesh a 2-D mask exactly as dw2d's own default
algorithm does (bitwise), with dw2d's default `min_distance` (4 pixels) unless one was given; (2) a configuration with
no 2-D meaning is refused with a message naming it, never meshed by a silent fallback; (3) after a 2-D mask the
algorithm's mesh, compressed segmentation, `.rec` file and reports are dw2d's, the 3-D-only reports are `None` and the
3-D-only operations refuse; a 3-D mask afterwards gives the 3-D state back; (4) a 3-D mask with an axis of length 1 is
refused with a message suggesting `mask[0]`, a mask of another dimension is refused, a float 3-D mask warns.
"""

import warnings

import numpy as np
import pytest

import dw2d
import dw3d
import dw3d.io


def three_cells() -> np.ndarray:
    labels = np.zeros((60, 60), dtype=np.int64)
    labels[10:50, 10:30] = 1
    labels[10:30, 30:50] = 2
    labels[30:50, 30:50] = 3
    return labels


def two_boxes() -> np.ndarray:
    mask = np.zeros((24, 24, 24), dtype=np.uint16)
    mask[4:20, 4:12, 4:20] = 1
    mask[4:20, 12:20, 4:20] = 2
    return mask


def _same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("getter", [dw3d.get_default_mesh_reconstruction_algorithm,
                                    dw3d.get_dithered_mesh_reconstruction_algorithm,
                                    dw3d.get_deterministic_mesh_reconstruction_algorithm])
def test_a_2d_mask_is_meshed_by_dw2d_at_its_default_spacing(getter):
    mask = three_cells()
    algorithm = getter()
    mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    assert _same(mesh, dw2d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(mask))
    assert algorithm.two_dimensional_algorithm.min_distance == 4
    assert mesh[0].shape[1] == 2 and mesh[1].shape[1] == 2


def test_an_explicit_min_distance_is_used_in_both_dimensions():
    mask = three_cells()
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    assert algorithm.placement_min_distance == 3 and algorithm.requested_min_distance == 3
    mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    assert _same(mesh, dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
                 .construct_mesh_from_segmentation_mask(mask))
    factory = dw3d.MeshReconstructionAlgorithmFactory().set_peak_local_points_placement_method(5)
    assert factory.make_algorithm().construct_mesh_from_segmentation_mask(mask)[0].shape == \
        dw2d.get_default_mesh_reconstruction_algorithm(min_distance=5).construct_mesh_from_segmentation_mask(mask)[0].shape


def test_the_3d_default_spacing_is_unchanged():
    assert dw3d.get_default_mesh_reconstruction_algorithm().placement_min_distance == 3
    assert dw3d.MeshReconstructionAlgorithmFactory().make_algorithm().placement_min_distance == 3


def test_relocation_and_canonical_order_follow_the_algorithm():
    mask = three_cells()
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm()
    algorithm.relocate_junctions = False
    algorithm.canonical_tetrahedron_order = False
    mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    expected = (dw2d.MeshReconstructionAlgorithmFactory().set_junction_relocation(False)
                .set_canonical_triangle_order(False).make_algorithm().construct_mesh_from_segmentation_mask(mask))
    assert _same(mesh, expected)
    assert algorithm.relocation_report is None


@pytest.mark.parametrize(("getter", "named"), [
    (dw3d.get_boundary_layer_mesh_reconstruction_algorithm, "peak_local_points_boundary_layer"),
    (dw3d.get_junction_protected_mesh_reconstruction_algorithm, "point placement"),
    (dw3d.get_link_checked_mesh_reconstruction_algorithm, "point placement"),
])
def test_a_configuration_without_a_2d_meaning_is_refused(getter, named):
    with pytest.raises(ValueError, match=named):
        getter().construct_mesh_from_segmentation_mask(three_cells())


def test_physical_image_faces_and_surgery_off_are_refused_in_2d():
    algorithm = dw3d.MeshReconstructionAlgorithmFactory().set_physical_image_faces("axis0_low").make_algorithm()
    with pytest.raises(ValueError, match="physical image faces"):
        algorithm.construct_mesh_from_segmentation_mask(three_cells())
    algorithm = dw3d.MeshReconstructionAlgorithmFactory(perform_mesh_postprocess_surgery=False).make_algorithm()
    with pytest.raises(ValueError, match="mesh surgery off"):
        algorithm.construct_mesh_from_segmentation_mask(three_cells())


def test_after_a_2d_mask_the_results_are_dw2d_s_and_a_3d_mask_restores_the_3d_state(tmp_path):
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(three_cells())
    two = algorithm.two_dimensional_algorithm
    assert _same(algorithm.last_constructed_mesh, (points, segments, labels))
    assert algorithm.last_compressed_segmentation.keys() == two.last_compressed_segmentation.keys()
    assert algorithm.relocation_report == two.relocation_report and algorithm.cut_cells == two.cut_cells == ()
    assert algorithm.medium_labels == (0,)
    for report in (algorithm.crop_faces, algorithm.crop_face_report, algorithm.cell_volume_report,
                   algorithm.enclosed_medium_report):
        assert report is None
    algorithm.save_to_rec_mesh(tmp_path / "mesh.rec")
    assert dw3d.io.rec_dimension(tmp_path / "mesh.rec") == 2
    for operation, args in (("set_pixel_size", (1.0, 1.0)), ("set_global_min_max", (0.0, 1.0)),
                            ("center_around_zero", ()), ("save_to_vtk_mesh", (tmp_path / "mesh.vtk",))):
        with pytest.raises(ValueError, match="no 2-D counterpart"):
            getattr(algorithm, operation)(*args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points3, triangles, _ = algorithm.construct_mesh_from_segmentation_mask(two_boxes())
    assert algorithm.two_dimensional_algorithm is None
    assert points3.shape[1] == 3 and triangles.shape[1] == 3
    assert algorithm.crop_face_report is not None


def test_physical_units_in_2d_take_two_values():
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    points, _, _ = algorithm.construct_mesh_in_physical_units(three_cells(), (0.5, 0.5), spacing_unit="um")
    pixels = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3).construct_mesh_from_segmentation_mask(
        three_cells())[0]
    assert np.array_equal(points, pixels * 0.5)
    assert algorithm.two_dimensional_algorithm.rec_metadata()["axis_order"] == "yx"


@pytest.mark.parametrize("shape", [(1, 24, 24), (24, 1, 24), (24, 24, 1)])
def test_a_3d_mask_with_an_axis_of_length_one_is_refused(shape):
    with pytest.raises(ValueError, match=r"mask\[0\]"):
        dw3d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(np.ones(shape, int))


@pytest.mark.parametrize("shape", [(24,), (2, 3, 4, 5)])
def test_a_mask_neither_2d_nor_3d_is_refused(shape):
    with pytest.raises(ValueError, match="2-D or 3-D"):
        dw3d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(np.ones(shape, int))


def test_a_float_3d_mask_warns_and_is_meshed():
    with pytest.warns(UserWarning, match="not an integer type"):
        points, _, _ = dw3d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(
            two_boxes().astype(np.float64))
    assert len(points)
