"""dw2d's names mirrored on dw3d's: same calls, same return layout, the 0.1.5 names deprecated, the reports and checks.

Pinned: (1) `get_default_mesh_reconstruction_algorithm(...).construct_mesh_from_segmentation_mask(mask)` returns, bit
for bit, the arrays of the deprecated `geometry_reconstruction_2d(mask, ...).return_mesh()`, split into `(points,
segments, labels)`, on golden cases, with and without junction relocation, and with a label expansion; (2) every name
dw2d 0.1.5 exported still resolves, with one `DeprecationWarning`, and the third-party names its star imports leaked
are gone; (3) the reports are frozen dataclasses; (4) the input checks (a 3-D array, an axis of length 1, a float mask);
(5) the warnings for an enclosed pocket of medium and a label absent from the mesh; (6) the compressed segmentation and
its rebuild under both key sets; (7) the `.rec` file records `dimension = 2`; (8) physical units, square pixels only;
(9) nothing is printed unless `print_info`.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest

import dw2d
from dw2d.functions import _Reconstruction2D

from .conftest import golden_cases


def three_cells() -> np.ndarray:
    """A 60 x 60 label image: three cells meeting at two junctions, in a medium."""
    labels = np.zeros((60, 60), dtype=np.int64)
    labels[10:50, 10:30] = 1
    labels[10:30, 30:50] = 2
    labels[30:50, 30:50] = 3
    return labels


def _legacy(mask, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return dw2d.geometry_reconstruction_2d(mask, **options)


CASES = [c for c in golden_cases() if c[0] in ("c_elegans__md3", "mask_Cells_2_expansion_6__md5", "doublet_03__md3")]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
@pytest.mark.parametrize("relocate", [True, False])
def test_the_mirrored_names_give_the_arrays_of_the_deprecated_route_bitwise(case, relocate):
    _, mask, min_distance, expansion = case
    verts, edges = _legacy(mask, min_dist=min_distance, expansion_labels=expansion,
                           relocate_junctions=relocate).return_mesh()
    factory = dw2d.MeshReconstructionAlgorithmFactory().set_peak_local_points_placement_method(min_distance)
    algorithm = factory.set_label_expansion(expansion).set_junction_relocation(relocate).make_algorithm()
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    assert points.dtype == np.float64 and segments.dtype == labels.dtype == np.int64
    assert segments.shape == labels.shape == (len(edges), 2)
    assert np.array_equal(points, verts)
    assert np.array_equal(np.hstack((segments, labels)), edges)
    assert np.all(labels[:, 0] < labels[:, 1])


def test_the_default_algorithm_is_the_default_route_at_four_pixels():
    mask = three_cells()
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    assert algorithm.min_distance == dw2d.DEFAULT_MIN_DISTANCE == 4
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    verts, edges = _legacy(mask).return_mesh()
    assert np.array_equal(points, verts) and np.array_equal(np.hstack((segments, labels)), edges)
    assert algorithm.last_constructed_mesh[0] is points


def test_one_object_meshes_several_masks_and_its_reports_follow_the_last():
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    first = algorithm.construct_mesh_from_segmentation_mask(three_cells())
    second = algorithm.construct_mesh_from_segmentation_mask(three_cells()[:, ::-1].copy())
    assert not np.array_equal(first[0], second[0])
    assert algorithm.last_constructed_mesh[0] is second[0]


def test_the_deprecated_reconstruction_warns_once_and_names_the_replacement():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dw2d.geometry_reconstruction_2d(three_cells())
    messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(messages) == 1
    assert "get_default_mesh_reconstruction_algorithm" in messages[0] and "0.9" in messages[0]


@pytest.mark.parametrize("name", ["compute_angles", "trace_line", "Delaunay_Graph", "retrieve_border_lines",
                                  "Clean_mesh", "seeded_watershed_map", "build_triangulation"])
def test_every_name_of_0_1_5_resolves_with_one_deprecation_warning(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(dw2d, name)
    assert callable(value)
    messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(messages) == 1 and f"dw2d.{name} is deprecated" in messages[0]


def test_the_names_0_1_5_leaked_from_its_star_imports_are_gone():
    for name in ("np", "plt", "pickle", "struct", "math", "dataclass", "field", "list2cmdline", "expand_labels"):
        with pytest.raises(AttributeError):
            getattr(dw2d, name)


def test_every_name_of_all_resolves_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in dw2d.__all__:
            assert getattr(dw2d, name) is not None


def test_the_reports_are_frozen_dataclasses_and_none_when_off():
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    algorithm.construct_mesh_from_segmentation_mask(three_cells())
    report = algorithm.relocation_report
    assert isinstance(report, dw2d.JunctionRelocationReport)
    assert report.as_dict()["moved"] == report.moved
    with pytest.raises(AttributeError):
        report.moved = 0
    assert algorithm.interface_projection_report is None
    plain = dw2d.MeshReconstructionAlgorithmFactory().set_junction_relocation(False).make_algorithm()
    plain.construct_mesh_from_segmentation_mask(three_cells())
    assert plain.relocation_report is None


def test_the_projection_report_when_asked():
    factory = dw2d.MeshReconstructionAlgorithmFactory().set_peak_local_points_placement_method(3)
    algorithm = factory.set_interface_projection(True).make_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(three_cells())
    assert isinstance(algorithm.interface_projection_report, dw2d.InterfaceProjectionReport)
    with pytest.raises(ValueError, match="requires the junction relocation"):
        factory.set_junction_relocation(False).make_algorithm()


def test_the_public_relocation_takes_dw3d_s_argument_order():
    mask = three_cells()
    plain = dw2d.MeshReconstructionAlgorithmFactory().set_peak_local_points_placement_method(3)
    plain = plain.set_junction_relocation(False).make_algorithm()
    points, segments, labels = plain.construct_mesh_from_segmentation_mask(mask)
    moved, report = dw2d.relocate_junction_vertices(mask, points, segments, labels,
                                                    fixed_vertices=plain.image_border_vertices)
    relocated = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    expected = relocated.construct_mesh_from_segmentation_mask(mask)[0]
    assert np.array_equal(moved, expected)
    assert report == relocated.relocation_report


@pytest.mark.parametrize("shape", [(1, 60, 60), (60, 1, 60), (60, 60, 1)])
def test_a_3d_array_with_an_axis_of_length_one_is_refused_never_squeezed(shape):
    with pytest.raises(ValueError, match=r"mask\[0\]"):
        dw2d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(np.zeros(shape, int))


def test_a_3d_or_1d_mask_is_refused():
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    with pytest.raises(ValueError, match="dw3d"):
        algorithm.construct_mesh_from_segmentation_mask(np.zeros((5, 6, 7), int))
    with pytest.raises(ValueError, match="2-D masks"):
        algorithm.construct_mesh_from_segmentation_mask(np.zeros(9, int))


def test_a_float_mask_warns_and_is_meshed_as_before():
    mask = three_cells()
    with pytest.warns(UserWarning, match="not an integer type"):
        points, _, _ = dw2d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(
            mask.astype(float))
    assert np.array_equal(points, dw2d.get_default_mesh_reconstruction_algorithm()
                          .construct_mesh_from_segmentation_mask(mask)[0])


def test_an_enclosed_pocket_of_medium_is_named():
    mask = three_cells()
    mask[18:24, 15:21] = 0  # a hole in cell 1
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    with pytest.warns(dw2d.EnclosedMediumWarning, match="36 px at"):
        algorithm.construct_mesh_from_segmentation_mask(mask)
    (pocket,) = algorithm.enclosed_medium_pockets
    assert pocket.pixels == 36 and pocket.first_pixel == (18, 15) and pocket.lining_labels == (1,)
    assert algorithm.medium_labels == (0,)


def test_a_label_absent_from_the_mesh_is_named():
    mask = three_cells()
    mask[20, 20] = 7  # one pixel inside cell 1: at min_distance 6 no triangle of the watershed carries it
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=6)
    # Unpadded: the cells are 10 px from every side, below the bound of 14 px at min_distance 6, so the default pads
    # them, which keeps cell 1 whole (0.999 of its pixels, 0.668 unpadded) and gives label 7 a sliver of 18 px^2
    # reaching the padded corner, named by ThinMediumMarginWarning instead.
    algorithm.pad_thin_medium_margins = False
    with pytest.warns(dw2d.UnresolvedCellWarning, match="7"):
        algorithm.construct_mesh_from_segmentation_mask(mask)
    assert algorithm.unresolved_cells == (7,)


def test_the_cut_cells_are_those_with_a_vertex_on_the_image_border():
    mask = three_cells()
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    algorithm.construct_mesh_from_segmentation_mask(mask)
    assert algorithm.cut_cells == ()
    mask[:, :10] = 1  # cell 1 now reaches the left border
    algorithm.construct_mesh_from_segmentation_mask(mask)
    assert algorithm.cut_cells == (1,)


def test_the_compressed_segmentation_and_its_rebuild_under_both_key_sets(tmp_path):
    mask = three_cells()
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    *_, compressed = algorithm.both_construct_mesh_and_compressed_segmentation(mask)
    assert set(compressed) == {"points", "segments", "labels", "seeds", "image_shape", "image_border_vertices"}
    rebuilt = dw2d.reconstruct_mask_from_dict(compressed)
    assert rebuilt.shape == mask.shape and np.mean(rebuilt == mask) > 0.95
    dw2d.save_compressed_segmentation(tmp_path / "compressed.npy", compressed)
    assert np.array_equal(dw2d.reconstruct_mask_from_saved_file_dict(tmp_path / "compressed.npy"), rebuilt)
    old = _legacy(mask, min_dist=3)
    old.export_segmentation(tmp_path / "old.npy")
    with pytest.warns(DeprecationWarning, match="reconstruct_mask_from_saved_file_dict"):
        from_path = dw2d.reconstruct_mask_from_dict(tmp_path / "old.npy")
    assert np.array_equal(from_path, rebuilt)


@pytest.mark.parametrize("binary_mode", [False, True])
def test_the_rec_file_records_its_dimension_and_reads_back(tmp_path, binary_mode):
    import dw2d.io

    mask = three_cells()
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    algorithm.save_to_rec_mesh(tmp_path / "mesh.rec", binary_mode)
    read = dw2d.io.load_rec(tmp_path / "mesh.rec")
    for a, b in zip(read, (points, segments, labels), strict=True):
        assert np.array_equal(a, b)
    appendix = dw2d.io.load_rec_appendix(tmp_path / "mesh.rec")
    assert appendix["dimension"] == 2 and appendix["axis_order"] == "yx" and appendix["source_shape"] == (60, 60)
    assert np.array_equal(appendix["image_border_vertices"], algorithm.image_border_vertices)


def test_physical_units_scale_square_pixels_and_refuse_others():
    mask = three_cells()
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    pixels = algorithm.construct_mesh_from_segmentation_mask(mask)[0].copy()
    points, _, _ = algorithm.construct_mesh_in_physical_units(mask, (0.25, 0.25), spacing_unit="um")
    assert np.array_equal(points, pixels * 0.25)
    metadata = algorithm.rec_metadata()
    assert metadata["coordinate_frame"] == "physical" and metadata["spacing"] == (0.25, 0.25)
    assert np.array_equal(algorithm.construct_mesh_in_physical_units(mask, (1.0, 1.0))[0], pixels)
    with pytest.raises(ValueError, match="square pixels"):
        algorithm.construct_mesh_in_physical_units(mask, (0.25, 0.5))
    with pytest.raises(ValueError, match="two positive"):
        algorithm.construct_mesh_in_physical_units(mask, (0.25, 0.25, 0.25))


def test_nothing_is_printed_unless_asked():
    mask = three_cells()
    for print_info, expect_output in ((False, False), (True, True)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            dw2d.get_default_mesh_reconstruction_algorithm(print_info=print_info).construct_mesh_from_segmentation_mask(mask)
        assert bool(out.getvalue()) == expect_output


def test_the_engine_is_the_deprecated_class_without_its_warning():
    assert issubclass(dw2d.functions.geometry_reconstruction_2d, _Reconstruction2D)
