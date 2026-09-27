"""Junction relocation on by default (0.5.0): the warning, the provenance, and the equivalence it rests on.

Pins four things a user of the default relies on:

* the default reconstruction is exactly the relocation-off reconstruction followed by
  `dw3d.relocate_junction_vertices` on the same mask -- the default path adds the post-process and
  nothing else, and connectivity and labels are untouched;
* `CoarseSpacingRelocationWarning` fires exactly once per reconstruction when relocation is on and
  the placement's `min_distance` is at least 5, with the documented text, and never otherwise --
  not below 5, not with relocation off, not for a placement that carries no `min_distance`;
* `relocation_provenance` records whether relocation ran, with which guard, and whether the warning
  fired;
* `relocate_junctions=False` is the way back to the 0.4 mesh (the golden masters under
  `tests/golden_without_relocation/` pin that on every configuration; here it is checked against
  the post-process equivalence).
"""

import warnings
from functools import partial

import numpy as np
import pytest

import dw3d
from dw3d.junction_relocation import COARSE_SPACING_RELOCATION_MESSAGE, CoarseSpacingRelocationWarning
from dw3d.points_on_edt import peak_local_points_dithered
from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm
from tests.conftest import load_image_or_skip

CASE_IMAGE = "3.tif"


def _coarse_warnings(caught: list) -> list:
    return [w for w in caught if issubclass(w.category, CoarseSpacingRelocationWarning)]


def _reconstruct(algorithm: MeshReconstructionAlgorithm, mask: np.ndarray) -> tuple[tuple, list]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = algorithm.construct_mesh_from_segmentation_mask(mask)
    return mesh, caught


def test_the_default_is_the_untreated_mesh_plus_the_post_process():
    mask = load_image_or_skip(CASE_IMAGE)
    default = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    untreated = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    untreated.relocate_junctions = False
    (points, triangles, labels), _ = _reconstruct(default, mask)
    (base_points, base_triangles, base_labels), _ = _reconstruct(untreated, mask)

    moved, report = dw3d.relocate_junction_vertices(mask, base_points, base_triangles, base_labels)
    assert np.array_equal(points, moved)
    assert np.array_equal(triangles, base_triangles)
    assert np.array_equal(labels, base_labels)
    assert report.n_accepted > 0, "the fixture must exercise the relocation, or this proves nothing"
    assert not np.array_equal(points, base_points)


@pytest.mark.parametrize(("min_distance", "expected"), [(3, 0), (4, 0), (5, 1)])
def test_the_coarse_spacing_warning_fires_once_from_min_distance_5(min_distance, expected):
    mask = load_image_or_skip(CASE_IMAGE)
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=min_distance)
    _, caught = _reconstruct(algorithm, mask)
    coarse = _coarse_warnings(caught)
    assert len(coarse) == expected
    if expected:
        assert str(coarse[0].message) == COARSE_SPACING_RELOCATION_MESSAGE
        assert coarse[0].filename == __file__, "stacklevel must point at the caller"
    assert algorithm.relocation_provenance["coarse_spacing_warning_emitted"] is bool(expected)


def test_the_warning_fires_again_on_the_next_reconstruction_not_twice_on_one():
    mask = load_image_or_skip(CASE_IMAGE)
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=5)
    _, first = _reconstruct(algorithm, mask)
    _, second = _reconstruct(algorithm, mask)
    assert len(_coarse_warnings(first)) == 1
    assert len(_coarse_warnings(second)) == 1


def test_no_warning_with_relocation_off():
    mask = load_image_or_skip(CASE_IMAGE)
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=5)
    algorithm.relocate_junctions = False
    _, caught = _reconstruct(algorithm, mask)
    assert not _coarse_warnings(caught)
    provenance = algorithm.relocation_provenance
    assert provenance["ran"] is False
    assert provenance["report"] is None
    assert provenance["coarse_spacing_warning_emitted"] is False


def test_no_warning_when_the_placement_carries_no_min_distance():
    """A custom placement callable (not a `functools.partial` with `min_distance`) cannot be read."""
    mask = load_image_or_skip(CASE_IMAGE)
    wrapped = partial(peak_local_points_dithered, min_distance=5, seed=42, print_info=False)
    reference = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=5)

    def custom_placement(segmented_image, edt_image):
        return wrapped(segmented_image, edt_image)

    algorithm = MeshReconstructionAlgorithm(
        print_info=False,
        edt_creation_function=reference.edt_creation_function,
        point_placing_function=custom_placement,
        tesselation_creation_function=reference.tesselation_creation_function,
        score_computation_function=reference.score_computation_function,
        perform_mesh_postprocess_surgery=True,
    )
    assert algorithm.placement_min_distance is None
    _, caught = _reconstruct(algorithm, mask)
    assert not _coarse_warnings(caught)
    assert algorithm.relocation_provenance["ran"] is True


def test_the_provenance_records_that_relocation_ran_and_how():
    mask = load_image_or_skip(CASE_IMAGE)
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=3)
    assert algorithm.relocation_provenance["ran"] is False, "nothing has run before the first mesh"
    _reconstruct(algorithm, mask)
    provenance = algorithm.relocation_provenance
    assert provenance["relocate_junctions"] is True
    assert provenance["ran"] is True
    assert provenance["min_distance"] == 3
    assert "nonlocal self-intersection guard" in provenance["guard"]
    assert provenance["report"]["n_accepted"] == algorithm.relocation_report.n_accepted
    assert provenance["report"]["n_self_intersections_after"] <= provenance["report"]["n_self_intersections_before"]


def test_the_warning_class_is_exported_and_filterable():
    assert issubclass(dw3d.CoarseSpacingRelocationWarning, UserWarning)
    mask = load_image_or_skip(CASE_IMAGE)
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm(min_distance=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", dw3d.CoarseSpacingRelocationWarning)
        with pytest.raises(dw3d.CoarseSpacingRelocationWarning):
            algorithm.construct_mesh_from_segmentation_mask(mask)
