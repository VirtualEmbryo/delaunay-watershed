"""The reconstruction surface in 0.5.0: three names removed, three exact synonyms deprecated.

`get_v0_3_algorithm`, `get_a1b_algorithm` and `get_a3_a5_algorithm` (and their top-level
`get_*_mesh_reconstruction_algorithm` aliases) warned from 0.4.0 that they would be removed in
0.5.0; they are gone, and their replacements are `get_dithered_algorithm`,
`get_deterministic_algorithm` and `get_junction_protected_algorithm`.

`get_offset_included_algorithm`, `get_offset_excluded_algorithm` and `get_cubic_score_algorithm`
are exact synonyms of keyword calls on `get_junction_protected_algorithm`. They had never warned, so
removing them outright would break users without notice: in 0.5.0 each emits a
`DeprecationWarning` naming the equivalent call and announcing removal in 0.6.0, and each must
still reproduce that call bit for bit -- same mesh, not a re-implementation that could drift.
"""

import warnings
from functools import partial

import numpy as np
import pytest

import dw3d
from dw3d import MeshReconstructionAlgorithmFactory as Factory
from tests.conftest import load_image_or_skip

MIN_DISTANCE = 3
CASE_IMAGE = "3.tif"

REMOVED = ("get_v0_3_algorithm", "get_a1b_algorithm", "get_a3_a5_algorithm")
REMOVED_TOP_LEVEL = (
    "get_v0_3_mesh_reconstruction_algorithm",
    "get_a1b_mesh_reconstruction_algorithm",
    "get_a3_a5_mesh_reconstruction_algorithm",
)

#: (deprecated synonym, the keyword call it stands for, text its warning must contain)
SYNONYMS = [
    (
        "get_offset_included_algorithm",
        partial(Factory.get_junction_protected_algorithm),
        "get_junction_protected_algorithm(",
    ),
    (
        "get_offset_excluded_algorithm",
        partial(Factory.get_junction_protected_algorithm, exclude_offsets_from_surface=True),
        "exclude_offsets_from_surface=True",
    ),
    ("get_cubic_score_algorithm", partial(Factory.get_junction_protected_algorithm, spline_order=3), "spline_order=3"),
]
SYNONYM_IDS = [name for name, _, _ in SYNONYMS]


@pytest.mark.parametrize("name", REMOVED)
def test_the_aliases_announced_for_removal_are_removed(name):
    assert not hasattr(Factory, name)


@pytest.mark.parametrize("name", REMOVED_TOP_LEVEL)
def test_their_top_level_exports_are_removed(name):
    assert not hasattr(dw3d, name)
    assert name not in dw3d.__all__


@pytest.mark.parametrize(("name", "replacement", "text"), SYNONYMS, ids=SYNONYM_IDS)
def test_the_synonym_warns_names_its_replacement_and_the_removal_version(name, replacement, text):  # noqa: ARG001
    with pytest.warns(DeprecationWarning, match="deprecated since 0.5.0") as caught:
        getattr(Factory, name)(min_distance=MIN_DISTANCE, print_info=False)
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 1
    assert text in messages[0]
    assert "0.6.0" in messages[0]


@pytest.mark.parametrize(("name", "replacement", "text"), SYNONYMS, ids=SYNONYM_IDS)
def test_the_synonym_reproduces_its_keyword_call_exactly(name, replacement, text):  # noqa: ARG001
    """On `data/Images/3.tif` at `min_distance=3`, under the 0.5.0 default (relocation on)."""
    mask = load_image_or_skip(CASE_IMAGE)
    with pytest.warns(DeprecationWarning, match="deprecated since 0.5.0"):
        old = getattr(Factory, name)(min_distance=MIN_DISTANCE, print_info=False)
    new = replacement(min_distance=MIN_DISTANCE, print_info=False)
    old_mesh = old.construct_mesh_from_segmentation_mask(mask)
    new_mesh = new.construct_mesh_from_segmentation_mask(mask)
    for old_array, new_array in zip(old_mesh, new_mesh, strict=True):
        assert np.array_equal(old_array, new_array)


@pytest.mark.parametrize(
    "name",
    [
        "get_default_algorithm",
        "get_dithered_algorithm",
        "get_deterministic_algorithm",
        "get_boundary_layer_algorithm",
        "get_junction_protected_algorithm",
        "get_link_checked_algorithm",
    ],
)
def test_the_six_recommended_configurations_do_not_warn(name):
    """`get_link_checked_algorithm` in particular no longer routes through a deprecated synonym."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(Factory, name)(min_distance=MIN_DISTANCE, print_info=False)
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecation_warnings, [str(w.message) for w in deprecation_warnings]
