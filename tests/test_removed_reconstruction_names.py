"""The reconstruction getters that were removed, and the six that remain.

`get_v0_3_algorithm`, `get_a1b_algorithm` and `get_a3_a5_algorithm` (and their top-level
`get_*_mesh_reconstruction_algorithm` aliases) warned from 0.4.0 that they would be removed in
0.5.0; they are gone, and their replacements are `get_dithered_algorithm`,
`get_deterministic_algorithm` and `get_junction_protected_algorithm`.

`get_offset_included_algorithm`, `get_offset_excluded_algorithm` and `get_cubic_score_algorithm`
were exact synonyms of keyword calls on `get_junction_protected_algorithm`, deprecated in 0.5.0 with
a `DeprecationWarning` naming the call; they were removed in 0.7.0, with their top-level aliases. The
calls that replace them are `get_junction_protected_algorithm()`,
`get_junction_protected_algorithm(exclude_offsets_from_surface=True)` and
`get_junction_protected_algorithm(spline_order=3)`; the golden-master tests pin each of those calls to
the reference meshes the removed names reproduced.
"""

import warnings

import pytest

import dw3d
from dw3d import MeshReconstructionAlgorithmFactory as Factory

MIN_DISTANCE = 3

REMOVED = (
    "get_v0_3_algorithm",
    "get_a1b_algorithm",
    "get_a3_a5_algorithm",
    "get_offset_included_algorithm",
    "get_offset_excluded_algorithm",
    "get_cubic_score_algorithm",
)
REMOVED_TOP_LEVEL = (
    "get_v0_3_mesh_reconstruction_algorithm",
    "get_a1b_mesh_reconstruction_algorithm",
    "get_a3_a5_mesh_reconstruction_algorithm",
    "get_offset_included_mesh_reconstruction_algorithm",
    "get_offset_excluded_mesh_reconstruction_algorithm",
    "get_cubic_score_mesh_reconstruction_algorithm",
)


@pytest.mark.parametrize("name", REMOVED)
def test_the_removed_getters_are_gone(name):
    assert not hasattr(Factory, name)


@pytest.mark.parametrize("name", REMOVED_TOP_LEVEL)
def test_their_top_level_exports_are_gone(name):
    assert not hasattr(dw3d, name)
    assert name not in dw3d.__all__


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
    """None of the remaining getters routes through a deprecated name."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(Factory, name)(min_distance=MIN_DISTANCE, print_info=False)
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecation_warnings, [str(w.message) for w in deprecation_warnings]
