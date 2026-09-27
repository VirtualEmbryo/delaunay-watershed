"""Why the tiled EDT is **not** wired into `get_link_checked_algorithm`.

The link-checked default was asked to use `set_tiled_edt_method` for its EDT "if that is
a clean composition". It is not, and this file is the measurement that says so, kept as a
standing fixture rather than as a paragraph of prose. `get_link_checked_algorithm`
(boundary layer + junction protection + link-condition-checked offset exclusion) was a
former default, when the default reverted to `dithered` on end-to-end accuracy grounds
(see `get_default_algorithm`'s docstring); this file's finding is about
`get_link_checked_algorithm`'s own pipeline, which is unaffected by which configuration
`get_default_algorithm` currently points at, so it is checked directly against
`get_link_checked_mesh_reconstruction_algorithm` rather than through the default.

**The finding, in one sentence:** `compute_edt_tiled` returns a field whose *values* are
bit-identical to `compute_edt_classical`'s — exactly as the EDT memory/scaling work
measured, `atol = 0` on all 14 cases it checked — but whose **dtype is `float32`**, and
`get_link_checked_algorithm` reads the EDT through two dtype-sensitive consumers, so the
finished reconstruction moves on 8 of 8 in-repo configurations.

**Why the EDT memory/scaling work could not have seen this.** That work branched at
`727a546`, before the default flipped to the junction-protected configuration, and its
acceptance criterion "the seeding output is identical" was measured against the
deterministic point placer, which reads the EDT only through an integer-plateau extremum
filter — insensitive to dtype on values that are exactly `float32`-representable, which
these are. The default has moved twice since. It now places points with
`peak_local_points_junction_protected`, which builds a 3x3x3 Hessian by central finite
differences and takes `np.linalg.eigh` of it to get the boundary layer's across-sheet
normals, and it scores faces by interpolating the EDT at 27 barycentric samples per face.
Both are float arithmetic on the field, and both give different last bits at `float32`.
That measurement is correct as stated; what does not transfer is its *scope*.

Nothing here is an argument against the tiled EDT. `compute_edt_tiled` is the variant to
prefer for the scaling goal — 3.41x lower peak RSS than `compute_edt_classical` at
1024^3, against dense `float32`'s 2.67x — and this file exists so that the *cost* of
adopting it is on the record: adopting it is a deliberate re-baselining of every golden
master, fingerprint and 51-case record, not a free drop-in.
"""

from functools import partial

import numpy as np
import pytest

from dw3d import get_link_checked_mesh_reconstruction_algorithm
from dw3d.edt import compute_edt_classical, compute_edt_float32, compute_edt_tiled
from tests.conftest import load_image_or_skip

CASE_IMAGE = "3.tif"
MIN_DISTANCE = 3
TILE_SIZE = 256
HALO = 16


def _tiled(mask: np.ndarray, **_kwargs: object) -> np.ndarray:
    return compute_edt_tiled(mask, tile_size=TILE_SIZE, halo=HALO, print_info=False, parallel=1)


def _tiled_as_float64(mask: np.ndarray, **_kwargs: object) -> np.ndarray:
    return _tiled(mask).astype(np.float64)


@pytest.fixture(scope="module")
def mask():
    return load_image_or_skip(CASE_IMAGE)


def _reconstruct(mask, edt_creation_function=None):
    algo = get_link_checked_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)
    if edt_creation_function is not None:
        algo.edt_creation_function = edt_creation_function
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    return {
        "points": points,
        "triangles": triangles,
        "labels": labels,
        "scores": algo._tesselation_graph.scores,
        "edt_dtype": algo._edt_image.dtype,
    }


def test_the_tiled_field_is_value_identical_and_dtype_different(mask):
    """The EDT memory/scaling work's claim, reproduced, and the half of it that the word "bit-identical" hides."""
    classical = compute_edt_classical(mask, print_info=False)
    tiled = _tiled(mask)
    dense_float32 = compute_edt_float32(mask, print_info=False, parallel=1)

    assert np.array_equal(classical, tiled), "the tiled EDT's core claim: its field values are exact"
    assert np.array_equal(classical, dense_float32)
    assert float(np.abs(classical - tiled).max()) == 0.0

    assert classical.dtype == np.float64
    assert tiled.dtype == np.float32
    assert dense_float32.dtype == np.float32


def test_link_checked_is_not_bit_identical_under_the_tiled_edt(mask):
    """The composition the link-checked default was asked about, measured rather than assumed.

    If this ever starts passing as an equality — because someone made the EDT contract
    `float32` end to end, or made the Hessian and score stages promote — then wiring
    `set_tiled_edt_method` into `get_link_checked_algorithm` becomes free and this test
    should be inverted, deliberately, with every fixture re-baselined in the same commit.
    """
    reference = _reconstruct(mask)
    tiled = _reconstruct(mask, partial(_tiled))

    assert reference["edt_dtype"] == np.float64
    assert tiled["edt_dtype"] == np.float32
    assert not np.array_equal(reference["scores"], tiled["scores"])
    assert (len(reference["points"]), len(reference["triangles"])) != (
        len(tiled["points"]),
        len(tiled["triangles"]),
    )


def test_the_divergence_is_the_dtype_and_not_the_tiling(mask):
    """The controlled experiment that makes the attribution a measurement.

    Same tiled field, promoted back to `float64` before anything consumes it: the
    reconstruction returns to bit-identity with `get_link_checked_algorithm`'s own output.
    So the tiling guarantee is not in question — `dw3d.edt`'s certified escalating halo does
    what it claims — and the difference is entirely the storage precision the field arrives
    in.
    """
    reference = _reconstruct(mask)
    promoted = _reconstruct(mask, _tiled_as_float64)

    assert promoted["edt_dtype"] == np.float64
    np.testing.assert_array_equal(promoted["points"], reference["points"])
    np.testing.assert_array_equal(promoted["triangles"], reference["triangles"])
    np.testing.assert_array_equal(promoted["labels"], reference["labels"])
    np.testing.assert_array_equal(promoted["scores"], reference["scores"])


def test_tiled_and_dense_float32_agree_with_each_other(mask):
    """The second half of the attribution: it is dtype, not tiling, so the two must agree.

    `compute_edt_float32` does no tiling at all. If the tiled path's divergence came from
    the halo escalation rather than from the storage precision, these two would differ.
    """
    tiled = _reconstruct(mask, partial(_tiled))
    dense = _reconstruct(mask, partial(compute_edt_float32, print_info=False, parallel=1))

    np.testing.assert_array_equal(tiled["points"], dense["points"])
    np.testing.assert_array_equal(tiled["triangles"], dense["triangles"])
    np.testing.assert_array_equal(tiled["labels"], dense["labels"])
    np.testing.assert_array_equal(tiled["scores"], dense["scores"])
