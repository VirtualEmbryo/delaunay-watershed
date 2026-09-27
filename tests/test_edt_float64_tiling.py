"""The tiled EDT in `float64` — does it compose with `get_link_checked_algorithm`?

**The question, and where it comes from.** `tests/test_edt_dtype_composition.py` pins the finding
that the tiled EDT does *not* compose with `get_link_checked_algorithm` (boundary layer +
junction protection + link-condition-checked offset exclusion, a former default, now the
opt-in "revised meshing" variant — see `get_default_algorithm`'s docstring): the tiled field's values are
bit-identical to
`compute_edt_classical`'s, but its **dtype is `float32`**, and `get_link_checked_algorithm`
reads the field's last bits (the boundary layer's Hessian-eigenvector normals via `np.linalg.eigh`, and the
27-barycentric-sample face score interpolation), so the finished reconstruction moves on 8 of 8
in-repo configurations. That file also records the controlled experiment: promoting the *same*
tiled field to `float64` before anything consumes it returns the reconstruction to bit-identity.
This composition question is about `get_link_checked_algorithm`'s own pipeline and is unaffected
by which configuration `get_default_algorithm` currently points at, so this file is checked
directly against `get_link_checked_mesh_reconstruction_algorithm` rather than through the default.

H. Turlier recorded the obvious follow-up as a single experiment rather than a larger
piece of work: the tiled EDT's memory win comes mostly
from never holding the whole volume at once, not from the narrower dtype, so **emitting the
tiled EDT in `float64` should compose and keep most of the saving.**
`dw3d.edt.compute_edt_tiled(..., dtype=np.float64)` is that path. This file is the
composition half of the answer; the memory half is `benchmarks/edt_scaling.py --variants
tiled_float64` at 1024^3.

**What is asserted, and why each assertion is here.**

1. The `float64` tiled field equals `compute_edt_classical`'s **exactly** (`atol = 0`) — the
   guarantee the EDT memory/scaling work established, restated for the new dtype rather than assumed to carry over.
2. It equals `compute_edt_tiled(...).astype(np.float64)` **bit for bit**. This is the assertion
   that protects the implementation's one real hazard: the field is assembled as
   `edt_1 + max_edt_2` and `max_edt_2 - edt_2`, and doing that arithmetic in `float64` rather than
   in `float32`-then-widen produces a *different* field (`float64(a) + float64(b)` is not
   `float64(float32(a) + float32(b))`), which would silently break the composition this variant
   exists to deliver.
3. The finished reconstruction is bit-identical to `get_link_checked_algorithm`'s own on points,
   triangles, labels **and scores** — the composition claim itself.
4. It is *not* the `float32` tiled path's reconstruction, which keeps the contrast explicit: the
   two differ only in the field's storage precision, and that is enough to move the mesh.

Unlike `test_edt_dtype_composition.py`, none of these tests is written to fail deliberately. If
one starts failing, something changed in the EDT assembly or in a consumer's precision, and the
`float64`-tiling route to 2048^3 needs re-measuring.
"""

import re
from functools import partial

import numpy as np
import pytest

from dw3d import get_link_checked_mesh_reconstruction_algorithm
from dw3d.edt import compute_edt_classical, compute_edt_tiled
from tests.conftest import load_image_or_skip

CASE_IMAGE = "3.tif"
MIN_DISTANCE = 3
TILE_SIZE = 256
HALO = 16


def _tiled(mask: np.ndarray, dtype: type, **_kwargs: object) -> np.ndarray:
    return compute_edt_tiled(mask, tile_size=TILE_SIZE, halo=HALO, print_info=False, parallel=1, dtype=dtype)


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


def test_the_float64_tiled_field_equals_the_classical_field_exactly(mask):
    """The EDT memory/scaling work's exactness guarantee, restated for the new dtype, not assumed."""
    classical = compute_edt_classical(mask, print_info=False)
    tiled64 = _tiled(mask, np.float64)

    assert tiled64.dtype == np.float64
    assert classical.dtype == np.float64
    assert np.array_equal(classical, tiled64)
    assert float(np.abs(classical - tiled64).max()) == 0.0


def test_the_float64_field_is_the_float32_field_widened_bit_for_bit(mask):
    """The assertion that guards the assembly's precision, not just its result.

    If the offsets were applied in `float64` (`out += max_edt_2` on a `float64` buffer) instead of
    in `float32`-then-widened, this would fail while test 1 above might still pass on this image —
    and the reconstruction would then diverge from `get_link_checked_algorithm`'s own for a reason
    no other test names.
    """
    tiled32 = _tiled(mask, np.float32)
    tiled64 = _tiled(mask, np.float64)

    assert tiled32.dtype == np.float32
    np.testing.assert_array_equal(tiled32.astype(np.float64), tiled64)


def test_link_checked_reconstruction_is_bit_identical_under_the_float64_tiled_edt(mask):
    """The composition claim: this is the variant that can be swapped into `get_link_checked_algorithm`."""
    reference = _reconstruct(mask)
    tiled64 = _reconstruct(mask, partial(_tiled, dtype=np.float64))

    assert reference["edt_dtype"] == np.float64
    assert tiled64["edt_dtype"] == np.float64
    np.testing.assert_array_equal(tiled64["points"], reference["points"])
    np.testing.assert_array_equal(tiled64["triangles"], reference["triangles"])
    np.testing.assert_array_equal(tiled64["labels"], reference["labels"])
    np.testing.assert_array_equal(tiled64["scores"], reference["scores"])


def test_the_float32_tiled_edt_still_does_not_compose(mask):
    """The contrast, kept in the same file so the two are never read apart.

    Same tiling, same certified halo, same values — only the storage precision differs, and the
    `float32` one still moves the mesh. This is what makes the `float64` variant's success a
    statement about dtype rather than about the tiling.
    """
    tiled32 = _reconstruct(mask, partial(_tiled, dtype=np.float32))
    tiled64 = _reconstruct(mask, partial(_tiled, dtype=np.float64))

    assert tiled32["edt_dtype"] == np.float32
    assert not np.array_equal(tiled32["scores"], tiled64["scores"])


def test_the_factory_exposes_the_float64_tiled_method(mask):
    """`set_tiled_edt_method(dtype=np.float64)` reaches the same field through the public factory.

    Checked because a variant that is only reachable by patching `algo.edt_creation_function` (as
    the tests above do, to isolate one stage) is not a variant a caller can select.
    """
    from dw3d import MeshReconstructionAlgorithmFactory

    algo = (
        MeshReconstructionAlgorithmFactory(print_info=False)
        .set_tiled_edt_method(tile_size=TILE_SIZE, halo=HALO, dtype=np.float64)
        .set_peak_local_points_placement_method(min_distance=MIN_DISTANCE)
        .make_algorithm()
    )
    algo.construct_mesh_from_segmentation_mask(mask)
    assert algo._edt_image.dtype == np.float64
    np.testing.assert_array_equal(algo._edt_image, _tiled(mask, np.float64))


def test_the_float64_tiled_path_rejects_an_unsupported_dtype(mask):
    """A silently-ignored `dtype` would be worse than an error, so it is an error."""
    with pytest.raises(ValueError, match=re.escape("np.float32 or np.float64")):
        compute_edt_tiled(mask, tile_size=TILE_SIZE, halo=HALO, dtype=np.float16)
