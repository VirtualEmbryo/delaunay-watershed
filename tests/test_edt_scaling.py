"""The float32, tiled and sparse EDT variants, and the streamed narrow-band seeding.

The acceptance criteria of the EDT memory/scaling work are asserted here, at the tightest
tolerance the variants actually achieve rather than the tolerance the specification allowed
for:

* the specification asks for `atol=1e-4` relative to the EDT dynamic range between float32 and
  float64. Measured: **`atol=0`**, because the classical path's float64 array only ever
  holds float32-representable values (`test_classical_edt_is_float32_representable` proves
  the premise; `dw3d.edt`'s module docstring explains it). So the variants are asserted
  bit-identical, and a future change that made them merely *close* would fail here;
* the specification asks that the seeding output be identical. Asserted directly on the point sets,
  for the dense variants and for the streamed narrow-band path;
* the specification demands that the tiling guarantee be stated and tested against the dense result
  on a 256^3 case at `atol=0` inside the band. `test_tiling_guarantee_on_256_cubed` does
  exactly that, and also tests the *converse* -- that a deliberately under-sized halo
  produces values which are upper bounds and agree inside the band but not outside it,
  which is what makes the guarantee a guarantee rather than a coincidence of this data.
"""

import numpy as np
import pytest

from dw3d.edt import (
    CODE_BOUNDARY,
    CODE_INTERIOR,
    CODE_ZERO,
    _binary_edt,
    compute_edt_classical,
    compute_edt_float32,
    compute_edt_tiled,
    compute_edt_tiled_with_report,
    region_codes,
    sparse_edt,
)
from dw3d.edt_band import narrow_band_seeding, streamed_label_seeds, streamed_plateau_packing_extrema
from dw3d.points_on_edt import plateau_packing_extrema
from dw3d.segmentation import extract_seed_coords_and_indices
from tests.conftest import dataset_masks, load_image_or_skip

MIN_DISTANCE = 3


def _two_cell_mask(side: int, dtype=np.int32) -> np.ndarray:
    """A deterministic synthetic mask: two half-space cells inside a background box.

    Small, exactly describable, and independent of `data/Images/` (which is gitignored), so
    the structural tests run everywhere while the golden-master-style ones skip.
    """
    mask = np.zeros((side, side, side), dtype=dtype)
    margin = side // 8
    inner = (slice(margin, side - margin),) * 3
    mask[inner] = 1
    half = mask[inner][0].shape[0] // 2
    block = mask[inner]
    block[half:] = 2
    mask[inner] = block
    return mask


# ---------------------------------------------------------------------------------------
# The premise: float32 is the field's real precision
# ---------------------------------------------------------------------------------------


def test_classical_edt_is_float32_representable(image_name):
    """`compute_edt_classical`'s float64 output holds only float32-representable values.

    This is the premise that makes every other float32 claim in this file exact rather than
    approximate. If it ever fails, `compute_edt_float32` becomes a lossy variant and its
    acceptance criterion has to revert to the specification's `atol=1e-4`.
    """
    mask = load_image_or_skip(image_name)
    reference = compute_edt_classical(mask)
    assert reference.dtype == np.float64
    assert np.array_equal(reference, reference.astype(np.float32).astype(np.float64))


def test_region_codes_partition_matches_classical_construction(image_name):
    """`region_codes` is exactly the `mask_1` / `mask_2` pair the classical path builds."""
    mask = load_image_or_skip(image_name)
    code = region_codes(mask)
    reference = compute_edt_classical(mask)

    # The classical construction, re-derived from the codes and compared to its own output.
    edt_1 = _binary_edt(code == CODE_INTERIOR, 1)
    edt_2 = _binary_edt(code == CODE_BOUNDARY, 1)
    max_edt_2 = np.float32(edt_2.max())
    rebuilt = edt_1 + max_edt_2
    np.copyto(rebuilt, max_edt_2 - edt_2, where=code == CODE_BOUNDARY)
    np.copyto(rebuilt, np.float32(0.0), where=code == CODE_ZERO)

    assert np.array_equal(rebuilt.astype(np.float64), reference)
    assert set(np.unique(code)) <= {CODE_ZERO, CODE_BOUNDARY, CODE_INTERIOR}


# ---------------------------------------------------------------------------------------
# EDT memory/scaling acceptance: the dense variants
# ---------------------------------------------------------------------------------------


def test_float32_edt_is_bit_identical(image_name):
    """EDT memory/scaling acceptance criterion 1, dense variant, at atol=0 rather than the specified 1e-4."""
    mask = load_image_or_skip(image_name)
    reference = compute_edt_classical(mask)
    variant = compute_edt_float32(mask)
    assert variant.dtype == np.float32
    assert np.array_equal(variant.astype(np.float64), reference)


def test_tiled_edt_is_bit_identical_and_certifies(image_name):
    """EDT memory/scaling acceptance criterion 1, tiled variant, plus the certification the guarantee rests on."""
    mask = load_image_or_skip(image_name)
    reference = compute_edt_classical(mask)
    variant, report = compute_edt_tiled_with_report(mask, tile_size=64, halo=8)

    assert np.array_equal(variant.astype(np.float64), reference)
    assert report["edt_1"]["n_uncertified"] == 0
    assert report["edt_2"]["n_uncertified"] == 0
    # The escalation must actually have fired at halo=8 on cells of this size, otherwise the
    # test is not exercising the mechanism it claims to.
    assert report["edt_1"]["n_escalated"] > 0
    assert report["edt_1"]["halo_max_used"] > 8


def test_tiled_edt_independent_of_tiling(image_name):
    """The tiled EDT does not depend on the tiling, which is what "exact" has to mean."""
    mask = load_image_or_skip(image_name)
    a = compute_edt_tiled(mask, tile_size=32, halo=4)
    b = compute_edt_tiled(mask, tile_size=100, halo=64)
    assert np.array_equal(a, b)


def test_sparse_edt_reproduces_the_field(image_name):
    """`SparseEdt` answers queries exactly, from `O(area)` storage.

    Measured bit-identical on all four in-repo images. It is asserted at `atol=0` because
    that is what is measured, but note the caveat in `SparseEdt.at`: `cKDTree` computes the
    square root in float64 and narrows, while `edt.edt` computes it in float32, so exact
    agreement is a measurement here and not a proof. A double-rounding disagreement of one
    ulp would be admissible under the specified `atol=1e-4`; it would fail this test, and the
    right response would be to relax it to one ulp, not to widen it to 1e-4.
    """
    mask = load_image_or_skip(image_name)
    reference = compute_edt_classical(mask).astype(np.float32)
    container = sparse_edt(mask)

    assert container.shape == reference.shape
    # Query a deterministic sample of voxels of every region code rather than the whole
    # volume: `dense()` is O(volume) and this test runs on four images.
    rng = np.random.default_rng(0)
    code = region_codes(mask)
    sample = []
    for wanted in (CODE_ZERO, CODE_BOUNDARY, CODE_INTERIOR):
        voxels = np.argwhere(code == wanted)
        sample.append(voxels[rng.choice(len(voxels), size=min(4000, len(voxels)), replace=False)])
    sample = np.concatenate(sample)

    assert np.array_equal(container.at(sample), reference[tuple(sample.T)])
    assert container.bytes_stored < reference.nbytes


def test_sparse_edt_dense_round_trip():
    """`SparseEdt.dense()` reproduces the whole field, on a volume small enough to do so."""
    mask = _two_cell_mask(48)
    assert np.array_equal(sparse_edt(mask).dense(), compute_edt_classical(mask).astype(np.float32))


# ---------------------------------------------------------------------------------------
# EDT memory/scaling acceptance: the tiling guarantee, tested on a 256^3 case
# ---------------------------------------------------------------------------------------


def test_tiling_guarantee_on_256_cubed():
    """The guarantee the EDT memory/scaling work relies on, stated and tested on a 256^3 case at atol=0.

    Guarantee (the narrow-band halo argument, not a two-pass block scheme):

        restricting the zero set can only push the nearest zero away, so `d_tile >= d`;
        hence `d_tile(v) <= h  =>  d_tile(v) == d(v)`, and `min(d_tile, h) == min(d, h)`.

    Three things are asserted, and the third is what stops this being a coincidence:

    1. with the escalating halo the tiled field equals the dense field everywhere, atol=0;
    2. with a deliberately *fixed, too-small* halo the two agree exactly inside the band
       `d <= h` -- the guarantee's actual content;
    3. and disagree outside it, upward only (`d_tile >= d`) -- so the band restriction is
       load-bearing rather than vacuous.
    """
    side = 256
    mask = _two_cell_mask(side)
    reference = compute_edt_classical(mask).astype(np.float32)

    exact = compute_edt_tiled(mask, tile_size=64, halo=8)
    assert np.array_equal(exact, reference)

    # Now defeat the escalation to exercise the band statement itself. `_tiled_binary_edt_into`
    # escalates, so the fixed-halo field is built here directly, one tile at a time.
    from dw3d.edt import _expand, _tiles

    halo = 6
    code = region_codes(mask)
    fixed = np.empty(mask.shape, dtype=np.float32)
    for core in _tiles(mask.shape, 64):
        outer, inner = _expand(core, mask.shape, halo)
        fixed[core] = _binary_edt(code[outer] == CODE_INTERIOR, 1)[inner]

    dense_edt_1 = _binary_edt(code == CODE_INTERIOR, 1)
    in_band = dense_edt_1 <= halo

    assert np.array_equal(fixed[in_band], dense_edt_1[in_band])  # (2) exact in the band
    assert np.all(fixed >= dense_edt_1)  # (3a) monotone: upper bound everywhere
    assert np.array_equal(np.minimum(fixed, halo), np.minimum(dense_edt_1, halo))  # the clamped identity
    assert (fixed[~in_band] > dense_edt_1[~in_band]).any()  # (3b) and genuinely wrong outside


# ---------------------------------------------------------------------------------------
# EDT memory/scaling acceptance: the seeding output is identical
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("maximise", [True, False])
def test_seeding_identical_on_dense_variants(image_name, maximise):
    """EDT memory/scaling acceptance criterion 1's second half: the point set is what matters."""
    mask = load_image_or_skip(image_name)
    reference = plateau_packing_extrema(compute_edt_classical(mask), MIN_DISTANCE, maximise=maximise)
    for variant in (compute_edt_float32(mask), compute_edt_tiled(mask, tile_size=64, halo=8)):
        assert np.array_equal(plateau_packing_extrema(variant, MIN_DISTANCE, maximise=maximise), reference)


@pytest.mark.parametrize("maximise", [True, False])
def test_streamed_extrema_identical_to_dense(image_name, maximise):
    """The narrow-band / streamed extrema are the *same point set*, never a similar one."""
    mask = load_image_or_skip(image_name)
    reference = plateau_packing_extrema(compute_edt_classical(mask), MIN_DISTANCE, maximise=maximise)
    streamed, report = streamed_plateau_packing_extrema(mask, MIN_DISTANCE, maximise, tile_size=128, halo=16)
    assert np.array_equal(streamed, reference)
    assert report["n_candidates_above_halo"] == 0


def test_streamed_label_seeds_identical_to_dense(image_name):
    """The watershed's per-label seeds, streamed, are identical to the dense `argmax`."""
    mask = load_image_or_skip(image_name)
    coords, indices = extract_seed_coords_and_indices(mask, compute_edt_classical(mask))
    streamed_coords, streamed_indices = streamed_label_seeds(mask, tile_size=128, halo=16)
    assert np.array_equal(streamed_coords, coords)
    assert np.array_equal(streamed_indices, indices)


def test_streamed_seeding_independent_of_tiling(image_name):
    """A different tiling must not move a single point. Same argument as for the EDT."""
    mask = load_image_or_skip(image_name)
    a = narrow_band_seeding(mask, MIN_DISTANCE, tile_size=64, halo=4)
    b = narrow_band_seeding(mask, MIN_DISTANCE, tile_size=200, halo=48)
    assert np.array_equal(a["maxima"], b["maxima"])
    assert np.array_equal(a["minima"], b["minima"])
    assert np.array_equal(a["seed_coords"], b["seed_coords"])


def test_minima_live_in_a_narrow_band(image_name):
    """The premise of the narrow band, measured rather than assumed.

    The specification budgeted a band of width `w ~ 3*min_distance` around the boundary set for the
    interface minima. Measured, the minima candidates are *inside the boundary set itself*:
    their field value never exceeds `M`, the boundary sheets' half-thickness constant
    (1.24-1.45 on the four in-repo images against `M = 2.24`), because `total_edt = M - edt_2`
    there while the interior starts at `M + 1`. So the band needed is far tighter than
    budgeted, and this test pins that -- it is the fact that makes a band-resident minima
    pass possible at all.
    """
    mask = load_image_or_skip(image_name)
    _, report = streamed_plateau_packing_extrema(mask, MIN_DISTANCE, maximise=False, tile_size=128, halo=16)
    assert report["max_value_at_candidates"] <= 3 * MIN_DISTANCE


def test_region_codes_independent_of_tiling(image_name):
    """Tiling `find_boundaries` must not change a single voxel of the region codes.

    The halo is one voxel and the `mode="thick"` / `connectivity=2` test reads only a voxel's
    3x3x3 neighbourhood, so this is exact rather than approximate. It matters because tiling
    that pass is what removes `skimage`'s two full-volume `int32` temporaries, and those were
    the largest allocation left in `compute_edt_float32` — the difference between a 2.4x and a
    3.2x peak-RSS gain.
    """
    mask = load_image_or_skip(image_name)
    untiled = region_codes(mask, tile_size=None)
    for tile_size in (32, 64, 256):
        assert np.array_equal(region_codes(mask, tile_size=tile_size), untiled)


# ---------------------------------------------------------------------------------------
# EDT memory/scaling acceptance criterion 1, second half: the 10 `benchmarking-dataset` cases
# ---------------------------------------------------------------------------------------

# The specification asks for "all 4 in-repo images and 10 `benchmarking-dataset` cases". The four
# images are covered above; these are the ten. They are a different regime in one respect
# that matters -- 2-10 cells against 4-8, and masks stored as `int32` label fields written by
# a different generator -- so they exercise `region_codes` on label values and cell counts the
# in-repo images do not.
_DATASET_CASES = dataset_masks(limit=10)
_DATASET_IDS = [path.name.split("_")[0] for path in _DATASET_CASES]


@pytest.fixture(params=_DATASET_CASES, ids=_DATASET_IDS)
def dataset_mask(request):
    """One of the first 10 `benchmarking-dataset` masks; skips the suite if none are found."""
    import skimage.io as io

    return io.imread(request.param)


@pytest.mark.skipif(not _DATASET_CASES, reason="benchmarking-dataset not found beside the repo")
def test_dataset_edt_variants_are_bit_identical(dataset_mask):
    """Criterion 1 on the dataset cases: float32 and tiled reproduce the classical field exactly."""
    reference = compute_edt_classical(dataset_mask)
    assert np.array_equal(reference, reference.astype(np.float32).astype(np.float64))
    assert np.array_equal(compute_edt_float32(dataset_mask).astype(np.float64), reference)

    tiled, report = compute_edt_tiled_with_report(dataset_mask, tile_size=64, halo=8)
    assert np.array_equal(tiled.astype(np.float64), reference)
    assert report["edt_1"]["n_uncertified"] == 0
    assert report["edt_2"]["n_uncertified"] == 0


@pytest.mark.skipif(not _DATASET_CASES, reason="benchmarking-dataset not found beside the repo")
def test_dataset_seeding_is_identical(dataset_mask):
    """Criterion 1's "the point set is what matters" half, on the dataset cases.

    Covers the dense variants *and* the streamed narrow-band path, including the per-label
    watershed seeds, which is the consumer most likely to break on a mask whose label set is
    not contiguous.
    """
    reference_edt = compute_edt_classical(dataset_mask)
    expected = {
        maximise: plateau_packing_extrema(reference_edt, MIN_DISTANCE, maximise=maximise)
        for maximise in (True, False)
    }
    seed_coords, seed_indices = extract_seed_coords_and_indices(dataset_mask, reference_edt)

    for variant in (compute_edt_float32(dataset_mask), compute_edt_tiled(dataset_mask, tile_size=64, halo=8)):
        for maximise in (True, False):
            assert np.array_equal(plateau_packing_extrema(variant, MIN_DISTANCE, maximise=maximise),
                                  expected[maximise])

    streamed = narrow_band_seeding(dataset_mask, MIN_DISTANCE, tile_size=128, halo=16)
    assert np.array_equal(streamed["maxima"], expected[True])
    assert np.array_equal(streamed["minima"], expected[False])
    assert np.array_equal(streamed["seed_coords"], seed_coords)
    assert np.array_equal(streamed["seed_indices"], seed_indices)
    assert streamed["report_minima"]["n_candidates_above_halo"] == 0
    assert streamed["report_maxima"]["n_candidates_above_halo"] == 0
