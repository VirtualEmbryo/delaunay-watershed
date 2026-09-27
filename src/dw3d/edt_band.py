"""Streamed, narrow-band seeding on the EDT.

Why this module exists
----------------------
`dw3d.edt`'s `compute_edt_float32` and `compute_edt_tiled` fix the EDT's *temporaries*: the
peak drops from ~7 full-volume `float64` arrays to 1-2 full-volume `float32` arrays. That is
the ~7x reduction the memory-scaling work set as its acceptance criterion, but it is still
`O(volume)`: at 2048^3 one `float32` array is 34 GB, so the stage remains the largest single
allocation in the pipeline. Removing it means never building the volume at all, which
requires the *consumers* to be band-aware, not just the producer.

This module makes three of the four consumers band-aware, and is explicit about the fourth.

What reads the EDT, and whether it needs the volume
---------------------------------------------------
Measured by reading `MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`:

1. **`plateau_packing_extrema(..., maximise=False)`** -- the interface minima. Needs the
   field only where it is *small*: a minimum candidate of `total_edt` cannot sit deep inside
   a material, because a distance function decreases towards its zero set faster than a
   `(2*min_distance+1)` box filter can miss. `streamed_plateau_packing_extrema` confirms
   this by measurement rather than by the argument alone -- it reports
   `max_value_at_candidates`, and the measurement is stronger than the argument: **1.236 on
   `data/Images/1-3.tif` and 1.449 on `4.tif`**, against `M = 2.236`. Since `total_edt` is
   `M - edt_2 <= M` on the boundary set and `>= M + 1` in the interior, that means *every*
   minimum candidate lies inside the 2-voxel-thick boundary set itself. The band originally
   budgeted for (`w = 3*min_distance = 9`) is therefore an order of magnitude wider than the
   minima need, which is why the sparse container in `dw3d.edt` stores the zero set rather
   than a dilation of it.
2. **`plateau_packing_extrema(..., maximise=True)`** -- the interior maxima, the cells'
   medial axis. These are *exactly* where the field is largest, so they cannot come from a
   narrow band around the interface; a band representation needs the per-label maxima as well.
   They come here from tile streaming instead: the field is built exactly on one tile at a
   time and only the candidate voxels survive the tile.
3. **`extract_seed_coords_and_indices`** -- one `argmax` of the EDT per label. Streamable:
   `streamed_label_seeds` keeps a running per-label best over tiles.
4. **`compute_scores_by_max_value` / `_by_mean_value`** -- the watershed scores, the EDT
   sampled at 27 barycentric points on every tesselation face. **This one genuinely needs
   random access to the far field**, because faces span cell interiors, and it is therefore
   the reason a fully band-resident pipeline is not delivered here. It is not a large
   *number* of reads (~1.2 M sample points against 7.0 M voxels on `3.tif`), so the
   remaining work is bounded and identified: either serve the reads from
   `dw3d.edt.SparseEdt.at` -- exact, `O(area)` storage, but a `cKDTree` query per sample and
   a re-implementation of the trilinear/cubic interpolant on top of it -- or tile the score
   computation itself by face bounding box. Both change no numbers if done correctly; both
   are more than a memory change, so they are recorded here as the identified remainder of
   the memory-scaling work rather than smuggled in.

Exactness
---------
Every field value this module uses is computed by the certified escalating-halo tiling
documented in `dw3d.edt` -- the narrow-band halo argument, not a two-pass block scheme -- so
the streamed extrema and seeds are not "exact inside a band", they are exact, and
`tests/test_edt_scaling.py` asserts the streamed point sets are **bit-identical** to the dense
ones on all four in-repo images. The filter footprint needs no band assumption at all: the
harvest region is the core grown by exactly `min_distance`, so the `(2*min_distance+1)` box
filter's footprint is inside it by construction. What is *checked* per run rather than assumed
is that no candidate's value exceeded the halo its tile certified at
(`n_candidates_above_halo`, which is 0 on every case measured).

"""

from time import time

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray

from dw3d.edt import (
    CODE_BOUNDARY,
    CODE_INTERIOR,
    CODE_ZERO,
    _expand,
    _tiles,
    certified_region_edt,
    region_codes,
)
from dw3d.points_on_edt import pack_plateau_candidates


def _global_max_edt_2(code: NDArray[np.uint8], tile_size: int, halo: int, parallel: int) -> tuple[np.float32, dict]:
    """The constant `M = max(edt_2)`, by tile streaming, without a full-volume `float32` array.

    `edt_2` is the distance transform of the boundary set, so it is bounded by the sheets'
    half-thickness (2.236068 on every in-repo image) and certifies at a very small halo. The
    running maximum is taken over certified cores only, which is what makes `M` exact.
    """
    best = np.float32(0.0)
    report = {"n_tiles": 0, "n_escalated": 0, "halo_max_used": int(halo)}
    for core in _tiles(code.shape, tile_size):
        report["n_tiles"] += 1
        distance, used_halo, _, escalations = certified_region_edt(code, core, CODE_BOUNDARY, halo, parallel)
        report["n_escalated"] += escalations
        report["halo_max_used"] = max(report["halo_max_used"], used_halo)
        best = max(best, np.float32(distance.max() if distance.size else 0.0))
    return best, report


def _total_edt_on_region(
    code: NDArray[np.uint8],
    region: tuple[slice, ...],
    max_edt_2: np.float32,
    halo: int,
    parallel: int,
) -> tuple[NDArray[np.float32], int]:
    """Exact `total_edt` on `region`, computed from `code` with the certified escalating halo.

    Returns the values and the halo actually used, so the caller can report escalations.
    Never allocates more than `O(|region| * ((r + 2h) / r)**3)`.

    `edt_2` needs no escalation of its own: it is bounded by `max_edt_2`, which the caller
    has already streamed, so `ceil(max_edt_2) + 1` is a provably sufficient halo for it.
    """
    edt_1, used_halo, _, _ = certified_region_edt(code, region, CODE_INTERIOR, halo, parallel)
    edt_2_halo = int(np.ceil(float(max_edt_2))) + 1
    edt_2, _, _, _ = certified_region_edt(code, region, CODE_BOUNDARY, edt_2_halo, parallel)

    sub_code = code[region]
    total = edt_1 + max_edt_2
    np.copyto(total, max_edt_2 - edt_2, where=sub_code == CODE_BOUNDARY)
    np.copyto(total, np.float32(0.0), where=sub_code == CODE_ZERO)
    return total, used_halo


def streamed_plateau_packing_extrema(
    segmentation_mask: NDArray[np.uint],
    min_distance: int,
    maximise: bool,
    tile_size: int = 128,
    halo: int = 16,
    parallel: int = 1,
    code: NDArray[np.uint8] | None = None,
    max_edt_2: np.float32 | None = None,
) -> tuple[NDArray[np.uint], dict]:
    """`plateau_packing_extrema`, without ever materialising the EDT volume.

    The candidate set is harvested tile by tile -- each tile builds `total_edt` exactly on
    `core + min_distance` (the box filter's footprint) and keeps only the voxels where the
    field equals its own extremum filter -- and the accepted set is then produced by
    `pack_plateau_candidates`, the *same* packing routine the dense path calls. The two
    therefore agree by construction; `tests/test_edt_scaling.py` asserts they do.

    Two details make the agreement exact rather than approximate:

    * **Raster order.** `plateau_packing_extrema` gets its candidates from `np.argwhere`,
      i.e. in global C order, and that order is the packing's outermost tie-break. Tiles are
      not visited in global raster order, so the harvested candidates are re-sorted by flat
      index before packing.
    * **The global extremum exclusion.** The dense path drops candidates equal to
      `image.min()` (maxima) or `image.max()` (minima). Those are global scalars, unknown
      until every tile has been seen, so the exclusion is applied *after* the harvest rather
      than inside the tile -- which is equivalent, because it is a per-voxel test.

    Args:
        segmentation_mask (NDArray[np.uint]): Segmentation mask (label field).
        min_distance (int): The algorithm's `min_distance`.
        maximise (bool): True for the interior maxima, False for the interface minima.
        tile_size (int, optional): Core side length. Defaults to 128.
        halo (int, optional): Initial EDT halo; tiles escalate as needed. Defaults to 16.
        parallel (int, optional): Threads for `edt.edt`. Defaults to 1.
        code (NDArray[np.uint8] | None, optional): Precomputed region codes, to share the
            one `uint8` volume between several calls. Computed here when None.
        max_edt_2 (np.float32 | None, optional): Precomputed `M`, likewise.

    Returns:
        tuple[NDArray[np.uint], dict]: the accepted voxel coordinates (identical to
            `plateau_packing_extrema`'s), and a report carrying the candidate count, the
            band width the candidates actually occupy, the escalation count and the
            region-edge certification counter.
    """
    if code is None:
        code = region_codes(segmentation_mask)
    if max_edt_2 is None:
        max_edt_2, _ = _global_max_edt_2(code, tile_size, halo=4, parallel=parallel)

    size = 2 * min_distance + 1
    shape = code.shape
    report = {
        "n_tiles": 0,
        "n_escalated": 0,
        "halo_max_used": int(halo),
        "n_candidates": 0,
        "n_candidates_above_halo": 0,
        "global_min": np.inf,
        "global_max": -np.inf,
        "max_value_at_candidates": -np.inf,
        "min_value_at_candidates": np.inf,
    }

    chunks: list[tuple[NDArray[np.int64], NDArray[np.float32]]] = []
    for core in _tiles(shape, tile_size):
        report["n_tiles"] += 1
        # The extremum filter reads `min_distance` voxels beyond the core; `mode="nearest"`
        # at the *global* border is reproduced because the region is clipped to the volume.
        region, inner = _expand(core, shape, min_distance)
        total, used_halo = _total_edt_on_region(code, region, max_edt_2, halo, parallel)
        if used_halo != halo:
            report["n_escalated"] += 1
            report["halo_max_used"] = max(report["halo_max_used"], used_halo)

        report["global_min"] = min(report["global_min"], float(total.min()))
        report["global_max"] = max(report["global_max"], float(total.max()))

        extremum = (ndi.maximum_filter if maximise else ndi.minimum_filter)(total, size=size, mode="nearest")
        hit = (total == extremum)[inner]
        if not hit.any():
            continue
        local = np.argwhere(hit)
        origin = np.array([sl.start for sl in core], dtype=np.int64)
        hit_values = total[inner][hit]
        chunks.append((local + origin, hit_values))

        # Certification. The *filter* needs no certifying: the region is the core grown by
        # exactly `min_distance`, which is the box filter's footprint radius, so every core
        # voxel's footprint is inside the region by construction (and where the region is
        # clipped by the global border, `mode="nearest"` is the dense path's own behaviour).
        # What can fail is the *value*: a candidate whose field value exceeds the halo used
        # for this tile would be an upper bound rather than the EDT. The escalation makes
        # that impossible, so this counter is a check on the code, not a fallback.
        report["n_candidates_above_halo"] += int(np.count_nonzero(hit_values > used_halo + float(max_edt_2)))

    if not chunks:
        return np.zeros((0, 3), dtype=np.uint), report

    coords = np.concatenate([c for c, _ in chunks])
    values = np.concatenate([v for _, v in chunks])
    del chunks

    # Restore global raster order: it is the packing's outermost tie-break.
    order = np.argsort(np.ravel_multi_index(tuple(coords.T), shape), kind="stable")
    coords, values = coords[order], values[order]

    keep = values > np.float32(report["global_min"]) if maximise else values < np.float32(report["global_max"])
    coords, values = coords[keep], values[keep]
    report["n_candidates"] = len(coords)
    if len(values):
        report["max_value_at_candidates"] = float(values.max())
        report["min_value_at_candidates"] = float(values.min())

    return pack_plateau_candidates(coords, values, min_distance, maximise), report


def streamed_label_seeds(
    segmentation_mask: NDArray[np.uint],
    tile_size: int = 128,
    halo: int = 16,
    parallel: int = 1,
    code: NDArray[np.uint8] | None = None,
    max_edt_2: np.float32 | None = None,
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """`dw3d.segmentation.extract_seed_coords_and_indices`, without the EDT volume.

    The dense version flattens the whole EDT and the whole label field and takes
    `argmax(where(labels == i, edt, 0))` per label. This keeps a running best per label over
    tiles instead. The tie-break is reproduced exactly: `np.argmax` returns the *first*
    maximum in raster order, so a tile only displaces the incumbent on a strict improvement,
    and tiles are visited in raster order.

    One inherited subtlety, preserved: the dense version compares against `0` rather than
    `-inf` (`np.where(flat_labels == i, flat_edt, 0)`), so a label whose every voxel has
    `total_edt == 0` resolves to flat index 0 rather than to one of its own voxels. That is
    reproduced here rather than fixed, because fixing it would change the seeds and hence
    the mesh. It can only bite a label lying entirely on the bounding-box shell.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint]]: seed voxel coordinates and label indices,
            in the same order as the dense version (`np.unique(mask)`).
    """
    if code is None:
        code = region_codes(segmentation_mask)
    if max_edt_2 is None:
        max_edt_2, _ = _global_max_edt_2(code, tile_size, halo=4, parallel=parallel)

    shape = code.shape
    labels = np.unique(segmentation_mask)
    label_slot = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
    label_slot[labels] = np.arange(len(labels))

    best_value = np.zeros(len(labels), dtype=np.float32)  # the dense version's `0` baseline
    best_flat = np.zeros(len(labels), dtype=np.int64)

    for core in _tiles(shape, tile_size):
        total, _ = _total_edt_on_region(code, core, max_edt_2, halo, parallel)
        slot = label_slot[segmentation_mask[core].reshape(-1)]
        value = total.reshape(-1)

        # Group by label, then by descending value. `np.lexsort` is stable, so ties keep the
        # tile's own raster order -- and within a tile, local raster order agrees with *global*
        # raster order, because the global flat index is monotone in `(i, j, k)`. So the first
        # row of each label's group is that label's winner with `np.argmax`'s "first maximum"
        # tie-break, without ever materialising a per-voxel index array (`np.indices` on a
        # tile costs 24 bytes/voxel, which made this the pipeline's largest allocation).
        order = np.lexsort((-value, slot))
        grouped = slot[order]
        group_start = np.flatnonzero(np.concatenate(([True], grouped[1:] != grouped[:-1])))
        for start in group_start:
            s = int(grouped[start])
            if s < 0:  # a label absent from `np.unique(mask)` cannot occur, but be explicit
                continue
            index = int(order[start])
            # Global flat index of this one winner. Tiles are *not* visited in global raster
            # order (tile `[0:64, 64:128, :]` interleaves with `[0:64, 0:64, :]`), so the
            # cross-tile tie-break has to compare real global indices, not tile order.
            local = np.unravel_index(index, total.shape)
            flat = int(np.ravel_multi_index([int(local[a]) + core[a].start for a in range(3)], shape))
            # Strict improvement only, so the lowest global raster index keeps the tie --
            # again matching `np.argmax` on the flattened volume.
            if value[index] > best_value[s] or (value[index] == best_value[s] and flat < best_flat[s]):
                best_value[s] = value[index]
                best_flat[s] = flat

    coords = np.array(np.unravel_index(best_flat, shape)).T
    return coords.astype(np.uint), labels


def narrow_band_seeding(
    segmentation_mask: NDArray[np.uint],
    min_distance: int = 3,
    tile_size: int = 128,
    halo: int = 16,
    parallel: int = 1,
    print_info: bool = False,
) -> dict:
    """Everything the deterministic seeding needs, computed without an EDT volume.

    Shares one `uint8` region-code volume, one streamed `M` **and one sweep over the tiles**
    between the maxima, the minima and the per-label seeds, so the peak is `1 byte/voxel` plus
    the tile working set plus the harvested candidates -- no `float32` volume anywhere.

    The single sweep is the point. Calling `streamed_plateau_packing_extrema` twice and
    `streamed_label_seeds` once, as an earlier draft of this function did, rebuilds the EDT on
    every tile three times over; measured on `data/Images/3.tif` that made the streamed path
    3.9x the dense one instead of the ~1.3x the tiling overhead alone accounts for. The
    per-tile field is the expensive part and the three consumers all want the same one.

    Returns:
        dict: `maxima`, `minima`, `seed_coords`, `seed_indices`, `max_edt_2`, and the
            streaming reports, for memory and band-width accounting.
    """
    t0 = time()
    code = region_codes(segmentation_mask)
    max_edt_2, report_m = _global_max_edt_2(code, tile_size, halo=4, parallel=parallel)

    shape = code.shape
    size = 2 * min_distance + 1
    labels = np.unique(segmentation_mask)
    label_slot = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
    label_slot[labels] = np.arange(len(labels))
    best_value = np.zeros(len(labels), dtype=np.float32)  # the dense `argmax`'s `0` baseline
    best_flat = np.zeros(len(labels), dtype=np.int64)

    reports = {
        True: _empty_extrema_report(halo),
        False: _empty_extrema_report(halo),
    }
    chunks: dict[bool, list[tuple[NDArray[np.int64], NDArray[np.float32]]]] = {True: [], False: []}

    for core in _tiles(shape, tile_size):
        region, inner = _expand(core, shape, min_distance)
        total, used_halo = _total_edt_on_region(code, region, max_edt_2, halo, parallel)
        core_total = total[inner]

        for maximise in (True, False):
            _harvest_tile(reports[maximise], chunks[maximise], total, core_total, inner, core,
                          size, maximise, used_halo, halo, max_edt_2)

        _accumulate_label_seeds(core_total, segmentation_mask[core], label_slot, best_value, best_flat, core, shape)

    maxima = _finish_extrema(chunks[True], reports[True], shape, min_distance, maximise=True)
    minima = _finish_extrema(chunks[False], reports[False], shape, min_distance, maximise=False)
    seed_coords = np.array(np.unravel_index(best_flat, shape)).T.astype(np.uint)

    if print_info:
        print(f"narrow-band seeding: {len(maxima)} maxima, {len(minima)} minima in {time() - t0:.2f} s")

    return {
        "maxima": maxima,
        "minima": minima,
        "seed_coords": seed_coords,
        "seed_indices": labels,
        "max_edt_2": float(max_edt_2),
        "wall_time_s": time() - t0,
        "report_max_edt_2": report_m,
        "report_maxima": reports[True],
        "report_minima": reports[False],
    }


def _empty_extrema_report(halo: int) -> dict:
    """The report dict `streamed_plateau_packing_extrema` and `narrow_band_seeding` both fill."""
    return {
        "n_tiles": 0,
        "n_escalated": 0,
        "halo_max_used": int(halo),
        "n_candidates": 0,
        "n_candidates_above_halo": 0,
        "global_min": np.inf,
        "global_max": -np.inf,
        "max_value_at_candidates": -np.inf,
        "min_value_at_candidates": np.inf,
    }


def _harvest_tile(  # one call site; bundling this state into an object would only hide it
    report: dict,
    chunks: list,
    total: NDArray[np.float32],
    core_total: NDArray[np.float32],
    inner: tuple[slice, ...],
    core: tuple[slice, ...],
    size: int,
    maximise: bool,
    used_halo: int,
    halo: int,
    max_edt_2: np.float32,
) -> None:
    """One tile's extremum-filter candidates, appended to `chunks`. See `streamed_plateau_packing_extrema`."""
    report["n_tiles"] += 1
    if used_halo != halo:
        report["n_escalated"] += 1
        report["halo_max_used"] = max(report["halo_max_used"], used_halo)
    report["global_min"] = min(report["global_min"], float(total.min()))
    report["global_max"] = max(report["global_max"], float(total.max()))

    extremum = (ndi.maximum_filter if maximise else ndi.minimum_filter)(total, size=size, mode="nearest")
    hit = (total == extremum)[inner]
    if not hit.any():
        return
    origin = np.array([sl.start for sl in core], dtype=np.int64)
    hit_values = core_total[hit]
    chunks.append((np.argwhere(hit) + origin, hit_values))
    report["n_candidates_above_halo"] += int(np.count_nonzero(hit_values > used_halo + float(max_edt_2)))


def _finish_extrema(chunks: list, report: dict, shape: tuple[int, ...], min_distance: int, maximise: bool) -> NDArray:
    """Restore global raster order, apply the global-extremum exclusion, and pack."""
    if not chunks:
        return np.zeros((0, 3), dtype=np.uint)
    coords = np.concatenate([c for c, _ in chunks])
    values = np.concatenate([v for _, v in chunks])

    order = np.argsort(np.ravel_multi_index(tuple(coords.T), shape), kind="stable")
    coords, values = coords[order], values[order]
    keep = values > np.float32(report["global_min"]) if maximise else values < np.float32(report["global_max"])
    coords, values = coords[keep], values[keep]

    report["n_candidates"] = len(coords)
    if len(values):
        report["max_value_at_candidates"] = float(values.max())
        report["min_value_at_candidates"] = float(values.min())
    return pack_plateau_candidates(coords, values, min_distance, maximise)


def _accumulate_label_seeds(  # one call site
    core_total: NDArray[np.float32],
    core_labels: NDArray[np.uint],
    label_slot: NDArray[np.int64],
    best_value: NDArray[np.float32],
    best_flat: NDArray[np.int64],
    core: tuple[slice, ...],
    shape: tuple[int, ...],
) -> None:
    """Fold one tile into the running per-label `argmax`. See `streamed_label_seeds`."""
    slot = label_slot[core_labels.reshape(-1)]
    value = np.ascontiguousarray(core_total).reshape(-1)
    order = np.lexsort((-value, slot))
    grouped = slot[order]
    for start in np.flatnonzero(np.concatenate(([True], grouped[1:] != grouped[:-1]))):
        s = int(grouped[start])
        if s < 0:
            continue
        index = int(order[start])
        local = np.unravel_index(index, core_total.shape)
        flat = int(np.ravel_multi_index([int(local[a]) + core[a].start for a in range(3)], shape))
        if value[index] > best_value[s] or (value[index] == best_value[s] and flat < best_flat[s]):
            best_value[s] = value[index]
            best_flat[s] = flat
