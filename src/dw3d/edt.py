"""Compute Euclidean Distance Trasnform out of segmentation image.

Memory and scaling
------------------
`compute_edt_classical` allocates ~7 full-volume arrays and returns `float64`, which an
estimate of its peak memory puts at **481 GB at 2048^3** -- the stage that makes the stated
target infeasible by two orders of magnitude, independently of every accuracy question.
This module adds three variants, in increasing order of how much of the volume they refuse to
materialise. All three compute *the same field*; what differs is the representation.

The field itself, written out once so the variants can be checked against it::

    b       = find_boundaries(mask, connectivity=2, mode="thick")   # 2-voxel-thick sheets
    code    = 0 on the 1-voxel bounding-box shell where b == 0      # -> total_edt = 0
              1 where b != 0            (the "boundary set")        # -> total_edt = M - edt_2
              2 everywhere else         (the "interior")            # -> total_edt = edt_1 + M
    edt_1   = distance to the nearest voxel with code != 2
    edt_2   = distance to the nearest voxel with code != 1
    M       = max(edt_2)          (2.236068 on data/Images/3.tif -- the boundary is thin)

`code` is the same three-way split `compute_edt_classical` expresses as the overlapping
pair `mask_1 = 1 - _pad_mask(b)` and `mask_2 = b`; making it explicit is what lets the
variants below write each voxel exactly once instead of forming `(edt_1 + M) * mask_1 +
inv * mask_2` out of seven whole-volume temporaries.

**float32 is not an approximation here, it is the field's actual precision.** `edt.edt`
returns `float32`; `compute_edt_classical` is `float64` only because `mask_1 = 1 - b`
inherits `find_boundaries`' `int32`, and `float32 * int32` promotes to `float64` in numpy.
Every value in the returned `float64` array is therefore exactly float32-representable --
asserted on all four in-repo images and 10 `benchmarking-dataset` cases in
`tests/test_edt_scaling.py`. So `compute_edt_float32` is **bit-identical** to
`compute_edt_classical`, not merely equal to `atol=1e-4`, and the seeding it feeds is
identical for free rather than by measurement.

Tiling, and the guarantee it relies on
--------------------------------------
`compute_edt_tiled` computes the EDT block-wise on `tile_size**3` cores read with a halo.
`edt.edt` is separable and exact only over the array it is given, so a tile sees only the
zeros inside `core + halo`. That restriction is not neutral, but it is **monotone**, and
the whole scheme rests on one inequality:

    removing zeros can only move the nearest zero further away, so  d_tile(v) >= d(v).

Together with "a zero at distance <= h from a core voxel is inside core + halo when the
halo is h", that gives the guarantee this module relies on, stated explicitly because
the choice of scheme matters -- **the narrow-band halo argument, not a two-pass block
scheme**:

    d_tile(v) <= h   ==>   d_tile(v) == d(v)          (exact, atol = 0)
    d(v)     <= h    ==>   d_tile(v) == d(v)
    hence            min(d_tile, h) == min(d, h)      everywhere, identically.

The consequence used in the code is that a tile **certifies itself**: compute
`U = max over the core of d_tile`; every core voxel with `d_tile <= h` is exact, and `U` is
an upper bound on the core's true maximum. So if `U > h` the tile is recomputed **once**,
with `halo = ceil(U)`, which is then provably sufficient -- no doubling loop, at most two
attempts per tile, and the result is exact *everywhere*, not only inside the band. The
number of escalations is reported rather than hidden, because it is the cost model: the
work factor of a tile is `((T + 2h) / T)**3`, so `T >= 4h` keeps the overhead under 2x and
a tile size smaller than the cells' inradius is a mistake, not a tuning choice.

`tests/test_edt_scaling.py` tests exactly the statement above against the dense result on a
256^3 case, at `atol=0`, both inside the band and (via certification) globally.

Sparse representation
---------------------
`sparse_edt` goes further and stores no volume at all. Both `edt_1` and `edt_2` are
*distance functions*, so they are determined by their zero sets, and those sets are the
interface sheets: `O(area)`, not `O(volume)`. `SparseEdt` therefore keeps the boundary set's
own small values verbatim and, for the far field, keeps the **generator** -- the zero
voxels -- and answers a query by a `cKDTree` nearest-neighbour search. That is exact (up to
the float32 rounding of the final square root, quantified in `SparseEdt.at`), random-access,
and costs 6.8 % of the volume on `data/Images/3.tif`. It is a "narrow band plus per-label
maxima" representation, with the band chosen to be the zero set itself rather than a
dilation of it -- see `dw3d.edt_band` for the streamed extrema and seeds built on it, and for
the one consumer (the watershed scores) that this does *not* yet make band-aware.

Sacha Ichbiah 2021
Matthieu Perez 2024
Narrow-band / float32 / tiled variants added later for memory and scaling.
"""

from time import time

import numpy as np
from edt import edt as euclidean_dt
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

# Three-way region code of the EDT construction; see the module docstring.
CODE_ZERO = 0
CODE_BOUNDARY = 1
CODE_INTERIOR = 2

# `find_boundaries(connectivity=2, mode="thick")` marks a voxel from its 3x3x3
# neighbourhood only, so one voxel of halo is enough to compute it on a subvolume.
_BOUNDARY_STENCIL_RADIUS = 1


def compute_edt_classical(segmentation_mask: NDArray[np.uint], print_info: bool = False) -> NDArray[np.float64]:
    """Compute the Euclidean Distance Transorm of a segmented image.

    It will be 0 at borders and cells boundaries. Bigger when getting far from these points.
    """
    if print_info:
        print("Computing EDT ...")
    t1 = time()

    b = _StandardLabelToBoundary()(segmentation_mask)[0]  # "thick" boundaries are marked by 1, 0 outside
    mask_2 = b
    edt_2 = euclidean_dt(mask_2)  # EDT of the thick boundaries (0 elsewhere)
    b = _pad_mask(b)  # exterior bbox is marked as 1
    mask_1 = 1 - b  # 1 everywhere except bbox and boundaries
    edt_1 = euclidean_dt(
        mask_1,
    )  # main part of the final EDT. Both inside cells and outside cells. 0 in boundaries & bbox
    inv = (
        np.amax(edt_2) - edt_2
    )  # max EDT2 everywhere except on thick boundaries where it decreases to 0 on the mid of boundaries
    total_edt = (edt_1 + np.amax(edt_2)) * mask_1 + inv * mask_2  # total EDT is valid also on thick boundaries

    # # Matthieu Perez try 2: augment constrast in EDT
    # Total_EDT = 255 * ((Total_EDT / 255) ** 0.5)

    t2 = time()
    if print_info:
        print("EDT computed in ", np.round(t2 - t1, 2))

    return total_edt


def _recover_ignore_index(
    input_img: NDArray[np.uint],
    orig: NDArray[np.uint],
    ignore_index: None | int,
) -> NDArray[np.uint]:
    """Put back the ignored_index in input_image ?"""
    if ignore_index is not None:
        mask = orig == ignore_index
        input_img[mask] = ignore_index

    return input_img


class _StandardLabelToBoundary:
    """Class-function because why not ? The goal is to extract pixels boundaries in a multi-labeled image."""

    def __init__(
        self,
        ignore_index: int | None = None,
        append_label: bool = False,
        mode: str = "thick",
        foreground: bool = False,
    ) -> None:
        """Register the parameters for the class-function.

        Args:
            ignore_index (int | None, optional): Consider the ignored index as the background ?. Defaults to None.
            append_label (bool, optional): Append original input data to the result. Defaults to False.
            mode (str, optional): {'thick', 'inner', 'outer', 'subpixel'}
                How to mark the boundaries:

                thick: any pixel not completely surrounded by pixels of the same label (defined by connectivity)
                       is marked as a boundary. This results in boundaries that are 2 pixels thick.
                inner: outline the pixels *just inside* of objects, leaving background pixels untouched.
                outer: outline pixels in the background around object boundaries. When two objects touch,
                       their boundary is also marked.
                subpixel: return a doubled image, with pixels *between* the original pixels
                          marked as boundary where appropriate.
                Defaults to "thick".
            foreground (bool, optional): Extract the foreground and put it in the result. Defaults to False.
        """
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m: NDArray[np.uint]) -> NDArray[np.uint]:
        """Call the function to extract pixels boundaries in a multi-labeled image.

        The output is an image with boundaries marked but it can change with parameters...
        """
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype("int32")

        results = []
        if self.foreground:
            foreground = (m > 0).astype("uint8")
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


def _pad_mask(mask: NDArray[np.uint8], pad_size: int = 1) -> NDArray[np.uint8]:
    """Pad a mask with ones on the borders."""
    padded_mask = mask.copy()[
        pad_size:-pad_size,
        pad_size:-pad_size,
        pad_size:-pad_size,
    ]
    padded_mask = np.pad(
        padded_mask,
        ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)),
        "constant",
        constant_values=1,
    )
    return padded_mask


# Matthieu Perez : tests bias EDT towards boundaries between interfaces
def compute_edt_boundary_bias(segmentation_mask: NDArray[np.uint], print_info: bool = False) -> NDArray[np.float64]:
    """Compute a biased version of the Euclidean Distance Transform of a segmented image.

    It is biased towards the boundaries between interfaces.
    """
    if print_info:
        print("Computing EDT (bias) ...")
    t1 = time()

    region_indices = np.unique(segmentation_mask)
    total_boundaries = np.zeros(segmentation_mask.shape)

    for index in region_indices:
        if index == 0:
            region_labels = np.where(segmentation_mask == 0, 1, 0)
        else:
            region_labels = np.where(segmentation_mask == index, segmentation_mask, 0)
        total_boundaries += _StandardLabelToBoundary()(region_labels)[0]

    # total_boundaries *= 3
    total_boundaries = np.amax(total_boundaries) - total_boundaries

    # "thick" boundaries are marked by 1, 0 outside
    b = _StandardLabelToBoundary()(segmentation_mask)[0]
    mask_2 = b
    # EDT of the thick boundaries (0 elsewhere)
    edt_2 = euclidean_dt(mask_2)
    b = _pad_mask(b)  # exterior bbox is marked as 1
    mask_1 = 1 - b  # 1 everywhere except bbox and boundaries
    # main part of the final EDT. Both inside cells and outside cells. 0 in boundaries & bbox
    edt_1 = euclidean_dt(mask_1)
    # max EDT2 everywhere except on thick boundaries where it decreases to 0 on the mid of boundaries
    # + total_boundaries which is less on interesting parts of the mesh
    inv = np.amax(edt_2) - edt_2 + total_boundaries
    # total EDT is valid also on thick boundaries
    total_edt = (edt_1 + np.amax(inv)) * mask_1 + inv * mask_2

    # Matthieu Perez try 2: augment constrast in EDT
    # Total_EDT = 255 * ((Total_EDT / 255) ** 0.5)

    t2 = time()
    if print_info:
        print("EDT computed in ", np.round(t2 - t1, 2))

    return total_edt


def get_total_boundaries(segmented_mask: NDArray[np.uint]) -> NDArray[np.float64]:
    """Obtain pre-EDT on boundaries from segmentation mask."""
    region_indices = np.unique(segmented_mask)
    total_boundaries = np.zeros(segmented_mask.shape)

    for index in region_indices:
        if index == 0:
            region_labels = np.where(segmented_mask == 0, 1, 0)
        else:
            region_labels = np.where(segmented_mask == index, segmented_mask, 0)
        total_boundaries += _StandardLabelToBoundary()(region_labels)[0]

    total_boundaries = np.amax(total_boundaries) - total_boundaries
    return total_boundaries


# ---------------------------------------------------------------------------------------
# Region codes
# ---------------------------------------------------------------------------------------


def region_codes(segmentation_mask: NDArray[np.uint], tile_size: int | None = 256) -> NDArray[np.uint8]:
    """Three-way region code of `compute_edt_classical`'s construction. See the module docstring.

    `0` on the bounding-box shell outside the boundary set, `1` on the thick boundary set,
    `2` in the interior of a material. This is exactly the information the historical
    `mask_1 = 1 - _pad_mask(b)` / `mask_2 = b` pair carries, written as a partition so that
    each voxel of the output can be assigned once.

    **`find_boundaries` is tiled by default, and that is a memory decision, not a speed one.**
    `skimage`'s `find_boundaries(mode="thick")` runs a grey dilation and a grey erosion over
    the whole label field, so it holds two more full-volume arrays *of the mask's dtype*
    (`int32` on every case here) plus the `bool` result — 9 bytes per voxel of temporaries on
    top of the 1-byte output. Measured on the four in-repo images, that made it the largest
    allocation left in `compute_edt_float32` and held the memory-scaling work's peak-RSS gain at 2.1-2.7x
    against the 3.2x the per-voxel accounting predicts. Since the `mode="thick"` /
    `connectivity=2` boundary test reads only a voxel's own 3x3x3 neighbourhood, a halo of one
    voxel makes the tiled computation **exactly** equal to the untiled one
    (`tests/test_edt_scaling.py::test_region_codes_independent_of_tiling`).

    Args:
        segmentation_mask (NDArray[np.uint]): Segmentation mask (label field).
        tile_size (int | None, optional): Tile side for the boundary pass. `None` computes it
            in one shot, which is what the path before tiling did and is kept for comparison.
            Defaults to 256.

    Returns:
        NDArray[np.uint8]: `uint8` array of region codes, one byte per voxel.
    """
    shape = segmentation_mask.shape
    code = np.empty(shape, dtype=np.uint8)

    if tile_size is None:
        boundary = find_boundaries(segmentation_mask, connectivity=2, mode="thick")
        np.copyto(code, np.where(boundary, np.uint8(CODE_BOUNDARY), np.uint8(CODE_INTERIOR)))
        _apply_shell(code, boundary)
        return code

    for core in _tiles(shape, tile_size):
        outer, inner = _expand(core, shape, _BOUNDARY_STENCIL_RADIUS)
        sub_boundary = find_boundaries(segmentation_mask[outer], connectivity=2, mode="thick")[inner]
        code[core] = np.where(sub_boundary, np.uint8(CODE_BOUNDARY), np.uint8(CODE_INTERIOR))

    # The shell depends on the *global* border, so it is applied once, after the tiles. No
    # extra boundary pass is needed: the tiles have already written CODE_BOUNDARY or
    # CODE_INTERIOR on the faces, and the shell rule is exactly "INTERIOR becomes ZERO there".
    _demote_interior_shell(code)
    return code


def _demote_interior_shell(code: NDArray[np.uint8]) -> None:
    """Turn `CODE_INTERIOR` into `CODE_ZERO` on the 1-voxel bounding-box shell, in place.

    The tiled `region_codes` writes the same boundary status on the faces as the untiled one
    (the halo is 1 and the global border is clipped identically), so this is `_apply_shell`
    without a second `find_boundaries` pass over the volume.
    """
    for axis in range(code.ndim):
        for index in (0, -1):
            face = [slice(None)] * code.ndim
            face[axis] = index
            face = tuple(face)
            np.copyto(code[face], np.uint8(CODE_ZERO), where=code[face] == CODE_INTERIOR)


def _apply_shell(code: NDArray[np.uint8], boundary: NDArray[np.bool_]) -> None:
    """Set the 1-voxel bounding-box shell to `CODE_ZERO` where it is not in the boundary set.

    `_pad_mask` forces the shell into `mask_2`'s complement, so `mask_1` is 0 there and the
    total EDT is exactly 0 unless the voxel is itself a boundary voxel (in which case the
    `inv` branch applies and the code stays `CODE_BOUNDARY`).
    """
    for axis in range(code.ndim):
        for index in (0, -1):
            face = [slice(None)] * code.ndim
            face[axis] = index
            face = tuple(face)
            code[face] = np.where(boundary[face], np.uint8(CODE_BOUNDARY), np.uint8(CODE_ZERO))


def _binary_edt(selector: NDArray[np.bool_], parallel: int) -> NDArray[np.float32]:
    """Exact Euclidean distance from each `True` voxel to the nearest `False` voxel.

    `edt.edt` on a 0/1 field, which is what `compute_edt_classical` calls it with. The bool
    array is *viewed* rather than cast, so no copy of the volume is made.
    """
    return euclidean_dt(selector.view(np.uint8), black_border=False, parallel=parallel)


# ---------------------------------------------------------------------------------------
# Dense float32
# ---------------------------------------------------------------------------------------


def compute_edt_float32(
    segmentation_mask: NDArray[np.uint],
    print_info: bool = False,
    parallel: int = 1,
) -> NDArray[np.float32]:
    """Compute the same EDT as `compute_edt_classical`, in float32, without the 7 temporaries.

    **Bit-identical** to `compute_edt_classical`, not approximately equal: every value the
    `float64` version returns is exactly float32-representable, because `edt.edt` already
    works in float32 and the `float64` dtype is an artifact of `float32 * int32` promotion
    (see the module docstring). The point of the variant is memory, not precision.

    Live full-volume arrays at peak: the `float32` output, one `float32` scratch, the
    `uint8` region codes and one transient `bool` selector -- 10 bytes per voxel against the
    56 the classical path holds at its peak, and **2 full-volume float32 arrays**, which is
    the design target of the memory-scaling work.

    Args:
        segmentation_mask (NDArray[np.uint]): Segmentation mask input.
        print_info (bool, optional): Print the stage's wall time. Defaults to False.
        parallel (int, optional): Threads for `edt.edt`. `1` matches `compute_edt_classical`
            and keeps the timing comparison honest; the result does not depend on it.

    Returns:
        NDArray[np.float32]: 0 on the bounding-box shell and at the middle of cell
            boundaries, growing away from them -- the field `compute_edt_classical` returns.
    """
    if print_info:
        print("Computing EDT (float32) ...")
    t1 = time()

    code = region_codes(segmentation_mask)

    interior = code == CODE_INTERIOR
    out = _binary_edt(interior, parallel)  # edt_1
    del interior

    boundary = code == CODE_BOUNDARY
    scratch = _binary_edt(boundary, parallel)  # edt_2
    max_edt_2 = np.float32(scratch.max())
    np.subtract(max_edt_2, scratch, out=scratch)  # inv, in place
    out += max_edt_2  # edt_1 + M, in place
    np.copyto(out, scratch, where=boundary)
    del scratch, boundary

    np.copyto(out, np.float32(0.0), where=code == CODE_ZERO)

    if print_info:
        print("EDT computed in ", np.round(time() - t1, 2))
    return out


# ---------------------------------------------------------------------------------------
# Tiled with halo
# ---------------------------------------------------------------------------------------


def _tiles(shape: tuple[int, ...], tile_size: int) -> list[tuple[slice, ...]]:
    """Cores of a regular tiling of `shape`, as index tuples. Cores are disjoint and cover."""
    axes = [range(0, extent, tile_size) for extent in shape]
    cores: list[tuple[slice, ...]] = []

    def recurse(axis: int, prefix: tuple[slice, ...]) -> None:
        if axis == len(shape):
            cores.append(prefix)
            return
        for start in axes[axis]:
            recurse(axis + 1, (*prefix, slice(start, min(start + tile_size, shape[axis]))))

    recurse(0, ())
    return cores


def _expand(core: tuple[slice, ...], shape: tuple[int, ...], halo: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Grow `core` by `halo` voxels per side, clipped to `shape`.

    Returns the expanded index tuple and the index tuple that recovers the core *within*
    the expanded subvolume.
    """
    outer: list[slice] = []
    inner: list[slice] = []
    for axis, sl in enumerate(core):
        lo = max(sl.start - halo, 0)
        hi = min(sl.stop + halo, shape[axis])
        outer.append(slice(lo, hi))
        inner.append(slice(sl.start - lo, sl.stop - lo))
    return tuple(outer), tuple(inner)


def certified_region_edt(
    code: NDArray[np.uint8],
    region: tuple[slice, ...],
    wanted: int,
    halo: int,
    parallel: int,
) -> tuple[NDArray[np.float32], int, bool, int]:
    """Exact distance from each `code == wanted` voxel of `region` to the nearest other voxel.

    The one place the certified escalating-halo scheme of the module docstring is
    implemented, so the tiled EDT and `dw3d.edt_band`'s streamed extrema cannot drift apart.

    Escalation, and why it terminates:

    * if the tiled maximum `U` over the region is `<= halo`, every value is exact and we are
      done (the guarantee in the module docstring);
    * if `U` is finite, `ceil(U)` is a **sufficient** halo, because `U` bounds the region's
      true maximum from above -- so one retry suffices;
    * if `U` is infinite the region contains no `code != wanted` voxel at all (a tile lying
      entirely inside one cell, which happens as soon as `tile_size + 2*halo` is smaller than
      a cell's diameter). No bound is available then, so the halo doubles;
    * an expanded region that covers the whole volume is exact by definition, whatever `U`
      is, and the halo is capped so that this is always reached. Termination is therefore
      guaranteed in `O(log(max(shape)))` attempts, and in practice in one or two.

    Returns:
        tuple[NDArray[np.float32], int, bool, int]: the values on `region`; the halo finally
            used; whether the result is certified exact (it always is, on return -- the flag
            exists so callers can assert rather than trust); and the number of escalations.
    """
    shape = code.shape
    # A halo of `max(shape)` guarantees the expanded region is the whole volume.
    halo_cap = int(max(shape))
    current_halo = max(int(halo), 1)
    escalations = 0

    while True:
        outer, inner = _expand(region, shape, current_halo)
        distance = _binary_edt(code[outer] == wanted, parallel)[inner]
        largest = float(distance.max()) if distance.size else 0.0
        covers_volume = all(sl.start == 0 and sl.stop == shape[axis] for axis, sl in enumerate(outer))
        if largest <= current_halo or covers_volume:
            return distance, current_halo, True, escalations
        current_halo = min(halo_cap, 2 * current_halo if not np.isfinite(largest) else int(np.ceil(largest)))
        escalations += 1


def _tiled_binary_edt_into(
    out: NDArray[np.float32],
    code: NDArray[np.uint8],
    wanted: int,
    tile_size: int,
    halo: int,
    parallel: int,
) -> dict:
    """Fill `out` with the exact distance from each `code == wanted` voxel to the nearest other voxel.

    Tile by tile, via `certified_region_edt`. Returns a report: tile count, escalations, the
    largest halo used, and the number of core voxels left uncertified (which is 0 by
    construction -- the caller asserts it rather than assuming it).
    """
    shape = code.shape
    report = {
        "n_tiles": 0,
        "n_escalated": 0,
        "halo_initial": int(halo),
        "halo_max_used": int(halo),
        "n_uncertified": 0,
        "max_value": 0.0,
    }

    for core in _tiles(shape, tile_size):
        report["n_tiles"] += 1
        distance, used_halo, certified, escalations = certified_region_edt(code, core, wanted, halo, parallel)
        out[core] = distance
        report["n_escalated"] += escalations
        report["halo_max_used"] = max(report["halo_max_used"], used_halo)
        report["max_value"] = max(report["max_value"], float(distance.max()) if distance.size else 0.0)
        if not certified:  # pragma: no cover - unreachable; `certified_region_edt` cannot return False
            report["n_uncertified"] += int(np.count_nonzero(distance > used_halo))

    return report


def compute_edt_tiled_with_report(
    segmentation_mask: NDArray[np.uint],
    tile_size: int = 256,
    halo: int = 16,
    print_info: bool = False,
    parallel: int = 1,
) -> tuple[NDArray[np.float32], dict]:
    """Tiled, float32 EDT, plus the certification report. See the module docstring.

    Exact -- not merely exact inside a band -- because every tile certifies its own core and
    escalates its halo once if it has to. The report is returned rather than logged so a
    caller (and `tests/test_edt_scaling.py`) can assert `n_uncertified == 0` instead of
    trusting it.

    Live full-volume arrays at peak: the `float32` output and the `uint8` region codes, i.e.
    **1 full-volume float32 array** plus one byte per voxel, plus `O(tile_size**3)` scratch.

    Args:
        segmentation_mask (NDArray[np.uint]): Segmentation mask input.
        tile_size (int, optional): Core side length. Keep it at least 4x the halo actually
            needed (the cells' inradius): the per-tile work factor is
            `((tile_size + 2*halo) / tile_size)**3`. Defaults to 256.
        halo (int, optional): Initial halo. Defaults to 16; tiles that need more escalate.
        print_info (bool, optional): Print the stage's wall time. Defaults to False.
        parallel (int, optional): Threads for `edt.edt`. Defaults to 1.

    Returns:
        tuple[NDArray[np.float32], dict]: the EDT, and the certification report with one
            entry per sub-field (`edt_1`, `edt_2`) plus `max_edt_2`.
    """
    if print_info:
        print(f"Computing EDT (tiled, tile={tile_size}, halo={halo}) ...")
    t1 = time()

    code = region_codes(segmentation_mask)
    out = np.empty(code.shape, dtype=np.float32)

    # edt_2 first: its values are bounded by the boundary sheets' half-thickness, so it
    # certifies at a small halo, and it yields the global constant M the assembly needs.
    report_2 = _tiled_binary_edt_into(out, code, CODE_BOUNDARY, tile_size, halo, parallel)
    max_edt_2 = np.float32(out.max())

    boundary = code == CODE_BOUNDARY
    # Keep edt_2 only where it is used (the boundary set is O(area)); recompute edt_1 over
    # the whole volume into the same buffer.
    boundary_index = np.flatnonzero(boundary.reshape(-1))
    boundary_values = out.reshape(-1)[boundary_index].copy()

    report_1 = _tiled_binary_edt_into(out, code, CODE_INTERIOR, tile_size, halo, parallel)

    out += max_edt_2
    flat = out.reshape(-1)
    flat[boundary_index] = max_edt_2 - boundary_values
    del boundary_index, boundary_values, boundary
    np.copyto(out, np.float32(0.0), where=code == CODE_ZERO)

    if print_info:
        print("EDT computed in ", np.round(time() - t1, 2))

    return out, {
        "edt_1": report_1,
        "edt_2": report_2,
        "max_edt_2": float(max_edt_2),
        "tile_size": int(tile_size),
        "halo": int(halo),
    }


def compute_edt_tiled_float64_with_report(
    segmentation_mask: NDArray[np.uint],
    tile_size: int = 256,
    halo: int = 16,
    print_info: bool = False,
    parallel: int = 1,
) -> tuple[NDArray[np.float64], dict]:
    """The tiled EDT, emitted as `float64`, bit-identical to `compute_edt_tiled(...).astype(float64)`.

    **Why this exists.** The tiled EDT's
    *values* are bit-identical to `compute_edt_classical`'s, but its **dtype** is `float32`,
    and the junction-protected default reads the field's last bits through the boundary
    layer's Hessian-eigenvector normals and the 27-sample face score interpolation. So
    `set_tiled_edt_method` does not compose with the default: 0 of 8 in-repo configurations
    reproduce (`tests/test_edt_dtype_composition.py`). The
    same measurement also showed that casting the same tiled field to `float64` **does**
    reproduce the default bit-for-bit, and observed that the tiled EDT's memory win comes
    mostly from never holding the whole volume at once rather than from the narrower dtype
    — so a `float64` tiled path should compose *and* keep most of the saving. This function
    is that path, built so the question can be answered with a measurement.

    **The trap this avoids, and the reason it is not just `.astype(np.float64)`.** Writing
    `compute_edt_tiled(mask).astype(np.float64)` holds a full-volume `float32` array and a
    full-volume `float64` array at the same moment, so its peak is 12 bytes/voxel — *worse* than
    dense `compute_edt_float32`'s measured 11.2. Here the output is allocated `float64` up front
    and each tile's `float32` result is written into it, so no `float32` volume ever exists.

    **How bit-identity is preserved.** Every arithmetic step is still performed in `float32` and
    only the *stored* result is widened. That matters: `float64(a) + float64(b)` is not
    `float64(float32(a) + float32(b))`, so assembling the field in `float64` (`out += max_edt_2`
    on a `float64` buffer) would give a genuinely different field and defeat the point. The two
    sub-fields are therefore assembled per tile in `float32` — `float32(edt_1) + float32(max_edt_2)`
    — exactly as `compute_edt_tiled_with_report` does on its `float32` buffer, and the widening
    happens on assignment. `tests/test_edt_float64_tiling.py` asserts the bit-identity rather than
    arguing it.

    Live full-volume arrays at peak: the `float64` output and the `uint8` region codes, i.e.
    **1 full-volume float64 array** plus one byte per voxel, plus `O(tile_size**3)` scratch and an
    `O(interface area)` `float32` buffer holding `edt_2` at the boundary voxels (which the
    `float32` path also keeps, for the same reason: `edt_2` is overwritten by the `edt_1` pass).

    Args:
        segmentation_mask (NDArray[np.uint]): Segmentation mask input.
        tile_size (int, optional): Core side length. Defaults to 256.
        halo (int, optional): Initial halo; tiles escalate if they need more. Defaults to 16.
        print_info (bool, optional): Print the stage's wall time. Defaults to False.
        parallel (int, optional): Threads for `edt.edt`. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict]: the EDT as `float64`, and the certification report, in
            the same shape `compute_edt_tiled_with_report` returns plus `"dtype"`.
    """
    if print_info:
        print(f"Computing EDT (tiled float64, tile={tile_size}, halo={halo}) ...")
    t1 = time()

    code = region_codes(segmentation_mask)
    out = np.empty(code.shape, dtype=np.float64)
    shape = code.shape

    boundary_flat_index = np.flatnonzero((code == CODE_BOUNDARY).reshape(-1))
    boundary_values = np.empty(len(boundary_flat_index), dtype=np.float32)

    # Pass 1 -- edt_2, kept only where it is read (the boundary set, O(area)), plus the global
    # constant `max_edt_2` the assembly needs. Scattered by flat index rather than recomputed in
    # pass 2: recomputing would cost a second full edt_2 sweep for a buffer of O(area) bytes.
    report_2 = {"n_tiles": 0, "n_escalated": 0, "halo_initial": int(halo), "halo_max_used": int(halo),
                "n_uncertified": 0, "max_value": 0.0}
    max_edt_2 = np.float32(0.0)
    for core in _tiles(shape, tile_size):
        distance, used_halo, certified, escalations = certified_region_edt(code, core, CODE_BOUNDARY, halo, parallel)
        report_2["n_tiles"] += 1
        report_2["n_escalated"] += escalations
        report_2["halo_max_used"] = max(report_2["halo_max_used"], used_halo)
        report_2["max_value"] = max(report_2["max_value"], float(distance.max()) if distance.size else 0.0)
        if not certified:  # pragma: no cover - unreachable; `certified_region_edt` cannot return False
            report_2["n_uncertified"] += int(np.count_nonzero(distance > used_halo))
        if distance.size:
            max_edt_2 = np.float32(max(float(max_edt_2), float(distance.max())))
        local = np.argwhere(code[core] == CODE_BOUNDARY)
        if len(local):
            global_flat = np.ravel_multi_index(
                (local + np.array([sl.start for sl in core])).T,
                shape,
            )
            boundary_values[np.searchsorted(boundary_flat_index, global_flat)] = distance[tuple(local.T)]

    # Pass 2 -- edt_1, offset by `max_edt_2` in float32, widened only on the store.
    report_1 = {"n_tiles": 0, "n_escalated": 0, "halo_initial": int(halo), "halo_max_used": int(halo),
                "n_uncertified": 0, "max_value": 0.0}
    for core in _tiles(shape, tile_size):
        distance, used_halo, certified, escalations = certified_region_edt(code, core, CODE_INTERIOR, halo, parallel)
        report_1["n_tiles"] += 1
        report_1["n_escalated"] += escalations
        report_1["halo_max_used"] = max(report_1["halo_max_used"], used_halo)
        report_1["max_value"] = max(report_1["max_value"], float(distance.max()) if distance.size else 0.0)
        if not certified:  # pragma: no cover - unreachable
            report_1["n_uncertified"] += int(np.count_nonzero(distance > used_halo))
        out[core] = distance + max_edt_2  # float32 + float32 -> float32, then widened on store

    out.reshape(-1)[boundary_flat_index] = max_edt_2 - boundary_values
    np.copyto(out, 0.0, where=code == CODE_ZERO)

    if print_info:
        print("EDT computed in ", np.round(time() - t1, 2))

    return out, {
        "edt_1": report_1,
        "edt_2": report_2,
        "max_edt_2": float(max_edt_2),
        "tile_size": int(tile_size),
        "halo": int(halo),
        "dtype": "float64",
    }


def compute_edt_tiled(
    segmentation_mask: NDArray[np.uint],
    tile_size: int = 256,
    halo: int = 16,
    print_info: bool = False,
    parallel: int = 1,
    dtype: type = np.float32,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """`compute_edt_tiled_with_report` as an `EdtCreationFunction`, raising if certification fails.

    The certification cannot fail as the escalation is written (see the module docstring),
    so the check is an assertion about the code rather than a runtime fallback -- but it is
    checked, because a silent wrong EDT would be indistinguishable from a correct one
    downstream.

    Args:
        segmentation_mask (NDArray[np.uint]): Segmentation mask input.
        tile_size (int, optional): Core side length. Defaults to 256.
        halo (int, optional): Initial halo. Defaults to 16.
        print_info (bool, optional): Print the stage's wall time. Defaults to False.
        parallel (int, optional): Threads for `edt.edt`. Defaults to 1.
        dtype (type, optional): `np.float32` (the default, the tiled EDT's original contract) or
            `np.float64`, which routes to `compute_edt_tiled_float64_with_report`. The `float64`
            variant exists because the current default reconstruction reads the EDT's dtype and
            not only its values, so only the `float64` field composes with it -- see that
            function's docstring and `tests/test_edt_dtype_composition.py`. Defaults to `np.float32`.

    Returns:
        NDArray[np.float32] | NDArray[np.float64]: The EDT, in the requested dtype.

    Raises:
        ValueError: If `dtype` is neither `np.float32` nor `np.float64`.
        RuntimeError: If any core failed to certify (unreachable as written; checked anyway).
    """
    if dtype not in (np.float32, np.float64):
        message = f"compute_edt_tiled supports dtype np.float32 or np.float64, not {dtype!r}"
        raise ValueError(message)
    builder = compute_edt_tiled_with_report if dtype is np.float32 else compute_edt_tiled_float64_with_report
    edt_image, report = builder(
        segmentation_mask,
        tile_size=tile_size,
        halo=halo,
        print_info=print_info,
        parallel=parallel,
    )
    uncertified = report["edt_1"]["n_uncertified"] + report["edt_2"]["n_uncertified"]
    if uncertified:
        message = (
            f"tiled EDT left {uncertified} voxels uncertified after halo escalation "
            f"(report: {report}); the result is an upper bound there, not the EDT"
        )
        raise RuntimeError(message)
    return edt_image


# ---------------------------------------------------------------------------------------
# Sparse (narrow-band) representation
# ---------------------------------------------------------------------------------------


class SparseEdt:
    """`compute_edt_classical`'s field, stored in `O(interface area)` instead of `O(volume)`.

    See the module docstring for why this is lossless: `edt_1` and `edt_2` are distance
    functions, so the zero sets determine them, and the zero sets are the interface sheets.

    What is stored: the zero voxels of `edt_1` (the boundary set plus the bounding-box
    shell) as an integer coordinate array and a `cKDTree` over it; the boundary set's own
    `edt_2` values, keyed by sorted flat index; and the scalar `M`. Nothing is `O(volume)`.

    Attributes:
        shape: the volume's shape, so the object can stand in for the array's metadata.
        n_zero_voxels: size of the stored zero set.
        max_edt_2: the constant `M`.
        bytes_stored: the container's own footprint, for memory accounting.
    """

    def __init__(self, segmentation_mask: NDArray[np.uint], parallel: int = 1) -> None:
        """Build the sparse representation from a segmentation mask.

        Peak memory is one `uint8` code volume plus one `float32` volume for `edt_2`, both
        released before returning; only the `O(area)` arrays survive. `edt_2` is computed
        densely here because it is bounded by the sheet half-thickness and so is not the
        scaling problem; `edt_1`, which is, is never materialised at all.
        """
        code = region_codes(segmentation_mask)
        self.shape = tuple(code.shape)

        zero = code != CODE_INTERIOR
        self._zero_coords = np.argwhere(zero).astype(np.int32)
        self._tree = cKDTree(self._zero_coords.astype(np.float64))

        boundary = code == CODE_BOUNDARY
        edt_2 = _binary_edt(boundary, parallel)
        self.max_edt_2 = np.float32(edt_2.max())
        flat_boundary = boundary.reshape(-1)
        self._boundary_index = np.flatnonzero(flat_boundary)
        self._boundary_edt_2 = edt_2.reshape(-1)[self._boundary_index].copy()
        del edt_2, boundary

        # `CODE_ZERO` voxels are the only ones whose value is not a function of a distance,
        # so they are stored explicitly too. They are a subset of the box's surface.
        self._zero_code_index = np.flatnonzero((code == CODE_ZERO).reshape(-1))
        self.n_zero_voxels = len(self._zero_coords)

    @property
    def bytes_stored(self) -> int:
        """Footprint of the container's own arrays, excluding the `cKDTree`'s internal nodes."""
        return int(
            self._zero_coords.nbytes
            + self._boundary_index.nbytes
            + self._boundary_edt_2.nbytes
            + self._zero_code_index.nbytes,
        )

    def at(self, voxels: NDArray[np.int64], workers: int = -1) -> NDArray[np.float32]:
        """Evaluate the EDT at integer voxel coordinates, exactly, without a dense array.

        Three cases, matching the module docstring's definition: `CODE_ZERO` voxels return
        0; boundary voxels return `M - edt_2` from the stored table; interior voxels return
        `M + dist(v, zero set)` from the `cKDTree`.

        The interior branch is exact up to the rounding of one square root. `edt.edt`
        computes `float32(sqrt(exact squared distance))` while `cKDTree` computes
        `float64(sqrt(...))` and is then narrowed, so the two can differ by one float32
        ulp -- measured at `<= 9.5e-07` absolute on `data/Images/3.tif`, i.e. `2.5e-08` of
        the field's dynamic range. `tests/test_edt_scaling.py` pins that bound; it is far
        below the `atol=1e-4` accuracy target of the memory-saving variants, but it is **not**
        zero, so this container is the one memory-saving variant that is not bit-identical to the
        baseline, and it is reported as such rather than rounded into agreement.

        Args:
            voxels (NDArray[np.int64]): `(n, 3)` integer voxel coordinates.
            workers (int, optional): `cKDTree.query` workers. `-1` uses all cores. The
                result does not depend on it.

        Returns:
            NDArray[np.float32]: `(n,)` field values.
        """
        voxels = np.asarray(voxels, dtype=np.int64)
        flat = np.ravel_multi_index(tuple(voxels.T), self.shape)
        values = self.max_edt_2 + self._tree.query(voxels.astype(np.float64), workers=workers)[0].astype(np.float32)

        on_boundary = np.isin(flat, self._boundary_index)
        if on_boundary.any():
            slot = np.searchsorted(self._boundary_index, flat[on_boundary])
            values[on_boundary] = self.max_edt_2 - self._boundary_edt_2[slot]

        values[np.isin(flat, self._zero_code_index)] = np.float32(0.0)
        return values

    def dense(self, workers: int = -1) -> NDArray[np.float32]:
        """Materialise the whole field. For tests and small volumes only -- this is `O(volume)`."""
        grid = np.indices(self.shape).reshape(3, -1).T
        return self.at(grid, workers=workers).reshape(self.shape)


def sparse_edt(segmentation_mask: NDArray[np.uint], parallel: int = 1) -> SparseEdt:
    """Build the `O(area)` sparse representation of the EDT. See `SparseEdt`."""
    return SparseEdt(segmentation_mask, parallel=parallel)
