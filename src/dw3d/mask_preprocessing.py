"""Mask preprocessing before meshing: resampling a mask with anisotropic voxels to cubic voxels.

Why this module exists
----------------------
dw3d meshes a label image on its own voxel grid. When the voxels are longer along one axis (typically z) than
along the other two, meshing on that grid and scaling the mesh afterwards samples the surfaces coarsely along the
long axis: the junction triangles come out about `k` times taller (`k` = long spacing / short spacing) and their
normals, hence every contact angle read from them, are off. Resampling the mask to cubic voxels by **shape-based
interpolation** before meshing removes most of that error. Measured on masks down-sampled along z by `k` (each
coarse voxel takes the label covering most of it), one meshing seed, as the median over cases of each mesh's median
line-mean contact-angle error minus that of the same mask meshed at full resolution:

* on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size (dw3d 0.7.0; not measured again for
  0.7.1, whose meshes differ slightly, because some of these runs need 8 to 40 GiB of memory): meshing the
  native grid costs +1.38 deg at `k = 5/3`, +1.97 at 2 and +4.10 at 3; resampling first, +0.83, +0.71 and +1.17;
* on foambryo's benchmark, whose cells are larger (42 cases, dw3d 0.7.1): native +1.59, +2.25 and
  +5.85 deg at `k = 5/3, 2, 3`; resampled +0.43, +1.46 and +1.27.

Tension errors follow. Repeating slices (nearest neighbour) is worse than shape-based interpolation at every
anisotropy above 2, and coarsening the two short axes, or passing the anisotropy to the distance transform alone,
is worse than not resampling at all. Nothing was measured above `k = 3` at real cell size. foambryo's quadric angle
reading, which helps on the benchmark's larger cells, made every route worse at real cell size (dw3d 0.7.0's
meshes): keep foambryo's default reading.

Conventions
-----------
* `spacing` is the voxel size along the array's axes `(axis 0, axis 1, axis 2)`, so `(dz, dy, dx)` for a
  `(z, y, x)` stack, in any one unit. Exactly one axis may be coarser than the other two, which must be equal (to a
  relative `SPACING_RELATIVE_TOLERANCE`); this is the case of a microscope stack with its z step larger than its
  pixel size. Other spacings raise.
* Frame: the centre of voxel `j` along an axis lies at `j * spacing` along that axis, so the first voxel of the input
  and of the output share their centre, and the output covers exactly the input's span between its first and last
  voxel centres. The output spacing is the short spacing `h` on every axis.
* The output has the input's dtype and labels; no label is created.
"""

from __future__ import annotations

from math import floor

import numpy as np
from edt import edt as euclidean_dt
from numpy.typing import NDArray
from scipy import ndimage

#: Resampling methods: `"shape"` (per-label shape-based interpolation, the one measured to work) and `"nearest"`
#: (nearest slice, kept for comparison; measured worse at every anisotropy above 2).
RESAMPLING_METHODS = ("shape", "nearest")

#: Two spacings closer than this, relatively, are equal (a stack's metadata rounds its pixel size).
SPACING_RELATIVE_TOLERANCE = 1e-6

#: Shape-based interpolation computes a label's signed distance only inside its bounding box grown by this many
#: voxels across the slices; outside it the label's value is minus infinity. A label below `-margin` in both
#: neighbouring slices cannot win a voxel that another label owns in either of them (that one is at `+1/2` there).
SHAPE_INTERPOLATION_MARGIN = 8

#: The largest anisotropy (long spacing / short spacing) at which resampling was measured at real cell size. Above
#: it `MeshReconstructionAlgorithm.construct_mesh_in_physical_units` still works but warns
#: (`UnmeasuredAnisotropyWarning`).
MEASURED_ANISOTROPY_RATIO_MAX = 3.0


class UnmeasuredAnisotropyWarning(UserWarning):
    """The voxels are more than `MEASURED_ANISOTROPY_RATIO_MAX` times longer along one axis: accuracy not measured.

    Issued by `MeshReconstructionAlgorithm.construct_mesh_in_physical_units`. The mesh is still built; filter this
    category to silence it.
    """


def isotropic_spacing(spacing: tuple[float, float, float]) -> float:
    """The cubic voxel size `resample_to_isotropic` produces: the shortest of the three spacings."""
    return float(min(spacing))


def coarse_axis_and_factor(spacing: tuple[float, float, float]) -> tuple[int, float] | None:
    """The one coarse axis and its factor `k = coarse / fine`, or `None` when the voxels are cubic.

    Raises:
        ValueError: if a spacing is not positive and finite, or if the two finer axes differ (only one coarse
            axis is supported, the case of a z stack).
    """
    s = np.asarray(spacing, dtype=np.float64)
    if s.shape != (3,) or not np.all(np.isfinite(s)) or np.any(s <= 0):
        message = f"spacing must be three positive finite numbers, got {spacing!r}"
        raise ValueError(message)
    fine = float(np.min(s))
    close = np.abs(s - fine) <= SPACING_RELATIVE_TOLERANCE * fine
    if close.all():
        return None
    if close.sum() != 2:
        message = (
            f"spacing {tuple(spacing)} has two different fine axes; resample_to_isotropic supports one coarse axis "
            "with the other two equal (a z stack)"
        )
        raise ValueError(message)
    axis = int(np.flatnonzero(~close)[0])
    return axis, float(s[axis] / fine)


def resample_to_isotropic(
    mask: NDArray,
    spacing: tuple[float, float, float],
    method: str = "shape",
) -> NDArray:
    """Resample a label mask with one coarse axis to cubic voxels of the short spacing.

    Args:
        mask: 3-D integer label image (`0` = medium or background).
        spacing: voxel size along axes 0, 1, 2, so `(dz, dy, dx)` for a `(z, y, x)` stack (see the module's
            conventions).
        method: `"shape"` (default) or `"nearest"`, see `RESAMPLING_METHODS`.

    Returns:
        The resampled mask, same dtype, on cubic voxels of size `isotropic_spacing(spacing)`. Along the coarse axis
        it has `floor((n - 1) k) + 1` voxels for `n` input voxels. **When the voxels are already cubic the input
        array itself is returned**, unchanged and uncopied, so an isotropic mask goes through bitwise.

    Raises:
        ValueError: for an unknown method, a mask that is not 3-D, or an unsupported spacing.

    The shape-based method (Raya & Udupa, 1990): for every label, including `0`, its 2-D signed distance is computed
    on every input slice across the coarse axis (positive inside, zero half-way between voxel centres) and
    interpolated linearly between slice centres; each output voxel takes the label with the largest value, ties to
    the smaller label. Where a label is present in one slice and absent from the next, the absent slice's value is
    `min(d - k, -1/2)` (`d` the present slice's): the surface keeps closing at one voxel per voxel, but never
    reaches the centre of a slice where the mask says the label is absent. Known bias: linear interpolation of a
    concave distance profile underestimates it, so small convex cells shrink slightly (measured on balls of radius
    9-14 voxels: -1 to -5 % of their volume at `k = 3`, -5 to -11 % at `k = 4`). Each label is computed only in its
    bounding box (`SHAPE_INTERPOLATION_MARGIN`), so the cost grows with the cells' volume, not with the number of
    labels times the volume: about 40 s and 23 GiB for a 1 866-cell tissue of 4.98e8 voxels at `k = 5/3`.
    The measurements behind the choice are summarised in the module docstring.
    """
    if method not in RESAMPLING_METHODS:
        message = f"unknown resampling method {method!r}; expected one of {RESAMPLING_METHODS}"
        raise ValueError(message)
    mask = np.asarray(mask)
    if mask.ndim != 3:
        message = f"mask must be 3-D, got shape {mask.shape}"
        raise ValueError(message)
    found = coarse_axis_and_factor(spacing)
    if found is None:
        return mask
    axis, k = found
    moved = np.moveaxis(mask, axis, 0)
    resampled = _nearest_along_axis_0(moved, k) if method == "nearest" else _shape_based_along_axis_0(moved, k)
    return np.ascontiguousarray(np.moveaxis(resampled, 0, axis))


def _output_positions(n: int, k: float) -> NDArray[np.float64]:
    """Output voxel centres along the coarse axis, in units of the short spacing: `0, 1, ..., floor((n-1) k)`."""
    # The tolerance keeps an exact multiple (e.g. (n - 1) * 3) from losing its last voxel to rounding.
    n_out = floor((n - 1) * k + 1e-9) + 1
    return np.arange(n_out, dtype=np.float64)


def _nearest_along_axis_0(mask: NDArray, k: float) -> NDArray:
    """Each output slice takes the input slice whose centre is nearest (`j = floor(i / k + 1/2)`)."""
    positions = _output_positions(mask.shape[0], k)
    source = np.minimum(np.floor(positions / k + 0.5).astype(np.int64), mask.shape[0] - 1)
    return mask[source]


def _signed_distance_2d(slices: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Per-slice 2-D signed distance in voxels, positive inside, zero half-way between centres; absent label: -inf."""
    out = np.empty(slices.shape, dtype=np.float32)
    for s, inside in enumerate(slices):
        if not inside.any():
            out[s] = -np.inf
            continue
        d_in = euclidean_dt(inside.astype(np.uint8), black_border=False)
        d_out = euclidean_dt((~inside).astype(np.uint8), black_border=False)
        out[s] = np.where(inside, d_in - 0.5, 0.5 - d_out)
    return out


def _shape_based_along_axis_0(mask: NDArray, k: float) -> NDArray:
    """Shape-based interpolation along axis 0 (see `resample_to_isotropic`)."""
    n, ny, nx = mask.shape
    positions = _output_positions(n, k)
    n_out = len(positions)
    centres = np.arange(n, dtype=np.float64) * k
    lower = np.clip(np.searchsorted(centres, positions, side="right") - 1, 0, n - 1)
    upper = np.clip(lower + 1, 0, n - 1)
    span = np.where(upper > lower, centres[upper] - centres[lower], 1.0)
    weight_upper = np.where(upper > lower, np.clip((positions - centres[lower]) / span, 0.0, 1.0), 0.0)

    best_value = np.full((n_out, ny, nx), -np.inf, dtype=np.float32)
    best_label = np.zeros((n_out, ny, nx), dtype=mask.dtype)
    margin = SHAPE_INTERPOLATION_MARGIN
    boxes = ndimage.find_objects(mask) if np.max(mask) > 0 else []
    for label in np.unique(mask):
        if label == 0:
            z0, z1, y0, y1, x0, x1 = 0, n, 0, ny, 0, nx
        else:
            bz, by, bx = boxes[int(label) - 1]
            z0, z1 = max(bz.start - 1, 0), min(bz.stop + 1, n)
            y0, y1 = max(by.start - margin, 0), min(by.stop + margin, ny)
            x0, x1 = max(bx.start - margin, 0), min(bx.stop + margin, nx)
        sd = _signed_distance_2d(mask[z0:z1, y0:y1, x0:x1] == label)
        filled = sd.copy()
        absent = [bool(np.all(sd[s] == -np.inf)) for s in range(sd.shape[0])]
        for s in range(sd.shape[0]):
            if not absent[s]:
                continue
            neighbours = [t for t in (s - 1, s + 1) if 0 <= t < sd.shape[0] and not absent[t]]
            if neighbours:
                filled[s] = np.minimum(np.max([sd[t] for t in neighbours], axis=0) - np.float32(k), np.float32(-0.5))
        i0 = int(np.ceil(centres[z0])) if z0 > 0 else 0
        i1 = min(int(np.floor(centres[z1 - 1])) + 1 if z1 < n else n_out, n_out)
        if i1 <= i0:
            continue
        lo = np.clip(lower[i0:i1] - z0, 0, filled.shape[0] - 1)
        hi = np.clip(upper[i0:i1] - z0, 0, filled.shape[0] - 1)
        w = weight_upper[i0:i1].astype(np.float32)[:, None, None]
        # Explicit end points: 0 * -inf is nan, so the blend is formed strictly inside only.
        with np.errstate(invalid="ignore"):
            blend = (1.0 - w) * filled[lo] + w * filled[hi]
        value = np.where(w <= 0.0, filled[lo], np.where(w >= 1.0, filled[hi], blend)).astype(np.float32)
        region = (slice(i0, i1), slice(y0, y1), slice(x0, x1))
        better = value > best_value[region]
        best_value[region] = np.where(better, value, best_value[region])
        best_label[region] = np.where(better, label, best_label[region])
    return best_label
