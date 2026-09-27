"""Module with methods to place points on an EDT image for future tesselation.

Plateau tie-breaking (the determinism fix)
--------------------------------------------
The EDT of a segmentation mask is massively plateau-valued: on `data/Images/3.tif` at
`min_distance=3`, 287 129 voxels are exact local minima of the distance field, forming
444 connected plateaus, the largest of which are whole interface sheets. `peak_local_max`
resolves such a plateau by returning *an* arbitrary representative of it (see the
`versionchanged:: 0.18` note in its docstring), and `dw3d` used to make that choice by
adding `U(0, 1e-5)` noise to a copy of the EDT under `np.random.seed(42)`.

That dither was measured to cause up to 7.3 % seed-to-seed
interface-area CV on `3.tif` and 22.8 % on `4.tif`, with a triple line vanishing outright
at seed 2. The seed hides the spread rather than removing it: it freezes one draw out of a
distribution the algorithm has no reason to prefer.

`peak_local_points` therefore now makes the choice **geometrically**, with no random
number involved anywhere. The rule, in full:

1. Take the same plateau set `peak_local_max` would take, from the **undithered** EDT: the
   voxels equal to a `(2*min_distance+1)**3` box extremum filter of the field, minus the
   global-extremum threshold `peak_local_max` applies by default.
2. Partition space into a regular lattice of cells of side `s = 2*min_distance+1` — the
   *same* box the extremum filter uses, so this introduces no new length scale.
3. Visit cells strongest-extremum-first, ties broken by lattice raster order. Within a
   cell, consider only candidates at Chebyshev distance `>= s` from every already-accepted
   point, and take the one whose **nearest accepted point is furthest away** in Euclidean
   distance — a farthest-point (Poisson-disk) choice. Ties fall back to the strongest
   extremum value, then to the candidate nearest that cell's candidate centroid, then to
   raster order. A cell whose candidates are all too close to an accepted point
   contributes no point.

The result is a **greedy maximal `s`-separated farthest-point packing of the plateau
set**, computed in a fixed deterministic order: identical on every run, on every platform,
and independent of numpy's RNG. Step 3 only ever has to test the 26 neighbouring cells,
because a cell of side `s` holds at most one accepted point and cells two apart are
automatically more than `s` apart, so the whole rule is O(number of plateau voxels).

Why farthest-point and not simply "the candidate nearest the cell centre"
-------------------------------------------------------------------------
Because the nearest-the-centre rule lays the samples out on a near-*cubic* lattice, and a
cubic arrangement is the degenerate case for a Delaunay triangulation: four points of a
square are cocircular, eight points of a cube cospherical, and Qhull resolves such cells
arbitrarily into zero-volume simplices. Measured on `3.tif` at `min_distance=3`, the
centre rule produced **15.7 % exactly-flat tetrahedra** (quality < 1e-6) against 0.18 %
for the dithered default, and drove the sliver fraction (quality < 0.01) from 14.1 % up to
20.6 %. The farthest-point rule approximates a hexagonal/blue-noise arrangement instead,
whose Delaunay is generically non-degenerate, and brings both back down — to 0.69 %
exactly-flat and **6.25 %** slivers, i.e. better than the dithered default it replaces
(and better at `min_distance=5` too, 5.2 % against 5.6 %). Tetrahedron quality improves in
the median as well (0.0232 -> 0.0293). None of that is the determinism fix's *purpose*, but
it is a real consequence of the tie-break and it moves two of the boundary-layer work's
acceptance metrics, so it is recorded and quantified here rather than
left for that work to be credited with.

Why `s = 2*min_distance+1` and not `min_distance`
-------------------------------------------------
Measured, not chosen to fit. `min_distance` is *not* the spacing the old default actually
produced: the dither's density on a plateau is one representative per
`|plateau ∩ filter footprint|` voxels, i.e. one per `(2*min_distance+1)**2` voxels of an
interface sheet — an effective spacing of 7 voxels at `min_distance=3`, not 3. Packing the
same plateaus at spacing `min_distance` instead yields **5.2x** as many interface points
and takes 43x as long in `skimage`'s `ensure_spacing` (measured on all four in-repo
images); packing at `2*min_distance+1` reproduces the old point budget to
**0.84-0.94x** on the interface and **0.86-0.96x** overall. Preserving the budget is the
point: rebalancing it is the boundary-layer work's job, and the determinism fix must not
do it silently.

Alternatives rejected, all measured on the four in-repo images at `min_distance` 3 and 5:

- **Plain lexicographic order on the raw EDT** (i.e. just delete the dither and call
  `peak_local_max` unchanged): 4.5-5.2x the interface points, 3.0-5.1x the interior
  points, and point placement goes from 0.3 s to 14.6 s on `3.tif`. Breaks both the point
  budget and the 20 % wall-time budget set for the change.
- **One centroid per connected plateau component**: collapses each interface sheet to a
  *single* point (444 components carry 287 129 candidate voxels on `3.tif`). Unusable.
- **One representative per lattice cell, without the packing constraint**: 1.15x interface
  points but 2.4-3.4x interior points, because a compact plateau straddling a cell
  boundary is split into two samples. The packing constraint in step 3 is exactly what
  fixes that — a second sample of the same small plateau is within `s` of the first.
- **Nearest-the-cell-centre instead of farthest-point** within the packing: same budget,
  but a cubic point lattice and 15.7 % zero-volume tetrahedra. See *Why farthest-point*,
  above.
- **A deterministic pseudo-random dither** keyed on the voxel coordinates (a hash instead
  of `np.random`). Reproducible and budget-neutral, but it only re-freezes an arbitrary
  draw under a different name, which is what the determinism fix set out to stop doing.

The dithered path is preserved verbatim as `peak_local_points_dithered`, reachable through
`MeshReconstructionAlgorithmFactory.set_dithered_peak_local_points_placement_method` and
`get_dithered_algorithm`, so the 20-seed determinism sweep and the v0.3 golden masters can
still be run. It no longer reseeds the *global* numpy RNG — an import-time-visible side
effect on every caller — but draws from its own `RandomState`, which is bit-identical.

Boundary layer
---------------
`peak_local_points_boundary_layer` is the boundary-layer point-placement scheme. It keeps the
deterministic interface minima and interior maxima and
*adds* a structured layer of **offset points** a fixed distance `delta` into each material
adjacent to a genuine interface, so that near-interface tetrahedra acquire a vertex at a
controlled non-zero distance from the interface instead of having all four vertices on it.

For each interface sample `p` (an EDT minimum):

1. The across-sheet **normal line** is the eigenvector of the largest-magnitude eigenvalue
   of the EDT's Hessian at `p`. The raw EDT *gradient* is not usable as a normal here: the
   field is a valley across the sheet, so at the trough the gradient is degenerate/tangent
   (measured: identical straddle yield whether the gradient or the Hessian is used, because
   the sign is what matters and that comes from the label image, not the field). The
   Hessian's dominant curvature direction is across the sheet by construction. Its sign is
   irrelevant because both `p + delta*n` and `p - delta*n` are probed.
2. The two candidates `p ± delta*n` are rounded to voxels and their **labels read from the
   segmentation mask** (not from the EDT, which is unsigned). If they are in-domain and
   carry *different* labels, both are emitted — one into each adjacent material. Otherwise
   the sample straddles nothing (it is a background/bounding-box plateau minimum, not a real
   interface) and yields no point. Adjacency is thus decided by the label image, never by the
   field.

Measured on `data/Images/3.tif`, `min_distance=3`: of 5228 EDT minima, **1172 (22.4 %)**
lie on a genuine >=2-label boundary, and **100 %** of those produce a correctly-oriented
straddling pair; the remaining 77.6 % are the outer bounding-box / background shell dw3d
also samples and correctly get no offset. The 2344 emitted offsets drop the all-surface-tet
fraction from the deterministic configuration's 42.0 % to ~8 % while holding the sliver
fraction. `delta` defaults to
`min_distance` (= `h_S`, the interface spacing). Offsets are deduplicated against one
another and against the corner/maximum/minimum points, so no tesselation vertex is repeated.

The offsets' EDT values are `~= delta` by construction and are what the regular-triangulation
work's weighted (regular) triangulation will use as weights; the boundary-layer scheme does
not yet plumb a weight channel through the `PointPlacingFunction` contract (that is the
regular-triangulation work's API change, which would alter a public signature), so they are
recomputed from the EDT there rather than returned here. This keeps the boundary-layer scheme a pure additive
point-placement method with the existing 2-tuple contract and the default unchanged.

Junction protection and shell coarsening
------------------------------------------
`peak_local_points_junction_protected` is the boundary layer plus the three things
junction protection adds, following Boltcheva, Yvinec & Boissonnat (MICCAI 2009). The strata
themselves are detected in `dw3d.junctions`; this module is where they enter the point budget.

**1. Junction samples with protecting balls.** The 0-strata (quadruple points) and 1-strata
(triple lines) are detected from the *label image*, thinned, and sampled at arclength
`h_J`. Each sample owns a protecting ball of radius `r_J = h_J / 2` — the radius at which
the balls of consecutive samples are tangent, so the balls cover the sampled curve — and
every *unstructured* point (interface minimum, interface boundary-layer offset) inside a
ball is dropped. That exclusion is not an extra knob layered on Boltcheva's scheme, it *is*
the scheme: a protecting ball is by definition empty of other samples.

It is **not** something the weights would have done by themselves, and an earlier draft of
this docstring claimed wrongly that it was. A weighted point is hidden only when its power
cell is empty, which needs the surrounding balls to *enclose* it; a single heavier
neighbour never suffices, because the power bisector of two weighted points is a plane and
each keeps a half-space (measured in `dw3d.tesselation`'s docstring and pinned in
`tests/test_junction_protection.py`). The exclusion is therefore a separate, necessary
step — and the ablation bears that out: dropping it takes
valence->=4 edges from 33 to 52 and abnormal non-manifold edges from 2 to 20 over the
11-case screen, whereas dropping the *weights* changes neither.

**2. `h_J = 2*min_distance + 1`, not `min_distance`.** The original specification was `h_J <= h_S`,
the interface spacing. The determinism fix measured that `dw3d`'s *effective* interface spacing is
`2*min_distance + 1` (7 at `min_distance = 3`), not `min_distance` — the parameter does not
mean what its name says (see *Why `s = 2*min_distance+1`* above). Sampling 1-junctions at
`min_distance` would therefore put 2.3x more points per unit length on the junctions than
on the interfaces, which is precisely the "more points locally -> more small tets" failure
M. Perez recorded. `h_J = 2*min_distance + 1` is the *density-matched* choice, not a tuned
one.

**3. The junction boundary layer.** The boundary layer's offset construction is applied around
junction samples too, into each of the 3 (or 4) adjacent materials, so that a junction
sample is surrounded by a structured shell instead of by whatever the interface sampling
left nearby. The adjacent materials, and the direction of each offset, are read from the
*label image* in a small stencil around the sample — never from the unsigned EDT. Junction
samples and their offsets form one *protected cluster* carrying the same weight `r_J**2`,
so they cannot hide one another (equal weights make the hiding test unsatisfiable) while
still clearing unstructured points out of the neighbourhood.

**4. Shell coarsening (a cheap win found during the boundary-layer work).**
`compute_edt_classical` calls `_pad_mask`, which marks the one-voxel bounding-box shell as
boundary, so the EDT is exactly 0 along the whole shell and the shell is one enormous
minimum plateau. Measurement showed the consequence: only
1172 of 5228 interface minima on `3.tif` lie on a genuine >=2-label interface — **77.6 % of
the "interface" point budget is spent on the bounding box and on background plateaus.** The
shell has to exist (it bounds the tesselation and lets border-touching cells close) but it
does not need interface density. `shell_coarsening` re-packs exactly the minima whose label
multiplicity is 1 — i.e. those on no interface at all, identified for free by the same
stratum detector — at `shell_coarsening * (2*min_distance+1)` and leaves every genuine
interface minimum alone.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from dw3d.edt import get_total_boundaries
from dw3d.junctions import label_multiplicity, labels_around, sample_junction_network
from dw3d.mesh_utilities import WELD

# Point-family codes. A point-placing scheme that returns the optional
# `point_metadata` element of the `PointPlacingFunction` contract labels every tesselation
# point with one of these, so the harness can attribute a measurement to the construction
# that placed the point instead of re-deriving the block boundaries from family sizes.
# The values are part of the metadata contract: append, never renumber.
FAMILY_CORNER = 0
FAMILY_MAXIMUM = 1
FAMILY_INTERFACE_OFFSET = 2
FAMILY_JUNCTION_SAMPLE = 3
FAMILY_JUNCTION_OFFSET = 4
FAMILY_INTERFACE_MINIMUM = 5
FAMILY_NAMES = {
    FAMILY_CORNER: "corners",
    FAMILY_MAXIMUM: "maxima",
    FAMILY_INTERFACE_OFFSET: "interface_offsets",
    FAMILY_JUNCTION_SAMPLE: "junction_samples",
    FAMILY_JUNCTION_OFFSET: "junction_offsets",
    FAMILY_INTERFACE_MINIMUM: "interface_minima",
}
# The families the offset-exclusion work treats as tesselation scaffolding rather than surface geometry.
OFFSET_FAMILIES = (FAMILY_INTERFACE_OFFSET, FAMILY_JUNCTION_OFFSET)

# 26-connectivity neighbour offsets of a lattice cell.
_NEIGHBOUR_OFFSETS = tuple(
    (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)
)


def peak_local_points(
    _segmented_mask: NDArray[np.uint] | None,
    edt_image: NDArray[np.float64],
    min_distance: int,
    print_info: bool = False,
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Deterministic local min and max points (+ corner points) from an EDT image.

    Plateaus of the distance field are resolved by the geometric rule documented at the
    top of this module — a greedy maximal `(2*min_distance+1)`-separated packing of the
    plateau set — with no random dither and no RNG seeding. This is the default point
    placement since point placement was made deterministic; `peak_local_points_dithered`
    is the historical (dithered) behaviour.

    Args:
        _segmented_mask (NDArray[np.uint] | None): (not used for this function). Give None.
        edt_image (NDArray[np.float64]): 3D image of an Euclidean Distance Transform.
        min_distance (int): Minimum distance between peaks (local min and local max)
        print_info (bool, optional): Print details about the algorithm. Defaults to False.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint]]:
           - an array of 3D pixel coordinates, stacked as [8 corners, local maxima, local minima];
           - the indices, into that array, of the local maxima sorted by decreasing EDT value.
    """
    corners = _give_corners(edt_image)

    if print_info:
        print("Searching local extremas ...")

    local_maxes = plateau_packing_extrema(edt_image, min_distance, maximise=True)
    if print_info:
        print("Number of local maxes :", len(local_maxes))

    local_mins = plateau_packing_extrema(edt_image, min_distance, maximise=False)
    if print_info:
        print("Number of local minimas :", len(local_mins))

    return _assemble_points(edt_image, corners, local_maxes, local_mins)


def peak_local_points_dithered(
    _segmented_mask: NDArray[np.uint] | None,
    edt_image: NDArray[np.float64],
    min_distance: int,
    print_info: bool = False,
    seed: int = 42,
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Peak local min and max points (+ corner points) from a randomly dithered EDT image.

    This is dw3d's historical default, kept reachable so that the seed-to-seed determinism
    sweep and the v0.3 golden masters can still be run; see the module docstring for why
    it is no longer the default. `seed` was hard-wired to 42 and is now a parameter, which
    is what lets `benchmarks/profiling.py` sweep it without monkeypatching numpy.

    The draw comes from a private `np.random.RandomState(seed)` rather than from the
    global RNG that `np.random.seed(seed)` mutates. numpy's legacy `RandomState` is
    version-stable by policy and `RandomState(seed).rand(...)` is the same stream as
    `np.random.seed(seed); np.random.rand(...)`, so this is bit-identical to the old
    behaviour while no longer resetting the caller's global RNG as a side effect.

    Args:
        _segmented_mask (NDArray[np.uint] | None): (not used for this function). Give None.
        edt_image (NDArray[np.float64]): 3D image of an Euclidean Distance Transform.
        min_distance (int): Minimum distance between peaks (local min and local max)
        print_info (bool, optional): Print details about the algorithm. Defaults to False.
        seed (int, optional): Seed of the dither. Defaults to 42, the historical value.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint]]:
           - an array of 3D pixel coordinates, stacked as [8 corners, local maxima, local minima];
           - the indices, into that array, of the local maxima sorted by decreasing EDT value.
    """
    corners = _give_corners(edt_image)

    if print_info:
        print("Searching local extremas ...")

    edt = _dithered(edt_image, seed)

    local_mins = peak_local_max(-edt, min_distance=min_distance, exclude_border=False).astype(np.uint)
    if print_info:
        print("Number of local minimas :", len(local_mins))

    local_maxes = peak_local_max(edt, min_distance=min_distance, exclude_border=False).astype(np.uint)
    if print_info:
        print("Number of local maxes :", len(local_maxes))

    return _assemble_points(edt, corners, local_maxes, local_mins)


def peak_local_points_bias_boundaries(
    segmented_mask: NDArray[np.uint],
    edt_image: NDArray[np.float64],
    min_distance: int,
    print_info: bool = False,
    seed: int = 42,
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Peak local min and max points (+ corner points) from an EDT image, biased by boundary thickness.

    Still a dithered method: only the default path was made deterministic.
    As in `peak_local_points_dithered`, the draw now comes from a private `RandomState`
    instead of the global RNG, which is bit-identical but has no side effect on the caller.

    Args:
        segmented_mask (NDArray[np.uint]): Original segmented image.
        edt_image (NDArray[np.float64]): 3D image of an Euclidean Distance Transform.
        min_distance (int): Minimum distance between peaks (local min and local max)
        print_info (bool, optional): Print details about the algorithm. Defaults to False.
        seed (int, optional): Seed of the dither. Defaults to 42, the historical value.

    Returns:
        tuple[NDArray[np.uint], NDArray[np.uint]]:
           - an array of 3D pixel coordinates, stacked as [8 corners, local maxima, boundary minima];
           - the indices, into that array, of the local maxima sorted by decreasing EDT value.
    """
    corners = _give_corners(edt_image)

    if print_info:
        print("Searching local extremas ...")

    edt = _dithered(edt_image, seed)

    # try Matthieu Perez add labels to peak local max
    total_boundaries = get_total_boundaries(segmented_mask)
    bv = np.unique(total_boundaries).astype(np.int64)

    boundary_mins = None
    for value in bv[:-2]:
        new_local_mins = peak_local_max(
            -edt,
            min_distance=value + 1,
            exclude_border=False,
            labels=(total_boundaries == value),
        ).astype(np.uint)
        boundary_mins = new_local_mins if boundary_mins is None else np.vstack((boundary_mins, new_local_mins))

    value = bv[-2]
    new_local_mins = peak_local_max(
        -edt,
        min_distance=min_distance,
        exclude_border=False,
        labels=(total_boundaries == value),
    ).astype(np.uint)
    boundary_mins = new_local_mins if boundary_mins is None else np.vstack((boundary_mins, new_local_mins))

    if print_info:
        print("Number of local minimas :", len(boundary_mins))

    local_maxes = peak_local_max(edt, min_distance=min_distance, exclude_border=False).astype(np.uint)

    if print_info:
        print("Number of local maxes :", len(local_maxes))

    return _assemble_points(edt, corners, local_maxes, boundary_mins)


def plateau_packing_extrema(
    image: NDArray[np.float64],
    min_distance: int,
    maximise: bool,
) -> NDArray[np.uint]:
    """Deterministic plateau representatives of the local extrema of `image`.

    Implements the rule documented at the top of this module. The candidate set is exactly
    the one `skimage.feature.peak_local_max(image, min_distance, exclude_border=False)`
    would work from (`maximise=True`), or the one it would work from on `-image`
    (`maximise=False`): the voxels equal to a `(2*min_distance+1)**3` box extremum filter,
    excluding the global-extremum threshold `peak_local_max` applies when `threshold_abs`
    is left at its default. The selection out of that set is the geometric packing rule
    rather than `peak_local_max`'s intensity-sorted greedy pass at spacing `min_distance`.

    The returned points are pairwise at Chebyshev distance `>= 2*min_distance+1`, which is
    strictly stronger than `peak_local_max`'s `>= min_distance` guarantee. Minima and
    maxima are packed independently, so a minimum and a maximum may be closer than that —
    as was already the case historically.

    Args:
        image: the scalar field (the EDT).
        min_distance: the algorithm's `min_distance`; the lattice and the filter both use
            a box of side `2*min_distance+1`.
        maximise: True for local maxima (cell interiors), False for local minima
            (interfaces).

    Returns:
        (n, 3) array of voxel coordinates, in acceptance order.
    """
    size = 2 * min_distance + 1

    if maximise:
        filtered = ndi.maximum_filter(image, size=size, mode="nearest")
        mask = (image == filtered) & (image > image.min())
    else:
        filtered = ndi.minimum_filter(image, size=size, mode="nearest")
        mask = (image == filtered) & (image < image.max())
    del filtered

    coords = np.argwhere(mask)  # C order, i.e. already in raster (lexicographic) order
    if len(coords) == 0:
        return coords.astype(np.uint)

    return pack_plateau_candidates(coords, image[mask], min_distance, maximise)


def pack_plateau_candidates(
    coords: NDArray[np.int64],
    values: NDArray[np.float64],
    min_distance: int,
    maximise: bool,
) -> NDArray[np.uint]:
    """The packing half of `plateau_packing_extrema`, split out for the EDT memory/scaling work.

    `plateau_packing_extrema` finds the candidate set by filtering a dense volume;
    `dw3d.edt_band.streamed_plateau_packing_extrema` finds the *same* candidate set by
    streaming tiles and never materialising the volume. Both then call this function, so the
    two paths are identical by construction rather than by a duplicated implementation that
    has to be kept in step — which matters because the packing is order-sensitive in three
    places (lattice-cell order, within-cell order, farthest-point tie-breaks) and a
    re-implementation would be a silent-divergence risk.

    Args:
        coords: `(n, 3)` candidate voxel coordinates, in **raster (C) order** — the order
            `np.argwhere` produces. The caller is responsible for that ordering; it is the
            outermost tie-break of the rule.
        values: the field value at each candidate.
        min_distance: the algorithm's `min_distance`; the lattice uses a box of side
            `2*min_distance+1`.
        maximise: True for maxima, False for minima.

    Returns:
        (n, 3) array of accepted voxel coordinates, in acceptance order.
    """
    size = 2 * min_distance + 1
    coords = np.asarray(coords, dtype=np.int64)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.uint)

    # Lower `strength` == stronger extremum, so one ascending sort serves both cases.
    values = np.asarray(values)
    strength = -values if maximise else values

    cells = coords // size
    dims = cells.max(axis=0) + 1
    cell_ids = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]

    # Group the candidates by lattice cell.
    by_cell = np.argsort(cell_ids, kind="stable")
    cell_ids, coords, strength = cell_ids[by_cell], coords[by_cell], strength[by_cell]
    group_start = np.flatnonzero(np.concatenate(([True], cell_ids[1:] != cell_ids[:-1])))
    group_size = np.diff(np.concatenate((group_start, [len(cell_ids)])))
    group_of = np.repeat(np.arange(len(group_start)), group_size)

    # Each cell's candidate centroid, used only as a tie-break: it puts the representative
    # in the middle of the plateau's intersection with the cell rather than at its corner.
    centroid_sum = np.zeros((len(group_start), 3))
    np.add.at(centroid_sum, group_of, coords)
    centroid = centroid_sum / group_size[:, None]
    to_centroid = ((coords - centroid[group_of]) ** 2).sum(axis=1)

    # Within a cell: strongest extremum, then nearest the centroid, then raster order.
    # `group_of` is the primary key, so groups stay contiguous and in the same order.
    within = np.lexsort((np.arange(len(coords)), to_centroid, strength, group_of))
    coords, strength = coords[within], strength[within]

    cell_of_group = coords[group_start] // size
    best_strength = np.full(len(group_start), np.inf)
    np.minimum.at(best_strength, group_of, strength)
    # Cells strongest-first, ties by lattice raster order.
    cell_order = np.lexsort(
        (cell_of_group[:, 2], cell_of_group[:, 1], cell_of_group[:, 0], best_strength),
    )
    group_end = np.concatenate((group_start[1:], [len(coords)]))

    accepted_in_cell: dict[tuple[int, int, int], NDArray[np.int64]] = {}
    chosen: list[NDArray[np.int64]] = []
    for group in cell_order:
        key = (int(cell_of_group[group, 0]), int(cell_of_group[group, 1]), int(cell_of_group[group, 2]))
        neighbours = [
            accepted_in_cell[neighbour_key]
            for offset in _NEIGHBOUR_OFFSETS
            if (neighbour_key := (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])) in accepted_in_cell
        ]
        candidates = coords[group_start[group] : group_end[group]]
        if neighbours:
            offsets = candidates[:, None, :].astype(np.int64) - np.array(neighbours, dtype=np.int64)[None, :, :]
            # Chebyshev distance to the nearest already-accepted point in a neighbouring cell.
            gaps = np.abs(offsets).max(axis=2).min(axis=1)
            far_enough = np.flatnonzero(gaps >= size)
            if len(far_enough) == 0:
                continue
            # Farthest-point rule: of the admissible candidates, take the one whose nearest
            # accepted neighbour is furthest away (Euclidean). `argmax` returns the first
            # maximum, so the within-cell order above resolves any tie.
            euclidean = np.sqrt((offsets.astype(float) ** 2).sum(axis=2)).min(axis=1)
            pick = candidates[far_enough[np.argmax(euclidean[far_enough])]]
        else:
            pick = candidates[0]
        accepted_in_cell[key] = pick
        chosen.append(pick)

    if not chosen:  # pragma: no cover - unreachable while the candidate set is non-empty
        return np.zeros((0, 3), dtype=np.uint)
    return np.array(chosen, dtype=np.uint)


def peak_local_points_boundary_layer(
    segmented_mask: NDArray[np.uint],
    edt_image: NDArray[np.float64],
    min_distance: int,
    print_info: bool = False,
    delta: float | None = None,
    exclude_offsets_from_surface: bool = False,
    guard_duplicate_faces: bool = True,
    collapse_rule: str = WELD,
) -> tuple[NDArray[np.uint], NDArray[np.uint]] | tuple[NDArray[np.uint], NDArray[np.uint], None, dict]:
    """Boundary-layer point placement: deterministic extrema plus a structured boundary layer.

    Keeps the deterministic interface minima and interior maxima and adds offset
    points `delta` into each material adjacent to a genuine interface (see the module
    docstring for the full construction and its rationale). No random number is involved,
    so the result is bit-identical across runs and platforms, exactly like the default.

    The offset points are stacked **between** the interior maxima and the interface minima
    and are counted as interior (non-interface) points in `indices_of_sorted_maxes`: they
    carry a non-zero EDT value (`~= delta`), so they are not on the interface, and the
    benchmark harness's all-surface-tet classification treats every vertex outside the
    trailing minima block as non-interface. This is what makes the boundary layer break up
    the all-surface tetrahedra rather than being miscounted as more interface points.

    Args:
        segmented_mask (NDArray[np.uint]): Original segmented image. **Required** here
            (unlike the default): the offsets' adjacency is read from it.
        edt_image (NDArray[np.float64]): 3D Euclidean Distance Transform image.
        min_distance (int): Minimum distance between peaks (local min and local max).
        print_info (bool, optional): Print details about the algorithm. Defaults to False.
        delta (float | None, optional): Offset distance into each material. Defaults to
            `min_distance` (the interface spacing `h_S`) when None.
        exclude_offsets_from_surface (bool, optional): Emit the optional
            `point_metadata` element so the extraction keeps the offsets in the tesselation
            and out of the extracted surface. See
            `peak_local_points_junction_protected`. Defaults to False.
        guard_duplicate_faces (bool, optional): Offset-exclusion ablation switch; see
            `peak_local_points_junction_protected`. Defaults to True.
        collapse_rule (str, optional): Offset-exclusion collapse rule; see
            `peak_local_points_junction_protected`. Defaults to `"weld"`.

    Returns:
        tuple: `(points, indices_of_sorted_maxes)`, or
            `(points, indices_of_sorted_maxes, None, point_metadata)` when
            `exclude_offsets_from_surface` is set (the `None` is the weight slot, which this
            scheme does not use). `guard_duplicate_faces` is forwarded into the metadata; see
            `peak_local_points_junction_protected`.
           - an array of 3D pixel coordinates, stacked as
             [8 corners, local maxima, boundary offsets, local minima];
           - the indices, into that array, of the non-interface interior points (maxima
             then offsets) sorted by decreasing EDT value.
    """
    families = boundary_layer_families(segmented_mask, edt_image, min_distance, delta, print_info)
    placed = _assemble_points_boundary_layer(
        edt_image,
        families["corners"],
        families["maxima"],
        families["offsets"],
        families["minima"],
    )
    if not exclude_offsets_from_surface:
        return placed
    metadata = _point_metadata(
        {
            FAMILY_CORNER: len(families["corners"]),
            FAMILY_MAXIMUM: len(families["maxima"]),
            FAMILY_INTERFACE_OFFSET: len(families["offsets"]),
            FAMILY_INTERFACE_MINIMUM: len(families["minima"]),
        },
        {FAMILY_INTERFACE_OFFSET: families["offset_parents"]},
        guard_duplicate_faces=guard_duplicate_faces,
        collapse_rule=collapse_rule,
    )
    return (*placed, None, metadata)


def boundary_layer_families(
    segmented_mask: NDArray[np.uint],
    edt_image: NDArray[np.float64],
    min_distance: int,
    delta: float | None = None,
    print_info: bool = False,
) -> dict:
    """Compute the four point families of the boundary-layer scheme, kept separate.

    The pipeline consumes these through `peak_local_points_boundary_layer`, which folds
    them into the single stacked array the `PointPlacingFunction` contract expects. The
    benchmark harness and the tests call this function directly so they can report the
    interior / interface / boundary-layer counts *separately* (the stacked array conflates
    the maxima and the offsets into one interior block) and verify the offsets' orientation
    against the label image without re-deriving the construction.

    Args:
        segmented_mask (NDArray[np.uint]): Original segmented image (label field).
        edt_image (NDArray[np.float64]): 3D Euclidean Distance Transform image.
        min_distance (int): Minimum distance between peaks.
        delta (float | None, optional): Offset distance; defaults to `min_distance`.
        print_info (bool, optional): Print details. Defaults to False.

    Returns:
        dict: keys `corners`, `maxima`, `minima`, `offsets` (each an (n, 3) uint array in
            voxel coordinates) and `orientation` (a dict of verification counts; see
            `_emit_boundary_layer_offsets`).
    """
    if segmented_mask is None:
        msg = "peak_local_points_boundary_layer needs the segmentation mask to place the boundary layer."
        raise ValueError(msg)

    corners = _give_corners(edt_image)
    if print_info:
        print("Searching local extremas ...")
    maxima = plateau_packing_extrema(edt_image, min_distance, maximise=True)
    minima = plateau_packing_extrema(edt_image, min_distance, maximise=False)
    if print_info:
        print("Number of local maxes :", len(maxima))
        print("Number of local minimas :", len(minima))

    offsets, offset_parents, orientation = _emit_boundary_layer_offsets(
        segmented_mask,
        edt_image,
        minima,
        min_distance,
        delta,
        forbidden=(corners, maxima, minima),
    )
    if print_info:
        print("Number of boundary-layer offsets :", len(offsets))

    return {
        "corners": corners,
        "maxima": maxima,
        "minima": minima,
        "offsets": offsets,
        # Index into `minima` of the interface sample each offset displaces.
        "offset_parents": offset_parents,
        "orientation": orientation,
    }


def peak_local_points_junction_protected(
    segmented_mask: NDArray[np.uint],
    edt_image: NDArray[np.float64],
    min_distance: int,
    print_info: bool = False,
    delta: float | None = None,
    junction_spacing: float | None = None,
    shell_coarsening: int = 1,
    protect_radius: float | None = None,
    junction_delta: float | None = None,
    protect_junctions: bool = True,
    junction_boundary_layer: bool = True,
    exclude_offsets_from_surface: bool = False,
    guard_duplicate_faces: bool = True,
    collapse_rule: str = WELD,
) -> tuple[NDArray[np.uint], NDArray[np.uint], NDArray[np.float64]] | tuple[
    NDArray[np.uint],
    NDArray[np.uint],
    NDArray[np.float64],
    dict,
]:
    """Junction-protected point placement: the boundary layer plus protected 0-/1-junctions.

    See the module docstring for the construction and the reasoning behind every default.
    Returns the **3-tuple** form of the `PointPlacingFunction` contract: the extra element
    is the per-point weight vector (squared protecting-ball radii, non-zero only on the
    protected junction cluster) that `weighted_delaunay_tesselation` consumes.

    Args:
        segmented_mask (NDArray[np.uint]): Original segmented image. Required: both the
            junction strata and the boundary layer's adjacency come from it.
        edt_image (NDArray[np.float64]): 3D Euclidean Distance Transform image.
        min_distance (int): Minimum distance between extrema.
        print_info (bool, optional): Print details about the algorithm. Defaults to False.
        delta (float | None, optional): Boundary-layer offset distance, for both the
            interface and the junction layers. Defaults to `min_distance` (the boundary
            layer's default).
        junction_spacing (float | None, optional): Arclength `h_J` between 1-junction
            samples. Defaults to `2*min_distance + 1`, the *effective* interface spacing.
        shell_coarsening (int, optional): Re-pack the label-multiplicity-1 minima (the
            bounding-box / background shell) at `shell_coarsening * (2*min_distance+1)`.
            `1` is off. Defaults to 1.
        protect_radius (float | None, optional): Protecting-ball radius; see
            `junction_protected_families`. Defaults to `junction_spacing / 2`.
        junction_delta (float | None, optional): Junction boundary-layer offset distance;
            see `junction_protected_families`. Defaults to `protect_radius`.
        protect_junctions (bool, optional): Ablation switch; see
            `junction_protected_families`. Defaults to True.
        junction_boundary_layer (bool, optional): Ablation switch; see
            `junction_protected_families`. Defaults to True.
        exclude_offsets_from_surface (bool, optional): Emit the optional
            `point_metadata` element, which tells the extraction to keep the boundary-layer
            offsets in the tesselation and out of the extracted surface. The points, the
            tesselation and the watershed labelling are bit-identical either way — this only
            changes what the surface extraction is allowed to use as a vertex, which is why
            it is a property of the point-placing scheme (the scheme knows which of its
            points are scaffolding) rather than a new algorithm parameter. Defaults to False.
        guard_duplicate_faces (bool, optional): Offset-exclusion ablation switch. Refuse the
            few merges that would put two triangles on one vertex triple; see
            `dw3d.mesh_utilities.exclude_offsets_from_surface` for the measured trade.
            Defaults to True. Ignored unless `exclude_offsets_from_surface` is set, and
            ignored under the `"link_condition"` collapse rule, which forbids the same thing.
        collapse_rule (str, optional): Which merges the extraction is allowed
            to make: `"weld"` is the unconditional merge and `"link_condition"` is the
            topology-preserving edge contraction that replaces it. See
            `dw3d.mesh_utilities.exclude_offsets_from_surface`. Defaults to `"weld"`, so
            the unconditional-merge records stay reproducible. Ignored unless
            `exclude_offsets_from_surface` is set.

    Returns:
        tuple: `(points, indices_of_sorted_maxes, weights)`, or the 4-tuple
            `(points, indices_of_sorted_maxes, weights, point_metadata)` when
            `exclude_offsets_from_surface` is set.
           - the points, stacked as
             `[8 corners, maxima, interface offsets, junction samples, junction offsets, minima]`;
           - the indices of the non-interface interior points sorted by decreasing EDT value;
           - the per-point weights;
           - the point metadata (`family`, `surface_merge_target`); see `_point_metadata`.
    """
    families = junction_protected_families(
        segmented_mask,
        edt_image,
        min_distance,
        delta=delta,
        junction_spacing=junction_spacing,
        shell_coarsening=shell_coarsening,
        protect_radius=protect_radius,
        junction_delta=junction_delta,
        protect_junctions=protect_junctions,
        junction_boundary_layer=junction_boundary_layer,
        print_info=print_info,
    )
    placed = _assemble_points_junction_protected(edt_image, families)
    if not exclude_offsets_from_surface:
        return placed
    metadata = _point_metadata(
        {
            FAMILY_CORNER: len(families["corners"]),
            FAMILY_MAXIMUM: len(families["maxima"]),
            FAMILY_INTERFACE_OFFSET: len(families["offsets"]),
            FAMILY_JUNCTION_SAMPLE: len(families["junction_points"]),
            FAMILY_JUNCTION_OFFSET: len(families["junction_offsets"]),
            FAMILY_INTERFACE_MINIMUM: len(families["minima"]),
        },
        {
            FAMILY_INTERFACE_OFFSET: families["offset_parents"],
            FAMILY_JUNCTION_OFFSET: families["junction_offset_parents"],
        },
        guard_duplicate_faces=guard_duplicate_faces,
        collapse_rule=collapse_rule,
    )
    return (*placed, metadata)


def junction_protected_families(
    segmented_mask: NDArray[np.uint],
    edt_image: NDArray[np.float64],
    min_distance: int,
    delta: float | None = None,
    junction_spacing: float | None = None,
    shell_coarsening: int = 1,
    protect_radius: float | None = None,
    junction_delta: float | None = None,
    protect_junctions: bool = True,
    junction_boundary_layer: bool = True,
    print_info: bool = False,
) -> dict:
    """Compute the six point families of the junction-protected scheme, kept separate.

    As with `boundary_layer_families`, the pipeline consumes these through
    `peak_local_points_junction_protected`; the benchmark harness and the tests call this
    directly so they can report each family's size, the junction network's sampled pairs
    (needed to *measure* junction preservation) and the shell-coarsening saving, without
    re-deriving the construction.

    Args:
        segmented_mask (NDArray[np.uint]): Original segmented image (label field).
        edt_image (NDArray[np.float64]): 3D Euclidean Distance Transform image.
        min_distance (int): Minimum distance between extrema.
        delta (float | None, optional): Offset distance; defaults to `min_distance`.
        junction_spacing (float | None, optional): `h_J`; defaults to `2*min_distance + 1`.
        shell_coarsening (int, optional): Shell re-packing factor; 1 is off. Defaults to 1.
        protect_radius (float | None, optional): Protecting-ball radius `r_J`. Defaults to
            `junction_spacing / 2`, at which consecutive samples' balls are tangent and so
            cover the sampled curve. `0` disables the exclusion (ablation).
        junction_delta (float | None, optional): Offset distance of the *junction* boundary
            layer. Defaults to `protect_radius`, the smallest value that leaves the
            protecting balls empty — see the module docstring. Not `delta`: a junction
            offset at `delta < r_J` sits inside the ball it is supposed to be protecting.
        protect_junctions (bool, optional): Emit junction samples at all. **Ablation switch,
            not a tuning parameter** — `False` gives the boundary layer plus shell
            coarsening only, which is how the ablation attributes each of the three
            junction-protection components. Defaults to True.
        junction_boundary_layer (bool, optional): Emit the boundary layer around junction
            samples (item 3 of the module docstring's junction-protection section). Ablation
            switch. Defaults to True.
        print_info (bool, optional): Print details. Defaults to False.

    Returns:
        dict: the point families `corners`, `maxima`, `minima`, `offsets`,
            `junction_points`, `junction_offsets`; `junction_pairs` (index pairs into
            `junction_points`); `is_corner_sample`; the protecting radius `protect_radius`;
            and the `orientation`, `junction`, `shell` info dicts.
    """
    if segmented_mask is None:
        msg = "peak_local_points_junction_protected needs the segmentation mask to place the junctions."
        raise ValueError(msg)

    spacing = 2 * min_distance + 1
    if delta is None:
        delta = float(min_distance)
    if junction_spacing is None:
        junction_spacing = float(spacing)
    if protect_radius is None:
        protect_radius = junction_spacing / 2.0
    if junction_delta is None:
        junction_delta = float(protect_radius)

    corners = _give_corners(edt_image)
    if print_info:
        print("Searching local extremas ...")
    maxima = plateau_packing_extrema(edt_image, min_distance, maximise=True)
    minima = plateau_packing_extrema(edt_image, min_distance, maximise=False)

    minima, shell_info = _coarsen_shell_minima(segmented_mask, minima, spacing, shell_coarsening)

    network = (
        sample_junction_network(segmented_mask, junction_spacing)
        if protect_junctions
        else {
            "points": np.zeros((0, 3), dtype=np.uint),
            "is_corner_sample": np.zeros(0, dtype=bool),
            "pairs": np.zeros((0, 2), dtype=np.int64),
            "strata": {},
            "n_skeleton_voxels": 0,
        }
    )
    junction_points = np.asarray(network["points"], dtype=np.int64)
    junction_info = {
        "n_junction_samples": len(junction_points),
        "n_corner_samples": int(np.count_nonzero(network["is_corner_sample"])),
        "n_sampled_pairs": len(network["pairs"]),
        "n_skeleton_voxels": network["n_skeleton_voxels"],
        "junction_spacing": float(junction_spacing),
        "protect_radius": float(protect_radius),
        "junction_delta": float(junction_delta),
        **network["strata"],
    }

    # The protecting balls clear the unstructured samples out of the junction neighbourhood.
    minima, n_minima_excluded = _outside_protecting_balls(minima, junction_points, protect_radius)
    junction_info["n_minima_excluded_by_protection"] = n_minima_excluded

    offsets, offset_parents, orientation = _emit_boundary_layer_offsets(
        segmented_mask,
        edt_image,
        minima,
        min_distance,
        delta,
        forbidden=(corners, maxima, minima, junction_points),
    )
    survives = _outside_protecting_balls_mask(offsets, junction_points, protect_radius)
    offsets, offset_parents = offsets[survives], offset_parents[survives]
    orientation["n_offsets_excluded_by_protection"] = int((~survives).sum())
    orientation["n_offsets_final"] = len(offsets)

    junction_offsets, junction_offset_parents, junction_layer_info = (
        _emit_junction_boundary_layer(
            segmented_mask,
            junction_points,
            junction_delta,
            forbidden=(corners, maxima, minima, offsets, junction_points),
        )
        if junction_boundary_layer
        else (np.zeros((0, 3), dtype=np.uint), np.zeros(0, dtype=np.int64), {"n_junction_offsets_final": 0})
    )
    junction_info.update(junction_layer_info)

    if print_info:
        print("Number of local maxes :", len(maxima))
        print("Number of local minimas :", len(minima))
        print("Number of boundary-layer offsets :", len(offsets))
        print("Number of junction samples :", len(junction_points))
        print("Number of junction boundary-layer offsets :", len(junction_offsets))

    return {
        "corners": corners,
        "maxima": maxima,
        "minima": minima,
        "offsets": offsets,
        # Offset parents: index into `minima` for an interface offset, into
        # `junction_points` for a junction offset.
        "offset_parents": offset_parents,
        "junction_points": junction_points.astype(np.uint),
        "junction_offsets": junction_offsets,
        "junction_offset_parents": junction_offset_parents,
        "junction_pairs": network["pairs"],
        "is_corner_sample": network["is_corner_sample"],
        "protect_radius": float(protect_radius),
        "orientation": orientation,
        "junction": junction_info,
        "shell": shell_info,
    }


def _coarsen_shell_minima(
    segmented_mask: NDArray[np.uint],
    minima: NDArray[np.uint],
    spacing: int,
    shell_coarsening: int,
) -> tuple[NDArray[np.uint], dict]:
    """Split the minima into genuine-interface and shell minima, and re-pack the latter coarsely.

    A minimum is "on an interface" when the largest label multiplicity `|L|` over the 2x2x2
    blocks containing its voxel is at least 2, i.e. at least two materials meet there. A
    multiplicity of 1 means the voxel is in the interior of a single material and is only a
    minimum of the EDT because `_pad_mask` zeroed the bounding-box shell — see the module
    docstring. Those are re-packed at `shell_coarsening * spacing` by the same greedy
    lattice-cell rule `plateau_packing_extrema` uses, in the minima's existing acceptance
    order, so the result is deterministic and the genuine interface samples are untouched.

    Returns the surviving minima (interface minima first, then the kept shell minima) and a
    report of what was dropped.
    """
    minima = np.asarray(minima, dtype=np.int64)
    info = {
        "n_minima_before": len(minima),
        "n_interface_minima": 0,
        "n_shell_minima": 0,
        "n_shell_minima_kept": 0,
        "n_minima_after": len(minima),
        "shell_coarsening": int(shell_coarsening),
    }
    if len(minima) == 0:
        return minima.astype(np.uint), info

    multiplicity = label_multiplicity(segmented_mask)
    on_interface = _max_multiplicity_at(multiplicity, minima) >= 2
    info["n_interface_minima"] = int(on_interface.sum())
    info["n_shell_minima"] = int((~on_interface).sum())

    interface_minima = minima[on_interface]
    shell_minima = minima[~on_interface]
    if shell_coarsening > 1 and len(shell_minima):
        shell_minima = shell_minima[_greedy_separated(shell_minima, shell_coarsening * spacing)]
    info["n_shell_minima_kept"] = len(shell_minima)

    kept = np.vstack((interface_minima, shell_minima)) if len(shell_minima) else interface_minima
    info["n_minima_after"] = len(kept)
    return kept.astype(np.uint), info


def _max_multiplicity_at(multiplicity: NDArray[np.uint8], voxels: NDArray[np.int64]) -> NDArray[np.uint8]:
    """Largest label multiplicity over the (up to 8) dual-lattice blocks containing each voxel.

    Dual index `(i, j, k)` covers voxels `i..i+1` per axis, so voxel `v` belongs to the dual
    blocks with index `v - 1` or `v` per axis, clipped to the dual lattice's extent.
    """
    upper = np.asarray(multiplicity.shape, dtype=np.int64) - 1
    best = np.zeros(len(voxels), dtype=np.uint8)
    for di in (-1, 0):
        for dj in (-1, 0):
            for dk in (-1, 0):
                index = np.clip(voxels + np.array([di, dj, dk]), 0, upper)
                np.maximum(best, multiplicity[index[:, 0], index[:, 1], index[:, 2]], out=best)
    return best


def _greedy_separated(points: NDArray[np.int64], separation: float) -> NDArray[np.int64]:
    """Indices of a greedy maximal `separation`-separated subset of `points`, in input order.

    Same rule as `plateau_packing_extrema`'s packing step, reduced to its essentials: visit
    the points in their existing (already deterministic) order and keep one when no kept
    point lies within `separation`. The lattice-cell hash bounds the neighbour test to 27
    cells, so this is O(n).
    """
    cell_size = max(float(separation), 1.0)
    accepted_in_cell: dict[tuple[int, int, int], list[NDArray[np.int64]]] = {}
    keep: list[int] = []
    separation_sq = float(separation) ** 2
    for index, point in enumerate(points):
        key = (int(point[0] // cell_size), int(point[1] // cell_size), int(point[2] // cell_size))
        too_close = False
        for offset in ((0, 0, 0), *_NEIGHBOUR_OFFSETS):
            neighbours = accepted_in_cell.get((key[0] + offset[0], key[1] + offset[1], key[2] + offset[2]))
            if neighbours and any(float(((point - other) ** 2).sum()) < separation_sq for other in neighbours):
                too_close = True
                break
        if not too_close:
            accepted_in_cell.setdefault(key, []).append(point)
            keep.append(index)
    return np.array(keep, dtype=np.int64)


def _outside_protecting_balls(
    points: NDArray[np.uint],
    junction_points: NDArray[np.int64],
    radius: float,
) -> tuple[NDArray[np.uint], int]:
    """Drop the points lying strictly inside a junction sample's protecting ball.

    Returns the survivors (order preserved) and how many were dropped. A point exactly on a
    ball's surface is kept: the hiding test in a regular triangulation is `|pq|**2 <= w_p`,
    but the boundary case is where Qhull's own degeneracy handling decides, so `dw3d` keeps
    the strictly-outside set and lets nothing depend on the tie.
    """
    points = np.asarray(points, dtype=np.int64)
    if len(points) == 0 or len(junction_points) == 0 or radius <= 0:
        return points.astype(np.uint), 0
    keep = _outside_protecting_balls_mask(points, junction_points, radius)
    return points[keep].astype(np.uint), int((~keep).sum())


def _outside_protecting_balls_mask(
    points: NDArray[np.int64],
    junction_points: NDArray[np.int64],
    radius: float,
) -> NDArray[np.bool_]:
    """The boolean survivor mask behind `_outside_protecting_balls`.

    Split out for the offset-exclusion work so the same filter can be applied to a parallel
    per-offset array (the parent index) without recomputing the query or duplicating the
    tie convention.
    """
    points = np.asarray(points, dtype=np.int64)
    if len(points) == 0 or len(junction_points) == 0 or radius <= 0:
        return np.ones(len(points), dtype=bool)
    distance, _ = cKDTree(np.asarray(junction_points, dtype=np.float64)).query(points.astype(np.float64))
    return distance >= radius


def _emit_junction_boundary_layer(
    segmented_mask: NDArray[np.uint],
    junction_points: NDArray[np.int64],
    delta: float,
    forbidden: tuple[NDArray[np.uint], ...],
) -> tuple[NDArray[np.uint], NDArray[np.int64], dict]:
    """Emit one boundary-layer offset per material adjacent to each junction sample (the junction boundary layer).

    For a junction sample `p`, the adjacent materials are the distinct labels of the mask in
    a radius-`ceil(delta)+1` ball around `p`. For each such material `m`, the direction is
    the mean displacement of the ball's `m`-labelled voxels from `p`, normalised, and the
    offset is `rint(p + delta * direction)`. The offset is kept only if it is in the domain
    **and the mask's label there really is `m`** — the same label-image verification the
    boundary layer applies to its interface offsets, so a direction spoiled by a concave
    material never
    produces a point in the wrong cell.

    Returns the deduplicated offsets, the index into `junction_points` of the sample each
    surviving offset displaces (its *parent*, on the 1-stratum rather than on an
    interface), and a report:
        n_junction_adjacent_materials — total (sample, material) pairs considered;
        n_junction_offsets_emitted    — offsets passing the label check, before dedup;
        n_junction_offsets_final      — after dedup against self and `forbidden`.
    """
    info = {
        "n_junction_adjacent_materials": 0,
        "n_junction_offsets_emitted": 0,
        "n_junction_offsets_final": 0,
    }
    empty = np.zeros((0, 3), dtype=np.uint)
    no_parents = np.zeros(0, dtype=np.int64)
    junction_points = np.asarray(junction_points, dtype=np.int64)
    if len(junction_points) == 0:
        return empty, no_parents, info

    shape = np.asarray(segmented_mask.shape, dtype=np.int64)
    radius = int(np.ceil(delta)) + 1
    stencil = np.array(
        [
            (i, j, k)
            for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1)
            for k in range(-radius, radius + 1)
            if i * i + j * j + k * k <= radius * radius
        ],
        dtype=np.int64,
    )

    neighbourhood = junction_points[:, None, :] + stencil[None, :, :]  # (n, s, 3)
    inside = np.all((neighbourhood >= 0) & (neighbourhood < shape), axis=2)
    clipped = np.clip(neighbourhood, 0, shape - 1)
    window_labels = segmented_mask[clipped[..., 0], clipped[..., 1], clipped[..., 2]]

    candidates: list[NDArray[np.int64]] = []
    candidate_parents: list[int] = []
    for sample_index, adjacent in enumerate(labels_around(segmented_mask, junction_points, radius=1)):
        info["n_junction_adjacent_materials"] += len(adjacent)
        for material in adjacent:
            selected = inside[sample_index] & (window_labels[sample_index] == material)
            if not selected.any():
                continue
            direction = stencil[selected].astype(np.float64).mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            offset = np.rint(junction_points[sample_index] + delta * direction / norm).astype(np.int64)
            if np.any(offset < 0) or np.any(offset >= shape):
                continue
            if int(segmented_mask[offset[0], offset[1], offset[2]]) != material:
                continue
            candidates.append(offset)
            candidate_parents.append(sample_index)

    info["n_junction_offsets_emitted"] = len(candidates)
    if not candidates:
        return empty, no_parents, info
    stacked = np.array(candidates, dtype=np.int64)
    kept = _dedup_against_index(stacked, forbidden)
    info["n_junction_offsets_final"] = len(kept)
    parents = np.asarray(candidate_parents, dtype=np.int64)[kept]
    return stacked[kept].astype(np.uint), parents, info


def _assemble_points_junction_protected(
    edt_image: NDArray[np.float64],
    families: dict,
) -> tuple[NDArray[np.uint], NDArray[np.uint], NDArray[np.float64]]:
    """Stack the six junction-protected families, index the interior block, and build the weight vector.

    Stacking is `[corners, maxima, interface offsets, junction samples, junction offsets,
    minima]`. The trailing block is the interface minima, exactly as in the deterministic
    and boundary-layer schemes, so the
    harness's "interface vertex" classification is unchanged. **The junction samples are
    counted as interior**, not interface: a sample sits on a 1- or 0-stratum, which is not
    the 2-stratum the all-surface-tet metric is about, and a tetrahedron whose four vertices
    are junction samples is not the degenerate near-interface sliver that metric detects.
    The junction offsets are interior for the same reason the boundary layer's are.

    The weights are `protect_radius**2` on the junction samples *and* their offsets — one
    protected cluster of equal weights, which is what stops the cluster hiding itself — and
    zero everywhere else.
    """
    corners = families["corners"]
    interior_parts = [
        np.asarray(families["maxima"], dtype=np.uint),
        np.asarray(families["offsets"], dtype=np.uint),
        np.asarray(families["junction_points"], dtype=np.uint),
        np.asarray(families["junction_offsets"], dtype=np.uint),
    ]
    non_empty = [part for part in interior_parts if len(part)]
    interior = np.vstack(non_empty) if non_empty else np.zeros((0, 3), dtype=np.uint)
    minima = np.asarray(families["minima"], dtype=np.uint)
    all_points = np.vstack((corners, interior, minima), dtype=np.uint)

    edt_at_interior = edt_image[interior[:, 0], interior[:, 1], interior[:, 2]] if len(interior) else np.zeros(0)
    indices_of_sorted_interior = (np.argsort(-edt_at_interior, kind="stable") + len(corners)).astype(np.uint)

    weights = np.zeros(len(all_points), dtype=np.float64)
    n_before_junctions = len(corners) + len(interior_parts[0]) + len(interior_parts[1])
    n_protected = len(interior_parts[2]) + len(interior_parts[3])
    weights[n_before_junctions : n_before_junctions + n_protected] = families["protect_radius"] ** 2

    return all_points, indices_of_sorted_interior, weights


def _point_metadata(
    block_sizes: dict[int, int],
    parents: dict[int, NDArray[np.int64]],
    guard_duplicate_faces: bool = True,
    collapse_rule: str = WELD,
) -> dict:
    """Build the offset-exclusion `point_metadata` from the stacking layout.

    Args:
        block_sizes: family code -> number of points in that block, **in stacking order**
            (dicts preserve insertion order, and that order is the layout).
        guard_duplicate_faces: forwarded into the metadata for
            `dw3d.mesh_utilities.exclude_offsets_from_surface`; see there for the trade.
        collapse_rule: forwarded to the same function; `"weld"` is the unconditional-merge
            rule and `"link_condition"` is the topology-preserving one.
        parents: family code -> per-point parent index *within its own parent family's
            block*, for the offset families only. The parent family is the interface minima
            for `FAMILY_INTERFACE_OFFSET` and the junction samples for
            `FAMILY_JUNCTION_OFFSET`, matching what `junction_protected_families` returns.

    Returns:
        dict with two `(n_points,)` arrays:
            `family` — the family code of each tesselation point;
            `surface_merge_target` — for each point, the index of the tesselation point the
                extracted surface should use in its place. Identity everywhere except on the
                offset families, where it is the offset's parent: the interface sample the
                offset is a `delta`-displacement of. The offset-exclusion work's whole
                construction is this one array; see
                `dw3d.mesh_utilities.exclude_offsets_from_surface`.
        and the scalar `guard_duplicate_faces`, forwarded to that function.
    """
    total = sum(block_sizes.values())
    family = np.empty(total, dtype=np.uint8)
    start: dict[int, int] = {}
    at = 0
    for code, size in block_sizes.items():
        family[at : at + size] = code
        start[code] = at
        at += size

    target = np.arange(total, dtype=np.int64)
    parent_family = {
        FAMILY_INTERFACE_OFFSET: FAMILY_INTERFACE_MINIMUM,
        FAMILY_JUNCTION_OFFSET: FAMILY_JUNCTION_SAMPLE,
    }
    for code, within_block in parents.items():
        size = block_sizes.get(code, 0)
        if size == 0:
            continue
        if len(within_block) != size:
            message = (
                f"family {FAMILY_NAMES[code]} has {size} points but {len(within_block)} parents; "
                "the parent bookkeeping and the stacking have drifted apart"
            )
            raise ValueError(message)
        target[start[code] : start[code] + size] = start[parent_family[code]] + np.asarray(
            within_block,
            dtype=np.int64,
        )
    return {
        "family": family,
        "surface_merge_target": target,
        "guard_duplicate_faces": guard_duplicate_faces,
        "collapse_rule": collapse_rule,
    }


def _emit_boundary_layer_offsets(
    segmented_mask: NDArray[np.uint],
    edt_image: NDArray[np.float64],
    minima: NDArray[np.uint],
    min_distance: int,
    delta: float | None,
    forbidden: tuple[NDArray[np.uint], ...],
) -> tuple[NDArray[np.uint], NDArray[np.int64], dict]:
    """Emit `p +- delta*n` offset points for each interface minimum straddling two labels.

    The normal `n` at `p` is the eigenvector of the largest-magnitude eigenvalue of the
    EDT Hessian (the across-sheet direction; see the module docstring). Both offsets are
    probed; a pair is kept only if both land in the domain and carry *different* labels in
    `segmented_mask` — i.e. it genuinely straddles an interface. Kept offsets are
    deduplicated against one another and against `forbidden` (the corner/maximum/minimum
    points) so no tesselation vertex is repeated.

    Also returns, for each surviving offset, the index **into `minima`** of the interface
    sample that emitted it (its *parent*). Added for the offset-exclusion work: the parent
    is the exact point on the interface that the offset is a `delta`-displacement of, and it
    is the target the offset-exclusion merge uses when excluding it from the extracted
    surface. Recording it here
    is what makes that exact rather than a re-derivation — the across-sheet normal cannot be
    recovered reliably at the offset itself, `delta` voxels inside a cell, where the EDT is
    smooth and its Hessian no longer sees the sheet.

    The returned `orientation` dict records, without assuming success:
        n_minima            — interface samples considered;
        n_on_interface      — samples on a genuine >=2-label boundary (straddling pairs);
        n_offsets_emitted   — 2 * n_on_interface, before dedup;
        n_offsets_final     — after dedup vs. self and `forbidden`;
        n_orientation_checked / n_orientation_ok — an independent re-verification that each
            straddling pair's two labels are distinct and both appear in the mask's
            neighbourhood of the sample (a check of the label image, not of our own maths).
    """
    if delta is None:
        delta = float(min_distance)
    shape = np.asarray(edt_image.shape)
    empty = np.zeros((0, 3), dtype=np.uint)
    base_info = {
        "n_minima": len(minima),
        "n_on_interface": 0,
        "n_offsets_emitted": 0,
        "n_offsets_final": 0,
        "n_orientation_checked": 0,
        "n_orientation_ok": 0,
    }
    no_parents = np.zeros(0, dtype=np.int64)
    if len(minima) == 0:
        return empty, no_parents, base_info

    normals = _edt_hessian_normals(edt_image, minima)
    minima_f = minima.astype(np.float64)
    plus = np.rint(minima_f + delta * normals).astype(np.int64)
    minus = np.rint(minima_f - delta * normals).astype(np.int64)

    in_plus = np.all((plus >= 0) & (plus < shape), axis=1)
    in_minus = np.all((minus >= 0) & (minus < shape), axis=1)
    labels_plus = segmented_mask[tuple(np.clip(plus, 0, shape - 1).T)]
    labels_minus = segmented_mask[tuple(np.clip(minus, 0, shape - 1).T)]

    straddle = in_plus & in_minus & (labels_plus != labels_minus)
    n_straddle = int(straddle.sum())  # python int so the info dict is JSON-serialisable
    base_info["n_on_interface"] = n_straddle
    base_info["n_offsets_emitted"] = 2 * n_straddle
    if not straddle.any():
        return empty, no_parents, base_info

    # Independent orientation check against the label image: for every straddling sample,
    # confirm the two offset labels are distinct and both present in a 3x3x3 mask window of
    # the sample. This verifies the layer rather than trusting the normal.
    base_info["n_orientation_checked"], base_info["n_orientation_ok"] = _verify_offset_orientation(
        segmented_mask,
        minima[straddle],
        labels_plus[straddle],
        labels_minus[straddle],
        shape,
    )

    # `plus` and `minus` are stacked in that order, so the parent of row i of the stack is
    # the straddling sample at i % n_straddle -- i.e. both offsets of one sample share a
    # parent, which is exactly what `exclude_offsets_from_surface` needs (each is a
    # delta-displacement of that sample).
    straddle_index = np.nonzero(straddle)[0]
    offsets = np.vstack((plus[straddle], minus[straddle])).astype(np.int64)
    parents = np.concatenate((straddle_index, straddle_index))
    kept = _dedup_against_index(offsets, forbidden)
    offsets, parents = offsets[kept], parents[kept]
    base_info["n_offsets_final"] = len(offsets)
    return offsets.astype(np.uint), parents.astype(np.int64), base_info


def _edt_hessian_normals(edt_image: NDArray[np.float64], coords: NDArray[np.uint]) -> NDArray[np.float64]:
    """Unit across-sheet normals at `coords`, from the EDT Hessian's dominant eigenvector.

    The Hessian is estimated by second-order central finite differences on the EDT array,
    with the stencil clipped to `[1, n-2]` per axis so it never reads out of bounds. The
    normal is the eigenvector of the eigenvalue of largest magnitude — the direction of
    strongest curvature, which is across the interface sheet (a valley in the EDT). The
    eigenvector's sign is arbitrary and irrelevant: the caller probes both `+n` and `-n`.
    """
    shape = np.asarray(edt_image.shape)
    c = np.clip(coords.astype(np.int64), 1, shape - 2)
    i, j, k = c[:, 0], c[:, 1], c[:, 2]

    def at(di: int, dj: int, dk: int) -> NDArray[np.float64]:
        return edt_image[i + di, j + dj, k + dk]

    e0 = at(0, 0, 0)
    hxx = at(1, 0, 0) - 2 * e0 + at(-1, 0, 0)
    hyy = at(0, 1, 0) - 2 * e0 + at(0, -1, 0)
    hzz = at(0, 0, 1) - 2 * e0 + at(0, 0, -1)
    hxy = (at(1, 1, 0) - at(1, -1, 0) - at(-1, 1, 0) + at(-1, -1, 0)) / 4
    hxz = (at(1, 0, 1) - at(1, 0, -1) - at(-1, 0, 1) + at(-1, 0, -1)) / 4
    hyz = (at(0, 1, 1) - at(0, 1, -1) - at(0, -1, 1) + at(0, -1, -1)) / 4

    hessian = np.empty((len(coords), 3, 3), dtype=np.float64)
    hessian[:, 0, 0], hessian[:, 1, 1], hessian[:, 2, 2] = hxx, hyy, hzz
    hessian[:, 0, 1] = hessian[:, 1, 0] = hxy
    hessian[:, 0, 2] = hessian[:, 2, 0] = hxz
    hessian[:, 1, 2] = hessian[:, 2, 1] = hyz

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    dominant = np.argmax(np.abs(eigenvalues), axis=1)
    normals = eigenvectors[np.arange(len(coords)), :, dominant]
    # eigh returns unit eigenvectors; a degenerate/zero Hessian would give a valid unit
    # vector too, and a wrong direction there simply fails the label straddle test and is
    # discarded, so no normalisation or fallback is needed.
    return normals


def _verify_offset_orientation(
    segmented_mask: NDArray[np.uint],
    samples: NDArray[np.int64],
    labels_plus: NDArray[np.uint],
    labels_minus: NDArray[np.uint],
    shape: NDArray[np.int64],
) -> tuple[int, int]:
    """Independently confirm each straddling pair against the label image.

    A pair is "ok" when its two offset labels differ (already ensured) *and* both appear in
    the 3x3x3 neighbourhood of the interface sample in `segmented_mask` — i.e. the offsets
    point into materials that are actually adjacent at the sample, not across a distant
    third region. Returns `(n_checked, n_ok)`.
    """
    samples = samples.astype(np.int64)
    shape = np.asarray(shape, dtype=np.int64)
    n_ok = 0
    for sample, lp, lm in zip(samples, labels_plus, labels_minus, strict=True):
        lo = np.maximum(sample - 1, 0)
        hi = np.minimum(sample + 2, shape)
        window = segmented_mask[int(lo[0]) : int(hi[0]), int(lo[1]) : int(hi[1]), int(lo[2]) : int(hi[2])]
        local = set(np.unique(window).tolist())
        if lp != lm and int(lp) in local and int(lm) in local:
            n_ok += 1
    return len(samples), n_ok


def _dedup_against(points: NDArray[np.int64], forbidden: tuple[NDArray[np.uint], ...]) -> NDArray[np.int64]:
    """Unique rows of `points`, minus any row already present in a `forbidden` array."""
    return points[_dedup_against_index(points, forbidden)]


def _dedup_against_index(points: NDArray[np.int64], forbidden: tuple[NDArray[np.uint], ...]) -> NDArray[np.int64]:
    """Indices *into* `points` of the rows `_dedup_against` keeps, in its output order.

    Split out for the offset-exclusion work, which needs to carry a per-offset parent index through the same
    de-duplication the coordinates go through. `points[_dedup_against_index(...)]` is
    `_dedup_against(...)` by construction, and a test pins that identity, so the two cannot
    drift apart. `np.unique(..., return_index=True)` reports the *first* occurrence of each
    duplicated row, so where two interface samples emit the same offset voxel the surviving
    parent is the lexicographically-first one — deterministic, and independent of the input
    order only in the sense that the coordinate array is what is sorted.
    """
    if not len(points):
        return np.zeros(0, dtype=np.int64)
    _, first = np.unique(points, axis=0, return_index=True)
    # np.unique sorts, so `first` is already ordered by the row it points at, which is the
    # order `_dedup_against`'s output is in.
    forbidden_rows = {tuple(row) for arr in forbidden for row in np.asarray(arr, dtype=np.int64).tolist()}
    if not forbidden_rows:
        return first
    keep = np.array([tuple(row) not in forbidden_rows for row in points[first].tolist()], dtype=bool)
    return first[keep]


def _assemble_points_boundary_layer(
    edt_image: NDArray[np.float64],
    corners: NDArray[np.uint],
    local_maxes: NDArray[np.uint],
    offsets: NDArray[np.uint],
    local_mins: NDArray[np.uint],
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Stack `[corners, maxima, offsets, minima]` and index the non-interface interior block.

    Offsets are stacked immediately after the maxima and before the minima, so the
    interface (minima) remain the trailing block the harness classifies as "interface".
    `indices_of_sorted_maxes` covers the whole interior block (maxima then offsets) sorted
    by decreasing EDT value, which is what marks the offsets as non-interface for the
    all-surface-tet metric. Nothing downstream consumes the order (see `_assemble_points`).
    """
    interior = np.vstack((local_maxes, offsets), dtype=np.uint) if len(offsets) else local_maxes.astype(np.uint)
    all_points = np.vstack((corners, interior, local_mins), dtype=np.uint)
    edt_at_interior = edt_image[interior[:, 0], interior[:, 1], interior[:, 2]] if len(interior) else np.zeros(0)
    indices_of_sorted_interior = np.argsort(-edt_at_interior, kind="stable") + len(corners)
    return all_points, indices_of_sorted_interior.astype(np.uint)


def _dithered(edt_image: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """The historical `U(0, 1e-5)` dither, drawn from a private RNG rather than the global one."""
    # Legacy RandomState is version-stable by numpy policy; see the docstrings above.
    rng = np.random.RandomState(seed)
    return edt_image + rng.rand(*edt_image.shape) * 1e-5


def _assemble_points(
    edt_image: NDArray[np.float64],
    corners: NDArray[np.uint],
    local_maxes: NDArray[np.uint],
    local_mins: NDArray[np.uint],
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """Stack the three point families and index the maxima by decreasing EDT value.

    The stacking order `[8 corners, maxima, minima]` is an invariant that
    `benchmarks/metrics.py` relies on to classify tesselation vertices, and that
    `indices_of_sorted_max_points` offsets into.

    The sort is `argsort(-values, kind="stable")` rather than the historical
    `argsort(values)[::-1]`. Both are descending; the historical form broke ties in
    *reverse* raster order and, being an unstable sort, broke them differently on
    different numpy builds. Many maxima share an EDT value exactly, so this matters in
    principle — but nothing has consumed the *order* since `_improve_tesselation` was
    deleted as dead code, so the change is output-neutral today.
    """
    all_points = np.vstack((corners, local_maxes, local_mins), dtype=np.uint)
    edt_at_maxes = edt_image[local_maxes[:, 0], local_maxes[:, 1], local_maxes[:, 2]]
    indices_of_sorted_max_points = np.argsort(-edt_at_maxes, kind="stable") + len(corners)
    return all_points, indices_of_sorted_max_points.astype(np.uint)


def _give_corners(img: NDArray) -> NDArray[np.uint]:
    """Give the eight corners pixels coordinates of a 3D image."""
    corners = np.zeros((8, 3), dtype=np.uint)
    index = 0
    a, b, c = img.shape
    for i in [0, a - 1]:
        for j in [0, b - 1]:
            for k in [0, c - 1]:
                corners[index] = np.array([i, j, k])
                index += 1
    return corners
