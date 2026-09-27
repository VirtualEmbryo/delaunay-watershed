r"""Multi-material junction (0-/1-strata) detection, thinning and sampling from a label image.

Junction protection, following Boltcheva, Yvinec & Boissonnat, *Mesh generation from 3D
multi-material images*, MICCAI 2009.

Why this module exists
----------------------
`dw3d` samples the interface (the 2-strata) densely and never samples the **junctions** at
all. Boltcheva's observation is that 2-junctions need no special treatment but 0- and
1-junctions do: a quadruple point that is not a mesh *vertex* and a triple line that is not
a chain of mesh *edges* have to be approximated by whatever the triangulation happens to
produce nearby, which in `dw3d` shows up as edges shared by four or more triangles
(a "quadjunction edge" — four materials meeting along a line, which cannot happen in a dry
foam and is therefore a meshing artifact).

The measured motivation, summed over the 51 benchmark cases: valence->=4 edges went
61 (v0.3) -> 133 (after the determinism fix) -> 520 (after the boundary-layer work), and
`_find_abnormal_non_manifold_edges` survivors 3 -> 5 -> 327. Both the determinism fix and the
boundary-layer work changed the *interface* sampling and both made the quadruple-point
neighbourhood worse, because nothing in the pipeline protects the 0-strata. This module
supplies the missing half.

The strata, and how they are detected
-------------------------------------
For each position of the **dual lattice** — the corners between voxels, i.e. each 2x2x2
block of the label image — let `|L|` be the number of distinct labels in that block:

| `\|L\|` | stratum | meaning |
|---|---|---|
| 1 | 3-stratum | interior of a material |
| 2 | 2-stratum | an interface between two materials |
| 3 | 1-stratum | a triple line |
| >= 4 | 0-stratum | a quadruple (or higher) point |

`label_multiplicity` computes `|L|` for the whole dual lattice in one vectorised pass. It
uses eight *views* of the label array (no copies) and 28 pairwise `!=` comparisons rather
than a sort, so its peak extra memory is two boolean arrays of the dual-lattice size and
its cost is a small multiple of one pass over the image. This is the "negligible-cost"
detection Boltcheva report, and it is the reason junction protection fits within the 20 %
wall-time budget set for it: **if this step is expensive, it is implemented wrong.**

An open question was whether `dw3d.edt.get_total_boundaries` already yields this
multiplicity so it could be reused. **It does not** — see `label_multiplicity`'s docstring
for the measurement. It is a per-label sum of `find_boundaries` masks, then inverted by
`max - x`, so it is correlated with `|L|` but not equal to it, it is dilated by one voxel in
every direction (`find_boundaries(mode="thick")` marks both sides), and it costs one
`find_boundaries` pass *per label*. `label_multiplicity` is both exact and cheaper.

Coordinates and the half-voxel convention
-----------------------------------------
Dual-lattice position `(i, j, k)` sits at voxel coordinate `(i + 0.5, j + 0.5, k + 0.5)`.
`dw3d`'s point-placement contract is integer voxel coordinates (`NDArray[np.uint]`), so
junction samples are emitted at `(i, j, k)`. That is a **uniform translation of the whole
detected junction network by `-(0.5, 0.5, 0.5)` voxels**, not a per-sample error: it leaves
every junction *length* and every junction *angle* exactly unchanged (a rigid translation),
and only shifts the junction network by half a voxel relative to the interface samples —
below the O(h) lattice quantisation that every other `dw3d` vertex already carries (they all
sit on integer voxel coordinates). Recorded here rather than corrected, because correcting it
would mean moving `dw3d` to non-integer seed coordinates, which is outside this module's scope.

Thinning and sampling
---------------------
`sample_junction_network` thins the union of the 0- and 1-strata (`|L| >= 3`) to a curve
skeleton with `skimage.morphology.skeletonize(method="lee")` — already a dependency, so no
new one is added — then:

* every connected component of the 0-stratum contributes **exactly one** sample: the
  *skeleton* voxel nearest that component's centroid. Snapping to the skeleton matters —
  thinning does not preserve the 0-stratum voxels themselves (measured on `3.tif`: only 4
  of the 13 `|L| >= 4` voxels survive it) but it does produce a degree-4 node where four
  materials meet, and putting the sample there is what makes the quadruple point a node of
  the sampled network rather than a point floating beside it;
* the skeleton is split at its degree-`!= 2` voxels into simple paths, and each path is
  sampled at approximately uniform arclength `spacing`, always including both endpoints and
  any 0-junction sample lying on it. Arclength uses the 26-connected step weights `1`,
  `sqrt(2)`, `sqrt(3)`.

Consecutive samples along a path are recorded as `line_pairs`. Those pairs are what
"junction preservation" is measured on downstream: Boltcheva's protecting-ball construction
*guarantees* that consecutive samples of a protected 1-feature are connected in the weighted
Delaunay triangulation, but their guarantee is stated for Delaunay **refinement** and `dw3d`
is one-shot, so we get a strong bias and no guarantee. The fraction of `line_pairs` that
actually appear as mesh edges is therefore measured, not assumed
(`benchmarks.metrics.junction_preservation_stats`).

Everything here is deterministic: no random number, no dependence on the global RNG, and
every ordering is fixed by raster order or by an explicit `lexsort`.

Herve Turlier's group, 2026.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# 26-connectivity neighbour offsets, in raster order, with their Euclidean step lengths.
_NEIGHBOURS_26 = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
    dtype=np.int64,
)
_STEP_LENGTHS = np.sqrt((_NEIGHBOURS_26**2).sum(axis=1))

# The eight corners of a 2x2x2 block, in raster order.
_CUBE_CORNERS = tuple((i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1))


def label_multiplicity(segmented_mask: NDArray[np.uint]) -> NDArray[np.uint8]:
    """Number of distinct labels in each 2x2x2 block of `segmented_mask`, in one vectorised pass.

    This is the label multiplicity `|L|` of the junction-detection step, evaluated on the dual
    lattice: entry `(i, j, k)` of the result counts the distinct labels among
    `segmented_mask[i:i+2, j:j+2, k:k+2]`, and corresponds to the voxel-frame position
    `(i + 0.5, j + 0.5, k + 0.5)`.

    Implementation note (this is the whole cost argument for junction protection). The eight corner
    arrays are *views*, so no copy of the label image is made. Distinctness is then counted
    by the 28 pairwise `!=` comparisons of the upper triangle — a corner is "new" when it
    differs from every earlier corner — which needs two boolean temporaries of the
    dual-lattice size. Sorting the eight values per position instead would need a
    `(8, n)` integer array, i.e. ~32x more memory on an int32 label image, for no gain.

    **Open question, resolved: `dw3d.edt.get_total_boundaries` is not a substitute.**
    It computes `max(S) - S` where `S` is the *sum* over labels of
    `find_boundaries(label_mask, mode="thick")`. Three differences, all disqualifying:
    it is defined on the primal voxel lattice and `mode="thick"` marks a voxel whenever any
    26-neighbour differs, so its support is dilated by one voxel relative to `|L|`; the sum
    counts *label masks touching the voxel's neighbourhood*, which coincides with `|L|` only
    where the neighbourhood is exactly a clean junction and not, for example, where one label
    appears twice in disconnected pieces; and it runs one `find_boundaries` pass per label, so
    it costs O(n_labels) passes over the volume against this function's O(1).

    Args:
        segmented_mask (NDArray[np.uint]): the 3D label image.

    Returns:
        NDArray[np.uint8]: `(nx-1, ny-1, nz-1)` array of label multiplicities, values in
            `[1, 8]`.
    """
    mask = np.asarray(segmented_mask)
    nx, ny, nz = (dimension - 1 for dimension in mask.shape)
    corners = [mask[i : i + nx, j : j + ny, k : k + nz] for i, j, k in _CUBE_CORNERS]

    count = np.ones(corners[0].shape, dtype=np.uint8)
    is_new = np.empty(corners[0].shape, dtype=bool)
    differs = np.empty(corners[0].shape, dtype=bool)
    for k in range(1, 8):
        is_new[...] = True
        for j in range(k):
            np.not_equal(corners[k], corners[j], out=differs)
            is_new &= differs
        count += is_new
    return count


def junction_strata(segmented_mask: NDArray[np.uint]) -> dict:
    """Boolean dual-lattice masks of the 0-, 1- and 2-strata, plus their voxel counts.

    Args:
        segmented_mask (NDArray[np.uint]): the 3D label image.

    Returns:
        dict: keys `multiplicity` (the `|L|` array), `is_corner_stratum` (`|L| >= 4`),
            `is_line_stratum` (`|L| == 3`), `is_interface_stratum` (`|L| == 2`) and the
            three integer counts `n_corner_voxels`, `n_line_voxels`, `n_interface_voxels`.
    """
    multiplicity = label_multiplicity(segmented_mask)
    is_corner = multiplicity >= 4
    is_line = multiplicity == 3
    is_interface = multiplicity == 2
    return {
        "multiplicity": multiplicity,
        "is_corner_stratum": is_corner,
        "is_line_stratum": is_line,
        "is_interface_stratum": is_interface,
        "n_corner_voxels": int(is_corner.sum()),
        "n_line_voxels": int(is_line.sum()),
        "n_interface_voxels": int(is_interface.sum()),
    }


def sample_junction_network(segmented_mask: NDArray[np.uint], spacing: float) -> dict:
    """Detect, thin and sample the 0- and 1-junctions of a label image.

    One sample per 0-junction (quadruple point) connected component, and arclength-uniform
    samples at `spacing` along the thinned 1-junction (triple line) network, with both
    endpoints of every path always kept. See the module docstring for the construction, the
    half-voxel convention and the determinism argument.

    Args:
        segmented_mask (NDArray[np.uint]): the 3D label image.
        spacing (float): target arclength between consecutive 1-junction samples, in voxels
            (the junction sample spacing `h_J`). Must be positive.

    Returns:
        dict:
            - `points` (n, 3) uint: every junction sample, 0-junctions first;
            - `is_corner_sample` (n,) bool: True for the 0-junction (quadruple point) samples;
            - `pairs` (m, 2) int64: index pairs into `points` of samples consecutive along a
              thinned path — the segments a protected triangulation should reproduce as
              edges, and what junction preservation is measured on;
            - `strata`: the `junction_strata` counts, without the bulky boolean arrays;
            - `n_skeleton_voxels`: size of the thinned `|L| >= 3` set.
    """
    if spacing <= 0:
        msg = f"junction sample spacing must be positive, got {spacing}"
        raise ValueError(msg)

    strata = junction_strata(segmented_mask)
    is_corner = strata["is_corner_stratum"]
    junction_set = is_corner | strata["is_line_stratum"]
    counts = {k: v for k, v in strata.items() if isinstance(v, int)}

    empty = np.zeros((0, 3), dtype=np.uint)
    if not junction_set.any():
        return {
            "points": empty,
            "is_corner_sample": np.zeros(0, dtype=bool),
            "pairs": np.zeros((0, 2), dtype=np.int64),
            "strata": counts,
            "n_skeleton_voxels": 0,
        }

    skeleton = skeletonize(junction_set, method="lee").astype(bool)
    skeleton_coords = np.argwhere(skeleton)
    corner_voxels = _corner_representatives_on_skeleton(is_corner, skeleton_coords)

    points, is_corner_sample, pairs = _sample_skeleton_paths(skeleton, skeleton_coords, spacing, corner_voxels)
    return {
        "points": points,
        "is_corner_sample": is_corner_sample,
        "pairs": pairs,
        "strata": counts,
        "n_skeleton_voxels": len(skeleton_coords),
    }


def _corner_representatives_on_skeleton(
    is_corner: NDArray[np.bool_],
    skeleton_coords: NDArray[np.int64],
) -> NDArray[np.int64]:
    """One skeleton voxel per 0-stratum component: the one nearest the component's centroid.

    Two components can in principle snap to the same skeleton voxel (two `|L| >= 4` blobs a
    voxel apart); duplicates are removed, so the count of 0-junction samples is a lower
    bound on the number of components. Deterministic — `cKDTree.query` returns the lowest
    index on an exact tie, and `skeleton_coords` is in raster order.
    """
    if not is_corner.any() or len(skeleton_coords) == 0:
        return np.zeros((0, 3), dtype=np.int64)

    from scipy.spatial import cKDTree

    labelled, n_components = ndi.label(is_corner, structure=np.ones((3, 3, 3), dtype=np.uint8))
    coords = np.argwhere(labelled)
    component_of = labelled[tuple(coords.T)] - 1
    centroid_sum = np.zeros((n_components, 3), dtype=np.float64)
    np.add.at(centroid_sum, component_of, coords)
    centroids = centroid_sum / np.bincount(component_of, minlength=n_components)[:, None]

    _, nearest = cKDTree(skeleton_coords).query(centroids)
    return np.unique(skeleton_coords[np.atleast_1d(nearest)], axis=0)


def _skeleton_adjacency(coords: NDArray[np.int64], shape: tuple[int, ...]) -> list[list[int]]:
    """26-connected adjacency lists over the skeleton voxels `coords` (indices into `coords`)."""
    index_of = np.full(shape, -1, dtype=np.int64)
    index_of[tuple(coords.T)] = np.arange(len(coords))

    adjacency: list[list[int]] = [[] for _ in range(len(coords))]
    upper = np.asarray(shape, dtype=np.int64)
    for offset in _NEIGHBOURS_26:
        shifted = coords + offset
        inside = np.all((shifted >= 0) & (shifted < upper), axis=1)
        if not inside.any():
            continue
        neighbour = index_of[tuple(shifted[inside].T)]
        found = neighbour >= 0
        for source, target in zip(np.flatnonzero(inside)[found].tolist(), neighbour[found].tolist(), strict=True):
            adjacency[source].append(target)
    for neighbours in adjacency:
        neighbours.sort()
    return adjacency


def _sample_skeleton_paths(  # noqa: C901 - one cohesive graph walk; splitting it would hide the traversal
    skeleton: NDArray[np.bool_],
    coords: NDArray[np.int64],
    spacing: float,
    corner_voxels: NDArray[np.int64],
) -> tuple[NDArray[np.uint], NDArray[np.bool_], NDArray[np.int64]]:
    """Split a curve skeleton into simple paths at its degree-!=2 voxels and sample each at `spacing`.

    Every path endpoint, every voxel of `corner_voxels` lying on the path, and
    arclength-uniform interior samples are emitted. A path shorter than `spacing` still
    contributes its two endpoints and the pair joining them, so short triple lines are not
    dropped. Closed loops (every voxel of degree 2) are cut at the lexicographically
    smallest voxel that starts an unvisited edge.

    Deterministic: paths are enumerated in raster order of their starting voxel, each
    path's direction is fixed by the sorted adjacency lists, and the emitted samples are
    finally re-ordered as `[0-junction samples, 1-junction samples]`, each in raster order.

    Returns:
        the samples, a boolean marking the 0-junction ones, and the consecutive-pair index
        array (into the samples).
    """
    adjacency = _skeleton_adjacency(coords, skeleton.shape)
    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    corner_keys = {tuple(int(v) for v in voxel) for voxel in np.asarray(corner_voxels, dtype=np.int64)}

    samples: list[tuple[int, int, int]] = []
    sample_index: dict[tuple[int, int, int], int] = {}
    pairs: set[tuple[int, int]] = set()

    def emit(voxel_index: int) -> int:
        key = (int(coords[voxel_index, 0]), int(coords[voxel_index, 1]), int(coords[voxel_index, 2]))
        if key not in sample_index:
            sample_index[key] = len(samples)
            samples.append(key)
        return sample_index[key]

    def sample_path(path: list[int]) -> None:
        """Emit the arclength-uniform samples of one simple path (a list of voxel indices)."""
        if len(path) < 2:
            return
        steps = np.linalg.norm(np.diff(coords[path].astype(np.float64), axis=0), axis=1)
        arclength = np.concatenate(([0.0], np.cumsum(steps)))
        total = float(arclength[-1])
        n_intervals = max(1, round(total / spacing))
        # `searchsorted` maps each target arclength to the first path voxel at or past it;
        # the two endpoints are exact by construction.
        picks = np.clip(np.searchsorted(arclength, np.linspace(0.0, total, n_intervals + 1)), 0, len(path) - 1)
        # Any 0-junction sample on this path is mandatory: it must be a node of the sampled
        # network, not something a uniform pick happens to miss.
        mandatory = [
            position
            for position, voxel_index in enumerate(path)
            if (int(coords[voxel_index, 0]), int(coords[voxel_index, 1]), int(coords[voxel_index, 2])) in corner_keys
        ]
        if mandatory:
            picks = np.concatenate((picks, np.array(mandatory, dtype=picks.dtype)))
        picks = np.unique(picks)

        previous = emit(path[int(picks[0])])
        for pick in picks[1:]:
            current = emit(path[int(pick)])
            if current != previous:
                pairs.add((min(previous, current), max(previous, current)))
                previous = current

    visited_edges: set[tuple[int, int]] = set()

    def walk(start: int, first: int) -> list[int]:
        """Follow the degree-2 chain from `start` through `first` until a non-degree-2 voxel."""
        path = [start, first]
        visited_edges.add((min(start, first), max(start, first)))
        while degree[path[-1]] == 2:
            previous, current = path[-2], path[-1]
            nexts = [n for n in adjacency[current] if n != previous]
            if not nexts:
                break
            following = nexts[0]
            edge = (min(current, following), max(current, following))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path.append(following)
        return path

    # Paths anchored at endpoints and branch nodes first (raster order), then residual loops.
    for node in np.flatnonzero(degree != 2).tolist():
        for neighbour in adjacency[node]:
            if (min(node, neighbour), max(node, neighbour)) in visited_edges:
                continue
            sample_path(walk(node, neighbour))
    for node in range(len(coords)):
        for neighbour in adjacency[node]:
            if (min(node, neighbour), max(node, neighbour)) in visited_edges:
                continue
            path = walk(node, neighbour)
            if path[-1] != node:  # close the loop so its arclength is complete
                path.append(node)
            sample_path(path)

    # Isolated skeleton voxels (degree 0) are still junction evidence: keep them.
    for node in np.flatnonzero(degree == 0).tolist():
        emit(node)

    if not samples:
        return np.zeros((0, 3), dtype=np.uint), np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=np.int64)

    sample_array = np.array(samples, dtype=np.int64)
    is_corner_sample = np.array([tuple(row) in corner_keys for row in sample_array.tolist()], dtype=bool)
    # Canonical order: 0-junctions first, then 1-junctions, each in raster order. Nothing
    # downstream depends on the order, but pinning it keeps the fingerprints readable.
    order = np.lexsort((sample_array[:, 2], sample_array[:, 1], sample_array[:, 0], ~is_corner_sample))
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    pair_array = (
        np.unique(np.sort(remap[np.array(sorted(pairs), dtype=np.int64)], axis=1), axis=0)
        if pairs
        else np.zeros((0, 2), dtype=np.int64)
    )
    return sample_array[order].astype(np.uint), is_corner_sample[order], pair_array


def labels_around(segmented_mask: NDArray[np.uint], points: NDArray[np.uint], radius: int = 1) -> list[list[int]]:
    """Sorted distinct labels within a `(2*radius+1)**3` window of each point, from the label image.

    Used to decide which materials a junction sample is adjacent to, so that the
    boundary layer can be emitted into each of them (the junction boundary layer). Read
    from the label image, never from the unsigned EDT.
    """
    shape = np.asarray(segmented_mask.shape, dtype=np.int64)
    out: list[list[int]] = []
    for point in np.asarray(points, dtype=np.int64):
        low = np.maximum(point - radius, 0)
        high = np.minimum(point + radius + 1, shape)
        window = segmented_mask[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
        out.append(sorted(int(v) for v in np.unique(window)))
    return out
