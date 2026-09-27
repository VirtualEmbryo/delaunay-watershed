r"""Trijunction curves read from the label mask and keyed to a reconstructed mesh's own junctions.

What this is
------------
`extract_junction_curves` returns one polyline per material triple `(a, b, c)`, computed from
the segmentation mask, and says for each whether the reconstructed mesh has exactly one
trijunction of that same triple. The curves are more accurate than the mesh's own junction
lines and less complete than them; both halves are measured, and the API keeps them apart so a
caller cannot get one while thinking it got the other.

**Nothing in the reconstruction pipeline calls this and no default changes.** It is an
additional read of the same mask the mesh was built from.

Where the criterion comes from
------------------------------
A dual-lattice point -- the corner shared by eight voxels -- lies on the junction line of
`(a, b, c)` **iff those eight voxels contain all three labels**. `junctions.junction_strata`
already computes exactly this predicate (`|L| == 3`) for point placement; what this module adds
is the split by material triple, the attribution of points carrying four or more labels to
*every* triple they contain (which carries a line into its quadruple point rather than stopping
a lattice step short), and the reduction of each 26-connected component to one ordered polyline
by the minimum spanning tree's diameter path. Label `0` is an ordinary material: the outer
medium's contact lines with the cells are genuine trijunctions.

The measured advantage, and the measured limitation
---------------------------------------------------
These curves were measured against a registered reference mesh on the 47-case synthetic
benchmark, reading each variant at its own median junction-edge spacing so that the comparison is
not a comparison of reading resolutions. The position error below is the median distance in voxels
from a curve's vertices to the reference polyline; lower is better.

| dw3d `min_distance` | these curves | the mesh's own lines | advantage |
|---|---:|---:|---:|
| 2 | 0.5595 | 0.7862 | 1.41x |
| 3 | 0.5855 | 0.8642 | 1.48x |
| 4 | 0.6173 | 0.9277 | 1.50x |
| 7 | 0.6817 | 1.2942 | 1.90x |

All 30 head-to-head comparisons on the 42-case equilibrium population -- 5 rungs x 6 metrics --
favour the extraction, on 1.000 of 10,000 case-level bootstrap draws. At `min_distance = 3` the
curves read tortuosity at 1.001 of the matched-spacing floor (the mesh: 1.034), curvature at
1.013 (1.172) and tangent error at 1.957x (4.529x).

**They are better geometry, not correct geometry.** The identifiability-floor work measured the mask's own
identifiability
floor -- the best any mask-based method can do -- at 0.107 voxels. These curves sit at 0.5855,
a factor of **4.8** above it, and that gap is open.

Coordinates
-----------
Everything is in mask voxel units. A dual-lattice point `(i, j, k)` is emitted at
`(i + 0.5, j + 0.5, k + 0.5)`, which is the frame a reconstructed `dw3d` mesh's `points` live
in; `extract_junction_curves` checks that the mesh it is given lies inside the mask's extent and
refuses otherwise, because a permuted axis order would put every curve tens of voxels away and
the result would silently measure the permutation.

Provenance
----------
The detection, the MST-diameter spine and the endpoint-preserving three-point filter are from
the operating-point work (see the project's research history); the per-case reading spacing,
the identity rule against the mesh's own trijunctions and the consistency test are from the
junction-curve-extraction work (see the project's research history). The sub-voxel estimator
lives in `junction_curve_estimator`. The project's research history records the port's
reproduction of the junction-curve-extraction work's numbers and this module's own
measurement.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

from dw3d import junction_curve_estimator as estimator_module

#: The operating-point work's filter: passes of an endpoint-preserving three-point moving
#: average over the lattice spine. Fixed at the value every published measurement used; not
#: a tuning knob.
SMOOTH_PASSES = 2
#: The operating-point work's own drop rule: a component of fewer than this many dual points is not a curve.
MIN_POINTS = 3
#: Half-length, in voxels of arclength, of the chord the estimator's line direction is read
#: from. The identifiability-floor work's window radius, so the tangent is read over the same scale the window spans.
TANGENT_HALF_LENGTH_VOXELS = float(estimator_module.DEFAULT_RADIUS_VOXELS)

#: The 13 forward 26-neighbour offsets; the backward 13 come free from the edge list's symmetry.
_FORWARD_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)

#: Plateau's rule for a dry foam: three films, three materials, three incident triangles.
_THREE = 3
_MIN_POLYLINE_VERTICES = 2

Estimator = Literal["plain", "anchor", "centroid"]


@dataclass(frozen=True)
class JunctionCurve:
    """One trijunction polyline read from the mask.

    Attributes:
        triple: the sorted material triple `(a, b, c)`, with label `0` an ordinary material.
        points: the ordered polyline, shape `(n, 3)`, in mask voxel coordinates.
        arclength: the polyline's total discrete arc length, in voxels.
        matched: whether the mesh has exactly one trijunction component of this same triple and
            this is the only component the mask gives for it.
        identity: `"matched"`, `"invented"` (no mesh trijunction of this triple at all),
            `"split"` (one mesh component, several here), `"merged"` (several mesh components,
            one here) or `"ambiguous"` (several on both sides).
        component: index of this component within its triple, ordered by decreasing arclength.
        n_estimated: how many of `points` the sub-voxel estimator moved. Zero under the
            default `"plain"` estimator, which runs no linear program.
        n_estimator_refusals: how many it refused, leaving the detected position in place.
    """

    triple: tuple[int, int, int]
    points: NDArray[np.float64]
    arclength: float
    matched: bool
    identity: str
    component: int
    n_estimated: int
    n_estimator_refusals: int


@dataclass(frozen=True)
class JunctionCurves:
    """Every trijunction curve of one case, split by whether the mesh agrees it exists.

    `matched` is the set a caller wanting usable geometry wants: the junction-curve-extraction
    work measured its curves a
    median 0.68 voxels from all three of their own interfaces at once, with a p99 of 4.50. The
    `invented` curves are structurally different -- median 4.32 voxels and **0 of 16 within one
    voxel** -- so they are never merged into the same list. `fragmented` holds the rest: real
    trijunctions the mask resolves as several pieces, or the reverse.

    Attributes:
        matched: curves one-to-one with a mesh trijunction.
        invented: curves whose material triple is not a trijunction of the mesh at all.
        fragmented: curves that are `split`, `merged` or `ambiguous` against the mesh.
        match_rate: matched mesh trijunctions over all mesh trijunctions. The
            junction-curve-extraction work measured 0.900 at
            `min_distance = 3` and 0.908 at `4`, with zero missed.
        counts: `matched`, `missed`, `invented`, `split`, `merged`, `ambiguous`, plus
            `mesh_triples` and `extracted_triples`.
        spacing_voxels: the reading spacing the polylines were resampled at -- the mesh's own
            median junction-edge length, so the curves are read at the resolution the mesh is.
        estimator: which sub-voxel estimator ran -- `"plain"`, `"anchor"` or `"centroid"`.
        diagnostics: dropped components, estimator statuses, timings and the frame check.
    """

    matched: tuple[JunctionCurve, ...]
    invented: tuple[JunctionCurve, ...]
    fragmented: tuple[JunctionCurve, ...]
    match_rate: float | None
    counts: dict[str, int]
    spacing_voxels: float | None
    estimator: str
    diagnostics: dict = field(default_factory=dict)

    def all_curves(self) -> tuple[JunctionCurve, ...]:
        """Every curve, matched first. Use `matched` unless completeness is what you need.

        Returns:
            tuple[JunctionCurve, ...]: the matched, then fragmented, then invented curves.
        """
        return self.matched + self.fragmented + self.invented


# --------------------------------------------------------------------------------------
# the mesh's own trijunctions -- dw3d's predicate, not a new one
# --------------------------------------------------------------------------------------
def _edge_to_triangles(triangles: NDArray[np.int64]) -> dict[tuple[int, int], list[int]]:
    """Map every undirected mesh edge to the indices of its incident triangles."""
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    edges_sorted = np.sort(edges, axis=1)
    triangle_index = np.tile(np.arange(len(triangles)), 3)
    mapping: dict[tuple[int, int], list[int]] = {}
    for edge, triangle in zip(map(tuple, edges_sorted.tolist()), triangle_index.tolist(), strict=True):
        mapping.setdefault(edge, []).append(triangle)
    return mapping


def _trijunction_edges_by_triple(
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> dict[tuple[int, int, int], list[tuple[int, int]]]:
    """Mesh edges passing dw3d's own trijunction predicate, grouped by their material triple."""
    by_triple: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for edge, triangle_ids in _edge_to_triangles(triangles).items():
        if len(triangle_ids) != _THREE:
            continue
        materials: set[int] = set()
        for triangle in triangle_ids:
            materials.update(int(x) for x in labels[triangle])
        if len(materials) != _THREE:
            continue
        by_triple[tuple(sorted(materials))].append(edge)
    return dict(by_triple)


def _connected_blocks(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Connected vertex blocks of one edge set, each sorted, visited in ascending vertex order."""
    adjacency: dict[int, list[int]] = defaultdict(list)
    for v1, v2 in edges:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)
    seen: set[int] = set()
    blocks: list[list[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, block = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            block.append(v)
            for w in adjacency[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        blocks.append(sorted(block))
    return blocks


def mesh_trijunction_components(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> dict[tuple[int, int, int], list[dict]]:
    """The mesh's own trijunctions, grouped by material triple and split into components.

    A trijunction edge is one shared by **exactly three triangles spanning exactly three
    distinct materials** -- Plateau's rule for a dry foam, and the same predicate
    `dw3d_benchmarks.metrics.edge_topology_stats` and `attributed_triple_line_angles` apply, so
    a valence->=4 defect contributes nothing here rather than contributing nonsense. This is the
    line identity the reconstruction's own angle metrics are keyed to, which is why the
    extracted curves are keyed to it too.

    Args:
        points (NDArray[np.float64]): mesh vertex coordinates, `(n_points, 3)`.
        triangles (NDArray[np.int64]): triangle vertex indices, `(n_triangles, 3)`.
        labels (NDArray[np.int64]): the two materials each triangle separates, `(n_triangles, 2)`.

    Returns:
        dict[tuple[int, int, int], list[dict]]: per sorted triple, its connected components,
        each `{"vertices", "edges", "points", "length"}`, ordered by decreasing length then by
        smallest vertex index so the ordering does not depend on dictionary iteration order.
    """
    components: dict[tuple[int, int, int], list[dict]] = {}
    for triple, edges in _trijunction_edges_by_triple(triangles, labels).items():
        built: list[dict] = []
        for block in _connected_blocks(edges):
            block_set = set(block)
            block_edges = [(v1, v2) for v1, v2 in edges if v1 in block_set]
            vertices = np.array(block, dtype=np.int64)
            length = sum(float(np.linalg.norm(points[v1] - points[v2])) for v1, v2 in block_edges)
            built.append({
                "vertices": vertices,
                "edges": block_edges,
                "points": points[vertices],
                "length": length,
            })
        built.sort(key=lambda c: (-c["length"], int(c["vertices"][0])))
        components[triple] = built
    return components


def mesh_junction_spacing(components: dict[tuple[int, int, int], list[dict]]) -> float | None:
    """The mesh's own median trijunction-edge length, in voxels.

    This is the resolution at which the mesh states its junction lines, and therefore the
    resolution at which a competing curve must be read for the comparison to be about accuracy
    rather than about how finely each side was sampled. It is derived from the mesh alone: no
    reference, no mask, no parameter.

    The statistic is the median over components of each component's own median edge length,
    which is the statistic the junction-curve validation computed from its per-line rows.

    Args:
        components (dict): `mesh_trijunction_components`'s output.

    Returns:
        float | None: the spacing in voxels, or `None` when the mesh has no trijunction edge.
    """
    per_component: list[float] = []
    for blocks in components.values():
        for block in blocks:
            lengths = [
                float(np.linalg.norm(block["points"][i] - block["points"][j]))
                for i, j in _local_edges(block)
            ]
            positive = [length for length in lengths if length > 0]
            if positive:
                per_component.append(float(np.median(positive)))
    return float(np.median(per_component)) if per_component else None


def _local_edges(block: dict) -> list[tuple[int, int]]:
    """A component's edges as indices into its own `points` array."""
    index_of = {int(v): i for i, v in enumerate(block["vertices"])}
    return [(index_of[v1], index_of[v2]) for v1, v2 in block["edges"]]


# --------------------------------------------------------------------------------------
# detection on the dual lattice
# --------------------------------------------------------------------------------------
def dual_label_sets(segmented_mask: NDArray[np.uint]) -> tuple[NDArray[np.int64], list[frozenset[int]]]:
    """Per dual-lattice point, an index into a table of the distinct labels around it.

    A dual point `(i, j, k)` is the corner shared by the eight voxels
    `segmented_mask[i:i+2, j:j+2, k:k+2]`. Points with fewer than three distinct labels are not
    junctions and carry `-1`.

    Args:
        segmented_mask (NDArray[np.uint]): the label image.

    Returns:
        tuple[NDArray[np.int64], list[frozenset[int]]]: `codes` of shape `(X-1, Y-1, Z-1)` and a
        table where `table[c]` is the frozenset of labels around a point carrying code `c`.
    """
    shape = segmented_mask.shape
    corners = [
        segmented_mask[a:, b:, c:][: shape[0] - 1, : shape[1] - 1, : shape[2] - 1]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]
    stacked = np.stack(corners, axis=-1)
    stacked_sorted = np.sort(stacked, axis=-1)
    n_distinct = 1 + (np.diff(stacked_sorted, axis=-1) != 0).sum(axis=-1)
    codes = np.full(n_distinct.shape, -1, dtype=np.int64)
    table: list[frozenset[int]] = []
    index_of: dict[frozenset[int], int] = {}
    for i, j, k in np.argwhere(n_distinct >= _THREE):
        materials = frozenset(int(v) for v in stacked[i, j, k])
        code = index_of.get(materials)
        if code is None:
            code = len(table)
            index_of[materials] = code
            table.append(materials)
        codes[i, j, k] = code
    return codes, table


def _points_by_triple(
    codes: NDArray[np.int64],
    table: list[frozenset[int]],
) -> dict[tuple[int, ...], NDArray[np.int64]]:
    """Every dual point belonging to each material triple's junction line."""
    occupied = np.argwhere(codes >= 0)
    if occupied.size == 0:
        return {}
    per_triple: dict[tuple[int, ...], list[NDArray[np.int64]]] = defaultdict(list)
    code_values = codes[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
    for code in np.unique(code_values):
        members = occupied[code_values == code]
        for triple in combinations(sorted(table[code]), 3):
            per_triple[triple].append(members)
    return {triple: np.concatenate(blocks, axis=0) for triple, blocks in per_triple.items()}


def _component_graph(coordinates: NDArray[np.int64]) -> coo_matrix:
    """26-connectivity graph on a set of lattice points, with Euclidean weights."""
    n = len(coordinates)
    lookup = {tuple(int(v) for v in row): i for i, row in enumerate(coordinates)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, row in enumerate(coordinates):
        for offset in _FORWARD_OFFSETS:
            neighbour = (int(row[0]) + offset[0], int(row[1]) + offset[1], int(row[2]) + offset[2])
            j = lookup.get(neighbour)
            if j is None:
                continue
            weight = float(np.sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2))
            rows += [i, j]
            cols += [j, i]
            data += [weight, weight]
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def _tree_diameter_path(tree: coo_matrix, n: int) -> list[int]:
    """The longest path of a tree, by double breadth-first search. Deterministic."""
    symmetric = (tree + tree.T).tocsr()

    def farthest(source: int) -> tuple[int, dict[int, int]]:
        order, predecessors = breadth_first_order(symmetric, source, directed=False, return_predecessors=True)
        return int(order[-1]), {int(v): int(p) for v, p in enumerate(predecessors)}

    start, _ = farthest(0)
    end, predecessors = farthest(start)
    path = [end]
    while path[-1] != start:
        previous = predecessors[path[-1]]
        if previous < 0:
            break
        path.append(previous)
    if n == 1:
        return [0]
    return path


def _spine_of_component(coordinates: NDArray[np.int64]) -> tuple[NDArray[np.float64], float]:
    """One ordered polyline through a component, and the fraction of its points kept.

    The minimum spanning tree's diameter path is used rather than a nearest-neighbour walk
    because it is deterministic and because it handles both topologies that occur: an open arc
    between two quadruple points, and a closed ring, on which the MST breaks exactly one edge
    and the diameter path recovers the ring minus that edge.
    """
    n = len(coordinates)
    if n == 1:
        return coordinates.astype(np.float64) + 0.5, 1.0
    tree = minimum_spanning_tree(_component_graph(coordinates).tocsr())
    path = _tree_diameter_path(tree, n)
    ordered = coordinates[np.asarray(path, dtype=np.int64)].astype(np.float64) + 0.5
    return ordered, len(path) / n


def _smooth(path: NDArray[np.float64], passes: int) -> NDArray[np.float64]:
    """Endpoint-preserving three-point moving average, `passes` times."""
    smoothed = path.astype(np.float64).copy()
    for _ in range(max(0, passes)):
        if len(smoothed) < _THREE:
            break
        interior = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = np.vstack([smoothed[:1], interior, smoothed[-1:]])
    return smoothed


def subsample_indices(path: NDArray[np.float64], spacing: float) -> NDArray[np.int64]:
    """Indices of a subset of a polyline's own vertices, spaced by about `spacing` in arclength.

    Every retained point lies exactly on the original polyline, so this coarsens the reading
    without adding positional error. Targets are laid out at exact multiples of `spacing` and
    the *nearest* own-vertex is taken for each; taking the first vertex past each threshold
    systematically overshoots the requested spacing.

    Indices rather than points, because the sub-voxel estimator has to read its line direction
    from the *un*-coarsened polyline -- at an 8-voxel spacing a three-voxel chord between
    retained vertices does not exist.

    Args:
        path (NDArray[np.float64]): the polyline, shape `(n, 3)`.
        spacing (float): the target spacing in voxels; `<= 0` keeps every vertex.

    Returns:
        NDArray[np.int64]: ascending indices into `path`, always including its two ends.
    """
    if len(path) < _THREE or spacing <= 0:
        return np.arange(len(path), dtype=np.int64)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(arclength[-1])
    if total <= spacing:
        return np.array([0, len(path) - 1], dtype=np.int64)
    n_intervals = max(1, round(total / spacing))
    targets = np.linspace(0.0, total, n_intervals + 1)
    keep = sorted({int(np.argmin(np.abs(arclength - t))) for t in targets} | {0, len(path) - 1})
    return np.array(keep, dtype=np.int64)


def _polyline_tangent(path: NDArray[np.float64], index: int, half_length: float) -> NDArray[np.float64] | None:
    """Unit chord between the vertices `half_length` voxels of arclength either side of `index`."""
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    here = arclength[index]
    before = int(np.searchsorted(arclength, here - half_length, side="left"))
    after = int(np.searchsorted(arclength, here + half_length, side="right")) - 1
    before = max(0, min(before, index))
    after = min(len(path) - 1, max(after, index))
    if after == before:
        return None
    vector = path[after] - path[before]
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-9 else vector / norm


def _arclength(path: NDArray[np.float64]) -> float:
    """Total discrete arc length of a polyline, in the units of its coordinates."""
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0


def detect_junction_spines(
    segmented_mask: NDArray[np.uint],
    *,
    min_points: int = MIN_POINTS,
    smooth_passes: int = SMOOTH_PASSES,
) -> dict:
    """Every material triple's smoothed lattice spine, before any reading spacing is chosen.

    This is the detection half of `extract_junction_curves`, exposed on its own because it is
    the part that reads the mask and nothing else.

    Args:
        segmented_mask (NDArray[np.uint]): the label image.
        min_points (int): components with fewer dual points than this are dropped and counted.
        smooth_passes (int): passes of the three-point filter. Defaults to `SMOOTH_PASSES`.

    Returns:
        dict: `by_triple` (triple -> list of polylines, longest first), `diagnostics` (one record
        per component) and `n_dual_junction_points`.
    """
    codes, table = dual_label_sets(segmented_mask)
    per_triple = _points_by_triple(codes, table)
    by_triple: dict[tuple[int, int, int], list[NDArray[np.float64]]] = {}
    diagnostics: list[dict] = []
    for triple, coordinates in sorted(per_triple.items()):
        n_components, membership = connected_components(_component_graph(coordinates).tocsr(), directed=False)
        kept: list[NDArray[np.float64]] = []
        for component in range(n_components):
            members = coordinates[membership == component]
            if len(members) < min_points:
                diagnostics.append({
                    "triple": list(triple),
                    "component": component,
                    "n_points": len(members),
                    "status": "dropped_too_small",
                })
                continue
            path, coverage = _spine_of_component(members)
            smoothed = _smooth(path, smooth_passes)
            # Ordered by the RAW spine's arclength, not the smoothed one: the filter shortens a
            # polyline by a little and a different amount per component, so ordering on the
            # smoothed length could reorder two near-equal components. The operating-point
            # work orders on the raw length and every published component index is that
            # ordering.
            kept.append((float(_arclength(path)), smoothed))
            diagnostics.append({
                "triple": list(triple),
                "component": component,
                "n_points": len(members),
                "n_spine": len(path),
                "spine_coverage": float(coverage),
                "status": "kept",
            })
        if kept:
            order = np.argsort([-length for length, _ in kept])
            by_triple[tuple(int(t) for t in triple)] = [kept[i][1] for i in order]
    return {
        "by_triple": by_triple,
        "diagnostics": diagnostics,
        "n_dual_junction_points": int((codes >= 0).sum()),
    }


# --------------------------------------------------------------------------------------
# the public API
# --------------------------------------------------------------------------------------
def _assert_mesh_is_in_the_mask_frame(segmented_mask: NDArray[np.uint], points: NDArray[np.float64]) -> dict:
    """Stop unless the mesh's coordinates index the mask axis for axis."""
    if points.size == 0:
        return {"checked": False, "reason": "no points"}
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    shape = np.asarray(segmented_mask.shape, dtype=np.float64)
    if not (lower >= -1.0).all() or not (upper <= shape).all():
        error = (
            f"mesh points do not lie in the mask frame (extent {lower}..{upper} against shape "
            f"{segmented_mask.shape}); extract_junction_curves needs both in mask voxel coordinates"
        )
        raise ValueError(error)
    return {"checked": True, "extent_low": lower.tolist(), "extent_high": upper.tolist()}


def _identity_of(n_extracted: int, n_mesh: int) -> str:
    """One triple's status: the same one-to-one rule every line-geometry metric uses."""
    if n_mesh == 0:
        return "invented"
    if n_extracted == 0:
        return "missed"
    if n_extracted == 1 and n_mesh == 1:
        return "matched"
    if n_mesh == 1:
        return "split"
    if n_extracted == 1:
        return "merged"
    return "ambiguous"


def extract_junction_curves(
    segmented_mask: NDArray[np.uint],
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    *,
    estimator: Estimator = "plain",
) -> JunctionCurves:
    """Trijunction curves from the mask, keyed to the mesh's own trijunctions.

    The one line a user writes::

        curves = extract_junction_curves(mask, *load_rec("000_mesh.rec"))

    `curves.matched` is the usable set. Every one of its members has exactly one counterpart
    among the mesh's own trijunctions of the same material triple, so it can be joined to the
    interfaces and the dihedral angles the reconstruction already reports.
    `curves.invented` is everything the mask says is a junction and the mesh does not have;
    the junction-curve-extraction work measured **0 of 16** of those within one voxel of all
    three of their interfaces, against a
    median of 0.68 voxels for the matched ones, so they are kept apart rather than merged.
    `curves.fragmented` holds real trijunctions the mask resolves as several pieces.

    Reading spacing is not a parameter: the polylines are resampled at the mesh's **own** median
    trijunction-edge length, so accuracy is compared at the resolution the mesh states its lines
    at rather than at whatever resolution the lattice happens to give.

    **Cost, and why `"plain"` is the default.** `"plain"` is detection only -- no linear
    program -- and costs about 0.8 s per benchmark case against the 1.49 s dw3d spends
    reconstructing that case. `"anchor"` solves one small linear program per returned vertex and
    costs roughly fifteen times as much, about eight times the whole reconstruction it
    annotates, and buys 9-11 % of the position error. Nobody meshing a large tissue should pay that by default,
    so they do not.

    `"anchor"`'s real argument is not that 9 %: it is that the polytope's deepest interior point
    carries **only** the identifiability-floor work's irreducible narrow-wedge term (a),
    where the detected position and the centroid both carry more. Reach for it when a
    systematic bias against narrow wedges would corrupt what you are inferring, not to shave
    a tenth of a voxel.
    `"centroid"` is exposed to reproduce the junction-curve-extraction work's measurement and
    for nothing else: that work measured it carrying
    (a) **+** (b), so it is the larger systematic, not the smaller one, and it costs about
    thirty times the anchor's linear programs on top.

    Args:
        segmented_mask (NDArray[np.uint]): the label image the mesh was reconstructed from.
        points (NDArray[np.float64]): mesh vertex coordinates in mask voxel units, `(n, 3)`.
        triangles (NDArray[np.int64]): triangle vertex indices, `(m, 3)`.
        labels (NDArray[np.int64]): the two materials each triangle separates, `(m, 2)`.
        estimator (Estimator): `"plain"` (default, no linear program), `"anchor"` or
            `"centroid"`. See the cost paragraph above before changing it.

    Raises:
        ValueError: if `estimator` is not one of the three, or if the mesh's coordinates do not
            lie inside the mask's extent.

    Returns:
        JunctionCurves: the curves, the match rate, the identity counts and the diagnostics.
    """
    if estimator not in ("plain", "anchor", "centroid"):
        error = f"estimator must be 'plain', 'anchor' or 'centroid', not {estimator!r}"
        raise ValueError(error)
    points = np.asarray(points, dtype=np.float64)
    frame_check = _assert_mesh_is_in_the_mask_frame(segmented_mask, points)

    mesh_components = mesh_trijunction_components(points, triangles, labels)
    spacing = mesh_junction_spacing(mesh_components)

    detected = detect_junction_spines(segmented_mask)
    by_triple = detected["by_triple"]

    counts: dict[str, int] = {
        "matched": 0,
        "missed": 0,
        "invented": 0,
        "split": 0,
        "merged": 0,
        "ambiguous": 0,
    }
    estimator_statuses: dict[str, int] = {}
    grouped: dict[str, list[JunctionCurve]] = {"matched": [], "invented": [], "fragmented": []}

    for triple in sorted(set(by_triple) | set(mesh_components)):
        spines = by_triple.get(triple, [])
        identity = _identity_of(len(spines), len(mesh_components.get(triple, [])))
        counts[identity] += 1
        for component, spine in enumerate(spines):
            polyline, n_estimated, n_refused = _read_one_polyline(
                segmented_mask, spine, triple, spacing, estimator, estimator_statuses,
            )
            if len(polyline) < _MIN_POLYLINE_VERTICES:
                continue
            curve = JunctionCurve(
                triple=triple,
                points=polyline,
                arclength=_arclength(polyline),
                matched=identity == "matched",
                identity=identity,
                component=component,
                n_estimated=n_estimated,
                n_estimator_refusals=n_refused,
            )
            bucket = identity if identity in ("matched", "invented") else "fragmented"
            grouped[bucket].append(curve)

    counts["mesh_triples"] = len(mesh_components)
    counts["extracted_triples"] = len(by_triple)
    match_rate = counts["matched"] / len(mesh_components) if mesh_components else None

    return JunctionCurves(
        matched=tuple(grouped["matched"]),
        invented=tuple(grouped["invented"]),
        fragmented=tuple(grouped["fragmented"]),
        match_rate=match_rate,
        counts=counts,
        spacing_voxels=spacing,
        estimator=estimator,
        diagnostics={
            "frame_check": frame_check,
            "n_dual_junction_points": detected["n_dual_junction_points"],
            "n_components_dropped_too_small": sum(
                1 for d in detected["diagnostics"] if d["status"] == "dropped_too_small"
            ),
            "estimator_statuses": estimator_statuses,
            "smooth_passes": SMOOTH_PASSES,
            "min_points": MIN_POINTS,
            "estimator_radius_voxels": estimator_module.DEFAULT_RADIUS_VOXELS,
        },
    )


def _read_one_polyline(
    segmented_mask: NDArray[np.uint],
    spine: NDArray[np.float64],
    triple: tuple[int, int, int],
    spacing: float | None,
    estimator: str,
    statuses: dict[str, int],
) -> tuple[NDArray[np.float64], int, int]:
    """Resample one spine at `spacing` and, unless asked not to, move each vertex sub-voxel."""
    kept = subsample_indices(spine, spacing if spacing else 0.0)
    positions = spine[kept].copy()
    if estimator == "plain":
        return positions, 0, 0
    n_estimated = 0
    n_refused = 0
    for slot, index in enumerate(kept):
        direction = _polyline_tangent(spine, int(index), TANGENT_HALF_LENGTH_VOXELS)
        if direction is None:
            statuses["no_tangent"] = statuses.get("no_tangent", 0) + 1
            n_refused += 1
            continue
        result = estimator_module.estimate_at_sample(
            segmented_mask,
            spine[int(index)],
            direction,
            triple,
            want_centroid=estimator == "centroid",
        )
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        key = "centroid_point" if estimator == "centroid" else "anchor_point"
        if result["status"] != "ok" or key not in result:
            n_refused += 1
            continue
        positions[slot] = np.asarray(result[key], dtype=np.float64)
        n_estimated += 1
    return positions, n_estimated, n_refused
