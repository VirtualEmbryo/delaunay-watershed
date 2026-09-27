"""Pure measurement functions over a constructed `MeshReconstructionAlgorithm`.

Used by both `tests/test_golden_master.py` and `benchmarks/run_case.py` so the two share
one definition of every metric. The tetrahedron-side metrics (all-surface tetrahedra, quality,
slivers, score-gap degeneracy) exist because the original algorithm placed interface and
interior points 44:1 on `3.tif` at `min_distance=3`, leaving 58.0 % of tetrahedra with all
four vertices on the interface. Mesh metrics are keyed by label-tuples rather than vertex
correspondence because a `dw3d` mesh and a ground-truth `.rec` mesh are two different
discretisations of the same object.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from dw3d.mesh_surgery import _find_abnormal_non_manifold_edges
from dw3d.points_on_edt import boundary_layer_families, peak_local_points

if TYPE_CHECKING:
    from dw3d.tesselation_graph import TesselationGraph

# ---------------------------------------------------------------------------
# Point-placement stage (recomputed: the pipeline does not store these counts)
# ---------------------------------------------------------------------------


def point_placement_counts(
    edt_image: NDArray,
    min_distance: int,
    point_placing_function: Callable | None = None,
    segmented_image: NDArray | None = None,
) -> dict:
    """Recompute the point-placement stage to expose interface/interior point counts.

    Pass the algorithm's own `point_placing_function` when measuring a non-default
    variant (`get_dithered_algorithm`, the boundary-biased placement, ...); otherwise the
    counts would describe the default rule rather than the one that built the mesh.
    With the default `None` the current default rule is used at `min_distance`.

    Re-running the stage is safe: the default rule is deterministic with no RNG at all,
    and the dithered variants draw from a private `RandomState`, so
    neither perturbs the already-completed reconstruction nor the global RNG.
    """
    if point_placing_function is None:
        placed = peak_local_points(None, edt_image, min_distance)
    else:
        placed = point_placing_function(segmented_image, edt_image)
    # The junction-protected scheme returns a third element (the per-point weights); the counts below
    # do not depend on it.
    all_points, indices_of_sorted_maxes = placed[0], placed[1]
    n_corners = 8
    n_interior = len(indices_of_sorted_maxes)
    n_interface = len(all_points) - n_corners - n_interior
    return {
        "n_interface_points": int(n_interface),
        "n_interior_points": int(n_interior),
        "n_corner_points": n_corners,
    }


def boundary_layer_orientation_stats(
    segmented_image: NDArray,
    edt_image: NDArray,
    min_distance: int,
    delta: float | None = None,
) -> dict:
    """Count the boundary-layer offsets and verify their orientation.

    Recomputes the boundary-layer families (cheap, deterministic, no RNG) and returns the
    orientation dict from `dw3d.points_on_edt.boundary_layer_families` — the interface
    sample count, how many lie on a genuine >=2-label boundary, how many offsets are
    emitted before/after dedup, and the independent orientation re-check against the label
    image (see `_emit_boundary_layer_offsets`). This is the metric the boundary-layer
    work's falsification clause asks for: *count* the emitted offsets and *check* their
    orientation against the
    label image rather than assuming the layer is present and correct.

    Also derives the two fractions worth reporting directly:
        interface_yield        — n_on_interface / n_minima (share of EDT minima that sit on
                                  a real interface; the rest are bounding-box/background
                                  plateau minima that correctly get no offset);
        orientation_ok_fraction — n_orientation_ok / n_orientation_checked (should be 1.0).
    """
    families = boundary_layer_families(segmented_image, edt_image, min_distance, delta)
    info = dict(families["orientation"])
    n_minima = info["n_minima"]
    n_checked = info["n_orientation_checked"]
    info["n_maxima"] = len(families["maxima"])
    info["n_offsets"] = len(families["offsets"])
    info["interface_yield"] = float(info["n_on_interface"] / n_minima) if n_minima else 0.0
    info["orientation_ok_fraction"] = float(info["n_orientation_ok"] / n_checked) if n_checked else 1.0
    return info


def classify_tesselation_points(n_points: int, n_interior_points: int) -> dict[str, NDArray[np.bool_]]:
    """Boolean masks classifying each tesselation vertex by how it was placed.

    The three classes are interior (a local EDT maximum), interface (a local EDT minimum)
    and corner.

    Relies on `simple_delaunay_tesselation` preserving input point order (verified:
    `scipy.spatial.Delaunay.points is` the input array), and on the point-placing
    functions stacking points as `[corners(8), interior, interface]` with the interface
    (minima) block trailing. For the boundary-layer variant the interior block is
    `[maxima, offsets]` (offsets stacked before the minima and counted in
    `indices_of_sorted_maxes`), so the boundary-layer offsets are classified as interior —
    which is correct for the all-surface-tet metric: an offset point carries EDT `~= delta`
    and is *not* on the interface. Per-family offset counts are reported separately by
    `boundary_layer_orientation_stats`.
    """
    is_corner = np.zeros(n_points, dtype=bool)
    is_corner[:8] = True
    is_interior = np.zeros(n_points, dtype=bool)
    is_interior[8 : 8 + n_interior_points] = True
    is_interface = ~is_corner & ~is_interior
    return {"is_corner": is_corner, "is_interior": is_interior, "is_interface": is_interface}


# ---------------------------------------------------------------------------
# Tetrahedron / watershed-input stats (operate on TesselationGraph)
# ---------------------------------------------------------------------------


def tetrahedron_surface_stats(
    tesselation_graph: TesselationGraph,
    point_classes: dict[str, NDArray[np.bool_]],
) -> dict:
    """Fraction of tetrahedra with all 4 vertices on the interface, vs. with an interior vertex.

    "All-surface" means exactly: all 4 vertices are strictly
    interface (local-min) points, excluding both interior (local-max) and corner
    vertices. This is why `fraction_all_surface_tets + fraction_tets_with_interior_vertex`
    does not sum to 1 (original algorithm, `3.tif`, `min_distance=3`: 58.0% + 41.5% = 99.5%) — the
    remainder touches a corner
    but no interior vertex (`fraction_tets_touching_corner_only`).
    """
    tetra = tesselation_graph.tetrahedrons
    tetra_is_interior = point_classes["is_interior"][tetra]
    tetra_is_interface = point_classes["is_interface"][tetra]
    has_interior = tetra_is_interior.any(axis=1)
    all_surface = tetra_is_interface.all(axis=1)
    corner_only = ~has_interior & ~all_surface
    return {
        "n_tetrahedra": len(tetra),
        "fraction_all_surface_tets": float(np.mean(all_surface)),
        "fraction_tets_with_interior_vertex": float(np.mean(has_interior)),
        "fraction_tets_touching_corner_only": float(np.mean(corner_only)),
        "_all_surface_mask": all_surface,
    }


def tetrahedron_quality(tesselation_graph: TesselationGraph) -> NDArray[np.float64]:
    """Tet quality `vol / rms_edge**3` (regular tet = 0.1178)."""
    volumes = tesselation_graph._compute_volumes()
    verts = tesselation_graph.vertices[tesselation_graph.tetrahedrons]  # (n_tet, 4, 3)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edge_lengths_sq = np.stack(
        [np.sum((verts[:, i] - verts[:, j]) ** 2, axis=1) for i, j in edge_pairs],
        axis=1,
    )
    rms_edge = np.sqrt(np.mean(edge_lengths_sq, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = np.where(rms_edge > 0, volumes / np.where(rms_edge > 0, rms_edge**3, 1.0), 0.0)
    return quality


def sliver_stats(quality: NDArray[np.float64], all_surface_mask: NDArray[np.bool_]) -> dict:
    """Sliver fraction (`qual < 0.01`), overall and split by the all-surface classification."""
    stats = {
        "tet_quality_median": float(np.median(quality)),
        "tet_quality_p10": float(np.percentile(quality, 10)),
        "tet_quality_p1": float(np.percentile(quality, 1)),
        "sliver_fraction": float(np.mean(quality < 0.01)),
    }
    if all_surface_mask.any():
        stats["sliver_fraction_all_surface"] = float(np.mean(quality[all_surface_mask] < 0.01))
    if (~all_surface_mask).any():
        stats["sliver_fraction_has_interior_vertex"] = float(np.mean(quality[~all_surface_mask] < 0.01))
    return stats


def score_gap_stats(tesselation_graph: TesselationGraph) -> dict:
    """Fraction of consecutive (sorted) watershed face-score gaps below 1e-3 and 1e-5.

    The 1e-5 figure is the fraction of decisions the
    `1e-5` seed dither can directly flip, as opposed to the looser 1e-3 fragility figure.
    """
    scores = np.sort(tesselation_graph.scores)
    gaps = np.diff(scores)
    return {
        "n_faces": len(scores),
        "fraction_score_gaps_below_1e-3": float(np.mean(gaps < 1e-3)),
        "fraction_score_gaps_below_1e-5": float(np.mean(gaps < 1e-5)),
    }


# ---------------------------------------------------------------------------
# Final-mesh geometry (operate on points, triangles, labels only)
# ---------------------------------------------------------------------------


def triangle_areas(points: NDArray[np.float64], triangles: NDArray) -> NDArray[np.float64]:
    """Heron's-formula triangle areas. Same formula as `TesselationGraph._compute_areas`."""
    p = points[triangles]
    sides = p - p[:, [2, 0, 1]]
    lengths = np.linalg.norm(sides, axis=2)
    s = np.sum(lengths, axis=1) / 2
    diff = s[:, None] - lengths
    return np.sqrt(np.clip(s * diff[:, 0] * diff[:, 1] * diff[:, 2], 0, None))


def interface_areas(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> dict[tuple[int, int], float]:
    """Total area per interface, keyed by sorted label pair (not by vertex correspondence)."""
    areas = triangle_areas(points, triangles)
    sorted_labels = np.sort(labels, axis=1)
    result: dict[tuple[int, int], float] = {}
    for (a, b), area in zip(map(tuple, sorted_labels.tolist()), areas, strict=True):
        key = (int(a), int(b))
        result[key] = result.get(key, 0.0) + float(area)
    return result


def cell_volumes_from_tetrahedra(
    tesselation_graph: TesselationGraph,
    map_label_to_nodes_ids: dict[int, list[int]],
) -> dict[int, float]:
    """Cell volume = sum of the tesselation's tetrahedron volumes assigned that label.

    This reuses the watershed's own tetrahedron labelling rather than re-deriving volume
    from the extracted surface (which would need the surface's normal-orientation
    convention); it is exactly what the algorithm considers "the cell", so it is the
    natural choice for a regression/QA metric.
    """
    volumes = tesselation_graph._compute_volumes()
    result = {}
    for label, node_ids in map_label_to_nodes_ids.items():
        if label == 0:
            continue  # exterior / background is not a cell
        result[int(label)] = float(np.sum(volumes[node_ids]))
    return result


def _edge_to_triangle_map(triangles: NDArray) -> dict[tuple[int, int], list[int]]:
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    edges_sorted = np.sort(edges, axis=1)
    tri_idx = np.tile(np.arange(len(triangles)), 3)
    edge_map: dict[tuple[int, int], list[int]] = {}
    for edge, tri in zip(map(tuple, edges_sorted.tolist()), tri_idx.tolist(), strict=True):
        edge_map.setdefault(edge, []).append(tri)
    return edge_map


def edge_topology_stats(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> dict:
    """Valence histogram, hole count (valence-1), valence->=4 edges, and triple-line lengths.

    A triple-line edge is one shared by exactly 3 triangles whose labels cover exactly 3
    distinct materials (Plateau: exactly 3 films meet along a line in a dry foam, so a
    valence-4+ edge is a meshing artifact, not a physical quadrijunction along a line).
    """
    edge_map = _edge_to_triangle_map(triangles)
    valence_histogram: Counter[int] = Counter()
    triple_line_lengths: dict[tuple[int, int, int], float] = {}
    n_holes = 0
    valence_geq4_count = 0
    valence_geq4_length = 0.0

    for (p1, p2), tri_ids in edge_map.items():
        valence = len(tri_ids)
        valence_histogram[valence] += 1
        length = float(np.linalg.norm(points[p1] - points[p2]))

        if valence == 1:
            n_holes += 1
        elif valence == 3:
            edge_labels: set[int] = set()
            for tri in tri_ids:
                edge_labels.update(int(x) for x in labels[tri])
            if len(edge_labels) == 3:
                key = tuple(sorted(edge_labels))
                triple_line_lengths[key] = triple_line_lengths.get(key, 0.0) + length
        elif valence >= 4:
            valence_geq4_count += 1
            valence_geq4_length += length

    return {
        "valence_histogram": dict(valence_histogram),
        "n_holes": n_holes,
        "n_valence_geq4_edges": valence_geq4_count,
        "length_valence_geq4_edges": valence_geq4_length,
        "triple_line_lengths": triple_line_lengths,
    }


def _pairwise_angles_about_edge(
    points: NDArray[np.float64],
    triangles: NDArray,
    tri_ids: list[int],
    p1: int,
    p2: int,
) -> list[float]:
    """Angles between the incident triangles' apex directions, measured about edge `p1-p2`.

    Each incident triangle contributes the unit vector from the edge to its third vertex,
    projected perpendicular to the edge; the returned angles are the pairwise angles
    between those vectors. Degenerate contributions (a zero-length edge, or an apex lying
    on the edge line) are dropped rather than producing a NaN angle.

    Args:
        points (NDArray[np.float64]): Mesh vertex positions.
        triangles (NDArray): Mesh triangles (vertex indices).
        tri_ids (list[int]): Indices of the triangles incident to this edge.
        p1 (int): First endpoint of the edge.
        p2 (int): Second endpoint of the edge.

    Returns:
        list[float]: Pairwise angles in radians; empty if the edge is degenerate.
    """
    e = points[p2] - points[p1]
    e_norm_sq = float(np.dot(e, e))
    if e_norm_sq == 0:
        return []

    apexes = []
    for tri in tri_ids:
        third = next(v for v in triangles[tri] if v not in (p1, p2))
        u = points[third] - points[p1]
        u_perp = u - (np.dot(u, e) / e_norm_sq) * e
        norm = np.linalg.norm(u_perp)
        if norm > 0:
            apexes.append(u_perp / norm)

    return [
        float(np.arccos(np.clip(np.dot(apexes[i], apexes[j]), -1.0, 1.0)))
        for i in range(len(apexes))
        for j in range(i + 1, len(apexes))
    ]


def dihedral_angle_stats_per_triple_line(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> dict:
    """Mean and spread of the local dihedral configuration at each triple-line edge.

    Simplification (recorded here, not a `dw3d` behaviour change): for each trijunction
    edge we compute the 3 pairwise angles-to-the-edge among its 3 incident triangles and
    average them, rather than attributing each of the 3 angles to a specific one of the
    3 interfaces (which needs a sign/orientation convention this harness does not need
    for a QA/regression metric). Per-interface attribution is what
    `attributed_triple_line_angles` does.
    """
    edge_map = _edge_to_triangle_map(triangles)
    per_line: dict[tuple[int, int, int], list[float]] = {}

    for (p1, p2), tri_ids in edge_map.items():
        if len(tri_ids) != 3:
            continue
        edge_labels: set[int] = set()
        for tri in tri_ids:
            edge_labels.update(int(x) for x in labels[tri])
        if len(edge_labels) != 3:
            continue

        angles = _pairwise_angles_about_edge(points, triangles, tri_ids, p1, p2)
        if angles:
            per_line.setdefault(tuple(sorted(edge_labels)), []).extend(angles)

    return {
        str(key): {"mean_deg": float(np.degrees(np.mean(vals))), "std_deg": float(np.degrees(np.std(vals)))}
        for key, vals in per_line.items()
    }


def triangle_quality_stats(points: NDArray[np.float64], triangles: NDArray) -> dict:
    """Triangle min/max angle distribution and radius ratio (equilateral = 1)."""
    p = points[triangles]
    sides = p - p[:, [2, 0, 1]]  # side opposite vertex i
    lengths = np.linalg.norm(sides, axis=2)  # (n_tri, 3): [a, b, c] opposite [0, 1, 2]
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]

    def _angle_opposite(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
        cos_val = np.clip((y**2 + z**2 - x**2) / (2 * y * z + 1e-300), -1.0, 1.0)
        return np.arccos(cos_val)

    angle_a = _angle_opposite(a, b, c)
    angle_b = _angle_opposite(b, a, c)
    angle_c = _angle_opposite(c, a, b)
    angles = np.stack([angle_a, angle_b, angle_c], axis=1)
    min_angles = np.degrees(np.min(angles, axis=1))
    max_angles = np.degrees(np.max(angles, axis=1))

    areas = triangle_areas(points, triangles)
    s = (a + b + c) / 2
    inradius = areas / np.where(s > 0, s, 1.0)
    circumradius = (a * b * c) / np.where(areas > 0, 4 * areas, 1.0)
    radius_ratio = np.where(circumradius > 0, 2 * inradius / circumradius, 0.0)

    return {
        "min_angle_median_deg": float(np.median(min_angles)),
        "min_angle_p10_deg": float(np.percentile(min_angles, 10)),
        "max_angle_median_deg": float(np.median(max_angles)),
        "max_angle_p90_deg": float(np.percentile(max_angles, 90)),
        "radius_ratio_median": float(np.median(radius_ratio)),
        "radius_ratio_p10": float(np.percentile(radius_ratio, 10)),
    }


def connected_components_and_euler_per_cell(triangles: NDArray, labels: NDArray) -> dict[int, dict]:
    """Per-cell: number of connected components of its full boundary surface, and Euler characteristic V-E+F."""
    cell_labels = np.unique(labels)
    result = {}
    for cell in cell_labels:
        cell = int(cell)
        if cell == 0:
            continue
        mask = (labels[:, 0] == cell) | (labels[:, 1] == cell)
        cell_triangles = triangles[mask]
        if len(cell_triangles) == 0:
            continue

        graph = nx.Graph()
        graph.add_edges_from(cell_triangles[:, [0, 1]].tolist())
        graph.add_edges_from(cell_triangles[:, [0, 2]].tolist())
        graph.add_edges_from(cell_triangles[:, [1, 2]].tolist())

        n_components = nx.number_connected_components(graph)
        n_vertices = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_faces = len(cell_triangles)
        result[cell] = {
            "n_connected_components": n_components,
            "euler_characteristic": n_vertices - n_edges + n_faces,
        }
    return result


def abnormal_non_manifold_edge_count(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> int:
    """Count of `_find_abnormal_non_manifold_edges` survivors (dw3d/mesh_surgery.py) post-surgery."""
    return len(_find_abnormal_non_manifold_edges(points, triangles, labels))


def valence_geq4_edges_by_material_count(triangles: NDArray, labels: NDArray) -> dict:
    """Split the valence->=4 edges by how many distinct materials actually meet along them.

    `edge_topology_stats` reports one `n_valence_geq4_edges` number, but that number
    conflates two different defects, and the junction-protection work measured that the
    split matters:

    * **4+ materials** — a genuine *quadjunction edge*. Four cells meeting along a line is
      impossible in a dry foam (Plateau: exactly three films meet along an edge), so this is
      the under-resolved quadruple point
      that junction protection exists to fix.
    * **exactly 3 materials** — an edge with four or more incident triangles but only three
      materials, i.e. one of the three films is *doubled* or pinched at that edge. That is a
      local non-manifold defect of a triple line, not a mis-resolved quadruple point, and it
      is fixed by different means.

    Reported as a first-class metric since junction protection so the two are never again
    averaged into one number.
    """
    edge_map = _edge_to_triangle_map(triangles)
    counts: Counter[int] = Counter()
    for tri_ids in edge_map.values():
        if len(tri_ids) < 4:
            continue
        materials: set[int] = set()
        for tri in tri_ids:
            materials.update(int(x) for x in labels[tri])
        counts[len(materials)] += 1
    return {
        "n_valence_geq4_edges": int(sum(counts.values())),
        "n_valence_geq4_with_4plus_materials": int(sum(v for k, v in counts.items() if k >= 4)),
        "n_valence_geq4_with_3_materials": int(counts.get(3, 0)),
        "valence_geq4_material_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
    }


# ---------------------------------------------------------------------------
# Junction metrics
# ---------------------------------------------------------------------------


def junction_preservation_stats(
    families: dict,
    tesselation_graph: TesselationGraph,
    points: NDArray[np.float64],
    triangles: NDArray,
) -> dict:
    """How much of the sampled junction network survives into the tesselation and the mesh.

    **This is measured, not assumed.** Boltcheva's protecting balls
    guarantee that consecutive samples of a protected 1-feature are joined in the weighted
    Delaunay triangulation, but that guarantee is stated for Delaunay *refinement*; `dw3d`
    is one-shot, so the construction gives a strong bias and no guarantee.

    Three fractions, in increasing order of what they demand:

    * `vertex_preservation` — junction samples that are still vertices of the *final mesh*.
      A sample can be a tesselation vertex and still be dropped by `filter_unused_points` if
      the watershed puts no interface through it.
    * `tesselation_edge_preservation` — sampled consecutive pairs that are edges of some
      tetrahedron. This is the quantity Boltcheva's theorem is about.
    * `mesh_edge_preservation` — sampled consecutive pairs that are edges of a mesh triangle.
      This is the one that matters for the output, and it is bounded above by the previous
      two: a pair can be a tesselation edge and still not lie on the reconstructed interface.

    Matching is by exact voxel coordinate: junction samples are integer-valued and both the
    tesselation vertices and the mesh points carry those coordinates verbatim (nothing in
    `run_case`'s path rescales or recentres the mesh).
    """
    junction_points = np.asarray(families["junction_points"], dtype=np.int64)
    pairs = np.asarray(families["junction_pairs"], dtype=np.int64)
    result = {
        "n_junction_samples": len(junction_points),
        "n_sampled_pairs": len(pairs),
        "n_corner_samples": int(np.count_nonzero(families["is_corner_sample"])),
    }
    if len(junction_points) == 0:
        return result

    vertices = np.rint(tesselation_graph.vertices).astype(np.int64)
    vertex_index = {tuple(int(v) for v in row): i for i, row in enumerate(vertices)}
    mesh_index = {tuple(int(v) for v in row): i for i, row in enumerate(np.rint(points).astype(np.int64))}
    sample_keys = [tuple(int(v) for v in row) for row in junction_points]

    in_tesselation = [key in vertex_index for key in sample_keys]
    in_mesh = [key in mesh_index for key in sample_keys]
    result["tesselation_vertex_preservation"] = float(np.mean(in_tesselation))
    result["vertex_preservation"] = float(np.mean(in_mesh))

    if len(pairs) == 0:
        return result

    tetrahedra = tesselation_graph.tetrahedrons
    tet_edges = {
        (min(a, b), max(a, b))
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for a, b in zip(tetrahedra[:, i].tolist(), tetrahedra[:, j].tolist(), strict=True)
    }
    mesh_edges = set(_edge_to_triangle_map(triangles))

    n_tesselation_edges = 0
    n_mesh_edges = 0
    for first, second in pairs.tolist():
        key_a, key_b = sample_keys[first], sample_keys[second]
        index_a, index_b = vertex_index.get(key_a), vertex_index.get(key_b)
        if index_a is not None and index_b is not None and (min(index_a, index_b), max(index_a, index_b)) in tet_edges:
            n_tesselation_edges += 1
        mesh_a, mesh_b = mesh_index.get(key_a), mesh_index.get(key_b)
        if mesh_a is not None and mesh_b is not None and (min(mesh_a, mesh_b), max(mesh_a, mesh_b)) in mesh_edges:
            n_mesh_edges += 1

    result["tesselation_edge_preservation"] = float(n_tesselation_edges / len(pairs))
    result["mesh_edge_preservation"] = float(n_mesh_edges / len(pairs))
    return result


def attributed_triple_line_angles(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> dict:
    """Length-weighted mean dihedral angle per (triple line, interface pair), in degrees.

    `dihedral_angle_stats_per_triple_line` deliberately averages the three angles at a
    trijunction edge together, because the QA metric did not need to know which angle
    belonged to which pair of films. Comparing a reconstruction's junction angles against a
    reference *does* need it, so this function does the attribution that the older one leaves
    open: each incident triangle is identified by its own label pair (available
    directly in `labels`), and the angle between two incident triangles is attributed to the
    unordered pair of their label pairs.

    Returns `{(a, b, c): {"ab|ac": degrees, ...}}` keyed by the sorted material triple, with
    the inner keys being the two interfaces written as sorted label pairs joined by `|`.
    Only edges with exactly 3 incident triangles spanning exactly 3 materials are used, so a
    valence->=4 defect contributes nothing rather than contributing nonsense.
    """
    edge_map = _edge_to_triangle_map(triangles)
    accumulated: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}

    for (p1, p2), tri_ids in edge_map.items():
        if len(tri_ids) != 3:
            continue
        pairs = [tuple(sorted(int(x) for x in labels[tri])) for tri in tri_ids]
        materials = sorted({m for pair in pairs for m in pair})
        if len(materials) != 3 or len(set(pairs)) != 3:
            continue

        e = points[p2] - points[p1]
        length = float(np.linalg.norm(e))
        if length == 0:
            continue
        apexes = []
        for tri in tri_ids:
            third = next(v for v in triangles[tri] if v not in (p1, p2))
            u = points[third] - points[p1]
            u_perp = u - (np.dot(u, e) / (length * length)) * e
            norm = np.linalg.norm(u_perp)
            apexes.append(None if norm == 0 else u_perp / norm)
        if any(a is None for a in apexes):
            continue

        line = accumulated.setdefault(tuple(materials), {})
        for i in range(3):
            for j in range(i + 1, 3):
                key = "|".join(sorted((f"{pairs[i][0]},{pairs[i][1]}", f"{pairs[j][0]},{pairs[j][1]}")))
                angle = float(np.degrees(np.arccos(np.clip(np.dot(apexes[i], apexes[j]), -1.0, 1.0))))
                line.setdefault(key, []).append((angle, length))

    return {
        key: {
            pair_key: float(np.average([a for a, _ in values], weights=[w for _, w in values]))
            for pair_key, values in line.items()
        }
        for key, line in accumulated.items()
    }


def neumann_angles_from_tensions(tensions: dict) -> dict:
    """Exact equilibrium dihedral angles at every triple line, from the ground-truth tensions.

    At a triple line where films `(a,b)`, `(b,c)` and `(a,c)` meet, force balance in the
    plane normal to the line is `sum_k gamma_k t_k = 0` with `t_k` the unit in-plane vector
    along film `k` away from the line. Squaring `gamma_ac t_ac = -(gamma_ab t_ab +
    gamma_bc t_bc)` gives Neumann's law in the form used here:

        cos(angle between films ab and bc) = (gamma_ac^2 - gamma_ab^2 - gamma_bc^2)
                                             / (2 gamma_ab gamma_bc)

    This is an **analytic, mesh-free, gauge-free** reference: the angles depend only on
    tension *ratios*, so no gauge matching is needed. It is
    exact for a foam at equilibrium; `benchmarking-dataset`'s meshes deviate from Neumann by
    0.03-0.59 degrees (foambryo's dataset-audit work), which is one to two orders below
    the errors measured here.

    Returns `{(a, b, c): {"ab|ac": degrees, ...}}` in the same shape as
    `attributed_triple_line_angles`, so the two can be differenced key by key. A triple whose
    tensions violate the triangle inequality has no equilibrium configuration and is omitted.
    """
    lookup = {tuple(sorted(k)): float(v) for k, v in tensions.items()}
    materials = sorted({m for key in lookup for m in key})
    out: dict[tuple[int, int, int], dict[str, float]] = {}
    for i, a in enumerate(materials):
        for j, b in enumerate(materials[i + 1 :], start=i + 1):
            for c in materials[j + 1 :]:
                sides = {
                    "ab": lookup.get((a, b)),
                    "ac": lookup.get((a, c)),
                    "bc": lookup.get((b, c)),
                }
                if any(v is None for v in sides.values()):
                    continue
                angles = {}
                names = {"ab": (a, b), "ac": (a, c), "bc": (b, c)}
                for first, second, opposite in (("ab", "ac", "bc"), ("ab", "bc", "ac"), ("ac", "bc", "ab")):
                    g1, g2, g3 = sides[first], sides[second], sides[opposite]
                    cosine = (g3 * g3 - g1 * g1 - g2 * g2) / (2 * g1 * g2)
                    if not -1.0 <= cosine <= 1.0:  # no equilibrium triangle for these tensions
                        angles = {}
                        break
                    first_name = f"{names[first][0]},{names[first][1]}"
                    second_name = f"{names[second][0]},{names[second][1]}"
                    key = "|".join(sorted((first_name, second_name)))
                    angles[key] = float(np.degrees(np.arccos(cosine)))
                if angles:
                    out[(a, b, c)] = angles
    return out


def compare_angle_dicts(measured: dict, reference: dict) -> dict:
    """Absolute angle error, in degrees, between two attributed-angle dicts.

    Only `(triple line, interface pair)` keys present in both are compared, so a triple line
    the reconstruction failed to produce is reported as *missing* rather than silently
    scored as zero error. Both the per-angle median and the per-triple-line median-of-medians
    are returned; the latter is the headline, because a triple line with many edges must not
    outvote a short one.
    """
    per_line_medians: list[float] = []
    all_errors: list[float] = []
    n_common_lines = 0
    for key, reference_angles in reference.items():
        measured_angles = measured.get(key)
        if not measured_angles:
            continue
        errors = [
            abs(measured_angles[pair_key] - value)
            for pair_key, value in reference_angles.items()
            if pair_key in measured_angles
        ]
        if not errors:
            continue
        n_common_lines += 1
        all_errors.extend(errors)
        per_line_medians.append(float(np.median(errors)))
    return {
        "n_reference_triple_lines": len(reference),
        "n_measured_triple_lines": len(measured),
        "n_common_triple_lines": n_common_lines,
        "angle_error_median_deg": float(np.median(per_line_medians)) if per_line_medians else None,
        "angle_error_mean_deg": float(np.mean(all_errors)) if all_errors else None,
        "angle_error_p90_deg": float(np.percentile(all_errors, 90)) if all_errors else None,
    }


def similarity_to_mask_frame(
    mask: NDArray,
    reference_points: NDArray[np.float64],
    reference_triangles: NDArray,
    reference_labels: NDArray,
) -> dict:
    """Fit the similarity taking a ground-truth `.rec` mesh into the mask's voxel frame.

    `benchmarking-dataset`'s `.rec` meshes are stored in a normalised frame with the axes in
    the opposite order to the `.tif` (measured: the fitted rotation has determinant `-1` on
    every case tried, i.e. it is the axis reversal), so a reconstruction in voxel coordinates
    and its ground truth cannot be compared metrically without this fit. Angles do not need
    it — a similarity preserves them — but **lengths do**.

    The correspondences are label-keyed, per the harness's standing rule (`AUDIT.md`
    Part III.3): each material's centroid, plus each interface's centroid. Mask-side
    interface centroids come from the dual-lattice label multiplicity (`|L| == 2` positions,
    with the pair read off as the min and max of the 2x2x2 block), which is free here because
    junction protection computes that array anyway; mesh-side ones are the mean of the interface's
    triangle vertices. The fit itself is Umeyama's closed-form similarity, with reflection
    allowed because the axis order genuinely differs.

    **This is independent of the algorithm being measured** — it registers the ground truth
    to the *mask* — so the same scale is applied to every variant and a variant cannot
    improve its score by moving the fit. Measured residuals on cases 000/007/023/040:
    mean 0.9-1.5 voxels out of a ~120-voxel object, and the isotropic scale is stable to
    0.5-1.5 % against a cells-only fit. That residual is centroid-definition noise (a
    dilated-voxel centroid is not a surface-area centroid); it bounds the systematic error on
    any junction-*length* comparison at the ~1 % level, well below the differences being
    judged, and it is common to every variant.

    Returns a dict with `scale`, `rotation` (3x3), `translation`, `n_correspondences` and the
    residual statistics, or `None` values when there are too few correspondences to fit.
    """
    source, destination = _label_keyed_correspondences(mask, reference_points, reference_triangles, reference_labels)
    if len(source) < 4:
        return {"scale": None, "n_correspondences": len(source)}

    mu_source, mu_destination = source.mean(axis=0), destination.mean(axis=0)
    centred_source, centred_destination = source - mu_source, destination - mu_destination
    u, singular, vt = np.linalg.svd(centred_destination.T @ centred_source / len(source))
    rotation = u @ vt
    scale = float(singular.sum() / ((centred_source**2).sum() / len(source)))
    translation = mu_destination - scale * rotation @ mu_source
    residual = np.linalg.norm((scale * (rotation @ source.T)).T + translation - destination, axis=1)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "n_correspondences": len(source),
        "residual_mean_voxels": float(residual.mean()),
        "residual_max_voxels": float(residual.max()),
    }


def _label_keyed_correspondences(
    mask: NDArray,
    reference_points: NDArray[np.float64],
    reference_triangles: NDArray,
    reference_labels: NDArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Matched (reference-frame, mask-frame) centroids: one per material, one per interface."""
    from dw3d.junctions import (
        label_multiplicity,
    )

    materials = [int(v) for v in np.unique(mask) if v != 0]
    source: list[NDArray[np.float64]] = []
    destination: list[NDArray[np.float64]] = []

    for material in materials:
        incident = reference_triangles[(reference_labels[:, 0] == material) | (reference_labels[:, 1] == material)]
        voxels = np.argwhere(mask == material)
        if len(incident) == 0 or len(voxels) == 0:
            continue
        source.append(reference_points[np.unique(incident)].mean(axis=0))
        destination.append(voxels.mean(axis=0))

    nx, ny, nz = (dimension - 1 for dimension in mask.shape)
    corners = np.stack(
        [mask[i : i + nx, j : j + ny, k : k + nz] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
    )
    is_interface = label_multiplicity(mask) == 2
    if is_interface.any():
        low, high = corners.min(axis=0)[is_interface], corners.max(axis=0)[is_interface]
        positions = np.argwhere(is_interface) + 0.5  # dual lattice -> voxel frame
        for a, b in sorted(set(zip(low.tolist(), high.tolist(), strict=True))):
            selected = (low == a) & (high == b)
            if selected.sum() < 20:
                continue
            matching = ((reference_labels[:, 0] == a) & (reference_labels[:, 1] == b)) | (
                (reference_labels[:, 0] == b) & (reference_labels[:, 1] == a)
            )
            if matching.sum() < 5:
                continue
            source.append(reference_points[reference_triangles[matching]].reshape(-1, 3).mean(axis=0))
            destination.append(positions[selected].mean(axis=0))

    if not source:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(source, dtype=np.float64), np.array(destination, dtype=np.float64)


def junction_length_error(measured: dict, reference: dict) -> dict:
    """Relative triple-line-length error per material triple, against a reference mesh.

    Both arguments are `{(a, b, c): total length}` dicts in the *same* frame — apply
    `similarity_to_mask_frame`'s scale to the reference first. Reported per triple line
    (median and p90 of `|L - L_ref| / L_ref`) and in total (`sum L / sum L_ref - 1`), over
    the triples present in both, plus how many reference triples the reconstruction missed
    entirely. A missing triple is a 100 % length error but is counted separately rather than
    folded in, so the two failure modes stay distinguishable.
    """
    common = [key for key in reference if key in measured]
    relative = [abs(measured[key] - reference[key]) / reference[key] for key in common if reference[key] > 0]
    total_measured = sum(measured[key] for key in common)
    total_reference = sum(reference[key] for key in common)
    return {
        "n_reference_triple_lines": len(reference),
        "n_measured_triple_lines": len(measured),
        "n_common_triple_lines": len(common),
        "n_missing_triple_lines": len(reference) - len(common),
        "length_error_median": float(np.median(relative)) if relative else None,
        "length_error_p90": float(np.percentile(relative, 90)) if relative else None,
        "total_length_ratio": float(total_measured / total_reference) if total_reference > 0 else None,
    }
