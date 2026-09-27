#!/usr/bin/env python
r"""The residual abnormal non-manifold edges, dumped as inspectable local patches.

**What is being made inspectable, and why.** The current default leaves **6** abnormal
non-manifold edges over the 51-case benchmark, one per case, each of valence exactly 4. Some of
them are a new category — a 2-material valence-4 edge, i.e. a single film pinched against
itself — and whether that category is genuine foam topology the pipeline cannot yet represent,
or a reconstruction artefact, had not been determined either way when this script was written.
That is a question about a *local* piece of geometry, and until this script nothing let anyone
look at one. (Run on the current default, it finds **2** of the 6 in the pinched-film category,
cases 030 and 032, and only 032 sits far from any junction of the mask: 10.43 voxels from the
nearest voxel where three materials meet.)

For every surviving abnormal edge, on the variant asked for, it writes:

* **`*_patch.rec` / `*_patch.vtk`** — the offending edge, its incident triangles, and two rings
  of neighbouring triangles, as a standalone multimaterial mesh. Per-face cell data marks the
  ring each triangle belongs to (`ring` = 0 for the incident fan, 1 and 2 for the two rings) and
  its material pair, so the patch can be read in ParaView without the manifest.
* **`*_edge.vtk`** — the offending edge alone, as a 1-cell line mesh, so it can be overlaid on
  the patch and there is no ambiguity about which edge is meant.
* **PNGs from three fixed angles** with the offending edge drawn as a thick highlighted curve
  and the incident fan colour-coded by material pair.
* **A diagnosis record**: the case, the distance to the nearest true 1-stratum and 0-stratum of
  the *mask* (i.e. how far the defect sits from any genuine junction, independent of the
  reconstruction), how many distinct materials meet there, whether it is one of the
  "2-material valence-4" pinched-film cases or the other kind, and what
  `dw3d.mesh_surgery`'s two repair searches return on its tetrahedron label cycle — so "surgery
  refused" is *shown* rather than asserted.

**Baselines, so there is something to compare against by eye.** For each case that carries a
defect, the same patch extraction is run on one **normal trijunction edge** and one **normal
quadruple-point edge** from the same mesh, chosen as the median-length edge of each kind so they
are typical rather than picked. Without those, a reader has no way to tell whether a patch looks
odd because it *is* odd or because all local patches look like that.

**This script presents evidence; it does not argue.** The classification it emits
(`n_materials`, `valence`, `dist_to_1_stratum`, whether the label cycle admits a repair) is
mechanical. Any reading of it belongs in a separate write-up, marked as such.

The diagnosis fields reproduce `benchmarks/diagnose_abnormal_edges.py` and one bug in it is
fixed here rather than inherited: that script only computes junction samples when the *variant
name* starts with `junction_protected`, so on `--variant default` — which **is** boundary layer
+ junction protection + link-checked offset exclusion — every edge was reported
`endpoints_are_junction_samples = [False, False]` whether or not it was. Here the junction
families are recomputed whenever the algorithm's point placer actually has
junction-protection keywords, so the answer does not depend on what the variant is called.

Usage (from `delaunay-watershed-3d/`)::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.abnormal_edge_patches \
        --variant default --out ../abnormal_edge_diagnostics/default

    # restrict to the cases a prior diagnosis already found defects in (much faster):
    PYTHONPATH="src:." .venv/bin/python -m benchmarks.abnormal_edge_patches \
        --variant default --only 007 026 --out ../abnormal_edge_diagnostics/default
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d_benchmarks.run_case import VARIANT_GETTERS
from dw3d.io import load_rec, save_rec
from dw3d.junctions import label_multiplicity
from dw3d.mesh_surgery import (
    _find_abnormal_non_manifold_edges,
    _find_candidate_for_label_switching,
    _find_candidate_for_two_labels_switching,
    _number_of_adjacent_cells_of_edge,
)
from dw3d.points_on_edt import junction_protected_families

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DATASET_DIR = WORKSPACE_ROOT / "benchmarking-dataset"
IN_REPO_IMAGES = [REPO_ROOT / "data" / "Images" / f"{i}.tif" for i in range(1, 5)]
MIN_DISTANCE = 3

#: Rings of neighbouring triangles kept around the offending edge, beyond the incident fan.
N_RINGS = 2

#: Fixed camera directions for the patch renders. A patch is a small, roughly isotropic blob, so
#: unlike the baseline panel rows these can be spread over the sphere; they are fixed (not framed per
#: patch) so two patches are photographed from the same three angles and are comparable by eye.
VIEWS: dict[str, tuple[float, float, float]] = {
    "a": (0.00, 0.35, 0.94),
    "b": (0.87, 0.30, -0.39),
    "c": (-0.55, -0.75, 0.37),
}


class ArtefactError(RuntimeError):
    """Raised when an artefact this script claims to have written is empty or degenerate."""


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def incident_triangles(triangles: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    """Indices of every triangle containing both endpoints of `edge`.

    Args:
        triangles (np.ndarray): `(nf, 3)` triangles.
        edge (tuple[int, int]): The two vertex indices.

    Returns:
        np.ndarray: Triangle indices, ascending.
    """
    a, b = int(edge[0]), int(edge[1])
    return np.flatnonzero((triangles == a).any(axis=1) & (triangles == b).any(axis=1))


def grow_rings(triangles: np.ndarray, seed_faces: np.ndarray, n_rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Grow a vertex-adjacent neighbourhood outwards from `seed_faces`.

    Vertex adjacency rather than edge adjacency, deliberately: the defect *is* an edge-adjacency
    anomaly, so growing by shared edges would follow the anomaly instead of describing the
    geometry around it, and on a non-manifold edge "the neighbouring triangle" is not
    well defined. Growing by shared vertices gives the complete local disc regardless.

    Args:
        triangles (np.ndarray): `(nf, 3)` triangles.
        seed_faces (np.ndarray): Face indices forming ring 0.
        n_rings (int): How many further rings to add.

    Returns:
        tuple[np.ndarray, np.ndarray]: The face indices kept (ascending), and the ring number of
            each of those faces.
    """
    ring_of: dict[int, int] = {int(f): 0 for f in seed_faces}
    frontier_vertices = set(np.unique(triangles[seed_faces]).tolist())
    for ring in range(1, n_rings + 1):
        touching = np.flatnonzero(np.isin(triangles, list(frontier_vertices)).any(axis=1))
        added = [int(f) for f in touching if int(f) not in ring_of]
        for f in added:
            ring_of[f] = ring
        if not added:
            break
        frontier_vertices = set(np.unique(triangles[added]).tolist())
    faces = np.array(sorted(ring_of), dtype=np.int64)
    rings = np.array([ring_of[int(f)] for f in faces], dtype=np.int64)
    return faces, rings


def extract_patch(
    points: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    edge: tuple[int, int],
    n_rings: int = N_RINGS,
) -> dict:
    """Cut a standalone local patch around one edge, reindexed to its own vertex set.

    Args:
        points (np.ndarray): `(nv, 3)` mesh vertices.
        triangles (np.ndarray): `(nf, 3)` triangles.
        labels (np.ndarray): `(nf, 2)` material pairs.
        edge (tuple[int, int]): The edge the patch is centred on.
        n_rings (int, optional): Rings beyond the incident fan. Defaults to `N_RINGS`.

    Returns:
        dict: `{"points", "triangles", "labels", "rings", "edge_local", "incident_faces",
            "n_incident", "valence"}` — a self-contained mesh plus the edge's local indices.

    Raises:
        ArtefactError: If the edge has no incident triangle (which would mean the edge list and
            the triangle list do not describe the same mesh).
    """
    incident = incident_triangles(triangles, edge)
    if len(incident) == 0:
        message = f"edge {edge} has no incident triangle in this mesh"
        raise ArtefactError(message)
    faces, rings = grow_rings(triangles, incident, n_rings)

    used_vertices = np.unique(triangles[faces])
    remap = {int(v): i for i, v in enumerate(used_vertices)}
    patch_triangles = np.vectorize(remap.__getitem__)(triangles[faces]).astype(np.int64)
    return {
        "points": np.asarray(points)[used_vertices].astype(np.float64),
        "triangles": patch_triangles,
        "labels": np.asarray(labels)[faces].astype(np.int64),
        "rings": rings,
        "edge_local": (remap[int(edge[0])], remap[int(edge[1])]),
        "incident_faces_local": np.flatnonzero(rings == 0).astype(np.int64),
        "n_incident": len(incident),
        "valence": len(incident),
    }


# ---------------------------------------------------------------------------
# Diagnosis (the mechanical description; no interpretation)
# ---------------------------------------------------------------------------


def nearest_stratum_distance(multiplicity: np.ndarray, point: np.ndarray, want: int, radius: int = 24) -> float:
    """Distance from `point` (voxel frame) to the nearest dual-lattice block of multiplicity `want`.

    Searched in a box of half-width `radius`; returns `inf` if there is none inside it, so a large
    answer reads as "no such stratum anywhere nearby" and not as a silently clipped number. The
    radius is 24 here rather than `diagnose_abnormal_edges.py`'s 12, because the claim being
    checked is that pinched-film edges can sit *more than 10 voxels* from any 1-stratum, and a
    12-voxel window cannot distinguish 11 from 40.

    Args:
        multiplicity (np.ndarray): `label_multiplicity(mask)`, on the dual lattice.
        point (np.ndarray): Query point in voxel coordinates.
        want (int): 3 for a 1-stratum (triple line), `>= 4` for a 0-stratum (quadruple point).
        radius (int, optional): Search half-width in voxels. Defaults to 24.

    Returns:
        float: Euclidean distance in voxels, or `inf`.
    """
    centre = np.rint(point).astype(np.int64)
    lo = np.maximum(centre - radius, 0)
    hi = np.minimum(centre + radius + 1, np.asarray(multiplicity.shape))
    if np.any(hi <= lo):
        return float("inf")
    window = multiplicity[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    hits = np.argwhere(window >= want) if want >= 4 else np.argwhere(window == want)
    if len(hits) == 0:
        return float("inf")
    positions = hits + lo + 0.5  # dual index (i,j,k) sits at voxel coordinate (i+.5, j+.5, k+.5)
    return float(np.min(np.linalg.norm(positions - point, axis=1)))


def classify(n_materials: int, valence: int) -> str:
    """Name the defect category, mechanically, from the two numbers that separate them.

    Args:
        n_materials (int): Distinct materials over the incident triangles' label pairs.
        valence (int): Number of incident triangles.

    Returns:
        str: The category name. The two *normal* configurations (a valence-3 edge where three
            materials meet, a valence-`k` edge where `k` materials meet) are named as normal, so
            the baseline patches do not read as anomalies in the index.
    """
    if valence == 3 and n_materials == 3:
        return "normal trijunction (3-material valence-3)"
    if valence == n_materials:
        return f"normal {n_materials}-fold junction ({n_materials}-material valence-{valence})"
    if valence == 4 and n_materials == 2:
        return "2-material valence-4 (film pinched against itself)"
    if valence == 4 and n_materials == 3:
        return "3-material valence-4 (pinched triple line)"
    return f"{n_materials}-material valence-{valence} (outside the named categories)"


def diagnose_edge(
    edge: tuple[int, int],
    points: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    multiplicity: np.ndarray,
    algo: object,
    junction_voxels: set[tuple[int, int, int]],
    kind: str,
) -> dict:
    """The mechanical description of one edge: geometry, materials, strata, surgery's verdict.

    Args:
        edge (tuple[int, int]): The edge.
        points (np.ndarray): Mesh vertices.
        triangles (np.ndarray): Mesh triangles.
        labels (np.ndarray): Mesh material pairs.
        multiplicity (np.ndarray): `label_multiplicity(mask)`.
        algo (object): The finished `MeshReconstructionAlgorithm`, for the tesselation graph.
        junction_voxels (set[tuple[int, int, int]]): The junction-protection samples, if this variant has
            any; used to say whether an endpoint is one.
        kind (str): `"abnormal"`, `"normal_trijunction"` or `"normal_quadruple"`.

    Returns:
        dict: Every field, JSON-safe.
    """
    which = incident_triangles(triangles, edge)
    pairs = [tuple(sorted(int(x) for x in labels[t])) for t in which]
    materials = sorted({m for pair in pairs for m in pair})
    p1, p2 = np.asarray(points)[int(edge[0])], np.asarray(points)[int(edge[1])]
    midpoint = 0.5 * (p1 + p2)

    # `filter_unused_points` runs after surgery, so the output mesh's point indices are not the
    # tesselation's. Map back by exact coordinate: mesh points are unrescaled copies of
    # tesselation vertices, so the match is exact and a miss is a real inconsistency, not a
    # tolerance question.
    vertex_index = {tuple(float(x) for x in row): i for i, row in enumerate(algo._tesselation_graph.vertices)}
    v1 = vertex_index.get(tuple(float(x) for x in p1))
    v2 = vertex_index.get(tuple(float(x) for x in p2))
    if v1 is None or v2 is None:
        tetra_cycle, label_cycle = [], []
    else:
        tetra_cycle = algo._tesselation_graph.find_tetra_cycle_around_edge((v1, v2))
        label_cycle = [int(x) for x in algo._map_node_id_to_label[tetra_cycle]]

    single = _find_candidate_for_label_switching(label_cycle)
    double = _find_candidate_for_two_labels_switching(label_cycle)
    return {
        "kind": kind,
        "edge": [int(edge[0]), int(edge[1])],
        "p1": [float(x) for x in p1],
        "p2": [float(x) for x in p2],
        "midpoint": [float(x) for x in midpoint],
        "edge_length_voxels": float(np.linalg.norm(p2 - p1)),
        "valence": len(which),
        "n_adjacent_cells_of_edge": int(_number_of_adjacent_cells_of_edge(triangles, labels, edge)),
        "n_materials": len(materials),
        "materials": materials,
        "interface_pairs": sorted(Counter("-".join(map(str, p)) for p in pairs).items()),
        "category": classify(len(materials), len(which)),
        "dist_to_1_stratum_voxels": nearest_stratum_distance(multiplicity, midpoint, want=3),
        "dist_to_0_stratum_voxels": nearest_stratum_distance(multiplicity, midpoint, want=4),
        "max_mask_multiplicity_within_3vox": int(
            multiplicity[
                max(int(midpoint[0]) - 3, 0) : int(midpoint[0]) + 4,
                max(int(midpoint[1]) - 3, 0) : int(midpoint[1]) + 4,
                max(int(midpoint[2]) - 3, 0) : int(midpoint[2]) + 4,
            ].max(),
        ),
        "endpoints_are_junction_samples": [
            tuple(round(float(x)) for x in p1) in junction_voxels,
            tuple(round(float(x)) for x in p2) in junction_voxels,
        ],
        "tetra_cycle_length": len(tetra_cycle),
        "label_cycle": label_cycle,
        "surgery_single_switch_candidates": [int(x) for x in single],
        "surgery_double_switch_candidates": [int(x) for x in double],
        "surgery_verdict": _surgery_verdict(label_cycle, single, double),
    }


def _surgery_verdict(label_cycle: list[int], single: list[int], double: list[int]) -> str:
    """State what `dw3d.mesh_surgery` did or refused to do on this label cycle, and why.

    Args:
        label_cycle (list[int]): Tetrahedron labels around the edge.
        single (list[int]): `_find_candidate_for_label_switching`'s result.
        double (list[int]): `_find_candidate_for_two_labels_switching`'s result.

    Returns:
        str: One sentence, mechanical.
    """
    if not label_cycle:
        return "no tetrahedron cycle recovered for this edge, so surgery was never offered a repair here"
    if single:
        return (
            f"a single-tetrahedron relabelling IS available ({len(single)} candidate block(s)); the edge "
            "survives because surgery stopped at max_iter, or because applying it recreates the defect "
            "elsewhere (a period-2 limit cycle between neighbouring edges)"
        )
    if double:
        return (
            f"no single-block relabelling works, but a two-label switch does ({len(double)} candidate(s)); "
            "the edge survives for the same max_iter/limit-cycle reason"
        )
    return (
        "neither repair search returns a candidate: no relabelling of one tetrahedron block, and no "
        "two-label switch, makes this cycle normal. Surgery declines, which is the correct behaviour for "
        "its vocabulary - a repair would need a whole-block merge "
        "of 2-3 tetrahedra, which moves cell volume and is not implemented"
    )


# ---------------------------------------------------------------------------
# Normal-junction baselines
# ---------------------------------------------------------------------------


def pick_normal_baseline_edges(triangles: np.ndarray, labels: np.ndarray, points: np.ndarray) -> dict:
    """Pick one *typical* normal trijunction edge and one normal quadruple-point edge.

    "Typical" is made concrete as the **median-length** edge of each kind, so the baseline is not
    cherry-picked to look good (or bad) next to the defects. A normal edge here means one whose
    incident-triangle count equals the number of cells meeting along it — the exact predicate
    `_find_abnormal_non_manifold_edges` uses to call an edge *not* abnormal — with valence 3 for a
    trijunction and 4 for a quadruple point.

    Args:
        triangles (np.ndarray): `(nf, 3)` triangles.
        labels (np.ndarray): `(nf, 2)` material pairs.
        points (np.ndarray): `(nv, 3)` vertices.

    Returns:
        dict: `{"normal_trijunction": (a, b) | None, "normal_quadruple": (a, b) | None}`, plus
            `"counts"` giving how many candidates each was chosen from.
    """
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    edges = np.sort(edges, axis=1)
    key = edges[:, 0] * (len(points) + 1) + edges[:, 1]
    unique_key, first, counts = np.unique(key, return_index=True, return_counts=True)
    del unique_key

    chosen: dict = {"counts": {}}
    for name, valence in (("normal_trijunction", 3), ("normal_quadruple", 4)):
        candidate_rows = first[counts == valence]
        candidates = []
        for row in candidate_rows:
            edge = (int(edges[row, 0]), int(edges[row, 1]))
            if _number_of_adjacent_cells_of_edge(triangles, labels, edge) == valence:
                candidates.append(edge)
        chosen["counts"][name] = len(candidates)
        if not candidates:
            chosen[name] = None
            continue
        lengths = np.array([np.linalg.norm(points[b] - points[a]) for a, b in candidates])
        chosen[name] = candidates[int(np.argsort(lengths)[len(lengths) // 2])]
    return chosen


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def _assert_patch_sane(name: str, patch: dict) -> dict:
    """Fail loudly if a patch about to be written is empty, degenerate or badly indexed.

    Args:
        name (str): Artefact name for the message.
        patch (dict): Output of `extract_patch`.

    Returns:
        dict: Counts and extrema worth recording (triangle areas, ring histogram).

    Raises:
        ArtefactError: On an empty patch, a non-finite coordinate, an out-of-range index, a
            repeated vertex inside a triangle, or an incident fan that lost its edge.
    """
    points, triangles = patch["points"], patch["triangles"]
    problems = []
    if len(points) == 0:
        problems.append("0 points")
    if len(triangles) == 0:
        problems.append("0 triangles")
    if len(points) and not np.isfinite(points).all():
        problems.append("non-finite coordinates")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(points)):
        problems.append(f"triangle index out of range [0, {len(points)})")
    if len(triangles) and (
        (triangles[:, 0] == triangles[:, 1]).any()
        or (triangles[:, 1] == triangles[:, 2]).any()
        or (triangles[:, 0] == triangles[:, 2]).any()
    ):
        problems.append("a triangle with a repeated vertex")
    a, b = patch["edge_local"]
    if len(incident_triangles(triangles, (a, b))) != patch["n_incident"]:
        problems.append("the patch's own incident-triangle count does not match the source mesh's")
    if problems:
        message = f"{name} is not a usable patch: {', '.join(problems)}"
        raise ArtefactError(message)

    p = points[triangles]
    areas = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return {
        "n_points": len(points),
        "n_triangles": len(triangles),
        "n_zero_area_triangles": int((areas == 0.0).sum()),
        "min_triangle_area": float(areas.min()),
        "total_area": float(areas.sum()),
        "ring_histogram": {str(k): int(v) for k, v in sorted(Counter(patch["rings"].tolist()).items())},
        "n_incident_triangles": patch["n_incident"],
        "bbox_extent_voxels": [float(x) for x in (points.max(axis=0) - points.min(axis=0))],
    }


def write_patch_rec(path: Path, patch: dict) -> dict:
    """Write the patch as a binary `.rec` and verify it round-trips."""
    save_rec(path, patch["points"], patch["triangles"], patch["labels"], binary_mode=True)
    back_points, back_triangles, back_labels = load_rec(path)
    if not (
        np.array_equal(back_triangles, patch["triangles"])
        and np.array_equal(back_labels, patch["labels"])
        and np.array_equal(back_points, patch["points"])
    ):
        message = f"{path} did not round-trip through load_rec"
        raise ArtefactError(message)
    return {"path": path.name, "bytes": int(path.stat().st_size)}


def write_patch_vtk(path: Path, patch: dict) -> dict:
    """Write the patch as an ASCII `.vtk` with the ring index and material pair as cell data."""
    import meshio

    cell_data = {
        "ring": [patch["rings"].astype(np.int32)],
        "label1": [patch["labels"][:, 0].astype(np.int32)],
        "label2": [patch["labels"][:, 1].astype(np.int32)],
        "is_incident_to_offending_edge": [(patch["rings"] == 0).astype(np.int32)],
    }
    meshio.Mesh(patch["points"], [("triangle", patch["triangles"])], cell_data=cell_data).write(path, binary=False)
    back = meshio.read(path)
    if len(back.get_cells_type("triangle")) != len(patch["triangles"]):
        message = f"{path} round-tripped the wrong triangle count"
        raise ArtefactError(message)
    return {"path": path.name, "bytes": int(path.stat().st_size),
            "cell_data_fields": ["ring", "label1", "label2", "is_incident_to_offending_edge"]}


def write_edge_vtk(path: Path, patch: dict) -> dict:
    """Write the offending edge alone as a 1-cell `line` mesh, for overlay."""
    import meshio

    a, b = patch["edge_local"]
    points = np.asarray([patch["points"][a], patch["points"][b]], dtype=np.float64)
    meshio.Mesh(points, [("line", np.array([[0, 1]]))]).write(path, binary=False)
    back = meshio.read(path)
    if len(back.get_cells_type("line")) != 1 or len(back.points) != 2:
        message = f"{path} did not round-trip as a single 2-point line"
        raise ArtefactError(message)
    return {"path": path.name, "bytes": int(path.stat().st_size)}


def assert_png_sane(path: Path, min_bytes: int = 3000) -> dict:
    """Reopen a PNG and check it is neither empty nor a single flat colour."""
    import matplotlib.image as mpimg

    if not path.exists():
        message = f"{path} was not written"
        raise ArtefactError(message)
    size = path.stat().st_size
    if size < min_bytes:
        message = f"{path} is {size} bytes, below the {min_bytes}-byte floor - almost certainly a blank render"
        raise ArtefactError(message)
    image = mpimg.imread(path)
    std = float(image[..., :3].std())
    flat = image[..., :3].reshape(-1, 3)
    stride = max(1, len(flat) // 20000)
    n_distinct = len(np.unique(np.round(flat[::stride] * 255).astype(np.uint8), axis=0))
    if std == 0.0 or n_distinct < 3:
        message = f"{path} is a flat image (std {std}, {n_distinct} sampled colours) - nothing rendered"
        raise ArtefactError(message)
    return {
        "path": path.name,
        "bytes": int(size),
        "shape": list(image.shape),
        "pixel_std": std,
        "n_distinct_colours_sampled": n_distinct,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def render_patch(patch: dict, diagnosis: dict, png_dir: Path, stem: str, resolution: tuple[int, int]) -> list[dict]:
    """Render one patch from the three fixed angles, with the offending edge highlighted.

    The incident fan is drawn opaque and coloured by its material pair; the outer rings are drawn
    translucent so the fan is not buried; the offending edge is a thick black curve network. That
    combination is what makes a pinched film distinguishable from a normal junction by eye: at a
    normal trijunction the three films leave the edge in three distinct directions, and at a
    pinched 2-material edge two of the four incident triangles carry the *same* label pair.

    Args:
        patch (dict): Output of `extract_patch`.
        diagnosis (dict): Output of `diagnose_edge`, used only for the structure names.
        png_dir (Path): Output directory.
        stem (str): File-name stem.
        resolution (tuple[int, int]): PNG size.

    Returns:
        list[dict]: One checked record per PNG.

    Raises:
        ArtefactError: If two views photograph identically (the camera never moved).
    """
    import matplotlib as mpl
    import polyscope as ps

    ps.init()
    ps.remove_all_structures()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("neg_y_up")
    ps.set_SSAA_factor(2)
    ps.set_window_size(*resolution)
    ps.set_transparency_mode("pretty")

    points, triangles, rings = patch["points"], patch["triangles"], patch["rings"]
    pairs = ["-".join(map(str, sorted(row))) for row in patch["labels"].tolist()]
    distinct = sorted(set(pairs))
    colour_of = {pair: mpl.colormaps["tab10"](i % 10)[:3] for i, pair in enumerate(distinct)}
    pair_colours = np.array([colour_of[p] for p in pairs])

    fan = np.flatnonzero(rings == 0)
    outer = np.flatnonzero(rings > 0)

    # The incident fan is the subject: opaque, saturated per material pair, wireframed, and shaded
    # identically front and back (an incident triangle of a non-manifold edge is routinely seen
    # from its far side, and the default policy would render it as a hole). The two outer rings are
    # context only, so they are a single neutral grey at high transparency -- giving them their own
    # material-pair colours, as a first attempt did, produces four overlapping translucent coloured
    # sheets in which the fan is invisible.
    fan_mesh = ps.register_surface_mesh("incident fan (ring 0)", points, triangles[fan], smooth_shade=False)
    fan_mesh.add_color_quantity("material pair", pair_colours[fan], defined_on="faces", enabled=True)
    fan_mesh.set_edge_width(1.4)
    fan_mesh.set_edge_color((0.05, 0.05, 0.05))
    fan_mesh.set_back_face_policy("identical")
    if len(outer):
        outer_mesh = ps.register_surface_mesh(
            f"neighbourhood (rings 1-{int(rings.max())})",
            points,
            triangles[outer],
            smooth_shade=False,
            color=(0.72, 0.74, 0.78),
        )
        outer_mesh.add_color_quantity("material pair", pair_colours[outer], defined_on="faces", enabled=False)
        outer_mesh.set_transparency(0.13)
        outer_mesh.set_edge_width(0.0)
        outer_mesh.set_back_face_policy("identical")
    else:
        outer_mesh = None

    a, b = patch["edge_local"]
    edge_points = np.asarray([points[a], points[b]])
    scene_extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0
    edge_net = ps.register_curve_network(
        f"offending edge ({diagnosis['category']})",
        edge_points,
        np.array([[0, 1]]),
        color=(0.05, 0.95, 0.25),
    )
    edge_net.set_radius(0.35 * float(np.linalg.norm(edge_points[1] - edge_points[0])) / scene_extent, relative=True)
    ps.register_point_cloud("edge endpoints", edge_points, color=(0.0, 0.0, 0.0), radius=0.012)

    up = np.array([0.0, -1.0, 0.0])
    aspect = resolution[0] / resolution[1]
    records: list[dict] = []
    seen: dict[str, str] = {}
    png_dir.mkdir(parents=True, exist_ok=True)
    # Framed on the incident fan, not the whole 2-ring patch: the fan is a handful of triangles
    # inside a neighbourhood an order of magnitude larger, so framing the patch shrinks the subject
    # to a few pixels. The rings still enter the frame as context, they just no longer set the zoom.
    subject = np.vstack([points[np.unique(triangles[fan])], edge_points])
    subject_centre = 0.5 * (subject.min(axis=0) + subject.max(axis=0))
    frame_points = subject_centre + 1.25 * (subject - subject_centre)
    for view_name, direction in VIEWS.items():
        camera, target = _camera_for(frame_points, direction, up, ps.get_vertical_fov_degrees(), aspect)
        ps.look_at(tuple(camera), tuple(target))
        # Two renders per angle: with the neighbourhood for context, and with the incident fan
        # alone. The fan-only render is the decisive one -- at a normal trijunction three
        # differently-coloured films leave the edge in three directions, at a 2-material valence-4
        # edge four *identically*-coloured triangles leave it in four -- and even a 13 %-opacity
        # neighbourhood tints it enough to make the colours harder to read.
        for suffix, show_outer in (("", True), ("_fanonly", False)):
            if outer_mesh is not None:
                outer_mesh.set_enabled(show_outer)
            path = png_dir / f"{stem}_{view_name}{suffix}.png"
            ps.screenshot(str(path), transparent_bg=False)
            record = assert_png_sane(path)
            record["view"] = view_name
            record["shows_neighbourhood"] = show_outer
            record["camera"] = [float(x) for x in camera]
            clash = seen.get(record["sha256"])
            if clash is not None:
                message = (
                    f"{path.name} is byte-identical to {clash} - either the camera did not move "
                    "between views or the neighbourhood was never drawn"
                )
                raise ArtefactError(message)
            seen[record["sha256"]] = path.name
            records.append(record)
    if outer_mesh is not None:
        outer_mesh.set_enabled(True)
    return records


def _camera_for(
    points: np.ndarray,
    direction: tuple[float, float, float],
    up: np.ndarray,
    fov_vertical_deg: float,
    aspect: float,
    margin: float = 1.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the camera so the whole patch fits, for a fixed view direction (see its baseline twin)."""
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    screen_up = np.cross(forward, right)
    target = 0.5 * (points.min(axis=0) + points.max(axis=0))
    centred = points - target
    tan_v = np.tan(np.radians(fov_vertical_deg) / 2.0)
    tan_h = aspect * tan_v
    offset_w = np.abs(centred @ right)
    offset_h = np.abs(centred @ screen_up)
    depth = centred @ forward
    distance = float(np.max(np.maximum(offset_w / tan_h, offset_h / tan_v) * margin + depth))
    return target + forward * distance, target


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def all_cases() -> list[tuple[str, Path]]:
    """Every benchmark case: the four in-repo images, then the 47 dataset masks."""
    cases = [(p.stem, p) for p in IN_REPO_IMAGES if p.exists()]
    cases += [(p.name.removesuffix("_labels_filled.tif"), p) for p in sorted(DATASET_DIR.glob("*_labels_filled.tif"))]
    return cases


def junction_voxels_of(algo: object, mask: np.ndarray) -> set[tuple[int, int, int]]:
    """The junction-protected samples for this algorithm, or an empty set if it does not place any.

    Keyed on whether the algorithm's point placer actually carries junction-protection keywords,
    **not** on the variant's name. `benchmarks/diagnose_abnormal_edges.py` keys on the name
    (`variant.startswith("junction_protected")`), which silently reports `[False, False]` for
    every edge on `--variant default` even though the default *is* boundary layer + junction
    protection + link-checked offset exclusion.

    Args:
        algo (object): The finished algorithm.
        mask (np.ndarray): The segmentation mask.

    Returns:
        set[tuple[int, int, int]]: Integer voxel coordinates of the junction samples.
    """
    keywords = getattr(algo.point_placing_function, "keywords", {}) or {}
    if "protect_radius" not in keywords and "junction_spacing" not in keywords:
        return set()
    families = junction_protected_families(
        mask,
        algo._edt_image,
        MIN_DISTANCE,
        delta=keywords.get("delta"),
        junction_spacing=keywords.get("junction_spacing"),
        shell_coarsening=keywords.get("shell_coarsening", 1),
        protect_radius=keywords.get("protect_radius"),
        junction_delta=keywords.get("junction_delta"),
        protect_junctions=keywords.get("protect_junctions", True),
        junction_boundary_layer=keywords.get("junction_boundary_layer", True),
    )
    return {tuple(int(v) for v in row) for row in np.asarray(families["junction_points"])}


def run_case(
    case: str,
    mask_path: Path,
    variant: str,
    out_dir: Path,
    write_png: bool,
    resolution: tuple[int, int],
) -> dict:
    """Reconstruct one case, and dump every abnormal edge plus the two normal baselines.

    Args:
        case (str): Case stem.
        mask_path (Path): The mask `.tif`.
        variant (str): A key of `VARIANT_GETTERS`.
        out_dir (Path): Artefact root.
        write_png (bool): Emit renders.
        resolution (tuple[int, int]): PNG size.

    Returns:
        dict: The case record; also written to `{case}/{case}_defects.json` when non-empty.
    """
    mask = io.imread(mask_path)
    algo = VARIANT_GETTERS[variant](min_distance=MIN_DISTANCE, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    abnormal = _find_abnormal_non_manifold_edges(points, triangles, labels)

    record: dict = {
        "case": case,
        "variant": variant,
        "min_distance": MIN_DISTANCE,
        "mask": {"path": mask_path.name, "shape": list(mask.shape),
                 "n_cells": len([v for v in np.unique(mask) if v != 0])},
        "mesh": {"n_points": len(points), "n_triangles": len(triangles)},
        "n_abnormal": len(abnormal),
        "entries": [],
    }
    if len(abnormal) == 0:
        return record

    multiplicity = label_multiplicity(mask)
    junction_voxels = junction_voxels_of(algo, mask)
    record["n_junction_samples"] = len(junction_voxels)

    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, tuple[int, int], str]] = [
        (f"{case}_abnormal{i:02d}", (int(e[0]), int(e[1])), "abnormal") for i, e in enumerate(abnormal)
    ]
    baselines = pick_normal_baseline_edges(triangles, labels, points)
    record["baseline_candidate_counts"] = baselines["counts"]
    for name in ("normal_trijunction", "normal_quadruple"):
        if baselines[name] is not None:
            targets.append((f"{case}_{name}", baselines[name], name))
        else:
            record.setdefault("baselines_unavailable", []).append(name)

    for stem, edge, kind in targets:
        patch = extract_patch(points, triangles, labels, edge)
        sanity = _assert_patch_sane(stem, patch)
        diagnosis = diagnose_edge(edge, points, triangles, labels, multiplicity, algo, junction_voxels, kind)
        entry = {
            **diagnosis,
            "stem": stem,
            "patch_sanity": sanity,
            "artefacts": {
                "rec": write_patch_rec(case_dir / f"{stem}_patch.rec", patch),
                "vtk": write_patch_vtk(case_dir / f"{stem}_patch.vtk", patch),
                "edge_vtk": write_edge_vtk(case_dir / f"{stem}_edge.vtk", patch),
            },
            "png": render_patch(patch, diagnosis, case_dir / "png", stem, resolution) if write_png else [],
        }
        record["entries"].append(entry)

    (case_dir / f"{case}_defects.json").write_text(json.dumps(_json_safe(record), indent=2) + "\n")
    return record


def _json_safe(obj: object) -> object:
    """Recursively make a record JSON-serialisable, turning non-finite floats into `null`."""
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=tuple(VARIANT_GETTERS), default="default")
    parser.add_argument("--only", nargs="*", default=None, help="Case stems to restrict to.")
    parser.add_argument("--out", type=Path, default=WORKSPACE_ROOT / "abnormal_edge_diagnostics" / "default")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--resolution", nargs=2, type=int, default=[1400, 1100])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for case, path in all_cases():
        if args.only and case not in args.only:
            continue
        record = run_case(case, path, args.variant, args.out, not args.no_png, tuple(args.resolution))
        records.append(record)
        if record["n_abnormal"]:
            print(f"{case}: {record['n_abnormal']} abnormal")
            for entry in record["entries"]:
                print(
                    f"    {entry['kind']:20s} {entry['category']:48s} "
                    f"valence={entry['valence']} materials={entry['materials']} "
                    f"len={entry['edge_length_voxels']:.2f} "
                    f"d1={entry['dist_to_1_stratum_voxels']} d0={entry['dist_to_0_stratum_voxels']} "
                    f"cycle={entry['label_cycle']} "
                    f"patch={entry['patch_sanity']['n_triangles']}tri/{entry['patch_sanity']['n_points']}pts",
                )
        else:
            print(f"{case}: clean")

    total = sum(r["n_abnormal"] for r in records)
    summary = {
        "variant": args.variant,
        "min_distance": MIN_DISTANCE,
        "n_cases": len(records),
        "n_abnormal_total": total,
        "cases_with_defects": [r["case"] for r in records if r["n_abnormal"]],
        "records": records,
    }
    (args.out / "d2_2_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    print(f"\n{total} abnormal edges over {len(records)} cases (variant={args.variant}); artefacts in {args.out}")


if __name__ == "__main__":
    main()
