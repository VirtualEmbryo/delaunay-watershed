#!/usr/bin/env python
"""Diagnose the abnormal non-manifold edges that survive `post_process_mesh_surgery`.

Boundary layer + junction protection leaves
**7** abnormal edges over the 51-case set against an acceptance target of <= 5 (down from
the boundary-layer-alone configuration's 327). This script answers the question the
junction-protection work left open: *what distinguishes the seven*, and is
the cause structural (a specific local topology) or incidental (a scattering of one-off
events)?

For each surviving abnormal edge it records, without assuming anything about the answer:

* **valence** (number of incident mesh triangles) and the **multiset of interface label
  pairs** of those triangles -- this is what separates a mis-resolved quadruple point
  (>= 4 materials) from a *pinched* triple line (3 materials) from a *doubled interface
  sheet* (2 materials);
* the **tetrahedron label cycle** around the edge, which is exactly the object
  `dw3d.mesh_surgery` tries to repair, plus a re-run of both of its repair searches
  (`_find_candidate_for_label_switching`, `_find_candidate_for_two_labels_switching`) so
  that "surgery could not fix it" is *shown* rather than inferred;
* whether either endpoint is a **junction sample**, and the distance from the edge to
  the nearest detected 0-stratum (|L| >= 4) and 1-stratum (|L| = 3) voxel of the label
  image -- i.e. whether the defect sits at a genuine quadruple point;
* the **label multiplicity** of the mask in a small ball around the edge midpoint, which
  says how many materials genuinely meet there, independent of the reconstruction.

Usage:
    PYTHONPATH="src:." .venv/bin/python -m benchmarks.diagnose_abnormal_edges \
        --variant junction_protected --out benchmarks/baseline/abnormal_edges_default.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d_benchmarks.run_case import VARIANT_GETTERS
from dw3d.junctions import label_multiplicity
from dw3d.mesh_surgery import (
    _find_abnormal_non_manifold_edges,
    _find_candidate_for_label_switching,
    _find_candidate_for_two_labels_switching,
)
from dw3d.points_on_edt import junction_protected_families

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DATASET_DIR = WORKSPACE_ROOT / "benchmarking-dataset"
IN_REPO_IMAGES = [REPO_ROOT / "data" / "Images" / f"{i}.tif" for i in range(1, 5)]
MIN_DISTANCE = 3


def _all_cases() -> list[tuple[str, Path]]:
    cases = [(p.stem, p) for p in IN_REPO_IMAGES if p.exists()]
    cases += [(p.stem, p) for p in sorted(DATASET_DIR.glob("*_labels_filled.tif"))]
    return cases


def _incident(triangles: np.ndarray, labels: np.ndarray, edge: tuple[int, int]) -> tuple[np.ndarray, list]:
    pid1, pid2 = int(edge[0]), int(edge[1])
    which = np.flatnonzero((triangles == pid1).any(axis=1) & (triangles == pid2).any(axis=1))
    pairs = [tuple(sorted(int(x) for x in labels[t])) for t in which]
    return which, pairs


def _nearest_stratum_distance(multiplicity: np.ndarray, point: np.ndarray, want: int, radius: int = 12) -> float:
    """Distance from `point` (voxel coords) to the nearest dual-lattice block with |L| == want.

    Searched in a box of half-width `radius` around the point; returns `inf` if none is
    within the box, so a large value means "no such stratum anywhere nearby" rather than a
    silently clipped number.
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
    # Dual index (i,j,k) sits at voxel coordinate (i+0.5, j+0.5, k+0.5).
    positions = hits + lo + 0.5
    return float(np.min(np.linalg.norm(positions - point, axis=1)))


def diagnose_case(case: str, mask_path: Path, variant: str) -> dict:
    """Reconstruct one case and describe every surviving abnormal non-manifold edge."""
    mask = io.imread(mask_path)
    algo = VARIANT_GETTERS[variant](min_distance=MIN_DISTANCE, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    abnormal = _find_abnormal_non_manifold_edges(points, triangles, labels)

    record: dict = {"case": case, "variant": variant, "n_abnormal": len(abnormal), "edges": []}
    if len(abnormal) == 0:
        return record

    multiplicity = label_multiplicity(mask)
    vertex_index = {
        tuple(float(x) for x in row): i for i, row in enumerate(algo._tesselation_graph.vertices)
    }
    junction_voxels: set[tuple[int, int, int]] = set()
    if variant.startswith("junction_protected"):
        keywords = getattr(algo.point_placing_function, "keywords", {})
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
        junction_voxels = {tuple(int(v) for v in row) for row in np.asarray(families["junction_points"])}

    for edge in abnormal:
        which, pairs = _incident(triangles, labels, edge)
        materials = sorted({m for pair in pairs for m in pair})
        p1, p2 = points[int(edge[0])], points[int(edge[1])]
        midpoint = 0.5 * (p1 + p2)

        # `filter_unused_points` runs *after* surgery, so the final mesh's point indices are
        # not the tesselation's. Map back by coordinate -- the mesh points are copies of
        # tesselation vertices, unrescaled, so the match is exact.
        v1 = vertex_index.get(tuple(float(x) for x in p1))
        v2 = vertex_index.get(tuple(float(x) for x in p2))
        if v1 is None or v2 is None:  # pragma: no cover - would mean the mesh was rescaled
            tetra_cycle, label_cycle = [], []
        else:
            tetra_cycle = algo._tesselation_graph.find_tetra_cycle_around_edge((v1, v2))
            label_cycle = [int(x) for x in algo._map_node_id_to_label[tetra_cycle]]

        record["edges"].append(
            {
                "edge": [int(edge[0]), int(edge[1])],
                "p1": [float(x) for x in p1],
                "p2": [float(x) for x in p2],
                "edge_length": float(np.linalg.norm(p2 - p1)),
                "valence": len(which),
                "n_materials": len(materials),
                "materials": materials,
                "interface_pairs": sorted(Counter("-".join(map(str, p)) for p in pairs).items()),
                "endpoints_are_junction_samples": [
                    tuple(round(x) for x in p1) in junction_voxels,
                    tuple(round(x) for x in p2) in junction_voxels,
                ],
                "dist_to_0_stratum": _nearest_stratum_distance(multiplicity, midpoint, want=4),
                "dist_to_1_stratum": _nearest_stratum_distance(multiplicity, midpoint, want=3),
                "max_multiplicity_within_3vox": int(
                    multiplicity[
                        max(int(midpoint[0]) - 3, 0) : int(midpoint[0]) + 4,
                        max(int(midpoint[1]) - 3, 0) : int(midpoint[1]) + 4,
                        max(int(midpoint[2]) - 3, 0) : int(midpoint[2]) + 4,
                    ].max(),
                ),
                "tetra_cycle_length": len(tetra_cycle),
                "label_cycle": label_cycle,
                "surgery_single_switch_candidates": _find_candidate_for_label_switching(label_cycle),
                "surgery_double_switch_candidates": _find_candidate_for_two_labels_switching(label_cycle),
            },
        )
    return record


def main() -> None:
    """Diagnose every case, or a filtered subset, and write one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANT_GETTERS), default="junction_protected")
    parser.add_argument("--only", nargs="*", default=None, help="case stems to restrict to")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for case, path in _all_cases():
        if args.only and case not in args.only:
            continue
        record = diagnose_case(case, path, args.variant)
        if record["n_abnormal"]:
            print(f"{case}: {record['n_abnormal']} abnormal")
            for e in record["edges"]:
                print(
                    f"    valence={e['valence']} materials={e['n_materials']}{e['materials']} "
                    f"len={e['edge_length']:.2f} cycle={e['label_cycle']} "
                    f"d0={e['dist_to_0_stratum']:.1f} d1={e['dist_to_1_stratum']:.1f} "
                    f"maxL={e['max_multiplicity_within_3vox']} "
                    f"jct_ends={e['endpoints_are_junction_samples']} "
                    f"pairs={e['interface_pairs']}",
                )
        records.append(record)

    total = sum(r["n_abnormal"] for r in records)
    print(f"\n{total} abnormal edges over {len(records)} cases (variant={args.variant})")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
