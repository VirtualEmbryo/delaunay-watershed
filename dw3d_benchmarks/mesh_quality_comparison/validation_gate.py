"""The four validation anchors: numbers this project already knows, from independent sources.

Anchors 3 and 4 are single-case facts, independent of cohort. Anchor 2 is a hard gate on the
cases actually run. Anchor 1 is a diagnostic: report it, note the cohort change, and continue
regardless of the outcome. Nothing here adjusts an
anchor to match a measurement -- a failing hard gate must stop the run, not be rewritten.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import skimage.io as io

from dw3d_benchmarks import metrics as dwm

from . import case_metrics, mesh_sets

ANCHOR_1_EXPECTED_PCT = 15.7
ANCHOR_2_MIN_DISTANCES = (2, 3, 4, 5, 7)


def _abnormal_edge_count_for(
    dataset_dir: str, case_id: str, configuration: str, min_distance: int,
) -> tuple[str, int, int]:
    mask = io.imread(Path(dataset_dir) / f"{case_id}_labels_filled.tif")
    algo = mesh_sets.get_algorithm(configuration, min_distance)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    n_abnormal = dwm.abnormal_non_manifold_edge_count(points, triangles, labels)
    return case_id, min_distance, n_abnormal


def check_anchor_2(dataset_dir: str, case_ids: list[str], *, configuration: str = "dithered", workers: int = 5) -> dict:
    """Population-level: zero abnormal non-manifold edges, all cases, md in {2,3,4,5,7}, default config.

    Hard gate, evaluated on the cases actually run (45 in the full run -- fewer than the
    original 47 the anchor's number came from).
    """
    t0 = time.perf_counter()
    jobs = [(case_id, md) for case_id in case_ids for md in ANCHOR_2_MIN_DISTANCES]
    offenders: list[dict] = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_abnormal_edge_count_for, dataset_dir, case_id, configuration, md): (case_id, md)
            for case_id, md in jobs
        }
        for future in as_completed(futures):
            case_id, min_distance, n_abnormal = future.result()
            n_done += 1
            if n_abnormal != 0:
                offenders.append({"case": case_id, "min_distance": min_distance, "n_abnormal": n_abnormal})
    wall = time.perf_counter() - t0
    reproduced = len(offenders) == 0
    return {
        "anchor": 2,
        "description": "zero abnormal non-manifold edges, all cases, min_distance in {2,3,4,5,7}, default config",
        "hard_gate": True,
        "configuration": configuration,
        "n_cases": len(case_ids),
        "min_distances": list(ANCHOR_2_MIN_DISTANCES),
        "n_reconstructions": len(jobs),
        "reproduced": reproduced,
        "offenders": offenders,
        "wall_time_s": wall,
    }


def check_anchor_3(
    dataset_dir: str, *, case_id: str = "011", min_distance: int = 3, configuration: str = "dithered",
) -> dict:
    """Single case: 0 abnormal edges under the current default at md=3 (was 1, pre-surgery-fixes)."""
    mask = io.imread(Path(dataset_dir) / f"{case_id}_labels_filled.tif")
    algo = mesh_sets.get_algorithm(configuration, min_distance)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    n_abnormal = dwm.abnormal_non_manifold_edge_count(points, triangles, labels)
    return {
        "anchor": 3,
        "description": (
            f"case {case_id} at md={min_distance}: 0 abnormal edges under the current default (was 1 pre-fix)"
        ),
        "hard_gate": True,
        "case": case_id,
        "min_distance": min_distance,
        "configuration": configuration,
        "value": n_abnormal,
        "expected": 0,
        "reproduced": n_abnormal == 0,
    }


def check_anchor_4(dataset_dir: str, *, case_id: str = "004") -> dict:
    """Single case: ground-truth mesh has exactly 2 malformed triple-line edges of 539."""
    from foambryo.validation.mesh_io import mesh_integrity

    from dw3d.io import load_rec

    v, f, regions = load_rec(Path(dataset_dir) / f"{case_id}_mesh.rec")
    report = mesh_integrity(v, f, regions)
    reproduced = report["n_triple_line_edges"] == 539 and report["n_malformed_triple_line_edges"] == 2
    return {
        "anchor": 4,
        "description": f"case {case_id} ground-truth mesh: exactly 2 malformed triple-line edges of 539",
        "hard_gate": True,
        "case": case_id,
        "n_triple_line_edges": report["n_triple_line_edges"],
        "n_malformed_triple_line_edges": report["n_malformed_triple_line_edges"],
        "expected": {"n_triple_line_edges": 539, "n_malformed_triple_line_edges": 2},
        "reproduced": reproduced,
    }


def check_anchor_1(dataset_dir: str, case_ids: list[str], *, configuration: str = "offset_included") -> dict:
    """Diagnostic only: +15.7% interface-area regression on offset_included vs ground truth.

    Measured on the full 47-case cohort originally; this run uses the cases actually run
    (45, or fewer under `--cases`), a cohort change that is expected to shift the number.
    Never a gate: report the value obtained and continue regardless of the outcome.
    """
    values = []
    for case_id in case_ids:
        gt = case_metrics.load_ground_truth(dataset_dir, case_id)
        if gt is None:
            continue
        mask = io.imread(Path(dataset_dir) / f"{case_id}_labels_filled.tif")
        algo = mesh_sets.get_algorithm(configuration, 3)
        points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
        block = case_metrics.compute_m2_interface_areas(points, triangles, labels, mask, gt)
        if "signed_relative_error_median" in block:
            values.append(block["signed_relative_error_median"])
    median_pct = 100 * float(np.median(values)) if values else None
    return {
        "anchor": 1,
        "description": "offset_included interface-area regression vs ground truth (expected +15.7%)",
        "hard_gate": False,
        "configuration": configuration,
        "n_cases": len(values),
        "expected_pct": ANCHOR_1_EXPECTED_PCT,
        "measured_pct": median_pct,
        "note": (
            "diagnostic only: the 15.7% figure was measured on the full 47-case cohort "
            "(including 031 and 038); this run excludes those two, so the number may "
            "legitimately shift. Not a gate; the run continues regardless."
        ),
    }


def run_all_anchors(dataset_dir: str, case_ids: list[str], *, workers: int = 5) -> dict:
    """Run anchors 3, 4 (cheap, single-case), then 2 (population, hard gate), then 1 (diagnostic).

    Ordered cheapest-and-most-decisive first: 3 and 4 are single reconstructions/mesh reads,
    so a hard-gate failure there is caught before spending time on anchor 2's ~225
    reconstructions or anchor 1's full-cohort pass.
    """
    anchor_3 = check_anchor_3(dataset_dir)
    anchor_4 = check_anchor_4(dataset_dir)
    if not (anchor_3["reproduced"] and anchor_4["reproduced"]):
        return {
            "anchor_3": anchor_3,
            "anchor_4": anchor_4,
            "anchor_2": None,
            "anchor_1": None,
            "all_hard_gates_passed": False,
            "stopped_early": True,
        }
    anchor_2 = check_anchor_2(dataset_dir, case_ids, workers=workers)
    if not anchor_2["reproduced"]:
        return {
            "anchor_3": anchor_3,
            "anchor_4": anchor_4,
            "anchor_2": anchor_2,
            "anchor_1": None,
            "all_hard_gates_passed": False,
            "stopped_early": True,
        }
    anchor_1 = check_anchor_1(dataset_dir, case_ids)
    return {
        "anchor_3": anchor_3,
        "anchor_4": anchor_4,
        "anchor_2": anchor_2,
        "anchor_1": anchor_1,
        "all_hard_gates_passed": True,
        "stopped_early": False,
    }
