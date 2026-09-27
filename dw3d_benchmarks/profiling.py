"""Cost and determinism profiling for the dw3d benchmark harness.

Kept separate from metrics.py because these functions re-run reconstruction multiple
times (expensive) rather than measuring a single already-constructed mesh.
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np
from numpy.typing import NDArray

from dw3d_benchmarks import metrics as m
from dw3d import get_default_mesh_reconstruction_algorithm, get_dithered_mesh_reconstruction_algorithm

# `ru_maxrss` is bytes on macOS (Darwin) and kilobytes on Linux.
_RSS_TO_MB = (1 / 1024**2) if sys.platform == "darwin" else (1 / 1024)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_TO_MB


def _algorithm(variant: str, min_distance: int):  # noqa: ANN202 - returns a MeshReconstructionAlgorithm
    from dw3d_benchmarks.run_case import VARIANT_GETTERS

    return VARIANT_GETTERS[variant](min_distance=min_distance, print_info=False)


def cost_profile(mask: NDArray[np.uint], min_distance: int, variant: str = "default") -> dict:
    """Overall wall time and peak RSS for one reconstruction, plus per-stage wall time.

    Per-stage timing replicates `MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`'s
    call sequence using the same algorithm's function references, purely for
    instrumentation; it does not replace the canonical run used for correctness metrics
    (see `run_full_case` in this package), and any of its own output is discarded.
    """
    algo = _algorithm(variant, min_distance)

    t_start = time.perf_counter()
    algo.construct_mesh_from_segmentation_mask(mask)
    total_wall_time_s = time.perf_counter() - t_start
    peak_rss_mb = _peak_rss_mb()

    # Second, separate instrumented pass: same function references, timed stage by stage.
    stage_algo = _algorithm(variant, min_distance)
    stage_times = {}

    t0 = time.perf_counter()
    edt_image = stage_algo.edt_creation_function(mask)
    stage_times["edt"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    placed = stage_algo.point_placing_function(mask, edt_image)
    stage_times["point_placing"] = time.perf_counter() - t0
    # Junction-protected point placement returns a third element, the per-point weights, and
    # offset exclusion's a fourth,
    # the point metadata; see `MeshReconstructionAlgorithm.construct_mesh_from_segmentation_mask`.
    points_for_tesselation, indices_of_sorted_maxes = placed[0], placed[1]
    point_weights = placed[2] if len(placed) > 2 else None

    t0 = time.perf_counter()
    tesselation_points, tesselation_tetrahedrons = stage_algo.tesselation_creation_function(
        points_for_tesselation,
        point_weights,
    )
    stage_times["tesselation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from dw3d.tesselation_graph import TesselationGraph

    tesselation_graph = TesselationGraph(
        tesselation_points,
        tesselation_tetrahedrons,
        indices_of_sorted_maxes,
        stage_algo.score_computation_function,
        edt_image,
        print_info=False,
    )
    stage_times["graph_and_scores"] = time.perf_counter() - t0

    return {
        "total_wall_time_s": total_wall_time_s,
        "peak_rss_mb": peak_rss_mb,
        "stage_wall_time_s": stage_times,
        "_stage_measurement_discarded_output": len(tesselation_graph.scores) >= 0,
    }


def determinism_profile(mask: NDArray[np.uint], min_distance: int, n_seeds: int = 20) -> dict:
    """Run the **v0.3 dithered** reconstruction over `n_seeds` dither seeds and report the spread.

    Reports the coefficient of variation of every interface area, cell volume and
    junction length that is present in every seed's mesh.

    This measures the *historical dithered* algorithm on purpose. The default algorithm
    has had no seed since point placement was made deterministic — its plateau tie-break
    is geometric — so a seed sweep of it would return zeros and say nothing. The sweep's
    remaining job is to supply the distribution against which the deterministic answer is
    judged representative rather than a lucky draw; `deterministic_vs_dither_sweep` does
    that comparison.

    Before point placement was made deterministic the seed was hard-wired as
    `np.random.seed(42)` inside `peak_local_points` and this function had to monkeypatch
    `numpy.random.seed` to move it. The dithered placement now takes `seed` as a
    parameter, so the sweep passes it directly and no longer patches numpy.
    """
    per_seed_interfaces: list[dict] = []
    per_seed_volumes: list[dict] = []
    per_seed_junctions: list[dict] = []
    mesh_sizes: list[tuple[int, int]] = []

    for seed in range(n_seeds):
        algo = get_dithered_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False, seed=seed)
        points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)

        per_seed_interfaces.append(m.interface_areas(points, triangles, labels))
        per_seed_volumes.append(m.cell_volumes_from_tetrahedra(algo._tesselation_graph, algo._map_label_to_nodes_ids))
        edge_stats = m.edge_topology_stats(points, triangles, labels)
        per_seed_junctions.append(edge_stats["triple_line_lengths"])
        mesh_sizes.append((len(points), len(triangles)))

    def _cv_per_key(dicts: list[dict]) -> dict:
        common_keys = set.intersection(*(set(d) for d in dicts)) if dicts else set()
        result = {}
        for key in common_keys:
            values = np.array([d[key] for d in dicts])
            mean = np.mean(values)
            result[str(key)] = float(np.std(values) / mean) if mean != 0 else float("nan")
        return result

    return {
        "n_seeds": n_seeds,
        "mesh_size_range": {
            "n_points": [min(s[0] for s in mesh_sizes), max(s[0] for s in mesh_sizes)],
            "n_triangles": [min(s[1] for s in mesh_sizes), max(s[1] for s in mesh_sizes)],
        },
        "interface_area_cv": _cv_per_key(per_seed_interfaces),
        "cell_volume_cv": _cv_per_key(per_seed_volumes),
        "junction_length_cv": _cv_per_key(per_seed_junctions),
        "n_common_interfaces": len(set.intersection(*(set(d) for d in per_seed_interfaces))),
        "n_common_junctions": len(set.intersection(*(set(d) for d in per_seed_junctions))),
    }


def _measure(algo, points, triangles, labels) -> dict[str, dict]:  # noqa: ANN001
    """The three per-key metric families the determinism comparison is built on."""
    return {
        "interface_area": m.interface_areas(points, triangles, labels),
        "cell_volume": m.cell_volumes_from_tetrahedra(algo._tesselation_graph, algo._map_label_to_nodes_ids),
        "junction_length": m.edge_topology_stats(points, triangles, labels)["triple_line_lengths"],
    }


def deterministic_vs_dither_sweep(mask: NDArray[np.uint], min_distance: int, n_seeds: int = 20) -> dict:
    """Locate the deterministic answer inside the historical dithered distribution.

    The determinism fix's acceptance criterion is not "the deterministic mesh equals the dithered
    mesh" — it cannot, since the dithered mesh is a different one for every seed. It is
    that the deterministic choice is **representative** of that distribution rather than
    sitting at its edge, i.e. the geometric tie-break is not a lucky or unlucky draw.

    For each interface area, cell volume and triple-line length present in all `n_seeds`
    dithered meshes, this reports the dither's mean/std/min/max and the deterministic
    value's position: `z = (deterministic - mean) / std` and whether it lies inside the
    sweep's observed range. Keys present in one but not the other are reported separately,
    because a *topology* difference (a triple line that exists in one and not the other)
    is a different and more serious kind of disagreement than a numerical one — the original
    determinism analysis recorded exactly that happening between dither seeds (on `3.tif`,
    triple line (0,2,5) disappears at seed 2).

    Read `frac_inside_range` and the `|z|` quantiles together with `dither_cv`: a large
    `|z|` on an interface whose dither CV is 1e-4 is a much weaker signal than the same
    `|z|` on one whose CV is 7 %.
    """
    per_seed: list[dict[str, dict]] = []
    for seed in range(n_seeds):
        algo = get_dithered_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False, seed=seed)
        per_seed.append(_measure(algo, *algo.construct_mesh_from_segmentation_mask(mask)))

    algo = get_default_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False)
    deterministic = _measure(algo, *algo.construct_mesh_from_segmentation_mask(mask))

    out: dict = {"n_seeds": n_seeds}
    for family in ("interface_area", "cell_volume", "junction_length"):
        common = set.intersection(*(set(s[family]) for s in per_seed))
        det_keys = set(deterministic[family])
        shared = sorted(common & det_keys, key=str)

        per_key = {}
        z_scores = []
        for key in shared:
            values = np.array([s[family][key] for s in per_seed], dtype=float)
            mean, std = float(values.mean()), float(values.std())
            value = float(deterministic[family][key])
            z = (value - mean) / std if std > 0 else float("inf") if value != mean else 0.0
            z_scores.append(abs(z))
            per_key[str(key)] = {
                "dither_mean": mean,
                "dither_cv": std / mean if mean != 0 else float("nan"),
                "dither_min": float(values.min()),
                "dither_max": float(values.max()),
                "deterministic": value,
                "z": z,
                "inside_range": bool(values.min() <= value <= values.max()),
                "relative_deviation": (value - mean) / mean if mean != 0 else float("nan"),
            }

        finite_z = [z for z in z_scores if np.isfinite(z)]
        out[family] = {
            "per_key": per_key,
            "n_compared": len(shared),
            "n_missing_from_deterministic": len(common - det_keys),
            "n_only_in_deterministic": len(det_keys - common),
            "missing_from_deterministic": sorted(map(str, common - det_keys)),
            "only_in_deterministic": sorted(map(str, det_keys - common)),
            "frac_inside_range": float(np.mean([v["inside_range"] for v in per_key.values()])) if per_key else None,
            "abs_z_median": float(np.median(finite_z)) if finite_z else None,
            "abs_z_p90": float(np.quantile(finite_z, 0.9)) if finite_z else None,
            "abs_z_max": float(np.max(finite_z)) if finite_z else None,
            "max_abs_relative_deviation": (
                float(np.max([abs(v["relative_deviation"]) for v in per_key.values()])) if per_key else None
            ),
            "worst_dither_cv": float(np.max([v["dither_cv"] for v in per_key.values()])) if per_key else None,
        }
    return out
