#!/usr/bin/env python
"""Sweep the junction-protection shell-coarsening factor and measure what it costs.

`dw3d.edt.compute_edt_classical` calls `_pad_mask`, which marks the one-voxel bounding-box
shell as boundary; the EDT is then exactly 0 all along that shell, making it one enormous
minimum plateau. Measured on `3.tif` at `min_distance=3`, only **1172 of 5228** interface
minima lie on a genuine >=2-label interface — **77.6 % of the interface point budget is the
bounding box and background plateaus**, not cell geometry. The shell must exist (it bounds
the tesselation, so border-touching cells can close) but it does not need interface density.

This script measures the two things that decide how far it can be coarsened:

* the **point-budget and wall-time saving**, and
* the **effect on border-touching cells** — the cells that actually rely on the shell.
  "Border-touching" is decided from the mask (a cell with a voxel within `border_margin` of
  the image edge), and the effect is measured as the relative error of the reconstructed
  cell volume against the mask's own voxel count, split border vs interior, plus the
  per-cell topology (connected components, Euler characteristic). Cell volume against the
  voxel count is the one quantity in `dw3d` with an external ground truth that needs no
  registration, which is why the determinism fix used it too.

Usage::

    python -m benchmarks.run_shell_sweep --factors 1 2 3 4 --n-dataset-cases 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import skimage.io as io
from scipy import ndimage as ndi

from dw3d_benchmarks import metrics as m
from dw3d import get_junction_protected_mesh_reconstruction_algorithm

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
MIN_DISTANCE = 3


def _shell_sensitive_cells(mask: np.ndarray, margin: int = 2) -> tuple[set[int], int]:
    """The cells whose reconstruction the bounding-box shell can actually affect.

    The first idea — "cells with a voxel within `margin` of the image edge" — turns out to
    select **nothing** on `benchmarking-dataset`: every embryo sits well inside its volume
    (`3.tif`'s non-zero bounding box is `[17,17,17]..[175,181,180]` in a 182x200x191 image),
    so no cell touches the border and the shell cannot distort one. That is itself the answer
    to "any effect on border-touching cells", and the count is returned so the report can say
    it rather than imply it.

    The set actually returned is the **outer** cells: those sharing a boundary with the
    background label 0. Those are the cells whose exterior surface is reconstructed from
    tetrahedra that reach out to the background and shell samples, so they are where a
    coarser shell could plausibly do damage.
    """
    border = np.zeros(mask.shape, dtype=bool)
    border[:margin] = border[-margin:] = True
    border[:, :margin] = border[:, -margin:] = True
    border[:, :, :margin] = border[:, :, -margin:] = True
    n_touching_image_edge = len({int(v) for v in np.unique(mask[border]) if v != 0})

    background = ndi.binary_dilation(mask == 0, np.ones((3, 3, 3), dtype=np.uint8))
    return {int(v) for v in np.unique(mask[background]) if v != 0}, n_touching_image_edge


def run_one(mask: np.ndarray, shell_coarsening: int) -> dict:
    """Reconstruct once at one shell-coarsening factor and measure the consequences."""
    algo = get_junction_protected_mesh_reconstruction_algorithm(
        min_distance=MIN_DISTANCE,
        shell_coarsening=shell_coarsening,
    )
    start = time.perf_counter()
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    wall_time = time.perf_counter() - start

    graph = algo._tesselation_graph
    volumes = m.cell_volumes_from_tetrahedra(graph, algo._map_label_to_nodes_ids)
    topology = m.connected_components_and_euler_per_cell(triangles, labels)
    outer, n_touching_image_edge = _shell_sensitive_cells(mask)

    errors: dict[str, list[float]] = {"border": [], "interior": []}
    defects = {"border": 0, "interior": 0}
    for cell, volume in volumes.items():
        cell = int(cell)
        if cell == 0:
            continue
        voxel_volume = float((mask == cell).sum())
        if voxel_volume == 0:
            continue
        group = "border" if cell in outer else "interior"
        errors[group].append(abs(volume - voxel_volume) / voxel_volume)
        stats = topology.get(cell)
        if stats and (stats["n_connected_components"] != 1 or stats["euler_characteristic"] != 2):
            defects[group] += 1

    point_placement = m.point_placement_counts(
        algo._edt_image,
        MIN_DISTANCE,
        point_placing_function=algo.point_placing_function,
        segmented_image=mask,
    )
    surface = m.tetrahedron_surface_stats(
        graph,
        m.classify_tesselation_points(len(graph.vertices), point_placement["n_interior_points"]),
    )
    edges = m.edge_topology_stats(points, triangles, labels)

    return {
        "shell_coarsening": shell_coarsening,
        "n_cells_touching_image_edge": n_touching_image_edge,
        "wall_time_s": wall_time,
        "n_tesselation_points": len(graph.vertices),
        "n_tetrahedra": int(surface["n_tetrahedra"]),
        "fraction_all_surface_tets": surface["fraction_all_surface_tets"],
        "n_valence_geq4_edges": edges["n_valence_geq4_edges"],
        "n_abnormal_non_manifold_edges": m.abnormal_non_manifold_edge_count(points, triangles, labels),
        "n_border_cells": len(errors["border"]),
        "n_interior_cells": len(errors["interior"]),
        "volume_error_border_median": float(np.median(errors["border"])) if errors["border"] else None,
        "volume_error_interior_median": float(np.median(errors["interior"])) if errors["interior"] else None,
        "volume_error_border_max": float(np.max(errors["border"])) if errors["border"] else None,
        "n_topology_defects_border": defects["border"],
        "n_topology_defects_interior": defects["interior"],
    }


def main() -> None:
    """Run the sweep and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--n-dataset-cases", type=int, default=6)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks" / "baseline" / "a5_shell_sweep.json")
    args = parser.parse_args()

    masks = [REPO_ROOT / "data" / "Images" / f"{i}.tif" for i in range(1, 5)]
    masks += sorted((WORKSPACE_ROOT / "benchmarking-dataset").glob("*_labels_filled.tif"))[: args.n_dataset_cases]
    masks = [path for path in masks if path.exists()]

    results: dict[str, list[dict]] = {}
    for path in masks:
        mask = io.imread(path)
        print(f"{path.stem}:", end=" ", flush=True)
        for factor in args.factors:
            record = run_one(mask, factor)
            results.setdefault(path.stem, []).append(record)
            print(f"s{factor}={record['wall_time_s']:.1f}s", end=" ", flush=True)
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"min_distance": MIN_DISTANCE, "cases": results}, indent=2, sort_keys=True) + "\n",
    )

    def _median(values: list[float | None]) -> float:
        """Median over the cases that have that group at all (a case may have no border cell)."""
        kept = [v for v in values if v is not None]
        return float(np.median(kept)) if kept else float("nan")

    print(
        f"\n{'factor':>7s} {'points':>9s} {'tets':>9s} {'allsurf':>9s} {'time x':>8s} {'v>=4':>6s} {'abn':>5s} "
        f"{'volerr border':>14s} {'volerr interior':>16s} {'topo defects b/i':>18s}",
    )
    baseline_time = sum(r[0]["wall_time_s"] for r in results.values())
    for index, factor in enumerate(args.factors):
        rows = [r[index] for r in results.values()]
        print(
            f"{factor:7d} {sum(r['n_tesselation_points'] for r in rows):9d} {sum(r['n_tetrahedra'] for r in rows):9d} "
            f"{_median([r['fraction_all_surface_tets'] for r in rows]) * 100:8.2f}% "
            f"{sum(r['wall_time_s'] for r in rows) / baseline_time:8.2f} "
            f"{sum(r['n_valence_geq4_edges'] for r in rows):6d} "
            f"{sum(r['n_abnormal_non_manifold_edges'] for r in rows):5d} "
            f"{_median([r['volume_error_border_median'] for r in rows]) * 100:13.3f}% "
            f"{_median([r['volume_error_interior_median'] for r in rows]) * 100:15.3f}% "
            f"{sum(r['n_topology_defects_border'] for r in rows):8d} /"
            f"{sum(r['n_topology_defects_interior'] for r in rows):8d}",
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
