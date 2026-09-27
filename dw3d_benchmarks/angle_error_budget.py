#!/usr/bin/env python
r"""Decompose `dw3d`'s 11-24 degrees of junction-angle error into its contributions.

**Sub-voxel refinement: diagnose before treating.** The junction-protection work measured
that the ground-truth `.rec` mesh reproduces Neumann's law to **0.34 degrees** (median over
the 47 ground-truth cases) while every `dw3d` variant sits at **11-24 degrees**. The original
audit of the pipeline predicted only **~3.5 degrees** from lattice quantisation, on the
estimate `error ~ epsilon / ell` with `epsilon = 0.5` voxel (rounding) and `ell = 8.3`
voxels (median edge, `3.tif`). Quantisation is therefore *not* the whole story, and sub-voxel
refinement as specified (sub-voxel seed placement) attacks exactly the part the audit
already said is small. This module measures the split before any of it is implemented.

The decomposition is `error = C * epsilon / ell` with three measurable factors, each
isolated by its own experiment:

1. **`C`, the estimator-and-geometry constant** -- `synthetic_wedge_sensitivity`. An exact
   three-film 120-degree trijunction is built analytically at a chosen edge length `ell`,
   its vertices are perturbed by i.i.d. noise of amplitude `epsilon`, and the *same*
   estimator (`metrics.attributed_triple_line_angles`) is run on it. No image, no EDT, no
   watershed: whatever error appears is the estimator's response to vertex noise and
   nothing else. Perturbing the **line** vertices and the **apex** vertices separately
   splits `C` into the part junction protection can reach and the part only interface
   placement can reach (sub-voxel refinement). This also settles the question the
   junction-protection work left open -- whether "some of the flat 11 degrees may be estimator
   noise rather than mesh error".

2. **`epsilon`, the effective vertex error** -- `effective_vertex_error`. Every `dw3d` mesh
   vertex's distance to the *registered ground-truth surface*, measured directly rather
   than assumed to be the 0.5-voxel rounding radius. This is the number sub-voxel
   refinement can move: a
   sub-voxel scheme removes the lattice-rounding component of `epsilon` and nothing else,
   so the reducible fraction is `1 - sqrt(epsilon^2 - sigma_round^2) / epsilon` with
   `sigma_round = 0.289` voxel (the s.d. of a uniform on +-1/2).

3. **`ell`** -- the measured median triple-line edge length of the mesh being scored.

`ground_truth_perturbation_ladder` is the end-to-end control for sub-voxel refinement: take
the ground-truth mesh, apply *one* perturbation at a time (round to the lattice; add noise
of amplitude `epsilon`), and measure the angle error each produces against Neumann. It
needs no reconstruction at all, so it separates "what a perturbed mesh looks like" from
"what `dw3d` produces".

Usage::

    PYTHONPATH="src:." .venv/bin/python -m benchmarks.angle_error_budget \
        --cases 000 007 023 040 --out benchmarks/baseline/angle_error_calibration.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import skimage.io as io
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from dw3d_benchmarks import metrics as m
from dw3d_benchmarks.run_case import VARIANT_GETTERS
from dw3d.io import load_rec
from dw3d.points_on_edt import FAMILY_NAMES, OFFSET_FAMILIES

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT.parent / "benchmarking-dataset"
MIN_DISTANCE = 3

# s.d. of a uniform distribution on [-1/2, 1/2]: the vertex-position error a perfectly
# sub-voxel scheme removes, and *only* that.
ROUNDING_SIGMA = 1.0 / np.sqrt(12.0)

# The wedge calibration the sub-voxel-refinement work measured at `ell = 8.3`: degrees of
# angle error per unit `epsilon / ell`, from an exact analytic 120-degree trijunction
# perturbed by i.i.d. vertex noise and read with this harness's own estimator. Recorded as
# a constant here so the offset-exclusion work can evaluate the same model on a different
# `epsilon` without re-running the calibration.
WEDGE_C_DEG = 33.2

_WEDGE_AZIMUTHS = (0.0, 120.0, 240.0)


# ---------------------------------------------------------------------------
# 1. The estimator's own sensitivity, on an exact analytic trijunction
# ---------------------------------------------------------------------------


def build_wedge(edge_length: float, n_segments: int = 12, n_rings: int = 1) -> tuple:
    """An exact three-film trijunction: 120 degrees between every pair, by construction.

    The triple line runs along `z` with vertices every `edge_length`; each film is a strip
    of `n_rings` rows of vertices at radii `edge_length, 2*edge_length, ...` in its own
    azimuthal half-plane. Films are labelled so that the three interfaces are `(1,2)`,
    `(2,3)` and `(1,3)`, which is what `attributed_triple_line_angles` keys on.

    Returns `(points, triangles, labels, line_vertex_mask)`. The mask marks the triple-line
    vertices, so a caller can perturb the line and the apexes independently.
    """
    line = np.stack(
        [np.zeros(n_segments + 1), np.zeros(n_segments + 1), edge_length * np.arange(n_segments + 1)],
        axis=1,
    )
    points = [line]
    is_line = [np.ones(len(line), dtype=bool)]
    film_rows = []
    for azimuth in _WEDGE_AZIMUTHS:
        direction = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth)), 0.0])
        rows = []
        for ring in range(1, n_rings + 1):
            row = line + ring * edge_length * direction
            rows.append(len(np.concatenate([p[:, 0] for p in points])) if False else None)
            points.append(row)
            is_line.append(np.zeros(len(row), dtype=bool))
        film_rows.append(rows)

    all_points = np.vstack(points)
    is_line_vertex = np.concatenate(is_line)

    # Index bookkeeping: the line occupies [0, n), then each film's rings follow in order.
    n = n_segments + 1
    triangles: list[list[int]] = []
    labels: list[list[int]] = []
    film_labels = [(1, 2), (2, 3), (1, 3)]
    for film, (label_a, label_b) in enumerate(film_labels):
        for ring in range(n_rings):
            inner = np.arange(n) if ring == 0 else np.arange(n) + n * (1 + film * n_rings + (ring - 1))
            outer = np.arange(n) + n * (1 + film * n_rings + ring)
            for i in range(n_segments):
                triangles.append([int(inner[i]), int(inner[i + 1]), int(outer[i])])
                labels.append([label_a, label_b])
                triangles.append([int(inner[i + 1]), int(outer[i + 1]), int(outer[i])])
                labels.append([label_a, label_b])

    return all_points, np.array(triangles), np.array(labels), is_line_vertex


def _wedge_angle_error(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> float:
    """Median |angle - 120| over the wedge's attributed trijunction angles, in degrees."""
    measured = m.attributed_triple_line_angles(points, triangles, labels)
    errors = [abs(value - 120.0) for line in measured.values() for value in line.values()]
    return float(np.median(errors)) if errors else float("nan")


def synthetic_wedge_sensitivity(
    edge_length: float = 8.3,
    amplitudes: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    n_repeats: int = 40,
    seed: int = 20260731,
) -> dict:
    """How many degrees of angle error does `epsilon` voxels of vertex noise produce?

    Three perturbation modes, so the total can be attributed:

    * `all` -- every vertex perturbed. This is what a lattice-quantised mesh looks like.
    * `line_only` -- only the triple-line vertices. **This is the part junction
      protection reaches**, and the measurement below is why it could not have improved the
      angle: it is the smaller of the two.
    * `apex_only` -- only the off-line vertices, i.e. the interface samples. This is the
      part sub-voxel refinement would have to move.

    Noise is i.i.d. uniform on `[-a, a]` per axis with `a = sqrt(3) * epsilon` chosen so the
    per-vertex displacement has r.m.s. magnitude `epsilon` -- the same convention the
    `epsilon` measured in `effective_vertex_error` uses, so the two compose.

    Fits `error = C * epsilon / edge_length` through the origin over the amplitudes and
    returns `C` in degrees per unit `epsilon/ell` ratio, with its standard error.
    """
    points, triangles, labels, is_line = build_wedge(edge_length)
    rng = np.random.default_rng(seed)
    modes = {
        "all": np.ones(len(points), dtype=bool),
        "line_only": is_line,
        "apex_only": ~is_line,
    }

    result: dict = {
        "edge_length": edge_length,
        "n_vertices": len(points),
        "exact_error_deg": _wedge_angle_error(points, triangles, labels),
        "amplitudes": list(amplitudes),
        "modes": {},
    }
    for mode, selector in modes.items():
        curve = []
        for epsilon in amplitudes:
            errors = []
            for _ in range(n_repeats if epsilon > 0 else 1):
                # uniform per-axis half-width giving r.m.s. displacement `epsilon`
                half_width = epsilon
                noise = rng.uniform(-half_width, half_width, size=points.shape) * np.sqrt(3.0)
                perturbed = points + noise * selector[:, None]
                errors.append(_wedge_angle_error(perturbed, triangles, labels))
            curve.append({"epsilon": epsilon, "mean_deg": float(np.mean(errors)), "sd_deg": float(np.std(errors))})
        x = np.array([point["epsilon"] for point in curve]) / edge_length
        y = np.array([point["mean_deg"] for point in curve])
        slope = float((x @ y) / (x @ x))
        residual = y - slope * x
        slope_se = float(np.sqrt((residual @ residual) / max(len(x) - 1, 1) / (x @ x)))
        result["modes"][mode] = {"curve": curve, "C_deg_per_ratio": slope, "C_se": slope_se}
    return result


# ---------------------------------------------------------------------------
# 2. The effective vertex error of a real reconstruction
# ---------------------------------------------------------------------------


def _sample_surface(points: NDArray[np.float64], triangles: NDArray, per_triangle: int = 6) -> NDArray[np.float64]:
    """A dense point sample of a triangle mesh: vertices plus fixed barycentric interior points."""
    barycentric = np.array(
        [(1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3),
         (1 / 2, 1 / 2, 0.0), (1 / 2, 0.0, 1 / 2)],
    )[:per_triangle]
    corners = points[triangles]  # (t, 3, 3)
    interior = np.einsum("bk,tkj->btj", barycentric, corners).reshape(-1, 3)
    return np.vstack([points, interior])


def effective_vertex_error(mask_path: Path, variant: str = "default") -> dict | None:
    """Distance from each reconstructed mesh vertex to the registered ground-truth surface.

    The ground truth is mapped into the mask's voxel frame with `similarity_to_mask_frame`,
    a fit that never sees the reconstruction, then densely sampled; each `dw3d` vertex is
    matched to its nearest sample. Reported split three ways, because the estimator treats
    them differently:

    * **triple-line vertices** -- those on an edge with three incident triangles spanning
      three materials. These are the `p1, p2` of the angle estimator, and they are what junction protection
      protects. Measured against the ground truth's *triple lines* only.
    * **apex vertices** -- every other interface vertex, measured against the full surface.
      These set the apex directions the angle is read from.
    * **all** -- for reference.

    Also returns the median triple-line edge length `ell`, so the `epsilon / ell` ratio the
    wedge calibration consumes can be formed. Returns `None` when the case has no ground
    truth or the registration cannot be fitted.
    """
    case = mask_path.stem.replace("_labels_filled", "")
    rec_path = mask_path.parent / f"{case}_mesh.rec"
    if not rec_path.exists():
        return None

    mask = io.imread(mask_path)
    algo = VARIANT_GETTERS[variant](min_distance=MIN_DISTANCE, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)

    gt_points, gt_triangles, gt_labels = load_rec(rec_path)
    registration = m.similarity_to_mask_frame(mask, gt_points, gt_triangles, gt_labels)
    if registration.get("scale") is None:
        return None
    gt_in_mask = (registration["scale"] * (registration["rotation"] @ gt_points.T)).T + registration["translation"]

    surface_tree = cKDTree(_sample_surface(gt_in_mask, gt_triangles))
    line_tree = cKDTree(_triple_line_points(gt_in_mask, gt_triangles, gt_labels))

    on_line, edge_lengths = _triple_line_vertices_and_lengths(points, triangles, labels)
    apex = np.ones(len(points), dtype=bool)
    apex[list(on_line)] = False
    line_index = np.array(sorted(on_line), dtype=np.int64)

    distances_all, _ = surface_tree.query(points)
    distances_line = line_tree.query(points[line_index])[0] if len(line_index) else np.zeros(0)
    distances_apex = distances_all[apex]

    return {
        "case": case,
        "variant": variant,
        "registration_residual_mean_voxels": registration["residual_mean_voxels"],
        "n_vertices": len(points),
        "n_line_vertices": len(line_index),
        "epsilon_all_rms": float(np.sqrt(np.mean(distances_all**2))),
        "epsilon_all_median": float(np.median(distances_all)),
        "epsilon_line_rms": float(np.sqrt(np.mean(distances_line**2))) if len(distances_line) else None,
        "epsilon_line_median": float(np.median(distances_line)) if len(distances_line) else None,
        "epsilon_apex_rms": float(np.sqrt(np.mean(distances_apex**2))) if distances_apex.size else None,
        "epsilon_apex_median": float(np.median(distances_apex)) if distances_apex.size else None,
        "ell_median_triple_line_edge": float(np.median(edge_lengths)) if len(edge_lengths) else None,
        "ell_median_all_edges": float(np.median(_all_edge_lengths(points, triangles))),
    }


def epsilon_by_point_family(mask_path: Path, variant: str = "default") -> dict | None:
    """The `epsilon` decomposition **by the point family that placed each mesh vertex**.

    The sub-voxel-refinement work did this by hand and it was the measurement that
    identified the offset-exclusion work: the boundary-layer offsets are 30-35 % of mesh
    vertices, carry `epsilon_rms = 2.51` voxels against the other families' 0.58-0.60, and
    account for **90 % of `epsilon^2`**. It is committed here, in the offset-exclusion
    work, because its whole claim is about one of those families and "attributable" means
    the split has to be reproducible, not re-derived.

    The family of each mesh vertex comes from the point-placing scheme itself
    (`point_metadata["family"]`, mapped through `algo._surface_point_ids`), not from
    re-deriving block boundaries from family sizes outside the library.

    Note that `epsilon^2` shares are reported as a fraction of the **sum over vertices**, so
    they weight a family by its size as well as by its error -- which is the right weighting,
    since every vertex contributes to the surface the estimator reads.

    Returns None when the case has no ground truth, the registration cannot be fitted, or the
    variant's point-placing scheme supplies no family metadata.
    """
    case = mask_path.stem.replace("_labels_filled", "")
    rec_path = mask_path.parent / f"{case}_mesh.rec"
    if not rec_path.exists():
        return None

    mask = io.imread(mask_path)
    algo = _algorithm_with_family_metadata(variant)
    if algo is None:
        return None
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)

    gt_points, gt_triangles, gt_labels = load_rec(rec_path)
    registration = m.similarity_to_mask_frame(mask, gt_points, gt_triangles, gt_labels)
    if registration.get("scale") is None:
        return None
    gt_in_mask = (registration["scale"] * (registration["rotation"] @ gt_points.T)).T + registration["translation"]
    distances, _ = cKDTree(_sample_surface(gt_in_mask, gt_triangles)).query(points)

    family_of_point = np.asarray(algo._point_metadata["family"])[np.asarray(algo._surface_point_ids)]
    total_sq = float(np.sum(distances**2))
    families = {}
    for code in np.unique(family_of_point):
        selected = distances[family_of_point == code]
        families[FAMILY_NAMES[int(code)]] = {
            "n": len(selected),
            "share_of_vertices": float(len(selected) / len(distances)),
            "epsilon_rms": float(np.sqrt(np.mean(selected**2))),
            "epsilon_median": float(np.median(selected)),
            "share_of_epsilon_squared": float(np.sum(selected**2) / total_sq) if total_sq > 0 else 0.0,
        }

    on_line, edge_lengths = _triple_line_vertices_and_lengths(points, triangles, labels)
    return {
        "case": case,
        "variant": variant,
        "n_vertices": len(points),
        "registration_residual_mean_voxels": registration["residual_mean_voxels"],
        "epsilon_all_rms": float(np.sqrt(np.mean(distances**2))),
        "ell_median_triple_line_edge": float(np.median(edge_lengths)) if len(edge_lengths) else None,
        "n_line_vertices": len(on_line),
        "families": families,
    }


def epsilon_of_the_estimator_stencil(mask_path: Path, variant: str = "default") -> dict | None:
    """`epsilon` restricted to the vertices the junction-angle estimator actually reads.

    **The offset-exclusion work's decisive measurement, and the correction it makes to
    the sub-voxel-refinement work's model.**

    That work wrote `error = C * epsilon / ell` with `epsilon` the r.m.s. vertex-to-surface distance
    over *all* mesh vertices, calibrated `C = 33.2 +- 1.2` deg on an analytic wedge, and it
    explained ~90 % of the measured angle error for both variants with no fitted parameter.
    But `attributed_triple_line_angles` does not read all mesh vertices. Per trijunction edge
    it reads **five**: the two endpoints of the edge (the *line* vertices) and the third vertex
    of each of the three incident triangles (the *apex* vertices). The population `epsilon` and
    the stencil `epsilon` are different quantities, and they respond differently to an
    intervention that changes only part of the mesh.

    This function measures the stencil one. It splits the same three ways as
    `effective_vertex_error` — line, apex, all — but over the stencil rather than the mesh, and
    it additionally reports the **point-family composition of the stencil**, which is what says
    how much of it any given construction can reach at all.

    Returns None when the case has no ground truth, the registration cannot be fitted, or the
    variant's point-placing scheme supplies no family metadata.
    """
    case = mask_path.stem.replace("_labels_filled", "")
    rec_path = mask_path.parent / f"{case}_mesh.rec"
    if not rec_path.exists():
        return None

    mask = io.imread(mask_path)
    algo = _algorithm_with_family_metadata(variant)
    if algo is None:
        return None
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)

    gt_points, gt_triangles, gt_labels = load_rec(rec_path)
    registration = m.similarity_to_mask_frame(mask, gt_points, gt_triangles, gt_labels)
    if registration.get("scale") is None:
        return None
    gt_in_mask = (registration["scale"] * (registration["rotation"] @ gt_points.T)).T + registration["translation"]
    surface_tree = cKDTree(_sample_surface(gt_in_mask, gt_triangles))
    line_tree = cKDTree(_triple_line_points(gt_in_mask, gt_triangles, gt_labels))

    line_ids, apex_ids, lengths = _estimator_stencil(np.asarray(triangles, dtype=np.int64), labels, points)
    if not len(line_ids) or not len(apex_ids):
        return None
    family_of_point = np.asarray(algo._point_metadata["family"])[np.asarray(algo._surface_point_ids)]

    # Line vertices are scored against the ground truth's triple lines, apex vertices against
    # its full surface -- the same convention as `effective_vertex_error`, so the two are
    # directly comparable.
    line_distance = line_tree.query(points[line_ids])[0]
    apex_distance = surface_tree.query(points[apex_ids])[0]
    stencil_distance = np.concatenate((line_distance, apex_distance))

    def composition(ids: NDArray[np.int64]) -> dict:
        families = family_of_point[ids]
        return {
            FAMILY_NAMES[int(code)]: float(np.mean(families == code))
            for code in np.unique(families)
        }

    ell = float(np.median(lengths))
    return {
        "case": case,
        "variant": variant,
        "n_trijunction_edges": len(lengths),
        "n_line_reads": len(line_ids),
        "n_apex_reads": len(apex_ids),
        "ell_median_triple_line_edge": ell,
        "epsilon_stencil_rms": float(np.sqrt(np.mean(stencil_distance**2))),
        "epsilon_stencil_line_rms": float(np.sqrt(np.mean(line_distance**2))),
        "epsilon_stencil_apex_rms": float(np.sqrt(np.mean(apex_distance**2))),
        "line_family_composition": composition(line_ids),
        "apex_family_composition": composition(apex_ids),
        "offset_share_of_apex_reads": float(np.mean(np.isin(family_of_point[apex_ids], OFFSET_FAMILIES))),
        "offset_share_of_line_reads": float(np.mean(np.isin(family_of_point[line_ids], OFFSET_FAMILIES))),
        # The model, evaluated on the stencil epsilon rather than the population one.
        "model_prediction_deg": float(WEDGE_C_DEG * np.sqrt(np.mean(stencil_distance**2)) / ell) if ell else None,
    }


def _estimator_stencil(
    triangles: NDArray[np.int64],
    labels: NDArray,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """The vertices `attributed_triple_line_angles` reads, with repetition.

    Repetition is deliberate: a vertex read by four trijunction edges contributes four times to
    the angle error, so the r.m.s. over reads — not over distinct vertices — is the quantity the
    model needs. The acceptance filter is the estimator's own (exactly 3 incident triangles,
    exactly 3 materials, 3 distinct label pairs), so the stencil is the estimator's, not an
    approximation of it.
    """
    edge_map = m._edge_to_triangle_map(triangles)
    line_ids: list[int] = []
    apex_ids: list[int] = []
    lengths: list[float] = []
    for (first, second), triangle_ids in edge_map.items():
        if len(triangle_ids) != 3:
            continue
        pairs = [tuple(sorted(int(x) for x in labels[t])) for t in triangle_ids]
        materials = sorted({m_ for pair in pairs for m_ in pair})
        if len(materials) != 3 or len(set(pairs)) != 3:
            continue
        length = float(np.linalg.norm(points[second] - points[first]))
        if length == 0:
            continue
        lengths.append(length)
        line_ids.extend((int(first), int(second)))
        apex_ids.extend(int(next(v for v in triangles[t] if v not in (first, second))) for t in triangle_ids)
    return np.array(line_ids, dtype=np.int64), np.array(apex_ids, dtype=np.int64), np.array(lengths)


def _algorithm_with_family_metadata(variant: str) -> object | None:
    """The variant's algorithm, forced to emit point-family metadata, or None if it cannot.

    Only the boundary-layer family of schemes carries families, and only when asked; asking
    changes nothing else (the points, tesselation and labelling are bit-identical), so the
    *same* metadata switch is used to measure the default as to run offset exclusion. That is what makes
    the before/after decomposition a comparison of one variable.
    """
    from dw3d.points_on_edt import (
        peak_local_points_boundary_layer,
        peak_local_points_junction_protected,
    )

    algo = VARIANT_GETTERS[variant](min_distance=MIN_DISTANCE, print_info=False)
    placer = algo.point_placing_function
    if getattr(placer, "func", None) not in (peak_local_points_boundary_layer, peak_local_points_junction_protected):
        return None
    algo.point_placing_function = partial(placer, exclude_offsets_from_surface=True)
    # `exclude_offsets_from_surface=True` also switches the *extraction* on, so measuring the
    # default this way would measure offset exclusion instead. Re-assert the variant's own behaviour by
    # dropping the merge target while keeping the families.
    if not placer.keywords.get("exclude_offsets_from_surface", False):
        algo.point_placing_function = _families_only(algo.point_placing_function)
    return algo


def _families_only(placer: Callable) -> Callable:
    """Wrap a point-placing function so its metadata carries `family` but no merge target."""

    def placed(*args: object, **kwargs: object) -> tuple:
        *head, metadata = placer(*args, **kwargs)
        return (*head, {"family": metadata["family"]})

    return placed


def _triple_line_points(points: NDArray[np.float64], triangles: NDArray, labels: NDArray) -> NDArray[np.float64]:
    """Densely sampled points along a mesh's triple lines (edges with 3 films, 3 materials)."""
    on_line, _ = _triple_line_vertices_and_lengths(points, triangles, labels)
    edge_map = m._edge_to_triangle_map(triangles)
    samples: list[NDArray[np.float64]] = []
    for (a, b), tri_ids in edge_map.items():
        if len(tri_ids) != 3:
            continue
        if len({int(x) for tri in tri_ids for x in labels[tri]}) != 3:
            continue
        samples.extend((1 - t) * points[a] + t * points[b] for t in np.linspace(0, 1, 5))
    return np.array(samples) if samples else points[sorted(on_line)] if on_line else points


def _triple_line_vertices_and_lengths(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
) -> tuple[set[int], NDArray[np.float64]]:
    edge_map = m._edge_to_triangle_map(triangles)
    on_line: set[int] = set()
    lengths: list[float] = []
    for (a, b), tri_ids in edge_map.items():
        if len(tri_ids) != 3:
            continue
        if len({int(x) for tri in tri_ids for x in labels[tri]}) != 3:
            continue
        on_line.update((int(a), int(b)))
        lengths.append(float(np.linalg.norm(points[b] - points[a])))
    return on_line, np.array(lengths)


def _all_edge_lengths(points: NDArray[np.float64], triangles: NDArray) -> NDArray[np.float64]:
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)


# ---------------------------------------------------------------------------
# 3. The end-to-end control: perturb the ground truth, one way at a time
# ---------------------------------------------------------------------------


def ground_truth_perturbation_ladder(
    mask_path: Path,
    amplitudes: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    n_repeats: int = 8,
    seed: int = 20260731,
) -> dict | None:
    """Angle error of the ground-truth mesh after exactly one perturbation, vs Neumann.

    The rungs, all in the mask's voxel frame so `epsilon` is in voxels:

    * `exact` -- the ground truth as stored. This is the 0.34-degree floor the junction-protection work measured.
    * `rounded` -- every vertex rounded to the integer voxel lattice. **This is the pure
      quantisation contribution the ~3.5-degree lattice estimate is about**, measured instead of
      estimated, and at the ground truth's own (fine) edge length.
    * `noise[epsilon]` -- i.i.d. displacement of r.m.s. magnitude `epsilon` voxels.

    Returns `None` when the case has no ground truth, no tensions, or no registration.
    """
    case = mask_path.stem.replace("_labels_filled", "")
    rec_path = mask_path.parent / f"{case}_mesh.rec"
    tensions_path = mask_path.parent / f"{case}_dict_tensions.npy"
    if not (rec_path.exists() and tensions_path.exists()):
        return None

    mask = io.imread(mask_path)
    gt_points, gt_triangles, gt_labels = load_rec(rec_path)
    registration = m.similarity_to_mask_frame(mask, gt_points, gt_triangles, gt_labels)
    if registration.get("scale") is None:
        return None
    in_mask = (registration["scale"] * (registration["rotation"] @ gt_points.T)).T + registration["translation"]

    neumann = m.neumann_angles_from_tensions(np.load(tensions_path, allow_pickle=True).item())

    def error_of(points: NDArray[np.float64]) -> float | None:
        return m.compare_angle_dicts(
            m.attributed_triple_line_angles(points, gt_triangles, gt_labels),
            neumann,
        )["angle_error_median_deg"]

    rng = np.random.default_rng(seed)
    rungs = {
        "exact": error_of(in_mask),
        "rounded": error_of(np.rint(in_mask)),
    }
    for epsilon in amplitudes:
        errors = [
            error_of(in_mask + rng.uniform(-epsilon, epsilon, size=in_mask.shape) * np.sqrt(3.0))
            for _ in range(n_repeats)
        ]
        errors = [e for e in errors if e is not None]
        rungs[f"noise_{epsilon}"] = float(np.mean(errors)) if errors else None

    return {
        "case": case,
        "n_gt_points": len(gt_points),
        "ell_median_gt_triple_line_edge": float(
            np.median(_triple_line_vertices_and_lengths(in_mask, gt_triangles, gt_labels)[1]),
        ),
        "ell_median_gt_all_edges": float(np.median(_all_edge_lengths(in_mask, gt_triangles))),
        "rungs": rungs,
    }


# ---------------------------------------------------------------------------


def _report_wedge(edge_lengths: list[float]) -> dict:
    """Section 1: the estimator's sensitivity constant at each edge length."""
    print("1. Estimator sensitivity on the exact analytic wedge (no image, no EDT):")
    out = {}
    for edge_length in edge_lengths:
        wedge = synthetic_wedge_sensitivity(edge_length=edge_length)
        out[str(edge_length)] = wedge
        print(f"  ell={edge_length}: exact error {wedge['exact_error_deg']:.2e} deg")
        for mode, data in wedge["modes"].items():
            at_half = next(p["mean_deg"] for p in data["curve"] if p["epsilon"] == 0.5)
            print(
                f"    {mode:10s} C={data['C_deg_per_ratio']:7.2f} +- {data['C_se']:.2f} deg per (eps/ell)"
                f"   error at eps=0.5: {at_half:5.2f} deg",
            )
    return out


def _report_vertex_error(cases: list[str], variants: list[str]) -> list[dict]:
    """Section 2: measured `epsilon` and `ell` for each (case, variant)."""
    print("\n2. Effective vertex error of the reconstruction against the registered ground truth:")
    out = []
    for case in cases:
        mask_path = DATASET_DIR / f"{case}_labels_filled.tif"
        if not mask_path.exists():
            print(f"  SKIP {case} (mask not found)")
            continue
        for variant in variants:
            record = effective_vertex_error(mask_path, variant=variant)
            if record is None:
                continue
            out.append(record)
            print(
                f"  {case} {variant:8s} eps_all={record['epsilon_all_rms']:.3f} "
                f"eps_line={record['epsilon_line_rms']:.3f} eps_apex={record['epsilon_apex_rms']:.3f} "
                f"ell_line={record['ell_median_triple_line_edge']:.2f} "
                f"ell_all={record['ell_median_all_edges']:.2f} voxels",
            )
    return out


def _report_family_epsilon(cases: list[str], variants: list[str]) -> list[dict]:
    """Section 2b (offset exclusion): `epsilon` split by the point family that placed the vertex."""
    print("\n2b. epsilon by point family (the decomposition the offset-exclusion work is judged on):")
    out = []
    for case in cases:
        mask_path = DATASET_DIR / f"{case}_labels_filled.tif"
        if not mask_path.exists():
            continue
        for variant in variants:
            record = epsilon_by_point_family(mask_path, variant=variant)
            if record is None:
                print(f"  SKIP {case} {variant} (no ground truth, or no family metadata)")
                continue
            out.append(record)
            print(f"  {case} {variant}: eps_all={record['epsilon_all_rms']:.3f} n={record['n_vertices']}")
            for name, block in sorted(record["families"].items()):
                print(
                    f"      {name:18s} n={block['n']:5d} {100 * block['share_of_vertices']:5.1f} % of vertices"
                    f"  eps_rms={block['epsilon_rms']:.3f}"
                    f"  {100 * block['share_of_epsilon_squared']:5.1f} % of eps^2",
                )
    return out


def _report_stencil_epsilon(cases: list[str], variants: list[str]) -> list[dict]:
    """Section 2c (offset exclusion): `epsilon` over the estimator's own stencil, and its composition."""
    print("\n2c. epsilon over the junction-angle estimator's stencil (the offset-exclusion work's correction):")
    out = []
    for case in cases:
        mask_path = DATASET_DIR / f"{case}_labels_filled.tif"
        if not mask_path.exists():
            continue
        for variant in variants:
            record = epsilon_of_the_estimator_stencil(mask_path, variant=variant)
            if record is None:
                print(f"  SKIP {case} {variant}")
                continue
            out.append(record)
            print(
                f"  {case} {variant}: eps_stencil={record['epsilon_stencil_rms']:.3f} "
                f"(line {record['epsilon_stencil_line_rms']:.3f}, apex {record['epsilon_stencil_apex_rms']:.3f}) "
                f"ell={record['ell_median_triple_line_edge']:.2f} "
                f"model={record['model_prediction_deg']:.2f} deg  "
                f"offsets in apex reads {100 * record['offset_share_of_apex_reads']:.1f} %",
            )
    return out


def _report_ladder(cases: list[str]) -> list[dict]:
    """Section 3: one perturbation at a time, applied to the ground-truth mesh."""
    print("\n3. Ground-truth perturbation ladder (no reconstruction involved):")
    out = []
    for case in cases:
        mask_path = DATASET_DIR / f"{case}_labels_filled.tif"
        if not mask_path.exists():
            continue
        record = ground_truth_perturbation_ladder(mask_path)
        if record is None:
            continue
        out.append(record)
        rungs = " ".join(f"{name}={value:.2f}" for name, value in record["rungs"].items() if value is not None)
        print(f"  {record['case']}: ell_gt={record['ell_median_gt_triple_line_edge']:.2f}  {rungs}")
    return out


def main() -> None:
    """Run the three experiments and write one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["000", "007", "023", "040"])
    parser.add_argument("--variants", nargs="+", default=["default", "a1b"])
    parser.add_argument("--wedge-edge-lengths", type=float, nargs="+", default=[4.0, 8.3, 16.0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {
        "wedge": _report_wedge(args.wedge_edge_lengths),
        "effective_vertex_error": _report_vertex_error(args.cases, args.variants),
        "epsilon_by_point_family": _report_family_epsilon(args.cases, args.variants),
        "epsilon_of_the_estimator_stencil": _report_stencil_epsilon(args.cases, args.variants),
        "ground_truth_ladder": _report_ladder(args.cases),
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 4. Is it the mesh or the estimator? A neighbourhood-averaged alternative
# ---------------------------------------------------------------------------


def neighbourhood_triple_line_angles(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    radius_factor: float = 2.5,
) -> dict:
    """Attributed trijunction angles from an **area-weighted neighbourhood**, not one apex.

    `metrics.attributed_triple_line_angles` reads each film's in-plane direction from the
    *single* apex vertex of the *single* incident triangle. That makes the estimate a
    direct function of one vertex position, so a vertex error `epsilon` at edge length
    `ell` produces `~C * epsilon / ell` degrees of angle error whether or not the surface
    it belongs to is in the right place. The junction-protection work left this as an open
    question -- "it is possible that some of the flat 11 degrees is estimator noise rather
    than mesh error" -- and said to settle it before sub-voxel refinement uses angle error as
    a convergence metric.

    This is the settling experiment. Same edges, same attribution, same output shape; the
    only change is how each film's direction is estimated: the area-weighted mean of
    `centroid - edge_midpoint` over *every* triangle of that film whose centroid lies
    within `radius_factor * ell` of the edge midpoint, projected perpendicular to the edge.
    Averaging over `O(radius_factor**2)` triangles suppresses i.i.d. vertex noise by roughly
    that factor while leaving a genuinely misplaced surface untouched — so if the two
    estimators disagree, the error is in the estimator, and if they agree, it is in the mesh.
    """
    edge_map = m._edge_to_triangle_map(triangles)
    centroids = points[triangles].mean(axis=1)
    areas = m.triangle_areas(points, triangles)
    film_of = [tuple(sorted(int(x) for x in row)) for row in labels]
    tree = cKDTree(centroids)

    accumulated: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}
    for (p1, p2), tri_ids in edge_map.items():
        if len(tri_ids) != 3:
            continue
        films = [tuple(sorted(int(x) for x in labels[tri])) for tri in tri_ids]
        materials = sorted({x for film in films for x in film})
        if len(materials) != 3 or len(set(films)) != 3:
            continue

        edge = points[p2] - points[p1]
        length = float(np.linalg.norm(edge))
        if length == 0:
            continue
        midpoint = 0.5 * (points[p1] + points[p2])
        neighbourhood = np.array(tree.query_ball_point(midpoint, radius_factor * length), dtype=np.int64)
        if len(neighbourhood) == 0:
            continue

        directions = [
            _mean_film_direction(film, neighbourhood, film_of, centroids, areas, midpoint, edge, length)
            for film in films
        ]
        if any(d is None for d in directions):
            continue

        line = accumulated.setdefault(tuple(materials), {})
        for i in range(3):
            for j in range(i + 1, 3):
                key = "|".join(sorted((f"{films[i][0]},{films[i][1]}", f"{films[j][0]},{films[j][1]}")))
                angle = float(np.degrees(np.arccos(np.clip(float(directions[i] @ directions[j]), -1.0, 1.0))))
                line.setdefault(key, []).append((angle, length))

    return {
        key: {
            pair_key: float(np.average([a for a, _ in values], weights=[w for _, w in values]))
            for pair_key, values in line.items()
        }
        for key, line in accumulated.items()
    }


def _mean_film_direction(
    film: tuple[int, int],
    neighbourhood: NDArray[np.int64],
    film_of: list,
    centroids: NDArray[np.float64],
    areas: NDArray[np.float64],
    midpoint: NDArray[np.float64],
    edge: NDArray[np.float64],
    length: float,
) -> NDArray[np.float64] | None:
    """Area-weighted mean in-plane direction of one film over an edge's neighbourhood."""
    selected = neighbourhood[np.array([film_of[i] == film for i in neighbourhood], dtype=bool)]
    if len(selected) == 0:
        return None
    offsets = centroids[selected] - midpoint
    offsets = offsets - np.outer(offsets @ edge, edge) / (length * length)
    norms = np.linalg.norm(offsets, axis=1)
    keep = norms > 1e-9
    if not keep.any():
        return None
    unit = offsets[keep] / norms[keep, None]
    mean = np.average(unit, axis=0, weights=areas[selected][keep])
    norm = float(np.linalg.norm(mean))
    return None if norm == 0 else mean / norm
