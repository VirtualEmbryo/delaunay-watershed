"""Per-`(mesh_set, case, min_distance)` mesh-quality measurement: M1-M7.

Every function here takes an already-constructed reconstruction (`points, triangles,
labels`, dw3d's own output format) and the case's ground truth, and returns plain,
JSON-serialisable numbers. No estimator here is new: M1 is `foambryo.geometry`'s production
`compute_angles_tri`; M3 is `foambryo.validation.mesh_io.signed_cell_volumes`; M4 is
`foambryo.curvature.compute_curvature_interfaces` applied identically to both meshes; M6's
integrity numbers are `foambryo.validation.mesh_io.mesh_integrity`; M2's registration and M7's
timing reuse `dw3d_benchmarks.metrics`/`profiling`. M5 (junction-line geometry) has no
existing implementation to reuse and no independent anchor; it is this
module's own, clearly-scoped construction, and its numbers are reported as provisional. M6's
nonlocal self-intersection counts are likewise this package's own construction, in
`self_intersection.py`, because no local or combinatorial predicate can see the property they
measure; that module's exact predicate is checked against an independent Moller-Trumbore oracle
rather than against itself.

Ground truth is read only here, in the analysis; nothing in the estimator's own call path
(`algo.construct_mesh_from_segmentation_mask(mask)`, called with the mask alone) touches it.
See `safeguards.py` for the check that truth is unreachable, not merely unused.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

# `foambryo` is a declared benchmark-only dependency (setup.cfg's `benchmarks` extra);
# `src/dw3d` itself never imports it.
from foambryo.curvature import compute_curvature_interfaces
from foambryo.dcel import DcelData
from foambryo.geometry import compute_angles_tri, compute_trijunction_edges
from foambryo.second_order_angle_reading import ORIENTED_CHORD_READING
from foambryo.validation.mesh_io import list_interfaces, mesh_integrity, signed_cell_volumes
from numpy.typing import NDArray

from dw3d.io import load_rec
from dw3d_benchmarks import metrics as dwm

from . import bootstrap, case_metadata
from .self_intersection import compute_self_intersections

_RSS_TO_MB = (1 / 1024**2) if sys.platform == "darwin" else (1 / 1024)
#: A trijunction wedge counts as reflex when it exceeds a straight angle by more than this.
#: Degrees. Not tuned: `arccos` is ill-conditioned at a straight wedge -- its derivative
#: diverges as the dot product approaches -1 -- so a wedge at exactly 180 degrees is returned
#: with an absolute error of order `sqrt(eps)`, about 8e-7 degrees, and a tolerance below that
#: would count rounding as geometry. Measured: reading a mesh's angle-sum deficit rather than
#: its oriented wedges disagrees on 12 junction edges out of ~1.3 million for exactly this
#: reason, all of them within 1.3e-6 degrees of straight.
REFLEX_TOLERANCE_DEG = 1e-6


def _as_int64(array: NDArray) -> NDArray[np.int64]:
    """Cast to int64, exactly and losslessly.

    dw3d returns `triangles`/`labels` as `uint64`; `foambryo.validation.mesh_io` requires
    `int64` and does not cast internally (unlike `DcelData.__init__`, which does). Values are
    always small vertex indices or material ids, so the cast is exact, never lossy.
    """
    return np.asarray(array, dtype=np.int64)


# ---------------------------------------------------------------------------
# Ground truth + registration
# ---------------------------------------------------------------------------


def load_ground_truth(dataset_dir: Path, case_id: str) -> dict | None:
    """Load a case's ground-truth mesh and (if present) tensions.

    Returns `None` if the case has no `.rec` file (should not happen inside the cohort, but
    checked rather than assumed).
    """
    rec_path = Path(dataset_dir) / f"{case_id}_mesh.rec"
    if not rec_path.exists():
        return None
    gt_points, gt_triangles, gt_labels = load_rec(rec_path)
    tensions_path = Path(dataset_dir) / f"{case_id}_dict_tensions.npy"
    tensions = np.load(tensions_path, allow_pickle=True).item() if tensions_path.exists() else None
    return {
        "points": gt_points,
        "triangles": gt_triangles,
        "labels": gt_labels,
        "tensions": tensions,
    }


def neumann_wedge_angles_deg(tensions: dict) -> dict[tuple[int, int, int], float]:
    """Exact equilibrium wedge angles from ground-truth tensions, keyed like `compute_angles_tri`.

    For triple `{a, b, c}`, the wedge inside material `f` (between the two films meeting at
    `f`) satisfies, by Neumann's law (force balance in the plane normal to the triple line):

        cos(theta_f) = (gamma_opposite^2 - gamma_1^2 - gamma_2^2) / (2 * gamma_1 * gamma_2)

    where `gamma_1, gamma_2` are the tensions of the two films incident on `f` and
    `gamma_opposite` is the tension of the film between the other two materials. This is the
    same identity as the equivalent form `cos(theta_A) = -(g_CA^2 + g_AB^2 - g_BC^2) / (2 g_CA
    g_AB)`, re-keyed to `(min(e, g), f, max(e, g))` -- `foambryo.geometry.compute_angles_tri`'s
    own wedge-key convention -- so the two can be diffed key-by-key with no re-derivation.

    A triple whose tensions violate the triangle inequality (no equilibrium configuration)
    contributes no key at all, for any of its three wedges: partial coverage of an
    inadmissible triple would compare an existing angle against a manufactured one.
    """
    lookup = {tuple(sorted(k)): float(v) for k, v in tensions.items()}
    materials = sorted({m for k in lookup for m in k})
    out: dict[tuple[int, int, int], float] = {}
    for i, a in enumerate(materials):
        for j, b in enumerate(materials[i + 1 :], start=i + 1):
            for c in materials[j + 1 :]:
                g_ab, g_ac, g_bc = lookup.get((a, b)), lookup.get((a, c)), lookup.get((b, c))
                if g_ab is None or g_ac is None or g_bc is None:
                    continue
                triple: dict[tuple[int, int, int], float] = {}
                admissible = True
                for f, g_opp, g1, g2 in ((a, g_bc, g_ab, g_ac), (b, g_ac, g_ab, g_bc), (c, g_ab, g_ac, g_bc)):
                    cosine = (g_opp**2 - g1**2 - g2**2) / (2 * g1 * g2)
                    if not -1.0 <= cosine <= 1.0:
                        admissible = False
                        break
                    others = sorted(x for x in (a, b, c) if x != f)
                    triple[(others[0], f, others[1])] = float(np.degrees(np.arccos(cosine)))
                if admissible:
                    out.update(triple)
    return out


# ---------------------------------------------------------------------------
# M1 -- contact angles
# ---------------------------------------------------------------------------


def _wedge_diff(measured: dict[tuple, float], reference: dict[tuple, float]) -> dict[tuple, float]:
    return {key: measured[key] - reference[key] for key in reference if key in measured}


#: Keys every descriptor below must carry, so that a statistic cannot be emitted without saying
#: what it is aggregated over. `measures`, `unit` and `definition` were the original three;
#: `aggregation_unit`, `pooling` and `weighting` were added because five different aggregation
#: conventions have been published for the same angle quantity and two published comparisons
#: crossed two of them. A number is not interpretable without all six.
_REQUIRED_DESCRIPTOR_KEYS: frozenset[str] = frozenset({
    "measures", "unit", "definition", "aggregation_unit", "pooling", "weighting",
})
#: The value of `aggregation_unit` for a statistic computed over individual wedges of individual
#: junction edges, and for one computed over trijunction-line means. A line mean averages many
#: edges before an error is ever formed, so the two populations are not comparable and a
#: statistic must say which it is.
_PER_WEDGE = "per junction edge and region (a wedge)"
_PER_LINE = "per trijunction line and region (a wedge key), the line mean"
#: The value of `pooling` for a statistic formed inside one case. Every per-case block here is
#: pooled within its own case by construction; whether an aggregating script then pools the cases
#: or takes a median over their medians is that script's choice and it differs by up to `0.6`
#: degrees on this dataset, so the descriptor says which level it is describing.
_WITHIN_CASE = "pooled within this case; the cohort convention is the aggregating script's"

#: What every M1 statistic measures, in what unit, over what population, and how aggregated.
#: Emitted as `_statistics` in each reference block so a reader (or a figure caption) never has to
#: guess whether a number is a bias or an accuracy, nor whether it is a per-wedge or a per-line
#: quantity. `net bias` is a signed offset; `differential bias` is a slope against the wedge's own
#: deviation from 120 degrees; `accuracy` is a magnitude-of-error; `spread` is a dispersion of that
#: magnitude.
_STATISTIC_DESCRIPTORS: dict[str, dict[str, str]] = {
    "signed_error_median_deg": {
        "measures": "net bias",
        "unit": "degrees",
        "definition": "median over wedges of the signed error",
        "note": "alternating errors of opposite sign cancel here; this is not an accuracy statistic",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "none",
    },
    "median_abs_error_deg": {
        "measures": "accuracy",
        "unit": "degrees",
        "definition": "median over wedges of the absolute error",
        "note": "how wrong a typical wedge is; the statistic comparable with the analytic-wedge floor",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "none",
    },
    "abs_error_iqr_deg": {
        "measures": "spread",
        "unit": "degrees",
        "definition": "p75 minus p25 over wedges of the absolute error",
        "note": "dispersion of the accuracy statistic, so its centre is not read alone",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "none",
    },
    "abs_error_p90_deg": {
        "measures": "accuracy (tail)",
        "unit": "degrees",
        "definition": "90th percentile over wedges of the absolute error",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "none",
    },
    "net_offset_deg": {
        "measures": "reading artefact: minus one third of the mean angle-sum deficit per line",
        "unit": "degrees",
        "definition": "intercept a of signed_error ~ a + b * (true - 120), production length weights",
        "note": (
            "identical to flattening_slope_length_weighted.intercept; a name, not a new "
            "computation. RESTATED: this is NOT a net bias at a symmetric wedge. A line's three "
            "reference wedge angles sum to 360 and its three wedge keys share one weight, so the "
            "weighted mean of (true - 120) is exactly zero and the intercept is exactly the "
            "weighted mean signed error. That mean is in turn exactly -(1/3) times the "
            "length-weighted mean per-line angle-sum deficit, and the deficit exists only where "
            "a wedge is reflex and the unoriented reading returns 360 - theta for it. So "
            "net_offset_deg can only be zero or negative, it is zero exactly when no junction "
            "edge carries a reflex wedge, and it is identically zero under an oriented reading "
            "(measured: |a| <= 5.5e-15 degrees on every arm). It measures how much of the mesh "
            "was clipped, not how the geometry is biased -- compare arms on "
            "n_reflex_junction_edges instead, which is ground-truth-free."
        ),
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "production equation weights: this line's length over the mesh total",
    },
    "net_offset_ci95": {
        "measures": "uncertainty of net bias",
        "unit": "degrees",
        "definition": "wedge-level bootstrap interval on net_offset_deg, within this one case",
        "note": "WEDGE-level, not case-level: a per-case block cannot compute a case-level interval",
        "aggregation_unit": _PER_LINE,
        "pooling": "wedge-level bootstrap within this case, never a case-level interval",
        "weighting": "production equation weights",
    },
    "flattening_slope_unweighted": {
        "measures": "differential bias",
        "unit": "dimensionless (degrees of error per degree of deviation)",
        "definition": "slope b of signed_error ~ a + b * (true - 120), equal weights",
        "note": "the mandated primary slope",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "none -- the mandated primary slope",
    },
    "flattening_slope_length_weighted": {
        "measures": "differential bias",
        "unit": "dimensionless (degrees of error per degree of deviation)",
        "definition": "slope b with production's equation weights (line length / total length)",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "production equation weights: this line's length over the mesh total",
    },
    "matched_spacing_floor_deg": {
        "measures": "accuracy floor of an analytic control",
        "unit": "degrees",
        "definition": (
            "median|angle - 120| on an exact analytic 120-degree wedge whose every vertex is "
            "displaced by i.i.d. noise of r.m.s. magnitude 0.5 voxels, at an edge length equal "
            "to this mesh's own median triple-line edge"
        ),
        "note": (
            "an ABSOLUTE per-wedge quantity, so comparable with median_abs_error_deg and with "
            "nothing else; the 0.5 voxels is asserted, not measured, and the value scales as "
            "1/ell, so it is lower for a coarser mesh and is NOT comparable across meshes of "
            "different spacing"
        ),
        "aggregation_unit": _PER_WEDGE,
        "pooling": "a single analytic wedge, not an aggregate over this mesh",
        "weighting": "none",
    },
    "n_common_wedges": {
        "measures": "count",
        "unit": "wedges",
        "definition": "wedge keys present in both meshes",
        "aggregation_unit": _PER_LINE,
        "pooling": _WITHIN_CASE,
        "weighting": "none",
    },
}

_incomplete_descriptors = sorted(
    name for name, descriptor in _STATISTIC_DESCRIPTORS.items()
    if not descriptor.keys() >= _REQUIRED_DESCRIPTOR_KEYS
)
if _incomplete_descriptors:
    # Enforced at import rather than in a test, because the failure this prevents is a statistic
    # reaching a table without saying what population it is aggregated over -- which is how two
    # published comparisons in this project came to cross two conventions.
    _message = (
        f"these M1 descriptors are missing one of {sorted(_REQUIRED_DESCRIPTOR_KEYS)}: "
        f"{_incomplete_descriptors}"
    )
    raise RuntimeError(_message)


def _bootstrap_weighted_intercept_ci(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    *,
    n_draws: int = bootstrap.N_DRAWS,
    seed: int = bootstrap.SEED,
    alpha: float = 0.05,
) -> dict:
    """Wedge-level bootstrap interval on the length-weighted intercept, within one case.

    Resamples this case's own wedge indices with replacement and refits `y ~ a + b x` with the
    production length weights on each draw. **Wedge-level, not case-level:** the enclosing
    function sees exactly one case, so the case-level interval every cohort statistic in this
    package uses (`bootstrap.bootstrap_median_ci`) is not computable here and is computed by
    the aggregating script instead. `_STATISTIC_DESCRIPTORS` records the distinction so the
    two intervals can never be read as the same thing.

    Returns:
        dict: `{"lo", "hi", "level", "n_wedges", "n_draws", "n_draws_fitted", "seed"}`, with
            `lo`/`hi` `None` when fewer than three wedges are available.
    """
    n = len(x)
    empty = {"lo": None, "hi": None, "level": "wedge", "n_wedges": int(n), "n_draws": n_draws, "seed": seed}
    if n < 3:
        return empty
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(n_draws, n))
    draws = [
        fit["intercept"]
        for fit in (_weighted_slope(x[row], y[row], w[row]) for row in index)
        if fit["intercept"] is not None
    ]
    if not draws:
        return empty
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "level": "wedge",
        "n_wedges": int(n),
        "n_draws": n_draws,
        "n_draws_fitted": len(draws),
        "seed": seed,
    }


def _weighted_slope(x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64]) -> dict:
    """Weighted least-squares slope and intercept of `y` on `x`, weights `w`."""
    if len(x) < 2 or not np.any(w > 0):
        return {"slope": None, "intercept": None, "n": len(x)}
    w_sum = w.sum()
    x_bar = float((w * x).sum() / w_sum)
    y_bar = float((w * y).sum() / w_sum)
    sxx = float((w * (x - x_bar) ** 2).sum())
    sxy = float((w * (x - x_bar) * (y - y_bar)).sum())
    if sxx <= 0:
        return {"slope": None, "intercept": None, "n": len(x)}
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    return {"slope": float(slope), "intercept": float(intercept), "n": len(x)}


def compute_m1_contact_angles(
    mesh: DcelData,
    gt_mesh: DcelData,
    neumann_wedges_deg: dict[tuple, float] | None,
    matched_spacing_floor_deg: float | None,
) -> dict:
    """M1: three material-attributed wedge angles per trijunction line, two references.

    Uses the production convention `compute_angles_tri(unique=False)` on both the
    reconstruction and the ground-truth mesh, diffed key-by-key (wedge key `(min(e,g), f,
    max(e,g))`). `dict_length` is the same object production weights lines by
    (`dict_length_tri / total_length_mesh` in `foambryo.tension_inference`), so the
    length-weighted slope here uses the identical weights, not a re-derived approximation.

    **The angle reducer and the equation weighting are two separate things, and only the
    second is a length weighting.** Pinned here because conflating them would silently change
    what "production convention" means:

    * the **angle reducer** is a plain arithmetic mean over the per-edge wedge samples
      accumulated for a wedge key -- `foambryo/src/foambryo/geometry.py:652`,
      `dict_mean_angles[key] = np.mean(dict_angles[key])`. The per-edge lengths are
      accumulated *separately*, into `dict_length`, and never enter this mean;
    * the **equation weighting** is a row scaling applied afterwards, when the
      Young-Dupre system is assembled: `factor = dict_length_tri[(a, b, c)] /
      total_length_mesh` in `_build_matrix_tension_symmetrical_yd`
      (`foambryo/src/foambryo/tension_inference.py`).

    So an **arclength-weighted angle mean is never the production convention** and must never
    be introduced under that name. `weights_length` below reproduces the *equation* weighting
    and is applied only to the regression rows, exactly as production applies it to the solve.

    Every emitted statistic is described by the `_statistics` block, which names what each one
    measures -- net bias, differential bias, accuracy, spread or count -- because the two
    quantities this comparison has historically traded off are **both bias statistics** and
    were read under accuracy-sounding names.

    Args:
        mesh: the reconstruction, as a `DcelData`.
        gt_mesh: the ground-truth mesh, as a `DcelData`.
        neumann_wedges_deg: exact Neumann wedge angles in degrees, keyed like
            `compute_angles_tri`, or `None` when the case has no tensions.
        matched_spacing_floor_deg: the analytic-wedge floor in degrees, or `None`. Reported
            verbatim; see `_STATISTIC_DESCRIPTORS` for what it is and is not comparable with.

    Returns:
        dict: wedge counts plus one block per available reference.
    """
    angles_rad, _angles_deg, dict_length = compute_angles_tri(mesh, unique=False)
    gt_angles_rad, _gt_angles_deg, _gt_dict_length = compute_angles_tri(gt_mesh, unique=False)

    result: dict = {"n_reconstruction_wedges": len(angles_rad), "n_ground_truth_wedges": len(gt_angles_rad)}

    def _report_against(reference_rad: dict, label: str) -> dict:
        common = sorted(set(angles_rad) & set(reference_rad))
        if not common:
            return {"n_common_wedges": 0}
        errors_deg = np.degrees(np.array([angles_rad[k] - reference_rad[k] for k in common]))
        true_minus_120 = np.degrees(np.array([reference_rad[k] for k in common])) - 120.0
        total_length = sum(dict_length.values()) / 3 if dict_length else 0.0
        weights_length = np.array(
            [dict_length.get(k, 0.0) / total_length if total_length > 0 else 0.0 for k in common],
        )
        weights_uniform = np.ones(len(common))
        absolute_deg = np.abs(errors_deg)
        length_weighted = _weighted_slope(true_minus_120, errors_deg, weights_length)
        block = {
            "n_common_wedges": len(common),
            "signed_error_median_deg": float(np.median(errors_deg)),
            "abs_error_p90_deg": float(np.percentile(absolute_deg, 90)),
            "flattening_slope_unweighted": _weighted_slope(true_minus_120, errors_deg, weights_uniform),
            "flattening_slope_length_weighted": length_weighted,
            # The accuracy statistic and its spread. Added because every statistic above is a
            # bias statistic: the slope is the differential bias against Neumann deviation and
            # the signed median is the net bias, so how wrong a typical wedge actually is was
            # not reported at all. Additive: nothing above is removed or renamed, since the
            # historical results JSON carries those keys and must stay readable.
            "median_abs_error_deg": float(np.median(absolute_deg)),
            "abs_error_iqr_deg": float(np.percentile(absolute_deg, 75) - np.percentile(absolute_deg, 25)),
            # The net bias, named: the fitted intercept is the bias a Neumann-symmetric wedge
            # would carry, separated from the slope. Same number as
            # `flattening_slope_length_weighted["intercept"]`, given a name that says what it is.
            "net_offset_deg": length_weighted["intercept"],
            "net_offset_ci95": _bootstrap_weighted_intercept_ci(true_minus_120, errors_deg, weights_length),
        }
        if matched_spacing_floor_deg is not None:
            block["matched_spacing_floor_deg"] = matched_spacing_floor_deg
        block["_statistics"] = {k: v for k, v in _STATISTIC_DESCRIPTORS.items() if k in block}
        block["_note"] = f"reference: {label}; slope/floor are this run's own values, not cross-checked."
        # Raw per-wedge pairs: the flattening figure is built from the results JSON only, never
        # from a re-run, so the scatter it needs must already be here.
        block["raw_true_minus_120_deg"] = true_minus_120.tolist()
        block["raw_signed_error_deg"] = errors_deg.tolist()
        block["raw_length_weight"] = weights_length.tolist()
        return block

    result["vs_ground_truth_mesh"] = _report_against(gt_angles_rad, "ground-truth mesh's own line-mean angles")
    if neumann_wedges_deg is not None:
        neumann_rad = {k: np.radians(v) for k, v in neumann_wedges_deg.items()}
        result["vs_neumann"] = _report_against(neumann_rad, "true Neumann angles from ground-truth tensions")
    return result


def compute_m1_component_coverage(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
) -> dict:
    """How much of M1's wedge population sits on a material triple with several separate lines.

    A **diagnostic count, not a correction.** Production's angle dictionaries are keyed by
    material triple with no connected-component identity, so two disconnected trijunction lines
    sharing a triple are averaged into one wedge angle. Representing production faithfully means
    keeping that pooling; it does not mean leaving the reader unable to see how often it happens.

    Returns:
        dict: counts of triples and of multi-component triples, and the fraction of total
            triple-line arclength carried by multi-component triples.
    """
    by_triple = _group_edges_by_triple(_triple_line_edges(triangles, labels))
    total = 0.0
    pooled = 0.0
    n_multi = 0
    n_components = 0
    for edges in by_triple.values():
        components = _connected_components_of_edges(edges)
        n_components += len(components)
        arclength = _edge_arclength(points, edges)
        total += arclength
        if len(components) > 1:
            n_multi += 1
            pooled += arclength
    return {
        "n_triples": len(by_triple),
        "n_components": n_components,
        "n_multi_component_triples": n_multi,
        "triple_line_arclength_total": total,
        "arclength_fraction_on_multi_component_triples": (pooled / total) if total > 0 else None,
        "_note": (
            "wedge angles on a multi-component triple pool several disconnected lines into one "
            "average, by production's own convention; this is a count of that, not a change to it"
        ),
    }


# ---------------------------------------------------------------------------
# M2 -- interface areas
# ---------------------------------------------------------------------------


def compute_m2_interface_areas(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    mask: NDArray,
    gt: dict,
) -> dict:
    """M2: signed relative area error per interface, plus the scale-free area-fraction error."""
    measured = dwm.interface_areas(points, triangles, labels)
    registration = dwm.similarity_to_mask_frame(mask, gt["points"], gt["triangles"], gt["labels"])
    if registration.get("scale") is None:
        return {"error": "registration_failed", "n_correspondences": registration.get("n_correspondences", 0)}
    scale_sq = float(registration["scale"]) ** 2
    gt_areas = dwm.interface_areas(gt["points"], gt["triangles"], gt["labels"])
    reference = {k: v * scale_sq for k, v in gt_areas.items()}
    common = sorted(set(measured) & set(reference))
    if not common:
        return {"error": "no_common_interfaces"}
    relative = np.array([(measured[k] - reference[k]) / reference[k] for k in common])

    total_measured, total_reference = sum(measured.values()), sum(gt_areas.values())
    frac_measured = {k: measured[k] / total_measured for k in common}
    frac_reference = {k: gt_areas[k] / total_reference for k in common}
    area_fraction_error = np.array([frac_measured[k] - frac_reference[k] for k in common])

    return {
        "registration_residual_mean_voxels": registration["residual_mean_voxels"],
        "n_common_interfaces": len(common),
        "n_measured_interfaces": len(measured),
        "n_reference_interfaces": len(reference),
        "signed_relative_error_median": float(np.median(relative)),
        "abs_relative_error_p90": float(np.percentile(np.abs(relative), 90)),
        "total_area_ratio": float(sum(measured[k] for k in common) / sum(reference[k] for k in common)),
        "area_fraction_error_median": float(np.median(area_fraction_error)),
        "area_fraction_error_p90": float(np.percentile(np.abs(area_fraction_error), 90)),
        # Raw (measured, reference) pairs for the geometry-fidelity y=x figure, built from the
        # results JSON only.
        "raw_measured_area": [float(measured[k]) for k in common],
        "raw_reference_area": [float(reference[k]) for k in common],
    }


# ---------------------------------------------------------------------------
# M3 -- cell volumes
# ---------------------------------------------------------------------------


def compute_m3_cell_volumes(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    mask: NDArray,
    gt: dict,
    exterior: int = 0,
) -> dict:
    """M3: signed relative cell-volume error, exterior reported separately from cells."""
    measured = signed_cell_volumes(points, _as_int64(triangles), _as_int64(labels))
    registration = dwm.similarity_to_mask_frame(mask, gt["points"], gt["triangles"], gt["labels"])
    if registration.get("scale") is None:
        return {"error": "registration_failed"}
    scale_cubed = float(registration["scale"]) ** 3
    gt_volumes = signed_cell_volumes(gt["points"], _as_int64(gt["triangles"]), _as_int64(gt["labels"]))
    reference = {k: v * scale_cubed for k, v in gt_volumes.items()}

    def _relative_errors(keys: list[int]) -> dict:
        common = [k for k in keys if k in measured and k in reference and reference[k] != 0]
        if not common:
            return {"n_common": 0}
        relative = np.array([(measured[k] - reference[k]) / reference[k] for k in common])
        return {
            "n_common": len(common),
            "signed_relative_error_median": float(np.median(relative)),
            "abs_relative_error_p90": float(np.percentile(np.abs(relative), 90)),
            "raw_measured_volume": [float(measured[k]) for k in common],
            "raw_reference_volume": [float(reference[k]) for k in common],
        }

    cells = sorted(k for k in reference if k != exterior)
    return {
        "registration_residual_mean_voxels": registration["residual_mean_voxels"],
        "cells": _relative_errors(cells),
        "exterior": _relative_errors([exterior]),
    }


# ---------------------------------------------------------------------------
# M4 -- interface mean curvature
# ---------------------------------------------------------------------------


def compute_m4_curvature(
    mesh: DcelData,
    gt_mesh: DcelData,
    mask: NDArray,
    gt: dict,
    crossover_1_over_voxel: float,
) -> dict:
    """M4: mean curvature per interface, one estimator (`compute_curvature_interfaces`) on both sides.

    Ground-truth curvature is computed in the `.rec` mesh's own frame then divided by the
    mask-registration scale (curvature has units 1/length, so it scales as `1/scale`, the
    inverse of a length). Below `crossover_1_over_voxel` the *true* curvature is close enough
    to zero that a relative error is not meaningful; absolute error is reported there instead.
    """
    measured = compute_curvature_interfaces(mesh, weighted=True)
    gt_curvature_own_frame = compute_curvature_interfaces(gt_mesh, weighted=True)
    registration = dwm.similarity_to_mask_frame(mask, gt["points"], gt["triangles"], gt["labels"])
    if registration.get("scale") is None:
        return {"error": "registration_failed"}
    scale = float(registration["scale"])
    reference = {k: v / scale for k, v in gt_curvature_own_frame.items()}

    # `compute_curvature_interfaces` returns nan for an interface with no non-trijunction
    # vertex to read (the function's own "TEMPORARY" note); those are undetermined, not zero
    # error, and must not enter a bare median silently (case_metadata.median_of_determined).
    common = sorted(k for k in reference if k in measured)
    if not common:
        return {"error": "no_common_interfaces"}

    near_zero = [k for k in common if abs(reference[k]) < crossover_1_over_voxel]
    away = [k for k in common if abs(reference[k]) >= crossover_1_over_voxel]

    block: dict = {
        "n_common_interfaces": len(common),
        "crossover_1_over_voxel": crossover_1_over_voxel,
        "n_below_crossover": len(near_zero),
        "n_above_crossover": len(away),
        # Raw (measured, reference) pairs for the geometry-fidelity y=x figure.
        "raw_measured_curvature": [float(measured[k]) for k in common],
        "raw_reference_curvature": [float(reference[k]) for k in common],
    }
    if near_zero:
        abs_err = [abs(measured[k] - reference[k]) for k in near_zero]
        block["below_crossover_abs_error"] = case_metadata.median_of_determined(abs_err, allow_partial=True)
    if away:
        rel_err = [(measured[k] - reference[k]) / reference[k] for k in away]
        block["above_crossover_signed_relative_error"] = case_metadata.median_of_determined(rel_err, allow_partial=True)
        abs_rel_err = [v for v in np.abs(rel_err) if np.isfinite(v)]
        block["above_crossover_abs_relative_error_p90"] = float(np.percentile(abs_rel_err, 90)) if abs_rel_err else None
    return block


# ---------------------------------------------------------------------------
# M5 -- junction line geometry (provisional; no independent anchor)
# ---------------------------------------------------------------------------


def _triple_line_edges(triangles: NDArray, labels: NDArray) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Edge (u<v) -> sorted material triple, for edges with exactly 3 incident triangles and 3 materials."""
    edge_map = dwm._edge_to_triangle_map(triangles)
    out: dict[tuple[int, int], tuple[int, int, int]] = {}
    for (u, v), tri_ids in edge_map.items():
        if len(tri_ids) != 3:
            continue
        materials = {int(x) for t in tri_ids for x in labels[t]}
        if len(materials) == 3:
            m1, m2, m3 = sorted(materials)
            out[(u, v)] = (m1, m2, m3)
    return out


def _chain_polylines(
    edges_by_triple: dict[tuple[int, int, int], list[tuple[int, int]]],
) -> dict[tuple, list[list[int]]]:
    """Reduce each triple's edge set to simple vertex chains (branches broken at degree != 2)."""
    result: dict[tuple, list[list[int]]] = {}
    for triple, edges in edges_by_triple.items():
        adjacency: dict[int, set[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)
        visited_edges: set[tuple[int, int]] = set()
        chains: list[list[int]] = []
        # Start from endpoints/branch points (degree != 2) first so interior cycles are the only leftovers.
        starts = [n for n, nbrs in adjacency.items() if len(nbrs) != 2] or list(adjacency)
        for start in starts:
            for nxt in list(adjacency.get(start, ())):
                edge_key = (min(start, nxt), max(start, nxt))
                if edge_key in visited_edges:
                    continue
                chain = [start, nxt]
                visited_edges.add(edge_key)
                prev, cur = start, nxt
                while len(adjacency.get(cur, ())) == 2:
                    nxts = [n for n in adjacency[cur] if n != prev]
                    if not nxts:
                        break
                    nxt2 = nxts[0]
                    edge_key2 = (min(cur, nxt2), max(cur, nxt2))
                    if edge_key2 in visited_edges:
                        break
                    visited_edges.add(edge_key2)
                    chain.append(nxt2)
                    prev, cur = cur, nxt2
                chains.append(chain)
        result[triple] = chains
    return result


def _connected_components_of_edges(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Partition an edge list into connected components, as edge lists.

    A **component** is a connected piece of one material triple's triple-line edge graph. It is
    not the same thing as a **chain**: `_chain_polylines` additionally cuts a component wherever
    a vertex has degree != 2, so one component can yield several chains. The distinction matters
    because `compute_m5_junction_geometry` scores `max(chains, key=len)` -- the longest *chain* --
    and the coverage this loses is a chain-level loss on top of the component-level one.
    """
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    seen: set[int] = set()
    components: list[list[tuple[int, int]]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack, member = [start], set()
        seen.add(start)
        while stack:
            node = stack.pop()
            member.add(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append([(u, v) for u, v in edges if u in member and v in member])
    return components


def _edge_arclength(points: NDArray[np.float64], edges: list[tuple[int, int]]) -> float:
    """Total length of an edge list, in the mesh's own units."""
    if not edges:
        return 0.0
    index = np.asarray(edges, dtype=np.int64)
    return float(np.linalg.norm(points[index[:, 1]] - points[index[:, 0]], axis=1).sum())


def _polyline_stats(points: NDArray[np.float64], chain: list[int]) -> dict:
    verts = points[chain]
    segments = np.diff(verts, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    arclength = float(seg_lengths.sum())
    chord = float(np.linalg.norm(verts[-1] - verts[0]))
    tangents = segments / np.clip(seg_lengths, 1e-12, None)[:, None]
    return {"arclength": arclength, "chord": chord, "tangents": tangents, "vertices": verts}


def _m5_triple_metrics(
    points: NDArray[np.float64],
    gt: dict,
    gt_points_in_mask_frame: NDArray[np.float64] | None,
    recon_longest: list[int],
    gt_longest: list[int],
) -> dict | None:
    """Per-triple M5 metrics for one shared triple-line triple, or `None` to skip it.

    Split out of `compute_m5_junction_geometry` (same computations, unchanged) to keep that
    function's branching within the project's complexity budget.

    Returns:
        dict | None: `{"tortuosity_ratio", "curvature_ratio", "tangent_errors_deg",
            "position_errors"}` (the last two lists of per-sample values, `curvature_ratio`
            `None` when `gt_points_in_mask_frame` is `None`), or `None` when either polyline
            is too short or degenerate (zero chord) to score.
    """
    r_stats = _polyline_stats(points, recon_longest)
    # Ground truth polyline in the mask/voxel frame when a registration is available (needed
    # for tangent/position, harmless for tortuosity since it is a frame-free ratio); in its
    # own frame otherwise, which still gives tortuosity but skips tangent/position/curvature.
    gt_frame_points = gt_points_in_mask_frame if gt_points_in_mask_frame is not None else gt["points"]
    g_stats = _polyline_stats(gt_frame_points, gt_longest)
    if r_stats["chord"] <= 0 or g_stats["chord"] <= 0:
        return None

    r_tortuosity = r_stats["arclength"] / r_stats["chord"]
    g_tortuosity = g_stats["arclength"] / g_stats["chord"]
    result = {
        "tortuosity_ratio": r_tortuosity / g_tortuosity if g_tortuosity > 0 else np.nan,
        "curvature_ratio": None,
        "tangent_errors_deg": [],
        "position_errors": [],
    }

    if gt_points_in_mask_frame is None:
        return result  # no registration: tangent/position/curvature need a shared, oriented frame

    # Curvature proxy: total absolute turning angle between consecutive segments, per unit
    # arclength -- a discrete bending energy, not a fitted circle; ratio measured/true. Both
    # stats are now in the same (mask) frame, so the ratio needs no further unit conversion.
    def _turning_per_length(stats: dict) -> float:
        tangents = stats["tangents"]
        if len(tangents) < 2:
            return 0.0
        cosines = np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
        turning = np.sum(np.arccos(cosines))
        return float(turning / stats["arclength"]) if stats["arclength"] > 0 else 0.0

    r_curv, g_curv = _turning_per_length(r_stats), _turning_per_length(g_stats)
    result["curvature_ratio"] = r_curv / g_curv if g_curv > 0 else np.nan

    # Tangent error and position error: resample both to a common arclength parameter. Both
    # polylines are already in the mask frame, so no further transform is needed here.
    n_samples = max(2, min(len(recon_longest), len(gt_longest)))
    r_param = np.linspace(0, 1, n_samples)
    g_param = np.linspace(0, 1, n_samples)
    r_cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(r_stats["vertices"], axis=0), axis=1))])
    g_cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(g_stats["vertices"], axis=0), axis=1))])
    r_cum = r_cum / r_cum[-1] if r_cum[-1] > 0 else r_cum
    g_cum = g_cum / g_cum[-1] if g_cum[-1] > 0 else g_cum
    r_samples = np.stack([np.interp(r_param, r_cum, r_stats["vertices"][:, d]) for d in range(3)], axis=1)
    g_samples = np.stack([np.interp(g_param, g_cum, g_stats["vertices"][:, d]) for d in range(3)], axis=1)

    # A chain has no canonical start/end (`_chain_polylines` picks one arbitrarily), so the
    # two polylines' traversal directions need not agree; comparing them without checking
    # would silently score a perfectly-matched line at ~180 degrees half the time. Try both
    # orientations of the ground-truth samples and keep whichever aligns better.
    def _tangent_errors(g: NDArray[np.float64], r_samples: NDArray[np.float64] = r_samples) -> NDArray[np.float64]:
        r_tan = np.diff(r_samples, axis=0)
        g_tan = np.diff(g, axis=0)
        r_tan_n = r_tan / np.clip(np.linalg.norm(r_tan, axis=1), 1e-12, None)[:, None]
        g_tan_n = g_tan / np.clip(np.linalg.norm(g_tan, axis=1), 1e-12, None)[:, None]
        cosines = np.clip(np.sum(r_tan_n * g_tan_n, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(cosines))

    forward_errors = _tangent_errors(g_samples)
    reversed_errors = _tangent_errors(g_samples[::-1])
    if np.mean(reversed_errors) < np.mean(forward_errors):
        g_samples, matched_errors = g_samples[::-1], reversed_errors
    else:
        matched_errors = forward_errors
    result["tangent_errors_deg"] = matched_errors.tolist()
    result["position_errors"] = np.linalg.norm(r_samples - g_samples, axis=1).tolist()
    return result


def _longest_chain_of_component(component_edges: list[tuple[int, int]]) -> list[int]:
    """The longest chain inside one component, using `_chain_polylines`' own chaining rule."""
    chains = _chain_polylines({("component",): component_edges}).get(("component",), [])
    return max(chains, key=len, default=[])


def _component_index(
    points: NDArray[np.float64],
    edges_by_triple: dict[tuple, list[tuple[int, int]]],
) -> dict[tuple, list[dict]]:
    """Per triple, its components with centroid, arclength, and longest chain.

    Returns:
        dict: `triple -> [{"edges", "arclength", "centroid", "longest_chain",
            "longest_chain_arclength"}, ...]`, components ordered by descending arclength so a
            reader sees the dominant piece first.
    """
    out: dict[tuple, list[dict]] = {}
    for triple, edges in edges_by_triple.items():
        blocks = []
        for component in _connected_components_of_edges(edges):
            vertices = sorted({v for edge in component for v in edge})
            chain = _longest_chain_of_component(component)
            blocks.append(
                {
                    "edges": component,
                    "n_edges": len(component),
                    "arclength": _edge_arclength(points, component),
                    "centroid": points[vertices].mean(axis=0),
                    "longest_chain": chain,
                    "longest_chain_arclength": _edge_arclength(
                        points, [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)],
                    ),
                },
            )
        out[triple] = sorted(blocks, key=lambda b: -b["arclength"])
    return out


def _match_components_mutual_nearest(candidate: list[dict], reference: list[dict]) -> list[tuple[int, int]]:
    """Match components one-to-one by **mutual nearest centroid**.

    Each side's nearest counterpart must be the other's nearest too, so a large candidate
    component cannot claim several reference ones. Unmatched components on either side are the
    caller's to count -- they are a coverage fact, not something to absorb into a metric.
    """
    if not candidate or not reference:
        return []
    distances = np.linalg.norm(
        np.array([b["centroid"] for b in candidate])[:, None, :] - np.array([b["centroid"] for b in reference])[None],
        axis=2,
    )
    matches = []
    for i in range(len(candidate)):
        j = int(np.argmin(distances[i]))
        if int(np.argmin(distances[:, j])) == i:
            matches.append((i, j))
    return matches


def compute_m5_component_geometry(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    gt: dict,
    registration: dict | None,
) -> dict:
    """M5 with **connected-line identity**, reported alongside the triple-keyed path, never instead.

    `compute_m5_junction_geometry` reproduces what production sees: angle and junction
    dictionaries keyed by material triple with no connected-component identity, and, per triple,
    only the single longest chain scored. That reproduction is the contract and is left exactly
    as it is. This function measures the same geometry with each connected trijunction line kept
    separate, so a triple carrying two disconnected lines contributes two comparisons instead of
    one, and matches candidate lines to reference lines by mutual nearest centroid.

    Both are reported. Where a triple has exactly one component on both sides -- the ordinary
    case -- the two paths score the identical chain and therefore agree exactly;
    `tests/test_mesh_quality_junction_component_identity.py` pins that and pins the case where
    they must differ.

    Returns:
        dict: per-component aggregates in the same shape as the triple-keyed block, plus
            `n_components_*` counts, `n_unmatched_*` counts and the arclength coverage the
            triple-keyed path achieves.
    """
    recon_by_triple = _group_edges_by_triple(_triple_line_edges(triangles, labels))
    gt_by_triple = _group_edges_by_triple(_triple_line_edges(gt["triangles"], gt["labels"]))

    gt_points_in_mask_frame = None
    if registration is not None and registration.get("scale") is not None:
        gt_points_in_mask_frame = (
            float(registration["scale"]) * (registration["rotation"] @ gt["points"].T)
        ).T + registration["translation"]
    gt_frame_points = gt_points_in_mask_frame if gt_points_in_mask_frame is not None else gt["points"]

    recon_components = _component_index(points, recon_by_triple)
    gt_components = _component_index(gt_frame_points, gt_by_triple)

    common = sorted(set(recon_components) & set(gt_components))
    tortuosity, tangent_errors, curvature, position_errors = [], [], [], []
    n_matched = n_unmatched_recon = n_unmatched_gt = n_none = 0
    for triple in common:
        matches = _match_components_mutual_nearest(recon_components[triple], gt_components[triple])
        n_unmatched_recon += len(recon_components[triple]) - len(matches)
        n_unmatched_gt += len(gt_components[triple]) - len(matches)
        for i, j in matches:
            recon_chain = recon_components[triple][i]["longest_chain"]
            gt_chain = gt_components[triple][j]["longest_chain"]
            if len(recon_chain) < 2 or len(gt_chain) < 2:
                n_none += 1
                continue
            per = _m5_triple_metrics(points, gt, gt_points_in_mask_frame, recon_chain, gt_chain)
            if per is None:
                n_none += 1
                continue
            n_matched += 1
            tortuosity.append(per["tortuosity_ratio"])
            if per["curvature_ratio"] is not None:
                curvature.append(per["curvature_ratio"])
            tangent_errors.extend(per["tangent_errors_deg"])
            position_errors.extend(per["position_errors"])

    recon_total = sum(b["arclength"] for blocks in recon_components.values() for b in blocks)
    recon_longest_chain_total = sum(
        blocks[0]["longest_chain_arclength"] for blocks in recon_components.values() if blocks
    )
    return {
        "n_components_reconstruction": sum(len(b) for b in recon_components.values()),
        "n_components_ground_truth": sum(len(b) for b in gt_components.values()),
        "n_multi_component_triples_reconstruction": sum(1 for b in recon_components.values() if len(b) > 1),
        "n_multi_component_triples_ground_truth": sum(1 for b in gt_components.values() if len(b) > 1),
        "n_matched_components": n_matched,
        "n_unmatched_components_reconstruction": n_unmatched_recon,
        "n_unmatched_components_ground_truth": n_unmatched_gt,
        "n_scored_none": n_none,
        "triple_line_arclength_total": recon_total,
        "triple_line_arclength_in_longest_chain_per_triple": recon_longest_chain_total,
        "arclength_fraction_scored_by_triple_keyed_path": (
            recon_longest_chain_total / recon_total if recon_total > 0 else None
        ),
        "tortuosity_ratio_median": float(np.nanmedian(tortuosity)) if tortuosity else None,
        "tangent_error_mean_deg": float(np.mean(tangent_errors)) if tangent_errors else None,
        "tangent_error_p90_deg": float(np.percentile(tangent_errors, 90)) if tangent_errors else None,
        "curvature_ratio_median": float(np.nanmedian(curvature)) if curvature else None,
        "line_position_error_median_voxels": float(np.median(position_errors)) if position_errors else None,
        "_matching_rule": "mutual nearest centroid, one-to-one; unmatched components counted, never dropped silently",
        "_provisional": True,
    }


def _group_edges_by_triple(edge_map: dict) -> dict[tuple, list[tuple[int, int]]]:
    """Invert `edge -> triple` into `triple -> [edge, ...]`."""
    grouped: dict[tuple, list[tuple[int, int]]] = {}
    for edge, triple in edge_map.items():
        grouped.setdefault(triple, []).append(edge)
    return grouped


def compute_m5_junction_geometry(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    gt: dict,
    registration: dict | None,
) -> dict:
    """M5 (provisional): tortuosity, tangent error, curvature ratio, line position error.

    Both meshes' own trijunction polylines are built the same way (chain triple-line edges;
    branches at a degree != 2 vertex end a chain), so the comparison is symmetric -- no mask
    dependence beyond the registration. Tortuosity is scale- and rotation-free (an arclength
    ratio) and does not need `registration` at all; tangent angles and line positions do,
    since the ground-truth `.rec` frame's axes are in the *opposite order* to the mask's
    (`metrics.similarity_to_mask_frame`'s docstring: the fitted rotation has determinant -1 on
    every case tried) -- comparing raw, unregistered tangents or positions would measure that
    axis reversal, not reconstruction error. `registration` is the dict
    `metrics.similarity_to_mask_frame` returns (`scale`, `rotation`, `translation`); when it is
    `None` (registration failed upstream), only tortuosity is reported. Matching between the
    two meshes' polylines of the same material triple is done by which is the **longest**
    component per triple, since the two discretisations do not share vertices.
    """
    recon_edges = _triple_line_edges(triangles, labels)
    gt_edges = _triple_line_edges(gt["triangles"], gt["labels"])

    def _group(edge_map: dict) -> dict[tuple, list[tuple[int, int]]]:
        grouped: dict[tuple, list[tuple[int, int]]] = {}
        for edge, triple in edge_map.items():
            grouped.setdefault(triple, []).append(edge)
        return grouped

    recon_chains = _chain_polylines(_group(recon_edges))
    gt_chains = _chain_polylines(_group(gt_edges))

    gt_points_in_mask_frame = None
    scale = None
    if registration is not None and registration.get("scale") is not None:
        scale = float(registration["scale"])
        gt_points_in_mask_frame = (scale * (registration["rotation"] @ gt["points"].T)).T + registration["translation"]

    common_triples = sorted(set(recon_chains) & set(gt_chains))
    tortuosity_ratio, tangent_errors_deg, curvature_ratio, position_errors = [], [], [], []
    n_missing = len(set(gt_chains) - set(recon_chains))
    n_invented = len(set(recon_chains) - set(gt_chains))

    for triple in common_triples:
        recon_longest = max(recon_chains[triple], key=len, default=None)
        gt_longest = max(gt_chains[triple], key=len, default=None)
        if recon_longest is None or gt_longest is None or len(recon_longest) < 2 or len(gt_longest) < 2:
            continue
        per_triple = _m5_triple_metrics(points, gt, gt_points_in_mask_frame, recon_longest, gt_longest)
        if per_triple is None:
            continue
        tortuosity_ratio.append(per_triple["tortuosity_ratio"])
        if per_triple["curvature_ratio"] is not None:
            curvature_ratio.append(per_triple["curvature_ratio"])
        tangent_errors_deg.extend(per_triple["tangent_errors_deg"])
        position_errors.extend(per_triple["position_errors"])

    return {
        "n_common_triples": len(common_triples),
        "n_missing_triples": n_missing,
        "n_invented_triples": n_invented,
        "tortuosity_ratio_median": float(np.nanmedian(tortuosity_ratio)) if tortuosity_ratio else None,
        "tangent_error_mean_deg": float(np.mean(tangent_errors_deg)) if tangent_errors_deg else None,
        "tangent_error_p90_deg": float(np.percentile(tangent_errors_deg, 90)) if tangent_errors_deg else None,
        "curvature_ratio_median": float(np.nanmedian(curvature_ratio)) if curvature_ratio else None,
        "line_position_error_median_voxels": float(np.median(position_errors)) if position_errors else None,
        "_provisional": True,
        "_note": "M5 has no independent anchor; treat as provisional.",
    }


# ---------------------------------------------------------------------------
# M6 -- validity
# ---------------------------------------------------------------------------


def compute_m6_validity(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    gt: dict,
) -> dict:
    """M6: watertightness, abnormal edges, degenerate/duplicate triangles, quadjunctions, identity.

    Also two blocks that need no ground truth: the reflex-junction-edge counts, see
    :func:`compute_reflex_junction_edges`; and the **nonlocal self-intersection** counts, see
    :func:`.self_intersection.compute_self_intersections`.

    Every other number in this block is local or combinatorial, so a mesh whose two distant sheets
    pass through each other satisfies all of them. Measured, and the reason the nonlocal count is
    here: a post-process guarded only on the triangles incident to the vertex it moved left all
    twelve local predicates bit-identical on 40 of 40 benchmark cases while 23 nonlocal crossings
    existed on 5 of them. A local guard cannot certify a nonlocal property.

    The self-intersection numbers are **counts, never scores**, and no threshold is proposed, in
    exactly the sense the reflex counts are not a gate either.
    """
    integrity = mesh_integrity(points, _as_int64(triangles), _as_int64(labels))
    n_abnormal = dwm.abnormal_non_manifold_edge_count(points, triangles, labels)
    valence_split = dwm.valence_geq4_edges_by_material_count(triangles, labels)
    quality = _tet_free_quality_proxy(points, triangles)

    measured_interfaces = set(list_interfaces(_as_int64(labels)))
    reference_interfaces = set(list_interfaces(_as_int64(gt["labels"])))
    invented = sorted(measured_interfaces - reference_interfaces)
    missed = sorted(reference_interfaces - measured_interfaces)
    shared = sorted(measured_interfaces & reference_interfaces)
    split_interfaces = []
    for pair in shared:
        n_measured_components = _n_connected_components_of_interface(triangles, labels, pair)
        n_reference_components = _n_connected_components_of_interface(gt["triangles"], gt["labels"], pair)
        if n_measured_components > n_reference_components:
            split_interfaces.append(
                {"interface": pair, "measured": n_measured_components, "reference": n_reference_components},
            )

    return {
        "watertight": bool(integrity["n_boundary_edges"] == 0 and not integrity["dangling_vertex_reference"]),
        "n_boundary_edges": integrity["n_boundary_edges"],
        "n_abnormal_non_manifold_edges": n_abnormal,
        "n_degenerate_faces": integrity["n_degenerate_faces"],
        "n_duplicate_faces": integrity["n_duplicate_faces"],
        "n_triple_line_edges": integrity["n_triple_line_edges"],
        "n_malformed_triple_line_edges": integrity["n_malformed_triple_line_edges"],
        "n_quadjunction_edges_note": "expected discretisation artefact, not corruption",
        "n_quadjunction_edges": integrity["n_quadjunction_edges"],
        **compute_reflex_junction_edges(points, triangles, labels),
        **compute_self_intersections(points, triangles),
        "valence_geq4_split": valence_split,
        "sliver_fraction": quality["sliver_fraction"],
        "identity": {
            "n_invented_interfaces": len(invented),
            "invented_interfaces": [f"{a},{b}" for a, b in invented],
            "n_missed_interfaces": len(missed),
            "missed_interfaces": [f"{a},{b}" for a, b in missed],
            "n_split_interfaces": len(split_interfaces),
            "split_interfaces": split_interfaces,
            "_note": "merge across differing label pairs is not checked; see module docstring",
        },
    }


def compute_reflex_junction_edges(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
) -> dict:
    """Count the mesh's reflex trijunction wedges, and describe how far past straight they go.

    A trijunctional edge's three wedge angles partition the plane normal to it and sum to 360
    degrees. A wedge wider than a straight angle -- **reflex** -- means one region's triangles
    occupy more than half the ring around that edge. Two facts make this worth counting:

    * it needs **no ground truth and no reference mesh**, so it is usable on experimental data
      and as a target for junction surgery, unlike every angle-error statistic here;
    * `foambryo`'s default angle reading is `arccos` of a dot product, range `[0, 180]`, so it
      reports a reflex wedge as `360 - theta`. Every line-averaged angle statistic computed
      under that reading is therefore reading a reflex wedge as *less* wrong than it is, and
      the rate at which that happens differs by a factor of six between reconstruction
      strategies. `net_offset_deg` is exactly `-(1/3)` of the resulting length-weighted mean
      per-line angle-sum deficit; see its entry in `_STATISTIC_DESCRIPTORS`.

    Measured for orientation, not as a threshold: the simulator's own reference meshes carry 5
    reflex edges in 14 183 (0.04 %); reconstructions carry 2.2-3.2 % on the dithered samplers
    and 13.6-31.2 % on the seed-free ones, with a similar 15-27 degree excess per reflex edge on
    both -- the families differ in the *count*, not the severity.

    **This is not a gate and no threshold is proposed here.** It is a validity count, reported
    like the other counts in this block and never as a score.

    Units: degrees. Angles come from `foambryo.geometry.compute_trijunction_edges` under the
    oriented chord reading, which is bitwise identical to the default reading on every
    non-reflex wedge, so adding this count moves no other number in this module.

    Failure modes: a mesh with no trijunctional edge at all returns zero counts and `None`
    fractions rather than dividing by zero. Quadrijunctional edges are skipped, exactly as every
    other angle quantity here skips them.

    Args:
        points (NDArray[np.float64]): vertex coordinates, mesh units.
        triangles (NDArray): triangle vertex indices.
        labels (NDArray): each triangle's two material labels.

    Returns:
        dict: `n_junction_edges`, `n_reflex_junction_edges`,
            `reflex_junction_edge_fraction`, and the per-edge excess distribution
            (`reflex_excess_mean_deg`, `reflex_excess_p90_deg`, `reflex_excess_max_deg`).
    """
    mesh = DcelData(points, _as_int64(triangles), _as_int64(labels))
    excess = [
        float(np.degrees(max(edge.angles))) - 180.0
        for edge in compute_trijunction_edges(mesh, reading=ORIENTED_CHORD_READING)
        if float(np.degrees(max(edge.angles))) - 180.0 > REFLEX_TOLERANCE_DEG
    ]
    n_edges = len(compute_trijunction_edges(mesh, reading=ORIENTED_CHORD_READING))
    return {
        "n_junction_edges": n_edges,
        "n_reflex_junction_edges": len(excess),
        "reflex_junction_edge_fraction": (len(excess) / n_edges) if n_edges else None,
        "reflex_excess_mean_deg": float(np.mean(excess)) if excess else None,
        "reflex_excess_p90_deg": float(np.percentile(excess, 90)) if excess else None,
        "reflex_excess_max_deg": float(np.max(excess)) if excess else None,
    }


def _tet_free_quality_proxy(points: NDArray[np.float64], triangles: NDArray) -> dict:
    """Triangle-only sliver fraction (no tetrahedralisation available post-hoc): min-angle < 2 deg."""
    stats = dwm.triangle_quality_stats(points, triangles)
    return {"sliver_fraction": float(stats["min_angle_p10_deg"] < 2.0)}


def _n_connected_components_of_interface(triangles: NDArray, labels: NDArray, pair: tuple[int, int]) -> int:
    import networkx as nx

    mask = (labels[:, 0] == pair[0]) & (labels[:, 1] == pair[1]) | (labels[:, 0] == pair[1]) & (labels[:, 1] == pair[0])
    selected = triangles[mask]
    if len(selected) == 0:
        return 0
    graph = nx.Graph()
    graph.add_edges_from(selected[:, [0, 1]].tolist())
    graph.add_edges_from(selected[:, [0, 2]].tolist())
    graph.add_edges_from(selected[:, [1, 2]].tolist())
    return nx.number_connected_components(graph)


# ---------------------------------------------------------------------------
# M7 -- budget and cost
# ---------------------------------------------------------------------------


def compute_m7_budget(
    points: NDArray[np.float64],
    triangles: NDArray,
    labels: NDArray,
    wall_time_s: float,
    peak_rss_mb: float,
) -> dict:
    """M7: points, triangles, triangles per cell, wall-clock, peak RSS."""
    cell_labels = sorted({int(x) for x in labels.flatten() if x != 0})
    tris_per_cell = []
    for cell in cell_labels:
        count = int(np.count_nonzero((labels[:, 0] == cell) | (labels[:, 1] == cell)))
        tris_per_cell.append(count)
    return {
        "n_points": len(points),
        "n_triangles": len(triangles),
        "n_cells": len(cell_labels),
        "triangles_per_cell_mean": float(np.mean(tris_per_cell)) if tris_per_cell else None,
        "triangles_per_cell_min": int(np.min(tris_per_cell)) if tris_per_cell else None,
        "wall_time_s": wall_time_s,
        "peak_rss_mb": peak_rss_mb,
    }


def peak_rss_mb() -> float:
    """Process peak resident set size, in MB (portable across macOS/Linux `ru_maxrss` units)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_TO_MB


def timed_reconstruction(algo, mask: NDArray) -> tuple[tuple, float, float]:  # noqa: ANN001
    """Run one reconstruction, returning `((points, triangles, labels), wall_time_s, peak_rss_mb)`."""
    t0 = time.perf_counter()
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)
    wall = time.perf_counter() - t0
    return (points, triangles, labels), wall, peak_rss_mb()
