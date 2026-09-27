r"""Sub-voxel junction-position estimation from the label mask alone.

Given a point believed to lie on the trijunction line of three materials, a direction along
that line, and the label mask, this module returns the position the mask's own evidence
prefers. It is the estimator half of `junction_curves.extract_junction_curves`; the detection
half lives there.

Provenance
----------
The linear program, the window rule, the wedge model and every numerical constant are from
the identifiability-floor work
(see the project's research history, functions `normal_plane_basis`, `voxels_in_window`,
`wedge_structure`, `signed_distance_rows`, `build_polytope`, `support_function`,
`polygon_measures`, `max_margin_point`, `apply_delta_to_configuration`, `rodrigues`),
ported here unchanged in behaviour. The
mask-only interface-azimuth initialisation and its coordinate-descent refinement are from
the junction-curve-extraction work
(see the project's research history), also unchanged.

The identifiability-floor work built the polytope to measure an *identifiability floor* and
read the reference mesh to orient its wedges. The junction-curve-extraction work replaced
that one ground-truth-reading step and turned the polytope into an estimator. This module
carries the junction-curve-extraction work's version, and only the part of the
identifiability-floor work that the two estimators need: the floor itself, its exact
non-linear validation, its bisection, its synthetic controls and its statistics are **not**
ported.

The model, in one paragraph
---------------------------
Around a sample point `p0` with line direction `d0`, take every integer voxel centre inside
the cylinder of radius `R = 3` voxels and half-length `R`. Model the local geometry as three
flat half-planes meeting on a line: seven parameters -- the line's in-plane position
`(u, v)`, its tilt `(a, b)`, and one azimuth offset per interface `dphi_1..3`. A voxel of
material `m` is explained when it lies inside `m`'s wedge, which is two linear inequalities in
those seven parameters. Stacking them gives a polytope `A delta <= b` of every flat
configuration that voxelises to exactly the labels observed.

Two points of that polytope are exposed:

* **`anchor`** -- its deepest interior point (largest Chebyshev margin, one LP). A later
  measurement in the junction-curve-extraction work found its displacement slope at
  `-0.01165` voxels/degree of narrow-wedge deficit, against the identifiability-floor
  work's decomposition term (a) `-0.01123`: the anchor carries the irreducible 79 % of the
  narrow-wedge systematic and not the 21 % that is the feasible set's internal asymmetry. **That
  smaller systematic is its argument**, not the 9-11 % of position error it also buys -- which
  is not worth its cost, which is why `extract_junction_curves` does not run it by default.
* **`centroid`** -- the area centroid of the feasible set's projection onto `(u, v)`, which
  needs the full support function (`N_DIRECTIONS = 32` LPs). The junction-curve-extraction
  work measured its slope at `-0.01435` against the identifiability-floor work's total
  `-0.01414`: it carries **(a) + (b)**, so it is the larger systematic, not the smaller one.
  It is exposed to reproduce that measurement and is not a recommendation.

Neither point reaches the mask's identifiability floor: the junction-curve-extraction work
measured 0.5236 (anchor) and 0.5151 (centroid) voxels against the identifiability-floor
work's exact floor of 0.107, a factor of 4.8.

No ground truth
---------------
Nothing here reads a reference mesh, a `.rec` file, a tension or a pressure. The only inputs
are the label mask, a point, a direction and a material triple. The junction-curve-extraction
work proved this operationally by running the estimator in a process where the `.rec` files
were structurally unreachable and obtaining a hash-identical result;
the project's research history repeats that proof for this port.

Cost
----
The estimator is the expensive half of the API and is reported as such rather than hidden.
Per sample the junction-curve-extraction work measured 75 ms for anchor and centroid
together; the anchor alone needs 1-7 LPs rather than 33-39. `extract_junction_curves`
therefore defaults to no estimator at all: detection costs about 0.8 s per benchmark case
against dw3d's own 1.49 s to reconstruct it, while the anchor path costs several times the
whole reconstruction. See the project's research history for where that time goes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

# --------------------------------------------------------------------------------------
# The identifiability-floor work's constants, carried as fixed documented defaults. None of
# these is tuned here.
# --------------------------------------------------------------------------------------
#: Window radius, in voxels. Derived from a 0.1-voxel sagitta budget against the
#: curvature work's measured median interface curvature: floor(sqrt(2 * 0.1 / 0.0217)) = 3.
DEFAULT_RADIUS_VOXELS = 3
#: Support directions for the centroid's polygon. 32 gives 11.25 deg spacing.
N_DIRECTIONS = 32
#: The wedge model requires convex wedges; these bound where it is claimed to hold.
WEDGE_WIDTH_MIN_DEG = 10.0
WEDGE_WIDTH_MAX_DEG = 170.0
#: A-priori box on the parameters so every LP is bounded. A hit on `u`/`v` is reported.
BOX_POSITION_VOXELS = 5.0
#: Derived from the same 0.1-voxel budget that fixed the window radius: sqrt(2 * 0.1 / 3) = 0.258 rad.
BOX_ANGLE_RAD = 0.25
#: Minimum voxel centres in a window before the LP is attempted (7 parameters, two rows each).
MIN_VOXELS_IN_WINDOW = 20
#: Re-anchoring iterations, and the step below which the anchor is called converged.
N_REANCHOR_ITERATIONS = 6
ANCHOR_CONVERGENCE_VOXELS = 0.01

# --------------------------------------------------------------------------------------
# The junction-curve-extraction work's constants for the mask-only azimuth initialisation.
# Also fixed, also documented.
# --------------------------------------------------------------------------------------
#: Half-width of the coordinate-descent scan on each interface azimuth, in degrees.
AZIMUTH_SEARCH_DEG = 30.0
#: Scan step. At `R = 3` voxels one degree subtends 0.05 voxels.
AZIMUTH_STEP_DEG = 1.0
#: Rounds of coordinate descent. Deterministic; no random restarts.
AZIMUTH_ROUNDS = 3

PARAMETER_NAMES = ("u", "v", "a", "b", "dphi_1", "dphi_2", "dphi_3")
N_PARAMETERS = 7

BOUNDS = [
    (-BOX_POSITION_VOXELS, BOX_POSITION_VOXELS),
    (-BOX_POSITION_VOXELS, BOX_POSITION_VOXELS),
    (-BOX_ANGLE_RAD, BOX_ANGLE_RAD),
    (-BOX_ANGLE_RAD, BOX_ANGLE_RAD),
    (-BOX_ANGLE_RAD, BOX_ANGLE_RAD),
    (-BOX_ANGLE_RAD, BOX_ANGLE_RAD),
    (-BOX_ANGLE_RAD, BOX_ANGLE_RAD),
]
_BOX = np.array([[low, high] for low, high in BOUNDS])
_ANGLES = np.arange(N_DIRECTIONS) * (2.0 * np.pi / N_DIRECTIONS)
_DIRECTIONS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)

_TINY = 1e-9


# --------------------------------------------------------------------------------------
# frame, window, wedges
# --------------------------------------------------------------------------------------
def normal_plane_basis(direction: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Right-handed `(e1, e2)` spanning the plane normal to `direction`, deterministically.

    The seed axis is the world axis least aligned with `direction`, so the basis is a function
    of `direction` alone and two runs report comparable `(u, v)` coordinates.

    Args:
        direction (NDArray[np.float64]): a non-zero 3-vector along the junction line.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: the two unit vectors `(e1, e2)`.
    """
    d = direction / np.linalg.norm(direction)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(d)))] = 1.0
    e1 = seed - np.dot(seed, d) * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2


def voxels_in_window(
    segmented_mask: NDArray[np.uint],
    centre: NDArray[np.float64],
    direction: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Integer voxel centres inside the cylinder of radius `radius` and half-length `radius`.

    Args:
        segmented_mask (NDArray[np.uint]): the label image; mesh coordinates are its voxel indices.
        centre (NDArray[np.float64]): the cylinder's centre, in voxel coordinates.
        direction (NDArray[np.float64]): the unit cylinder axis.
        radius (float): both the radius and the half-length, in voxels.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64]]: offsets from `centre`, and the label at each.
    """
    low = np.maximum(np.floor(centre - radius - 1.0).astype(int), 0)
    high = np.minimum(np.ceil(centre + radius + 1.0).astype(int) + 1, np.array(segmented_mask.shape))
    if np.any(high <= low):
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    grid = np.stack(
        np.meshgrid(
            np.arange(low[0], high[0]),
            np.arange(low[1], high[1]),
            np.arange(low[2], high[2]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    offsets = grid.astype(float) - centre
    axial = offsets @ direction
    radial = np.linalg.norm(offsets - axial[:, None] * direction, axis=1)
    keep = (radial <= radius) & (np.abs(axial) <= radius)
    kept = grid[keep]
    materials = segmented_mask[kept[:, 0], kept[:, 1], kept[:, 2]].astype(np.int64)
    return offsets[keep], materials


def wedge_structure(interfaces: list[dict], triple: tuple[int, int, int]) -> dict | None:
    """Order the three interfaces by azimuth and assign a material to each wedge.

    Args:
        interfaces (list[dict]): three `{"pair": (m, n), "azimuth": radians}` records.
        triple (tuple[int, int, int]): the material triple the wedges must cover.

    Returns:
        dict | None: `{"wedges": [...], "order": [...]}`, or `None` when the three azimuths do
        not separate the three materials into convex wedges -- i.e. when the flat
        three-half-plane model does not describe this neighbourhood at all. Those samples are
        refused and counted, never forced.
    """
    order = sorted(range(3), key=lambda i: interfaces[i]["azimuth"])
    wedges = []
    for k in range(3):
        start = order[k]
        end = order[(k + 1) % 3]
        width = (interfaces[end]["azimuth"] - interfaces[start]["azimuth"]) % (2.0 * np.pi)
        shared_materials = set(interfaces[start]["pair"]) & set(interfaces[end]["pair"])
        if len(shared_materials) != 1:
            return None
        material = int(next(iter(shared_materials)))
        if not (np.radians(WEDGE_WIDTH_MIN_DEG) < width < np.radians(WEDGE_WIDTH_MAX_DEG)):
            return None
        wedges.append({"material": material, "start": start, "end": end, "width_rad": float(width)})
    if {w["material"] for w in wedges} != {int(t) for t in triple}:
        return None
    return {"wedges": wedges, "order": order}


# --------------------------------------------------------------------------------------
# the mask-only azimuth estimate (the junction-curve-extraction work's replacement for
# the identifiability-floor work's reference-reading one)
# --------------------------------------------------------------------------------------
def mask_interface_azimuths(
    offsets: NDArray[np.float64],
    materials: NDArray[np.int64],
    triple: tuple[int, int, int],
    frame: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> list[dict] | None:
    """The three interface azimuths, from the window's voxel labels alone.

    Each material's own voxels give a circular mean direction -- its wedge's centre. An
    interface between two angularly adjacent wedges starts at the angular midpoint between
    their centres, and the three azimuths are then refined by coordinate descent on the number
    of window voxels the wedge model classifies correctly.

    Args:
        offsets (NDArray[np.float64]): voxel offsets from the sample point.
        materials (NDArray[np.int64]): the label at each offset.
        triple (tuple[int, int, int]): the material triple.
        frame (tuple): `(e1, e2, direction)`.

    Returns:
        list[dict] | None: three `{"pair", "azimuth"}` records, or `None` when a material has no
        voxel in the window or its voxels carry no net direction.
    """
    e1, e2, _direction = frame
    x = offsets @ e1
    y = offsets @ e2
    radius = np.hypot(x, y)
    keep = radius > _TINY  # a voxel centre exactly on the line carries no azimuth
    if keep.sum() < MIN_VOXELS_IN_WINDOW:
        return None
    theta = np.arctan2(y[keep], x[keep])
    labels = materials[keep]

    centres: dict[int, float] = {}
    for material in triple:
        selected = labels == material
        if not selected.any():
            return None
        mean_x = float(np.cos(theta[selected]).mean())
        mean_y = float(np.sin(theta[selected]).mean())
        if np.hypot(mean_x, mean_y) < _TINY:
            return None
        centres[int(material)] = float(np.arctan2(mean_y, mean_x))

    ordered = sorted(centres, key=lambda m: centres[m])
    interfaces: list[dict] = []
    for k in range(3):
        first, second = ordered[k], ordered[(k + 1) % 3]
        gap = (centres[second] - centres[first]) % (2.0 * np.pi)
        interfaces.append(
            {"pair": tuple(sorted((first, second))), "azimuth": float(centres[first] + 0.5 * gap)},
        )
    return _refine_azimuths(interfaces, theta, labels, triple)


def _wedge_material_at(azimuths: NDArray[np.float64], interfaces: list[dict]) -> NDArray[np.int64]:
    """For each azimuth, the material the wedge model assigns it, or `-1` if undefined."""
    order = sorted(range(3), key=lambda i: interfaces[i]["azimuth"])
    boundaries = np.array([interfaces[i]["azimuth"] for i in order])
    assigned = np.full(len(azimuths), -1, dtype=np.int64)
    for k in range(3):
        start_index, end_index = order[k], order[(k + 1) % 3]
        shared_materials = set(interfaces[start_index]["pair"]) & set(interfaces[end_index]["pair"])
        if len(shared_materials) != 1:
            return assigned
        material = int(next(iter(shared_materials)))
        start = boundaries[k]
        width = (boundaries[(k + 1) % 3] - start) % (2.0 * np.pi)
        within = ((azimuths - start) % (2.0 * np.pi)) < width
        assigned[within] = material
    return assigned


def _score(azimuths: NDArray[np.float64], labels: NDArray[np.int64], interfaces: list[dict]) -> int:
    """How many window voxels the wedge model classifies correctly."""
    return int((_wedge_material_at(azimuths, interfaces) == labels).sum())


#: The six orderings of three interfaces, and for each the two-interface pairs that bound each
#: wedge. Precomputed because the ordering is all that decides which material a wedge carries.
_ORDERINGS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
)


def _materials_per_ordering(interfaces: list[dict], triple: tuple[int, int, int]) -> dict:
    """Per ordering of the three interfaces, the material of each wedge, or `None` if inconsistent.

    The material a wedge carries is the one its two bounding interfaces share, which depends on
    the interfaces' *order* and not at all on their azimuths -- so for a coordinate-descent scan
    over one azimuth there are only six answers, and they can be computed once.
    """
    per_ordering: dict[tuple[int, int, int], list[int] | None] = {}
    wanted = {int(t) for t in triple}
    for ordering in _ORDERINGS:
        materials: list[int] = []
        for k in range(3):
            shared = set(interfaces[ordering[k]]["pair"]) & set(interfaces[ordering[(k + 1) % 3]]["pair"])
            if len(shared) != 1:
                materials = []
                break
            materials.append(int(next(iter(shared))))
        per_ordering[ordering] = materials if materials and set(materials) == wanted else None
    return per_ordering


def _scan_one_azimuth(
    azimuths: NDArray[np.float64],
    labels: NDArray[np.int64],
    current: list[dict],
    index: int,
    trials: NDArray[np.float64],
    per_ordering: dict,
) -> tuple[int, float]:
    """Score every trial azimuth for interface `index` at once, and return the best.

    This is the scalar loop over `wedge_structure` + `_score` rewritten as array work, and it is
    the same computation in the same order: the widths are the same expression, the assignment
    is the same `(theta - start) mod 2pi < width` test applied for `k = 0, 1, 2` with later
    wedges overwriting earlier ones, and ties are broken by the first trial in ascending order,
    exactly as `if value > best_score` does. `tests/test_junction_curves.py` asserts the two
    agree exactly on random configurations; the module keeps `_score` and `_wedge_material_at`
    as the readable definition this is checked against.

    Measured on 360 real samples, the scalar loop was 42 % of the anchor path's wall time --
    549 Python calls per scan, each rebuilding three dictionaries -- against 5 % for assembling
    the polytope and 48 % for the linear programs themselves (see the project's research
    history).

    Returns:
        tuple[int, float]: the best score and its azimuth; the score is `-1` when no trial gives
        a consistent wedge structure, which the caller turns into a refusal.
    """
    n_trials = len(trials)
    candidate = np.empty((n_trials, 3), dtype=np.float64)
    candidate[:, 0] = current[0]["azimuth"]
    candidate[:, 1] = current[1]["azimuth"]
    candidate[:, 2] = current[2]["azimuth"]
    candidate[:, index] = trials

    order = np.argsort(candidate, axis=1, kind="stable")
    boundaries = np.take_along_axis(candidate, order, axis=1)
    widths = (np.roll(boundaries, -1, axis=1) - boundaries) % (2.0 * np.pi)

    materials = np.zeros((n_trials, 3), dtype=np.int64)
    valid = np.zeros(n_trials, dtype=bool)
    for ordering, per_wedge in per_ordering.items():
        selected = (order == np.asarray(ordering)).all(axis=1)
        if per_wedge is None or not selected.any():
            continue
        materials[selected] = np.asarray(per_wedge, dtype=np.int64)
        valid |= selected
    valid &= (widths > np.radians(WEDGE_WIDTH_MIN_DEG)).all(axis=1)
    valid &= (widths < np.radians(WEDGE_WIDTH_MAX_DEG)).all(axis=1)
    if not valid.any():
        return -1, float(current[index]["azimuth"])

    offsets = (azimuths[None, None, :] - boundaries[:, :, None]) % (2.0 * np.pi)
    within = offsets < widths[:, :, None]
    assigned = np.full((n_trials, len(azimuths)), -1, dtype=np.int64)
    for k in range(3):
        assigned = np.where(within[:, k, :], materials[:, k, None], assigned)
    scores = (assigned == labels[None, :]).sum(axis=1)
    scores = np.where(valid, scores, -1)
    best = int(np.argmax(scores))
    return int(scores[best]), float(trials[best])


def _refine_azimuths(
    interfaces: list[dict],
    azimuths: NDArray[np.float64],
    labels: NDArray[np.int64],
    triple: tuple[int, int, int],
) -> list[dict] | None:
    """Coordinate descent on the classification score. Deterministic; no restarts."""
    current = [dict(interface) for interface in interfaces]
    per_ordering = _materials_per_ordering(current, triple)
    steps = np.radians(
        np.arange(-AZIMUTH_SEARCH_DEG, AZIMUTH_SEARCH_DEG + AZIMUTH_STEP_DEG, AZIMUTH_STEP_DEG),
    )
    for _round in range(AZIMUTH_ROUNDS):
        for index in range(3):
            base = current[index]["azimuth"]
            best_score, best_azimuth = _scan_one_azimuth(
                azimuths, labels, current, index, base + steps, per_ordering,
            )
            if best_score < 0:
                return None
            current[index]["azimuth"] = best_azimuth
    return current if wedge_structure(current, triple) is not None else None


def _refine_azimuths_scalar(
    interfaces: list[dict],
    azimuths: NDArray[np.float64],
    labels: NDArray[np.int64],
    triple: tuple[int, int, int],
) -> list[dict] | None:
    """The junction-curve-extraction work's own loop, kept as the definition `_refine_azimuths` is checked against.

    Not called in production. `tests/test_junction_curves.py` asserts the two agree exactly.
    """
    current = [dict(interface) for interface in interfaces]
    steps = np.radians(
        np.arange(-AZIMUTH_SEARCH_DEG, AZIMUTH_SEARCH_DEG + AZIMUTH_STEP_DEG, AZIMUTH_STEP_DEG),
    )
    for _round in range(AZIMUTH_ROUNDS):
        for index in range(3):
            base = current[index]["azimuth"]
            best_score, best_azimuth = -1, base
            for step in steps:
                trial = [dict(i) for i in current]
                trial[index]["azimuth"] = float(base + step)
                if wedge_structure(trial, triple) is None:
                    continue
                value = _score(azimuths, labels, trial)
                if value > best_score:
                    best_score, best_azimuth = value, float(base + step)
            if best_score < 0:
                return None
            current[index]["azimuth"] = best_azimuth
    return current if wedge_structure(current, triple) is not None else None


# --------------------------------------------------------------------------------------
# the linear model and its polytope
# --------------------------------------------------------------------------------------
def signed_distance_rows(
    offsets: NDArray[np.float64],
    azimuth: float,
    interface_index: int,
    frame: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`(gradient, constant)` of one interface's signed distance, linearised in the 7 parameters.

    Args:
        offsets (NDArray[np.float64]): voxel offsets `r = x - p0`, shape `(n, 3)`.
        azimuth (float): the interface's azimuth in the normal plane, in radians.
        interface_index (int): which of the three `dphi` parameters this interface owns.
        frame (tuple): `(e1, e2, direction)`.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: gradient `(n, 7)` and constant `(n,)`.
    """
    e1, e2, _direction = frame
    m = np.cos(azimuth) * e1 + np.sin(azimuth) * e2
    n = -np.sin(azimuth) * e1 + np.cos(azimuth) * e2

    constant = offsets @ n
    cross = np.cross(np.broadcast_to(n, offsets.shape), offsets)

    gradient = np.zeros((len(offsets), N_PARAMETERS))
    gradient[:, 0] = np.sin(azimuth)
    gradient[:, 1] = -np.cos(azimuth)
    gradient[:, 2] = cross @ e2
    gradient[:, 3] = -(cross @ e1)
    gradient[:, 4 + interface_index] = -(offsets @ m)
    return gradient, constant


def build_polytope(
    offsets: NDArray[np.float64],
    voxel_materials: NDArray[np.int64],
    interfaces: list[dict],
    structure: dict,
    frame: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> dict:
    """Assemble `A delta <= b`: label identity over the window, plus the wedge-width rows.

    Every voxel of a known material contributes two rows, one per bounding interface of its
    own wedge. Voxels of a foreign material are excluded and counted; voxels the anchor
    configuration itself misclassifies are flagged in `violated_at_anchor` but their rows are
    kept, because dropping them would report the feasible set of a model that does not
    describe the data.

    `n_excluded_anchor_violating` counts those. The identifiability-floor work called the
    same count
    `n_excluded_truth_violating`, where "truth" named the *anchor configuration* and never any
    ground truth; the field is renamed here so that nothing in a production module reads as
    though it had seen a reference mesh.

    Args:
        offsets (NDArray[np.float64]): voxel offsets from the sample point.
        voxel_materials (NDArray[np.int64]): the label at each offset.
        interfaces (list[dict]): the three `{"pair", "azimuth"}` records.
        structure (dict): `wedge_structure`'s output for those interfaces.
        frame (tuple): `(e1, e2, direction)`.

    Returns:
        dict: `A`, `b`, the boolean masks `used` and `violated_at_anchor`, and three counts.
    """
    wedge_of_material = {w["material"]: w for w in structure["wedges"]}
    rows: list[NDArray[np.float64]] = []
    rhs: list[NDArray[np.float64]] = []
    n_excluded_anchor_violating = 0
    n_used = 0
    used = np.zeros(len(offsets), dtype=bool)
    violated = np.zeros(len(offsets), dtype=bool)

    for material, wedge in wedge_of_material.items():
        selected = voxel_materials == material
        if not selected.any():
            continue
        local_offsets = offsets[selected]
        start, end = wedge["start"], wedge["end"]
        grad_start, const_start = signed_distance_rows(local_offsets, interfaces[start]["azimuth"], start, frame)
        grad_end, const_end = signed_distance_rows(local_offsets, interfaces[end]["azimuth"], end, frame)
        satisfied = (const_start >= 0.0) & (const_end <= 0.0)
        n_excluded_anchor_violating += int((~satisfied).sum())
        n_used += int(selected.sum())
        indices = np.flatnonzero(selected)
        used[indices] = True
        violated[indices[~satisfied]] = True
        rows.append(-grad_start)
        rhs.append(const_start)
        rows.append(grad_end)
        rhs.append(-const_end)

    known = np.isin(voxel_materials, list(wedge_of_material))
    n_excluded_foreign = int((~known).sum())

    width_rows = []
    width_rhs = []
    for wedge in structure["wedges"]:
        row = np.zeros(N_PARAMETERS)
        row[4 + wedge["end"]] = 1.0
        row[4 + wedge["start"]] = -1.0
        width_rows.append(row)
        width_rhs.append(np.radians(WEDGE_WIDTH_MAX_DEG) - wedge["width_rad"])
        width_rows.append(-row)
        width_rhs.append(wedge["width_rad"] - np.radians(WEDGE_WIDTH_MIN_DEG))
    rows.append(np.asarray(width_rows))
    rhs.append(np.asarray(width_rhs))

    return {
        "A": np.vstack(rows),
        "b": np.concatenate(rhs),
        "used": used,
        "violated_at_anchor": violated,
        "n_voxels_used": n_used,
        "n_excluded_foreign": n_excluded_foreign,
        "n_excluded_anchor_violating": n_excluded_anchor_violating,
    }


def max_margin_point(polytope: dict) -> dict:
    """The deepest interior point of `A delta <= b`, with its margin in voxels.

    Maximise `m` subject to `(a_i / |a_i|) . delta + m <= b_i / |a_i|` over every row. Row
    normalisation makes `m` a distance in voxels rather than a mixture of voxels and radians.

    Args:
        polytope (dict): `build_polytope`'s output.

    Returns:
        dict: `{"margin_voxels": float | None, "delta": NDArray | None}`; `None` when the LP fails.
    """
    a_ub, b_ub = polytope["A"], polytope["b"]
    norms = np.linalg.norm(a_ub, axis=1)
    keep = norms > 1e-12
    a_normalised = a_ub[keep] / norms[keep, None]
    b_normalised = b_ub[keep] / norms[keep]
    augmented = np.column_stack([a_normalised, np.ones(len(a_normalised))])
    cost = np.zeros(N_PARAMETERS + 1)
    cost[-1] = -1.0
    result = linprog(cost, A_ub=augmented, b_ub=b_normalised, bounds=[*BOUNDS, (None, None)], method="highs")
    if not result.success:
        return {"margin_voxels": None, "delta": None}
    return {"margin_voxels": float(result.x[-1]), "delta": np.asarray(result.x[:-1], dtype=float)}


def support_function(polytope: dict) -> dict | None:
    """Maximise `w . (u, v)` over the polytope for all `N_DIRECTIONS` support directions.

    Args:
        polytope (dict): `build_polytope`'s output.

    Returns:
        dict | None: support values, the optimal 7-vectors, and whether the a-priori box was
        active at any optimum. `None` when any LP fails.
    """
    a_ub, b_ub = polytope["A"], polytope["b"]
    values = np.full(N_DIRECTIONS, np.nan)
    solutions = np.full((N_DIRECTIONS, N_PARAMETERS), np.nan)
    box_hits = np.zeros(N_PARAMETERS, dtype=int)
    for k, direction in enumerate(_DIRECTIONS):
        cost = np.zeros(N_PARAMETERS)
        cost[0] = -direction[0]
        cost[1] = -direction[1]
        result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=BOUNDS, method="highs")
        if not result.success:
            return None
        values[k] = float(-result.fun)
        solutions[k] = result.x
        box_hits += (np.abs(np.abs(result.x) - np.abs(_BOX).max(axis=1)) < 1e-7).astype(int)
    return {
        "support": values,
        "solutions": solutions,
        "box_hits": box_hits.tolist(),
        "box_active_position": bool(box_hits[0] or box_hits[1]),
        "box_active_any": bool(box_hits.any()),
    }


def polygon_measures(points: NDArray[np.float64]) -> dict:
    """Area, area-centroid and half-diameter of the inner polygon through the support points.

    Args:
        points (NDArray[np.float64]): the `(u, v)` support points, in angular order.

    Returns:
        dict: `area_voxels2`, `centroid_voxels`, `half_diameter_voxels`, `max_displacement_voxels`.
    """
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        centroid = points.mean(axis=0)
    else:
        centroid = np.array([
            float(((x + x_next) * cross).sum() / (6.0 * area)),
            float(((y + y_next) * cross).sum() / (6.0 * area)),
        ])
    separations = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return {
        "area_voxels2": abs(area),
        "centroid_voxels": [float(centroid[0]), float(centroid[1])],
        "half_diameter_voxels": float(separations.max() / 2.0),
        "max_displacement_voxels": float(np.linalg.norm(points, axis=1).max()),
    }


def rodrigues(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Rotation matrix about a unit `axis` by `angle`, exactly.

    Args:
        axis (NDArray[np.float64]): a unit 3-vector.
        angle (float): the rotation angle in radians.

    Returns:
        NDArray[np.float64]: the `(3, 3)` rotation matrix.
    """
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def apply_delta_to_configuration(
    centre: NDArray[np.float64],
    frame: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    interfaces: list[dict],
    delta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], tuple, list[dict]]:
    """Move a configuration by `delta` exactly -- the new anchor for re-linearisation.

    The frame is carried by the exact minimal rotation, so the result is a genuine
    three-half-plane geometry rather than a linearised approximation of one.

    Args:
        centre (NDArray[np.float64]): the current sample position.
        frame (tuple): `(e1, e2, direction)`.
        interfaces (list[dict]): the three `{"pair", "azimuth"}` records.
        delta (NDArray[np.float64]): the 7-vector `(u, v, a, b, dphi_1, dphi_2, dphi_3)`.

    Returns:
        tuple: the new centre, the new frame and the new interfaces.
    """
    e1, e2, d0 = frame
    u, v, a, b = delta[:4]
    new_centre = centre + u * e1 + v * e2
    new_direction = d0 + a * e1 + b * e2
    new_direction = new_direction / np.linalg.norm(new_direction)
    axis = np.cross(d0, new_direction)
    axis_norm = np.linalg.norm(axis)
    rotation = (
        np.eye(3)
        if axis_norm < 1e-12
        else rodrigues(axis / axis_norm, float(np.arctan2(axis_norm, float(np.dot(d0, new_direction)))))
    )
    new_frame = (rotation @ e1, rotation @ e2, new_direction)
    new_interfaces = [
        {**interface, "azimuth": float(interface["azimuth"] + delta[4 + i])}
        for i, interface in enumerate(interfaces)
    ]
    return new_centre, new_frame, new_interfaces


# --------------------------------------------------------------------------------------
# the estimator itself
# --------------------------------------------------------------------------------------
def estimate_at_sample(  # noqa: C901 - one guarded pipeline; each early return names its own refusal
    segmented_mask: NDArray[np.uint],
    centre: NDArray[np.float64],
    direction: NDArray[np.float64],
    triple: tuple[int, int, int],
    radius: float = DEFAULT_RADIUS_VOXELS,
    *,
    want_centroid: bool = False,
) -> dict:
    """The anchor (and optionally the centroid) estimate of one junction position.

    Never raises and never reads ground truth. Every refusal returns a `status` naming it, so a
    caller can count refusals rather than discover them as exceptions.

    The re-anchoring loop is the identifiability-floor work's: the linearisation is moved
    to the feasible set's own
    deepest point before anything is read off it, because the detected curve the loop starts
    from is itself up to a voxel from the junction and linearising there would report the
    feasible set of a model that misclassifies part of its own window.

    Args:
        segmented_mask (NDArray[np.uint]): the label image.
        centre (NDArray[np.float64]): the starting position, in voxel coordinates.
        direction (NDArray[np.float64]): a unit vector along the junction line at `centre`.
        triple (tuple[int, int, int]): the three materials meeting on that line.
        radius (float): the window radius in voxels. Defaults to `DEFAULT_RADIUS_VOXELS`.
        want_centroid (bool): also solve the full support function and return `centroid_point`.
            Roughly thirty times the LP cost. Defaults to False. The junction-curve-extraction
            work always solved it, so a
            sample whose support LP fails is refused there and accepted here; `anchor_point`
            itself is computed before the support function and is unaffected.

    Returns:
        dict: `{"status": "ok", "anchor_point": [...], ...}` on success, otherwise
        `{"status": <reason>}`.
    """
    e1, e2 = normal_plane_basis(direction)
    frame = (e1, e2, direction)
    offsets, materials = voxels_in_window(segmented_mask, centre, direction, radius)
    if len(offsets) < MIN_VOXELS_IN_WINDOW:
        return {"status": "too_few_voxels", "n_voxels": len(offsets)}
    interfaces = mask_interface_azimuths(offsets, materials, triple, frame)
    if interfaces is None:
        return {"status": "no_azimuths"}
    if wedge_structure(interfaces, triple) is None:
        return {"status": "inconsistent_wedges"}

    anchor_centre, anchor_frame, anchor_interfaces = centre, frame, interfaces
    trace: list[dict] = []
    for _iteration in range(N_REANCHOR_ITERATIONS):
        probe_offsets, probe_materials = voxels_in_window(segmented_mask, anchor_centre, anchor_frame[2], radius)
        if len(probe_offsets) < MIN_VOXELS_IN_WINDOW:
            return {"status": "anchor_left_window"}
        structure = wedge_structure(anchor_interfaces, triple)
        if structure is None:
            return {"status": "anchor_wedges_inconsistent"}
        polytope = build_polytope(probe_offsets, probe_materials, anchor_interfaces, structure, anchor_frame)
        deepest = max_margin_point(polytope)
        if deepest["delta"] is None:
            return {"status": "anchor_lp_failed"}
        step = deepest["delta"]
        step_size = float(np.linalg.norm(step[:2]))
        trace.append({
            "margin_voxels": deepest["margin_voxels"],
            "n_misclassified": int(polytope["n_excluded_anchor_violating"]),
            "step_uv_voxels": step_size,
        })
        anchor_centre, anchor_frame, anchor_interfaces = apply_delta_to_configuration(
            anchor_centre, anchor_frame, anchor_interfaces, step,
        )
        if step_size < ANCHOR_CONVERGENCE_VOXELS and (deepest["margin_voxels"] or 0.0) > 0.0:
            break

    offsets, materials = voxels_in_window(segmented_mask, anchor_centre, anchor_frame[2], radius)
    if len(offsets) < MIN_VOXELS_IN_WINDOW:
        return {"status": "anchor_left_window"}
    structure = wedge_structure(anchor_interfaces, triple)
    if structure is None:
        return {"status": "anchor_wedges_inconsistent"}
    polytope = build_polytope(offsets, materials, anchor_interfaces, structure, anchor_frame)
    unexplained = polytope["violated_at_anchor"]
    n_unexplained = int(unexplained.sum())
    if n_unexplained:
        # The identifiability-floor work's own rule: voxels the best-fitting flat
        # configuration still misclassifies are
        # dropped, and only those, because they are evidence the flat model does not describe
        # them -- not evidence about its feasible set. The count is reported as the price.
        retained = ~unexplained
        if int(retained.sum()) < MIN_VOXELS_IN_WINDOW:
            return {"status": "too_few_explained_voxels"}
        offsets, materials = offsets[retained], materials[retained]
        polytope = build_polytope(offsets, materials, anchor_interfaces, structure, anchor_frame)
        if polytope["n_excluded_anchor_violating"]:
            return {"status": "anchor_still_infeasible_after_pruning"}

    anchor_e1, anchor_e2, anchor_direction = anchor_frame
    widths = np.array([w["width_rad"] for w in structure["wedges"]])
    narrow = int(np.argmin(widths))
    narrow_wedge = structure["wedges"][narrow]
    bisector_azimuth = anchor_interfaces[narrow_wedge["start"]]["azimuth"] + 0.5 * narrow_wedge["width_rad"]
    bisector = np.cos(bisector_azimuth) * anchor_e1 + np.sin(bisector_azimuth) * anchor_e2

    result = {
        "status": "ok",
        "triple": [int(t) for t in triple],
        "extracted_point": [float(v) for v in centre],
        "anchor_point": [float(v) for v in anchor_centre],
        "narrow_wedge_bisector": [float(v) for v in bisector],
        "direction": [float(v) for v in anchor_direction],
        "anchor_step_from_extracted_voxels": float(np.linalg.norm(anchor_centre - centre)),
        "n_voxels": len(offsets),
        "n_voxels_used": int(polytope["n_voxels_used"]),
        "n_unexplained_by_flat_model": n_unexplained,
        "n_reanchor_iterations": len(trace),
        "anchor_margin_voxels": trace[-1]["margin_voxels"] if trace else None,
        "wedge_widths_deg": [float(np.degrees(w)) for w in widths],
        "narrow_wedge_width_deg": float(np.degrees(widths[narrow])),
    }
    if not want_centroid:
        return result

    lp = support_function(polytope)
    if lp is None:
        return {"status": "lp_infeasible"}
    measures = polygon_measures(lp["solutions"][:, :2])
    centroid_uv = np.asarray(measures["centroid_voxels"], dtype=float)
    result["centroid_point"] = [
        float(v) for v in anchor_centre + centroid_uv[0] * anchor_e1 + centroid_uv[1] * anchor_e2
    ]
    result["centroid_uv_voxels"] = [float(centroid_uv[0]), float(centroid_uv[1])]
    result["box_active_position"] = bool(lp["box_active_position"])
    result["half_diameter_voxels"] = float(measures["half_diameter_voxels"])
    result["area_voxels2"] = float(measures["area_voxels2"])
    return result
