r"""Move a 2D mesh's interface vertices onto the mask's own interface locus (opt-in, after the junction relocation).

The 2D port of dw3d's locus projection. Connectivity is frozen: only the
coordinates of interface vertices move, under the guards of :mod:`dw2d.junction_relocation`.

Why, and why only after the junction relocation
-----------------------------------------------
dw2d's vertices are pixel centres, while the label image's interface between labels a and b lies
between pixel centres. On the equilibrium-foam benchmark (exact rasterisations of certified
circular-arc foams with known tensions; held-out split, 159 images, ``min_dist`` 3), the projection **alone** made the case angle error
worse (+0.46 deg [+0.19, +0.56]) and the tension error worse (+0.014), but applied **after** the
junction relocation it lowered the angle error by 2.45 deg [2.07, 2.68] from 9.15 deg and the
gauge-matched tension error by 0.029, and it stayed better than the legacy mesh with smooth boundary
noise of 0.5 and 1 px RMS (-1.96 deg and -1.29 deg; tensions -0.023 and -0.021). The interaction
(harmful alone, helpful composed) is measured, its mechanism is not. Hence
``geometry_reconstruction_2d(..., project_interface_vertices=True)`` requires
``relocate_junctions=True``.

What it does
------------
For every vertex of exactly one interface (a, b) — degree 2, both incident edges carrying the same
pair, not a junction, not on the image border — the target is the zero of
psi_ab(x) = (D_a(x) - D_b(x)) / 2, D_x the Euclidean distance to the nearest pixel *centre* of label x
(``scipy.ndimage.distance_transform_edt``), interpolated bilinearly. Newton steps along grad psi, at most
``MAX_ITERATIONS``, stop at |psi| < ``TOLERANCE_PX``; a vertex whose iteration does not converge, or
would move farther than ``MAX_DISPLACEMENT_PX``, is left in place. Moves are applied in ascending
vertex index under the four guards (short edge, crossing count, edge contact, signed area); whole-mesh
recounts of crossings and of contacts before and after certify the crossing and contact guards
globally (a rise raises). The contact guard is the one that matters here: on the equilibrium-foam
benchmark at 40 px, the projection without it flattened 141 Z-shaped turns of plain interface chains
into an edge touching another and a turn of exactly 180 degrees; with it those moves are refused
and the vertex stays where it was (:mod:`dw2d.junction_relocation`, "Edge contacts").

Units and conventions: pixels, (row, col), the frame of ``geometry_reconstruction_2d``'s vertices; the
label image the mesh was built from (after ``expansion_labels``). Nothing reads ground truth;
deterministic. Cost: one distance transform per label present.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from dw2d.junction_relocation import GuardedVertexMover, whole_mesh_contacts, whole_mesh_crossings

if TYPE_CHECKING:  # imported lazily at run time
    from collections.abc import Iterable

    from dw2d.reports import InterfaceProjectionReport

#: Newton iterations per vertex (fixed, not a tuning parameter).
MAX_ITERATIONS = 10
#: Convergence of |psi| in pixels (fixed).
TOLERANCE_PX = 1e-6
#: Largest displacement a projection may apply to one vertex, in pixels (fixed).
MAX_DISPLACEMENT_PX = 1.5


def _bilinear_with_gradient(F, p):
    """Bilinear value and gradient of the pixel-grid field F at continuous (row, col) p."""
    i, j = p
    H, W = F.shape
    i0 = int(np.clip(np.floor(i), 0, H - 2)); j0 = int(np.clip(np.floor(j), 0, W - 2))
    di, dj = i - i0, j - j0
    f00, f01, f10, f11 = F[i0, j0], F[i0, j0 + 1], F[i0 + 1, j0], F[i0 + 1, j0 + 1]
    val = f00 * (1 - di) * (1 - dj) + f01 * (1 - di) * dj + f10 * di * (1 - dj) + f11 * di * dj
    gi = (f10 - f00) * (1 - dj) + (f11 - f01) * dj
    gj = (f01 - f00) * (1 - di) + (f11 - f10) * di
    return val, np.array([gi, gj])


def _project_on_edges(verts, edges, labels, fixed_vertices=()):
    """Project every eligible interface vertex onto its pair's mask locus; connectivity frozen.

    Inputs: ``verts`` (V, 2) float (row, col) px, ``edges`` (E, 4) int as ``return_mesh`` gives them,
    ``labels`` the label image, ``fixed_vertices`` vertices never moved (``geometry_reconstruction_2d`` passes
    the image border vertices; the vertices on the border were already left in place by position). Returns (new_verts, report) with report = dict(candidates, moved,
    not_converged, refused {guard: count}, crossings_before, crossings_after, contacts_before,
    contacts_after, median_displacement_px). Raises RuntimeError if the whole-mesh crossing count or
    contact count rises.
    """
    L = np.asarray(labels)
    H, W = L.shape
    mover = GuardedVertexMover(verts, edges)
    pairs_at = defaultdict(set); degree = defaultdict(int)
    for v1, v2, m1, m2 in np.asarray(edges):
        for v in (int(v1), int(v2)):
            degree[v] += 1; pairs_at[v].add(tuple(sorted((int(m1), int(m2)))))
    fixed = {int(v) for v in fixed_vertices}
    candidates = [v for v in sorted(degree) if degree[v] == 2 and len(pairs_at[v]) == 1 and v not in fixed]
    distance = {}

    def dist(x):
        if x not in distance:
            distance[x] = ndimage.distance_transform_edt(L != x)
        return distance[x]

    before = whole_mesh_crossings(mover.v, edges)
    contacts_before = whole_mesh_contacts(mover.v, edges)
    refused = defaultdict(int); moved = 0; not_converged = 0; disp = []
    for v in candidates:
        p0 = mover.v[v].copy()
        if p0[0] <= 0 or p0[1] <= 0 or p0[0] >= H - 1 or p0[1] >= W - 1:
            continue
        (a, b), = pairs_at[v]
        Da, Db = dist(a), dist(b)
        p = p0.copy(); converged = False
        for _ in range(MAX_ITERATIONS):
            va, ga = _bilinear_with_gradient(Da, p); vb, gb = _bilinear_with_gradient(Db, p)
            psi = 0.5 * (va - vb); g = 0.5 * (ga - gb)
            if abs(psi) < TOLERANCE_PX:
                converged = True; break
            gg = float(g @ g)
            if gg < 1e-12:
                break
            p = p - psi * g / gg
            if np.linalg.norm(p - p0) > MAX_DISPLACEMENT_PX or not (0 < p[0] < H - 1 and 0 < p[1] < W - 1):
                break
        if not converged:
            not_converged += 1
            continue
        why = mover.try_move(v, p)
        if why:
            refused[why] += 1
        else:
            moved += 1; disp.append(float(np.linalg.norm(p - p0)))
    after = whole_mesh_crossings(mover.v, edges)
    if after > before:
        raise RuntimeError(f"crossing count rose {before} -> {after}: the guard is wrong")
    contacts_after = whole_mesh_contacts(mover.v, edges)
    if contacts_after > contacts_before:
        raise RuntimeError(f"contact count rose {contacts_before} -> {contacts_after}: the contact guard is wrong")
    return mover.v.copy(), dict(candidates=len(candidates), moved=moved, not_converged=not_converged, refused=dict(refused),
                                crossings_before=before, crossings_after=after,
                                contacts_before=contacts_before, contacts_after=contacts_after,
                                median_displacement_px=float(np.median(disp)) if disp else None)


def project_interface_vertices(
    mask: np.ndarray,
    points: np.ndarray,
    segments: np.ndarray,
    labels: np.ndarray,
    *,
    fixed_vertices: Iterable[int] = (),
) -> tuple[np.ndarray, InterfaceProjectionReport]:
    """Move every eligible interface vertex of a 2D mesh onto its pair's mask locus (connectivity frozen).

    This module's docstring says what it does, under which guards, and why it is offered only after the junction
    relocation. `dw2d.MeshReconstructionAlgorithm` runs it when asked (`set_interface_projection`).

    Args:
        mask: the 2-D label image the mesh was built from (after any label expansion).
        points: `(n, 2)` float vertex positions in pixels, `(row, column)`.
        segments: `(m, 2)` int vertex indices of each edge.
        labels: `(m, 2)` int, the two materials each edge separates.
        fixed_vertices: vertices never moved (the algorithm passes its `image_border_vertices`).

    Returns:
        tuple: `(points, report)`: a moved copy of `points` (float64) and a `dw2d.InterfaceProjectionReport`.

    Raises:
        RuntimeError: if the whole-mesh crossing or contact count rose.
    """
    from dw2d.reports import InterfaceProjectionReport

    edges = np.hstack((np.asarray(segments), np.asarray(labels)))
    moved, report = _project_on_edges(np.asarray(points, float), edges, np.asarray(mask), fixed_vertices)
    return moved, InterfaceProjectionReport(**report)
