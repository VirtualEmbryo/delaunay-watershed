"""Module to compute euclidean distance transforms, place points and build tesselation from image segmentation.

Regular (weighted) triangulation
----------------------------------
`regular_tesselation` is the weighted counterpart of `simple_delaunay_tesselation`, added
for the junction-protection work. A regular (a.k.a. weighted
Delaunay, or power) triangulation of weighted points is the projection of the lower convex
hull of the points lifted to `(x, ||x||**2 - w)`: with all weights zero this is the ordinary
Delaunay lifting to the paraboloid, and a positive weight `w_i` pushes point `i` down,
enlarging its power cell. Boltcheva's "protecting balls" *are* weighted points, so this is
the channel through which a junction sample keeps its neighbourhood.

Two things about it that are measured, not assumed:

* **`weights == 0` does not reproduce `scipy.spatial.Delaunay` bit-for-bit.** `dw3d`'s seed
  set lives on the integer voxel lattice and is massively cospherical; `ConvexHull` (Qhull
  `Qt`) triangulates those degenerate facets differently from `Delaunay` (`Qz`/`Qc`). This
  was found independently by a separate regular-triangulation experiment (which measured ~852
  of ~20 900 simplices differing on `3.tif` at `min_distance=3`), and settles in the negative the
  open question of whether zero weights reproduce the plain Delaunay triangulation. Callers
  that need the exact plain-Delaunay baseline must therefore dispatch to `simple_delaunay_tesselation`, which is what
  `weighted_delaunay_tesselation` does when every weight is zero.
* **Hiding needs enclosure, not a heavy neighbour.** A weighted point is *hidden* — absent
  from every simplex — exactly when its power cell is empty, and one nearby heavier point
  can never do that: the power bisector of two weighted points is a plane, so each of them
  always keeps a half-space. Measured here (`tests/test_junction_protection.py`): a
  zero-weight point 6 voxels from each of six heavy points survives while their radius is 5
  and disappears only at radius 7, when the balls together enclose it. **Junction protection
  therefore does not rely on weights to clear a junction's neighbourhood — it excludes the crowding
  points explicitly** (see `dw3d.points_on_edt`), and the weights only bias the
  triangulation. Where a point *is* hidden it stays in the returned array as an
  unreferenced vertex; `TesselationGraph` indexes vertices and `filter_unused_points` drops
  the unused ones at the end of the pipeline, so nothing downstream needs them removed here.
* **Uniform weights change nothing.** Adding the same constant to every weight translates
  the whole lift vertically, so the lower hull — and hence the triangulation — is exactly
  the Delaunay one. Only weight *differences* matter, which is why a protected cluster of
  equal-weight points is triangulated internally just as it would have been unweighted.

**Note for whoever resumes the regular-triangulation work.** Its WIP widened
`TesselationCreationFunction`'s second argument to the *EDT image*, deriving one global
`weight_scale * EDT` weight per seed; the junction-protection work widened it to an explicit
*per-point weight vector*, because "weight only the junction samples" cannot be expressed as
a global scale of the EDT. The per-point vector is the more general contract — the
regular-triangulation work's weights are `(weight_scale * edt[seed])**2`, which the
point-placing function can return directly — so the reconciliation is to keep the
junction-protection signature and move the regular-triangulation work's weight derivation
into its point-placing function.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Delaunay


def simple_delaunay_tesselation(
    points_for_tesselation: NDArray[np.uint],
    _weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a Delaunay tesselation on points at extrema of the Euclidean Distance Transform of the segmented image.

    Args:
        points_for_tesselation (NDArray[np.uint]): Points to construct the tesselation.
        _weights (NDArray[np.float64] | None, optional): Ignored. The parameter exists only
            so this function satisfies the widened `TesselationCreationFunction` contract
            (the junction-protection work), under which `weighted_delaunay_tesselation` *does* read per-point
            weights. Passing weights or not leaves the output bit-identical.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64]]:
            - tesselation's points
            - tesselation's tetrahedrons as array of point indices.
    """
    tesselation = Delaunay(points_for_tesselation)

    return tesselation.points, tesselation.simplices.astype(np.int64)


def regular_tesselation(
    points: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Regular (weighted Delaunay / power) tetrahedralisation via a lifted lower convex hull.

    The regular triangulation of weighted points `(x_i, w_i)` is the projection of the lower
    convex hull of the lifted points `(x_i, ||x_i||**2 - w_i)` in R^4. A hull facet belongs
    to the lower hull when its outward normal has a negative component along the lift axis.
    See the module docstring for the two behaviours that matter downstream (no bit-for-bit
    agreement with `Delaunay` at `w = 0`, and hidden points).

    Args:
        points (NDArray[np.float64]): (n, 3) point coordinates.
        weights (NDArray[np.float64]): (n,) per-point weights, i.e. squared protecting-ball
            radii. Non-negative for the power-ball interpretation, but the construction is
            well defined for any real weights.

    Returns:
        NDArray[np.int64]: (m, 4) array of tetrahedra as indices into `points`.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    lifted = np.column_stack((points, (points**2).sum(axis=1) - weights))

    hull = ConvexHull(lifted)
    # `equations` rows are (a, b, c, d, offset); the lift-axis normal component is column -2.
    return hull.simplices[hull.equations[:, -2] < 0].astype(np.int64)


def weighted_delaunay_tesselation(
    points_for_tesselation: NDArray[np.uint],
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tesselate with per-point weights, falling back to plain Delaunay when there are none.

    `weights is None`, or all-zero weights, dispatches to `simple_delaunay_tesselation` so
    the unweighted baseline is reproduced *exactly* rather than approximately (the lifted
    hull cannot match Qhull's `Delaunay` on the cospherical integer lattice; see the module
    docstring).

    Args:
        points_for_tesselation (NDArray[np.uint]): (n, 3) seed coordinates.
        weights (NDArray[np.float64] | None, optional): (n,) per-point weights (squared
            protecting-ball radii). Defaults to None.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64]]:
            - the tesselation points, as float64, in the input order;
            - the tetrahedra as (m, 4) indices into those points.
    """
    if weights is None or not np.any(weights):
        return simple_delaunay_tesselation(points_for_tesselation, weights)

    points = np.asarray(points_for_tesselation, dtype=np.float64)
    return points, regular_tesselation(points, weights)
