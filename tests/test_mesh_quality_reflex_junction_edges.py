"""M6 must count reflex trijunction wedges, and must count them without any ground truth.

A trijunctional edge's three wedge angles partition the plane normal to it and sum to 360
degrees, so at most one of them can exceed a straight angle. When one does -- a **reflex** wedge
-- one region's triangles occupy more than half the ring around that edge, and `foambryo`'s
default angle reading, `arccos` of a dot product, reports it as `360 - theta`. Every
line-averaged angle statistic in this package inherits the resulting shortfall, and the rate at
which it happens differs by a factor of six between the reconstruction strategies this package
compares, so part of what an angle comparison measured was how much each mesh was clipped.

`net_offset_deg` is exactly `-(1/3)` of the length-weighted mean per-line shortfall, and is
therefore not a "net bias at a symmetric wedge" -- its descriptor says so now, and this file
pins that it keeps saying so.

Pinned here:

1. the count and the fraction are emitted by `compute_m6_validity`, with the excess
   distribution, and nothing else in the block moved;
2. the count needs no ground truth -- `compute_reflex_junction_edges` takes only the mesh;
3. an analytic mesh with a known reflex wedge is counted, and one without is not;
4. `net_offset_deg` is zero exactly when the count is zero, and negative when it is not;
5. `net_offset_deg`'s descriptor states the identity rather than calling the quantity a bias.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from dw3d_benchmarks.mesh_quality_comparison import case_metrics

#: `(interior wedge of region 0, of region 1, of region 2)` for the two analytic phantoms below.
_SYMMETRIC_RAYS_DEG = (0.0, 120.0, 240.0)
_REFLEX_RAYS_DEG = (0.0, 100.0, 160.0)
_REFLEX_WEDGE_DEG = 200.0
_ADDED_KEYS = (
    "n_junction_edges",
    "n_reflex_junction_edges",
    "reflex_junction_edge_fraction",
    "reflex_excess_mean_deg",
    "reflex_excess_p90_deg",
    "reflex_excess_max_deg",
)


def _three_film_phantom(rays_deg: tuple[float, float, float], rings: int = 4, n_axis: int = 7):
    """Three flat films leaving one junction line at prescribed angles; wedges known exactly.

    Region `k` is the sector between rays `k` and `k + 1` and carries material `k`, so its
    interior wedge is exactly the gap between those rays. Each strip is wound so its right-hand
    normal is the ray turned a quarter turn clockwise about `+z`, i.e. into the sector *before*
    the ray, which is why film `k` is labelled `(material k, material k - 1)`: `foambryo`'s
    convention is that a normal points away from `material_1` and into `material_2`.

    Args:
        rays_deg: the three in-plane film directions, degrees, strictly increasing in `[0, 360)`.
        rings: triangle rings per film away from the junction.
        n_axis: vertices along the junction line.

    Returns:
        tuple: `(points, triangles, labels)`.
    """
    axis_positions = np.linspace(0.0, 20.0, n_axis)
    spacing = 4.0
    junction = np.stack([np.zeros(n_axis), np.zeros(n_axis), axis_positions], axis=1)
    blocks = [junction]
    offset = n_axis
    film_columns = []
    for angle in rays_deg:
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        columns = []
        for ring in range(1, rings + 1):
            radius = ring * spacing
            blocks.append(
                np.stack(
                    [
                        np.full(n_axis, radius * direction[0]),
                        np.full(n_axis, radius * direction[1]),
                        axis_positions,
                    ],
                    axis=1,
                ),
            )
            columns.append(np.arange(offset, offset + n_axis))
            offset += n_axis
        film_columns.append(columns)

    triangles, labels = [], []
    for k, columns in enumerate(film_columns):
        strips = [np.arange(n_axis), *columns]
        for left, right in itertools.pairwise(strips):
            for i in range(n_axis - 1):
                triangles.append([left[i], right[i], left[i + 1]])
                labels.append([k, (k - 1) % 3])
                triangles.append([right[i], right[i + 1], left[i + 1]])
                labels.append([k, (k - 1) % 3])
    return (
        np.concatenate(blocks, axis=0),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


def test_a_symmetric_junction_has_no_reflex_wedge():
    """The phantom's own correctness check: three rays 120 degrees apart, three 120-degree wedges."""
    counts = case_metrics.compute_reflex_junction_edges(*_three_film_phantom(_SYMMETRIC_RAYS_DEG))
    assert counts["n_junction_edges"] > 0
    assert counts["n_reflex_junction_edges"] == 0
    assert counts["reflex_junction_edge_fraction"] == 0.0
    assert counts["reflex_excess_mean_deg"] is None


def test_a_known_reflex_wedge_is_counted_and_its_excess_measured():
    """Rays at 0, 100 and 160 degrees enclose 100, 60 and 200: every edge is reflex by 20 degrees."""
    counts = case_metrics.compute_reflex_junction_edges(*_three_film_phantom(_REFLEX_RAYS_DEG))
    assert counts["n_reflex_junction_edges"] == counts["n_junction_edges"]
    assert counts["reflex_junction_edge_fraction"] == 1.0
    expected_excess = _REFLEX_WEDGE_DEG - 180.0
    for key in ("reflex_excess_mean_deg", "reflex_excess_p90_deg", "reflex_excess_max_deg"):
        assert counts[key] == pytest.approx(expected_excess, abs=1e-9), key


def test_the_count_reads_nothing_but_the_mesh():
    """Ground-truth-free is the property that makes this usable on experimental data.

    `compute_reflex_junction_edges` takes `(points, triangles, labels)` and nothing else, so
    there is no parameter through which a reference mesh could reach it.
    """
    import inspect

    parameters = list(inspect.signature(case_metrics.compute_reflex_junction_edges).parameters)
    assert parameters == ["points", "triangles", "labels"]


def test_m6_emits_the_counts_alongside_its_other_validity_numbers():
    """The counts belong in the validity block, reported like every other count there."""
    points, triangles, labels = _three_film_phantom(_REFLEX_RAYS_DEG)
    gt = {"points": points, "triangles": triangles, "labels": labels}
    block = case_metrics.compute_m6_validity(points, triangles, labels, gt)
    for key in _ADDED_KEYS:
        assert key in block, key
    assert block["n_reflex_junction_edges"] == block["n_junction_edges"]
    # and the historical contents of the block are untouched
    for key in ("watertight", "n_abnormal_non_manifold_edges", "n_quadjunction_edges", "identity"):
        assert key in block, key


def test_the_net_offset_descriptor_states_the_identity_rather_than_calling_it_a_bias():
    """`net_offset_deg` is minus a third of the mean angle-sum deficit, not a bias at a symmetric wedge.

    The wedges that *are* near-symmetric were measured to carry a *positive* mean error on 14 of
    15 reconstruction arms while the fitted intercept is negative on all 15; reading the
    intercept as their bias is what this wording exists to prevent.
    """
    descriptor = case_metrics._STATISTIC_DESCRIPTORS["net_offset_deg"]
    assert "deficit" in descriptor["measures"]
    assert "reflex" in descriptor["note"]
    assert "n_reflex_junction_edges" in descriptor["note"]
    assert descriptor["definition"].startswith("intercept a of signed_error")


def test_every_angle_statistic_descriptor_names_its_aggregation():
    """A statistic must say what population it is aggregated over, not only what it measures.

    Five different aggregation conventions have been published in this project for the same angle
    quantity -- per wedge against per trijunction line, pooled against a median over case medians,
    length-weighted against unweighted -- and two published comparisons crossed two of them. The
    descriptor block is what a table or a figure caption reads to avoid that, so every descriptor
    must carry `aggregation_unit`, `pooling` and `weighting` as well as `measures`, `unit` and
    `definition`. `case_metrics` enforces it at import; this pins the contract so the enforcement
    cannot be removed silently.
    """
    from dw3d_benchmarks.mesh_quality_comparison import case_metrics

    assert {
        "measures", "unit", "definition", "aggregation_unit", "pooling", "weighting",
    } <= case_metrics._REQUIRED_DESCRIPTOR_KEYS
    missing = {
        name: sorted(case_metrics._REQUIRED_DESCRIPTOR_KEYS - descriptor.keys())
        for name, descriptor in case_metrics._STATISTIC_DESCRIPTORS.items()
        if not descriptor.keys() >= case_metrics._REQUIRED_DESCRIPTOR_KEYS
    }
    assert not missing, missing
    units = {descriptor["aggregation_unit"] for descriptor in case_metrics._STATISTIC_DESCRIPTORS.values()}
    assert units <= {case_metrics._PER_WEDGE, case_metrics._PER_LINE}
