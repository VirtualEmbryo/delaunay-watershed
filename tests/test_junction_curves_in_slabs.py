"""Junction detection in slabs gives exactly the whole-volume detection's codes, table and curves.

`dual_label_sets` stacked the eight corner labels of every dual-lattice point of the whole volume,
sorted them and counted the distinct ones, about 92 bytes per voxel (+32 GiB on a 1 866-cell
tissue, that run's memory peak). The detection now runs slab by slab along axis 0, with one plane
of overlap, and `detect_junction_spines` never builds the whole-volume code grid. A pure
memory change: these tests compare it, bitwise, with the original functions kept here verbatim,
on random label volumes (with three, four and more labels per corner) cut into slabs of every
size down to one plane, and on the curves `extract_junction_curves` returns for a reconstructed
mesh.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray

import dw3d
from dw3d import junction_curves
from dw3d.junction_curves import _THREE, detect_junction_spines, dual_label_sets, extract_junction_curves
from tests import conftest


# ---------------------------------------------------------------- the original functions, verbatim
def reference_dual_label_sets(segmented_mask: NDArray[np.uint]) -> tuple[NDArray[np.int64], list[frozenset[int]]]:
    """Per dual-lattice point, an index into a table of the distinct labels around it.

    A dual point `(i, j, k)` is the corner shared by the eight voxels
    `segmented_mask[i:i+2, j:j+2, k:k+2]`. Points with fewer than three distinct labels are not
    junctions and carry `-1`.

    Args:
        segmented_mask (NDArray[np.uint]): the label image.

    Returns:
        tuple[NDArray[np.int64], list[frozenset[int]]]: `codes` of shape `(X-1, Y-1, Z-1)` and a
        table where `table[c]` is the frozenset of labels around a point carrying code `c`.
    """
    shape = segmented_mask.shape
    corners = [
        segmented_mask[a:, b:, c:][: shape[0] - 1, : shape[1] - 1, : shape[2] - 1]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]
    stacked = np.stack(corners, axis=-1)
    stacked_sorted = np.sort(stacked, axis=-1)
    n_distinct = 1 + (np.diff(stacked_sorted, axis=-1) != 0).sum(axis=-1)
    codes = np.full(n_distinct.shape, -1, dtype=np.int64)
    table: list[frozenset[int]] = []
    index_of: dict[frozenset[int], int] = {}
    for i, j, k in np.argwhere(n_distinct >= _THREE):
        materials = frozenset(int(v) for v in stacked[i, j, k])
        code = index_of.get(materials)
        if code is None:
            code = len(table)
            index_of[materials] = code
            table.append(materials)
        codes[i, j, k] = code
    return codes, table


def reference_points_by_triple(
    codes: NDArray[np.int64],
    table: list[frozenset[int]],
) -> dict[tuple[int, ...], NDArray[np.int64]]:
    """Every dual point belonging to each material triple's junction line."""
    occupied = np.argwhere(codes >= 0)
    if occupied.size == 0:
        return {}
    per_triple: dict[tuple[int, ...], list[NDArray[np.int64]]] = defaultdict(list)
    code_values = codes[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
    for code in np.unique(code_values):
        members = occupied[code_values == code]
        for triple in combinations(sorted(table[code]), 3):
            per_triple[triple].append(members)
    return {triple: np.concatenate(blocks, axis=0) for triple, blocks in per_triple.items()}


# ---------------------------------------------------------------- the tests


def random_labels(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = tuple(int(v) for v in rng.integers(2, 12, size=3))
    coarse = rng.integers(0, int(rng.integers(2, 7)), size=tuple(max(1, s // 2) for s in shape))
    labels = np.repeat(np.repeat(np.repeat(coarse, 2, 0), 2, 1), 2, 2)[: shape[0], : shape[1], : shape[2]]
    return labels.astype([np.uint16, np.int32, np.int64][seed % 3])


@pytest.mark.parametrize("seed", range(30))
def test_codes_and_table_match_the_whole_volume_detection(seed, monkeypatch):
    labels = random_labels(seed)
    expected_codes, expected_table = reference_dual_label_sets(labels)
    plane = (labels.shape[1] - 1) * (labels.shape[2] - 1)
    for slab in (1, plane, 3 * plane + 1, 2**23):
        monkeypatch.setattr(junction_curves, "_DUAL_SLAB_POINTS", slab)
        codes, table = dual_label_sets(labels)
        assert codes.dtype == expected_codes.dtype
        assert codes.shape == expected_codes.shape
        assert np.array_equal(codes, expected_codes)
        assert table == expected_table


@pytest.mark.parametrize("seed", range(30))
def test_points_by_triple_match_the_per_code_selection(seed, monkeypatch):
    labels = random_labels(seed)
    expected = reference_points_by_triple(*reference_dual_label_sets(labels))
    monkeypatch.setattr(junction_curves, "_DUAL_SLAB_POINTS", 5)
    occupied, code_values, table = junction_curves._dual_junction_points(labels)
    got = junction_curves._points_by_triple_from_points(occupied, code_values, table)
    assert list(got) == list(expected)
    for triple in expected:
        assert got[triple].dtype == expected[triple].dtype
        assert np.array_equal(got[triple], expected[triple])


def test_masks_too_thin_for_a_dual_point():
    for shape in ((1, 5, 5), (5, 1, 5), (5, 5, 1)):
        labels = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
        expected_codes, expected_table = reference_dual_label_sets(labels)
        codes, table = dual_label_sets(labels)
        assert codes.shape == expected_codes.shape
        assert table == expected_table


def test_spines_and_curves_do_not_depend_on_the_slab(monkeypatch):
    mask = conftest.load_image_or_skip("1.tif")
    algorithm = dw3d.get_default_mesh_reconstruction_algorithm()
    algorithm.relocate_junctions = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm.construct_mesh_from_segmentation_mask(mask)
    points, triangles, labels = algorithm._points, algorithm._triangles, algorithm._labels
    results = []
    for slab in (2**23, 3 * (mask.shape[1] - 1) * (mask.shape[2] - 1) + 7, 1):
        monkeypatch.setattr(junction_curves, "_DUAL_SLAB_POINTS", slab)
        spines = detect_junction_spines(mask)
        curves = extract_junction_curves(mask, points, triangles, labels)
        results.append((spines, curves))
    (spines0, curves0) = results[0]
    reference_codes, _ = reference_dual_label_sets(mask)
    assert spines0["n_dual_junction_points"] == int((reference_codes >= 0).sum())
    for spines, curves in results[1:]:
        assert spines["n_dual_junction_points"] == spines0["n_dual_junction_points"]
        assert spines["diagnostics"] == spines0["diagnostics"]
        assert list(spines["by_triple"]) == list(spines0["by_triple"])
        for triple, polylines in spines0["by_triple"].items():
            for a, b in zip(polylines, spines["by_triple"][triple], strict=True):
                assert a.tobytes() == b.tobytes()
        for bucket in ("matched", "invented", "fragmented"):
            for a, b in zip(getattr(curves0, bucket), getattr(curves, bucket), strict=True):
                assert (a.triple, a.identity, a.component) == (b.triple, b.identity, b.component)
                assert a.points.tobytes() == b.points.tobytes()
                assert a.arclength == b.arclength
        assert curves.counts == curves0.counts
        assert curves.diagnostics == curves0.diagnostics
