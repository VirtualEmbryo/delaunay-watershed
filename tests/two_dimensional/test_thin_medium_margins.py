"""A cell a few pixels from the image border no longer takes the medium between them, nor collapses; empty meshes.

Before 0.8.0, dw2d's field was 0 on the image border, so the membrane facing a side a few pixels away lost its points:
at `min_distance` 4 a 20-px square 2, 3 and 4 px from every side meshed to 1.25, 1.37 and 1.28 times its pixels (0.97
at 16 px), and a 12-px square 3 px from one side collapsed to a triangle of 0.23 times its pixels at `min_distance` 5
(0.84 at 12 px). Since 0.8.0 every side that no cell reaches and whose margin is below `2 * min_distance + 2` pixels is
padded with label 0 before meshing, and the mesh is returned in the mask's frame (`dw2d.thin_medium_margins`). A mesh
without an edge raised IndexError; it is now returned empty and its cells named. These tests fail on the 0.7 code: the
area tests by the ratio, the others because `pad_thin_medium_margins`, `medium_margin_padding` and
`ThinMediumMarginWarning` did not exist, or by the IndexError.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dw2d
from dw2d.thin_medium_margins import (
    IMAGE_SIDES,
    cells_on_each_image_side,
    padding_of_thin_margins,
    side_margins,
    thin_margin_bound,
)

BAND = (0.5, 1.5)


def square(side: int, margin: int) -> np.ndarray:
    mask = np.zeros((side + 2 * margin,) * 2, dtype=np.int64)
    mask[margin : margin + side, margin : margin + side] = 1
    return mask


def square_near_one_side(side: int, margin: int) -> np.ndarray:
    """A square `margin` px from the side axis0_low and 16 px from the three others."""
    mask = np.zeros((side + margin + 16, side + 32), dtype=np.int64)
    mask[margin : margin + side, 16 : 16 + side] = 1
    return mask


def reconstruct(mask: np.ndarray, min_distance: int = 4, **settings: bool):
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm(min_distance=min_distance)
    for name, value in settings.items():
        setattr(algorithm, name, value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    return algorithm, (points, segments, labels), [w.category for w in caught]


def area_over_pixels(mask: np.ndarray, points, segments, labels, cell: int = 1) -> float:
    start, end = points[segments[:, 0]], points[segments[:, 1]]
    cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
    area = abs(0.5 * (cross[labels[:, 0] == cell].sum() - cross[labels[:, 1] == cell].sum()))
    return area / np.count_nonzero(mask == cell)


@pytest.mark.parametrize("margin", [2, 3, 4, 5])
@pytest.mark.parametrize("min_distance", [3, 4, 5])
def test_a_square_near_the_image_border_keeps_its_area(margin, min_distance) -> None:
    """A 20-px square 2 to 5 px from every side meshes inside the band (before 0.8.0: up to 1.55 times its pixels)."""
    mask = square(20, margin)
    algorithm, mesh, caught = reconstruct(mask, min_distance)
    assert BAND[0] <= area_over_pixels(mask, *mesh) <= BAND[1]
    assert algorithm.medium_margin_padding == dict.fromkeys(IMAGE_SIDES, thin_margin_bound(min_distance) - margin)
    assert dw2d.ThinMediumMarginWarning not in caught


@pytest.mark.parametrize("margin", [3, 4])
def test_a_small_square_near_one_side_no_longer_collapses(margin) -> None:
    """A 12-px square 3 or 4 px from one side meshes as it does 12 px from it (before 0.8.0: a triangle, 0.23)."""
    mask = square_near_one_side(12, margin)
    algorithm, mesh, _ = reconstruct(mask, 5)
    _, far_mesh, _ = reconstruct(square_near_one_side(12, 12), 5)
    assert algorithm.medium_margin_padding == {"axis0_low": 12 - margin}
    far = area_over_pixels(square_near_one_side(12, 12), *far_mesh)
    assert area_over_pixels(mask, *mesh) == pytest.approx(far, abs=1e-12)


def test_the_padded_mesh_is_that_of_the_same_tissue_with_a_wider_margin() -> None:
    """With no dither in 2D, a square 3 px from every side gets the mesh of the square 10 px away, moved by 7 px."""
    algorithm, (points, segments, labels), _ = reconstruct(square(20, 3))
    _, (wide_points, wide_segments, wide_labels), _ = reconstruct(square(20, 10))
    assert algorithm.medium_margin_padding == dict.fromkeys(IMAGE_SIDES, 7)
    np.testing.assert_array_equal(segments, wide_segments)
    np.testing.assert_array_equal(labels, wide_labels)
    np.testing.assert_array_equal(points, wide_points - 7.0)


def test_switching_the_padding_off_gives_the_previous_mesh() -> None:
    """Padding off meshes the mask as given: the 20-px square 4 px from the border at 1.28 times its pixels."""
    mask = square(20, 4)
    algorithm, mesh, _ = reconstruct(mask, pad_thin_medium_margins=False)
    assert algorithm.medium_margin_padding == {}
    assert area_over_pixels(mask, *mesh) == pytest.approx(1.278, abs=1e-3)


@pytest.mark.parametrize("margin", [10, 16])
def test_a_mask_without_a_thin_margin_is_meshed_as_before(margin) -> None:
    """At or beyond the bound (10 px at `min_distance` 4), the mesh is bitwise that of the unpadded route."""
    mask = square(20, margin)
    algorithm, (points, segments, labels), _ = reconstruct(mask)
    _, (old_points, old_segments, old_labels), _ = reconstruct(mask, pad_thin_medium_margins=False)
    assert algorithm.medium_margin_padding == {}
    assert np.array_equal(points, old_points)
    assert np.array_equal(segments, old_segments)
    assert np.array_equal(labels, old_labels)


def test_a_side_a_cell_reaches_is_never_padded_and_the_rest_are() -> None:
    """A cell cut by the side axis1_high: that side stays as it is; the three sides of medium are padded."""
    mask = np.zeros((30, 26), dtype=np.int64)
    mask[3:27, 3:26] = 1
    algorithm, (points, segments, _), _ = reconstruct(mask)
    assert algorithm.medium_margin_padding == {"axis0_low": 7, "axis0_high": 7, "axis1_low": 7}
    used = points[np.unique(segments)]
    assert used[:, 1].max() <= 25 + 1e-12  # the crop side: the mesh stops on it
    assert 1 in algorithm.cut_cells


def test_the_compressed_segmentation_is_in_the_frame_of_the_mask() -> None:
    """The rebuilt mask has the mask's shape and agrees with it on 94.8 % of its pixels (unpadded: 62.0 %)."""
    mask = square(20, 3)
    algorithm, _, _ = reconstruct(mask)
    compressed = algorithm.last_compressed_segmentation
    assert tuple(compressed["image_shape"]) == mask.shape
    seeds = np.asarray(compressed["seeds"])
    assert np.all(seeds >= 0)
    assert np.all(seeds < np.array(mask.shape))
    rebuilt = dw2d.reconstruct_mask_from_dict(compressed)
    assert rebuilt.shape == mask.shape
    assert np.mean(rebuilt == mask) > 0.9
    assert algorithm.rec_metadata()["source_shape"] == mask.shape


def test_the_margin_helpers() -> None:
    mask = np.zeros((20, 30), dtype=np.int64)
    mask[2:15, 0:10] = 1
    assert side_margins(mask) == (2, 5, 0, 20)
    assert cells_on_each_image_side(mask) == ((), (), (1,), ())
    assert thin_margin_bound(4) == 10
    assert padding_of_thin_margins(side_margins(mask), cells_on_each_image_side(mask), 10) == ((8, 5), (0, 0))
    assert side_margins(np.zeros((5, 5), dtype=np.int64)) == (None, None, None, None)
    assert padding_of_thin_margins((None,) * 4, ((),) * 4, 10) is None


@pytest.mark.parametrize("pad", [True, False])
def test_a_mesh_without_an_edge_is_empty_and_its_cell_named(pad) -> None:
    """A 12-px square 1 px from every side at `min_distance` 5 unpadded, and a mask without a cell: no IndexError."""
    for mask, min_distance in ((square(12, 1), 5), (np.zeros((20, 20), dtype=np.int64), 4)):
        algorithm, (points, segments, labels), caught = reconstruct(mask, min_distance, pad_thin_medium_margins=pad)
        if not pad or not mask.any():
            assert np.shape(segments) == (0, 2)
            assert np.shape(labels) == (0, 2)
            assert np.shape(points)[1:] == (2,)
            assert (dw2d.UnresolvedCellWarning in caught) == bool(mask.any())
            with pytest.raises(ValueError, match="the mesh has no edge"):
                _ = algorithm.last_constructed_dcel
        else:  # padded, the square is meshed
            assert 1 in labels


def test_the_warning_gives_each_named_cells_area_ratio():
    """Each named cell's mesh area over its pixel count (the shoelace sum over its edges), to two decimals."""
    import re
    import warnings

    import dw2d
    from dw2d.reports import ThinMediumMarginWarning

    mask = np.zeros((48, 48), dtype=np.uint16)
    mask[2:26, 12:36] = 1
    mask[26:46, 12:36] = 2
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    algorithm.pad_thin_medium_margins = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    (message,) = [str(w.message) for w in caught if issubclass(w.category, ThinMediumMarginWarning)]
    for cell in (1, 2):
        rows = labels == cell
        start, end = points[segments[:, 0]], points[segments[:, 1]]
        cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
        area = 0.5 * (cross[rows[:, 0]].sum() - cross[rows[:, 1]].sum())
        found = re.findall(rf"cell {cell} lies on the image side \w+, .*?, mesh area ([0-9.]+) times its pixel count",
                           message)
        assert found and {float(r) for r in found} == {round(area / np.count_nonzero(mask == cell), 2)}
