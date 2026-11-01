"""Pockets of medium that the tissue encloses: their own regions, flagged as media, instead of part of a cell.

Why this exists
---------------
The watershed places one seed per label value, at the label's largest EDT value. The medium, label 0, is one
value, so before 0.7.1 it had one seed, in whichever of its pieces was deepest. An enclosed pocket of label 0 (a
lumen, a cavity) had none: its tetrahedra formed one unseeded group, which the first seeded cell it reached took
whole. On a 697-cell embryo with a 389 918-voxel lumen, one lining cell came out at 2.40 times its voxel volume,
and no lining cell kept a membrane against the lumen. When a pocket was deeper than the exterior (a tight crop),
label 0's seed fell in the pocket and the exterior was absorbed instead.

What it does
------------
Pieces of label 0 are 6-connected (two voxels sharing a face), as foambryo's mask report counts them. A piece
touching no face of the array is an **enclosed pocket**. A pocket becomes its own region, flagged as a medium,
when its inscribed radius is at least `min_distance + 1` voxels and at least two cells line it:

* the inscribed radius is the largest Euclidean distance, in voxels, from a pocket voxel to the nearest voxel
  outside the pocket. Below about 3 voxels at `min_distance` 3 a pocket holds no tetrahedron, and a seed placed
  in it would take a neighbouring cell's tetrahedron instead, so it is left as before;
* a pocket lined by one cell is a hole in that cell's segmentation, not a lumen between cells: as its own region
  it would be a closed surface with no triple line, whose tension no method can infer. It is left as before.

The new labels are `max + 1, max + 2, ...`, in the order of each pocket's lowest raster (C-order) index. Pockets
left as before are reported, and `EnclosedMediumWarning` names the cell that took each.

Every piece of label 0 that touches the image box is exterior and gets its own label-0 seed (see
`dw3d.reconstruction_algorithm`). A mask with no pocket at or above the bound and one exterior piece is
reconstructed bitwise as before.

Herve Turlier's group, 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

#: Reasons a pocket is left to the cells.
BELOW_THE_SIZE_BOUND = "below the size bound"
LINED_BY_ONE_CELL = "lined by one cell"
SWITCHED_OFF = "switched off"

#: Pockets named in the text of `EnclosedMediumWarning`, largest first.
_POCKETS_NAMED_IN_THE_WARNING = 5

_SIX_CONNECTED = ndimage.generate_binary_structure(3, 1)


class UnresolvedCellWarning(UserWarning):
    """Some regions of the mask have no tetrahedron of their own and are absent from the mesh.

    A region's watershed seed is the tetrahedron whose centroid is nearest to the region's deepest voxel. When no
    tetrahedron centroid lies in the region's voxels (a cell a few voxels thick, or smaller than the point
    placement's `min_distance`), that tetrahedron belongs to a neighbour; before 0.7.1 the region grew from it inside
    the neighbour, as a phantom. Since 0.7.1 the seed is dropped and the region is absent from the mesh. Emitted once
    per reconstruction that drops a seed, naming each region, its voxel count and the `min_distance`; the regions are
    the reconstruction's `enclosed_medium_report.unresolved`. A smaller `min_distance` may resolve them.
    """


class EnclosedMediumWarning(UserWarning):
    """Enclosed pockets of medium were left to the cells around them.

    Emitted once per reconstruction that leaves at least one pocket of label 0 to the cells (below the size bound,
    or lined by one cell). The pockets are the reconstruction's `enclosed_medium_report.pockets_left_to_cells`. A
    subclass of :class:`UserWarning`, so it can be silenced or promoted to an error on its own with
    :func:`warnings.filterwarnings`.
    """


@dataclass(frozen=True)
class EnclosedMedium:
    """An enclosed pocket of medium that became its own region.

    Attributes:
        label: its label in the reconstructed mask and mesh.
        voxels: its voxel count.
        inscribed_radius: the largest distance from one of its voxels to the nearest voxel outside it, in voxels of
            the grid that was meshed.
        lining_cells: the labels 6-adjacent to it, ascending.
    """

    label: int
    voxels: int
    inscribed_radius: float
    lining_cells: tuple[int, ...]


@dataclass(frozen=True)
class UnresolvedRegion:
    """A region of the mask with no tetrahedron of its own, absent from the mesh.

    Attributes:
        label: its label in the mask that was meshed.
        voxels: its voxel count.
        kind: `"cell"`, `"enclosed medium"` or `"exterior"` (a piece of label 0 touching the image box).
    """

    label: int
    voxels: int
    kind: str


@dataclass(frozen=True)
class PocketLeftToCells:
    """An enclosed pocket of medium left to the cells around it, as before 0.7.1.

    Attributes:
        voxels: its voxel count.
        inscribed_radius: as `EnclosedMedium.inscribed_radius`.
        lining_cells: the labels 6-adjacent to it, ascending.
        reason: `"below the size bound"`, `"lined by one cell"`, or `"switched off"` (the reconstruction's
            `label_enclosed_media` is `False`).
        absorbed_by: the region that took most of the volume of the tetrahedra whose centroid lies in the pocket;
            when none does, the region of the tetrahedron whose centroid is nearest to the pocket's deepest voxel.
            Usually a lining cell; it can be 0 (a pocket of a few voxels next to the exterior). `None` before the
            reconstruction's watershed has run.
    """

    voxels: int
    inscribed_radius: float
    lining_cells: tuple[int, ...]
    reason: str
    absorbed_by: int | None = None


@dataclass
class EnclosedMediumReport:
    """The pieces of medium of one reconstruction: the enclosed media made, the pockets left, the exterior.

    Attributes:
        media: the enclosed pockets that became their own regions, by label.
        pockets_left_to_cells: the enclosed pockets left to the cells, in the order of their lowest raster index.
        exterior_pieces: the pieces of label 0 that touch a face of the array, each of which gets a seed.
        min_inscribed_radius: the size bound used, in voxels.
        unresolved: the regions whose seed had no tetrahedron centroid in their own voxels, so that they are absent
            from the mesh (filled by the reconstruction; `UnresolvedCellWarning`).
        min_distance: the point placement's `min_distance` the reconstruction used (`None` when unknown).
        moved_seeds: labels whose seed's nearest tetrahedron lay outside their own voxels and was replaced by the
            nearest one inside (filled by the reconstruction).
        exterior_seeds_changed: whether label 0's seeds differ from the one seed a single label gets (two or more
            exterior pieces, or the deepest voxel of label 0 in a pocket left to the cells; filled by the
            reconstruction).
    """

    media: tuple[EnclosedMedium, ...] = ()
    pockets_left_to_cells: tuple[PocketLeftToCells, ...] = ()
    exterior_pieces: int = 0
    min_inscribed_radius: float = 0.0
    unresolved: tuple[UnresolvedRegion, ...] = ()
    min_distance: int | None = None
    moved_seeds: tuple[int, ...] = ()
    exterior_seeds_changed: bool = False
    has_medium: bool = field(default=False, repr=False)

    @property
    def dropped_seeds(self) -> tuple[int, ...]:
        """The labels of the unresolved regions, ascending."""
        return tuple(region.label for region in self.unresolved)

    @property
    def unresolved_cells(self) -> tuple[int, ...]:
        """The labels of the unresolved regions that are cells, ascending."""
        return tuple(region.label for region in self.unresolved if region.kind == "cell")

    @property
    def changes_the_mesh(self) -> bool:
        """Whether any of the three rules acted, so that the mesh may differ from 0.7.0's on the same tesselation.

        `False` means: no enclosed medium made, label 0 seeded as before, every seed already inside its label. The
        reconstruction is then bitwise what it was (pockets left to the cells are left as before).
        """
        return bool(self.media or self.dropped_seeds or self.moved_seeds or self.exterior_seeds_changed)

    @property
    def medium_labels(self) -> tuple[int, ...]:
        """The labels that are media: 0 first when the mask has it, then the enclosed media, ascending."""
        enclosed = tuple(medium.label for medium in self.media)
        return ((0,) if self.has_medium else ()) + enclosed

    def as_dict(self) -> dict:
        """A JSON-serialisable copy of the report."""
        return {
            "medium_labels": list(self.medium_labels),
            "media": [
                {"label": m.label, "voxels": m.voxels, "inscribed_radius": m.inscribed_radius,
                 "lining_cells": list(m.lining_cells)}
                for m in self.media
            ],
            "pockets_left_to_cells": [
                {"voxels": p.voxels, "inscribed_radius": p.inscribed_radius, "lining_cells": list(p.lining_cells),
                 "reason": p.reason, "absorbed_by": p.absorbed_by}
                for p in self.pockets_left_to_cells
            ],
            "exterior_pieces": self.exterior_pieces,
            "min_inscribed_radius": self.min_inscribed_radius,
            "unresolved": [{"label": u.label, "voxels": u.voxels, "kind": u.kind} for u in self.unresolved],
            "min_distance": self.min_distance,
            "moved_seeds": list(self.moved_seeds),
            "exterior_seeds_changed": self.exterior_seeds_changed,
            "changes_the_mesh": self.changes_the_mesh,
        }


@dataclass
class _Pocket:
    """One enclosed piece of label 0, with what the reconstruction needs after the relabelling."""

    first_index: int
    voxels: int
    inscribed_radius: float
    lining_cells: tuple[int, ...]
    flat_indices: NDArray[np.int64]
    deepest_index: int


@dataclass
class MediumPieces:
    """The pieces of label 0 of a mask, kept for the reconstruction's seeding and its report.

    Attributes:
        exterior_count: the pieces touching a face of the array.
        exterior_components: the component image, kept only when there are two or more exterior pieces (to seed
            each), else `None`; `exterior_ids` are its exterior components, in raster order of their first voxel.
        left: the enclosed pockets left to the cells.
    """

    exterior_count: int = 0
    exterior_components: NDArray[np.int32] | None = None
    exterior_ids: tuple[int, ...] = ()
    left: list[_Pocket] = field(default_factory=list)


def find_enclosed_media(
    mask: NDArray[np.integer],
    min_inscribed_radius: float,
) -> tuple[NDArray, EnclosedMediumReport]:
    """Give each enclosed pocket of label 0 lined by two or more cells, and deep enough, its own label.

    Args:
        mask: the 3-D integer label mask; label 0 is the medium. Never modified.
        min_inscribed_radius: the smallest inscribed radius, in voxels, of a pocket that becomes its own region
            (the reconstruction uses its placement's `min_distance + 1`).

    Returns:
        tuple: the mask to mesh (`mask` itself when no pocket becomes a region, else a relabelled copy whose dtype
        is widened if the new labels need it) and the report (`absorbed_by` of each pocket left is `None` here).
    """
    relabelled, report, _ = label_enclosed_media(mask, min_inscribed_radius)
    return relabelled, report


def label_enclosed_media(
    mask: NDArray[np.integer],
    min_inscribed_radius: float,
    *,
    make_regions: bool = True,
) -> tuple[NDArray, EnclosedMediumReport, MediumPieces]:
    """`find_enclosed_media`, plus the pieces the reconstruction's seeding and report need afterwards.

    With `make_regions=False` no pocket becomes a region (every pocket is left, reason `"switched off"`), so the
    mask is returned as is; the pieces are still found, for the exterior seeding and the report.
    """
    report = EnclosedMediumReport(min_inscribed_radius=float(min_inscribed_radius))
    pieces = MediumPieces()
    is_medium = mask == 0
    if not is_medium.any():
        return mask, report, pieces
    report.has_medium = True
    components, count = ndimage.label(is_medium, structure=_SIX_CONNECTED, output=np.int32)
    del is_medium
    faces = (components[0], components[-1], components[:, 0], components[:, -1], components[:, :, 0],
             components[:, :, -1])
    on_box = np.unique(np.concatenate([np.unique(face) for face in faces]))
    on_box = on_box[on_box > 0]
    report.exterior_pieces = pieces.exterior_count = len(on_box)
    if count == len(on_box):  # no enclosed pocket
        if len(on_box) > 1:
            pieces.exterior_components = components
            pieces.exterior_ids = _in_raster_order(components, on_box)
        return mask, report, pieces

    enclosed = np.setdiff1d(np.arange(1, count + 1), on_box)
    boxes = ndimage.find_objects(components)
    pockets = [_describe_pocket(mask, components, int(piece), boxes[piece - 1]) for piece in enclosed]
    pockets.sort(key=lambda pocket: pocket.first_index)

    def becomes_a_region(pocket: _Pocket) -> bool:
        return make_regions and pocket.inscribed_radius >= min_inscribed_radius and len(pocket.lining_cells) >= 2

    def reason(pocket: _Pocket) -> str:
        if not make_regions:
            return SWITCHED_OFF
        return BELOW_THE_SIZE_BOUND if pocket.inscribed_radius < min_inscribed_radius else LINED_BY_ONE_CELL

    chosen = [p for p in pockets if becomes_a_region(p)]
    left = [p for p in pockets if not becomes_a_region(p)]
    pieces.left = left
    report.pockets_left_to_cells = tuple(
        PocketLeftToCells(voxels=p.voxels, inscribed_radius=p.inscribed_radius, lining_cells=p.lining_cells,
                          reason=reason(p))
        for p in left
    )
    if len(on_box) > 1:
        pieces.exterior_components = components
        pieces.exterior_ids = _in_raster_order(components, on_box)
    else:
        del components
    if not chosen:
        return mask, report, pieces

    top = int(np.max(mask))
    dtype = np.promote_types(mask.dtype, np.min_scalar_type(top + len(chosen)))
    relabelled = mask.astype(dtype, copy=True)
    flat = relabelled.reshape(-1)
    media = []
    for k, pocket in enumerate(chosen, start=1):
        flat[pocket.flat_indices] = top + k
        media.append(EnclosedMedium(top + k, pocket.voxels, pocket.inscribed_radius, pocket.lining_cells))
    report.media = tuple(media)
    return relabelled, report, pieces


def _in_raster_order(components: NDArray[np.int32], ids: NDArray[np.integer]) -> tuple[int, ...]:
    """The component ids in the order of their first voxel in C order."""
    boxes = ndimage.find_objects(components)
    firsts = []
    for piece in ids:
        box = boxes[int(piece) - 1]
        local = np.flatnonzero((components[box] == piece).reshape(-1))[0]
        firsts.append((_global_index(components.shape, box, local), int(piece)))
    return tuple(piece for _, piece in sorted(firsts))


def _global_index(shape: tuple[int, ...], box: tuple[slice, ...], local_flat: int) -> int:
    local = np.unravel_index(local_flat, tuple(s.stop - s.start for s in box))
    return int(np.ravel_multi_index(tuple(int(c) + s.start for c, s in zip(local, box, strict=True)), shape))


def _describe_pocket(mask: NDArray, components: NDArray[np.int32], piece: int, box: tuple[slice, ...]) -> _Pocket:
    """Voxels, inscribed radius, lining cells, first and deepest voxels of one enclosed piece."""
    inside = components[box] == piece
    local = np.flatnonzero(inside.reshape(-1))
    flat_indices = np.array(
        np.ravel_multi_index(
            tuple(c + s.start for c, s in zip(np.unravel_index(local, inside.shape), box, strict=True)),
            components.shape,
        ),
        dtype=np.int64,
    )
    distance = ndimage.distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1, 1:-1]
    deepest_local = int(np.argmax(distance))  # first in C order among equal maxima
    # An enclosed piece never touches a face of the array, so one voxel around its box is inside the array.
    grown = tuple(slice(s.start - 1, s.stop + 1) for s in box)
    around = ndimage.binary_dilation(np.pad(inside, 1), structure=_SIX_CONNECTED) & ~np.pad(inside, 1)
    lining = np.unique(mask[grown][around])
    return _Pocket(
        first_index=int(flat_indices[0]),
        voxels=len(flat_indices),
        inscribed_radius=float(distance.reshape(-1)[deepest_local]),
        lining_cells=tuple(int(v) for v in lining if v != 0),
        flat_indices=flat_indices,
        deepest_index=_global_index(components.shape, box, deepest_local),
    )


def unresolved_message(report: EnclosedMediumReport) -> str:
    """The text of `UnresolvedCellWarning` for one report."""
    md = "unknown" if report.min_distance is None else str(report.min_distance)
    named = [f"{u.kind} {u.label} ({u.voxels} voxels)" for u in sorted(report.unresolved, key=lambda u: u.label)]
    n = len(named)
    return (
        f"{n} region{'s' if n != 1 else ''} of the mask {'have' if n != 1 else 'has'} no tetrahedron whose centroid "
        f"lies in {'their' if n != 1 else 'its'} voxels at min_distance {md}, so {'they are' if n != 1 else 'it is'} "
        f"absent from the mesh: " + ", ".join(named)
        + ". Before 0.7.1 such a region grew inside a neighbour; a smaller min_distance may mesh it."
    )


def warning_message(report: EnclosedMediumReport) -> str:
    """The text of `EnclosedMediumWarning` for one report (with `absorbed_by` filled)."""
    left = sorted(report.pockets_left_to_cells, key=lambda p: -p.voxels)
    named = []
    for pocket in left[:_POCKETS_NAMED_IN_THE_WARNING]:
        lining = ", ".join(str(c) for c in pocket.lining_cells)
        if pocket.absorbed_by == 0:
            taker = "the exterior medium"
        elif pocket.absorbed_by in report.medium_labels:
            taker = f"the enclosed medium {pocket.absorbed_by}"
        else:
            taker = f"cell {pocket.absorbed_by}"
        named.append(
            f"{pocket.voxels} voxels (inscribed radius {pocket.inscribed_radius:.3g}) left to {taker} "
            f"(lined by cell{'s' if len(pocket.lining_cells) != 1 else ''} {lining}): {pocket.reason}",
        )
    more = f"; and {len(left) - len(named)} more" if len(left) > len(named) else ""
    return (
        f"{len(left)} enclosed pocket{'s' if len(left) != 1 else ''} of medium left to the cells around "
        f"{'them' if len(left) != 1 else 'it'} (a pocket becomes its own region when its inscribed radius is at "
        f"least {report.min_inscribed_radius:g} voxels and two or more cells line it): " + "; ".join(named) + more
        + ". Give a pocket its own label in the mask if it is a lumen."
    )
