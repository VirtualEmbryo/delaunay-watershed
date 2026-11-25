"""What dw2d's reconstruction did and what it found, as frozen records and warning classes.

The names follow dw3d's where the concept is the same (`JunctionRelocationReport`, `EnclosedMediumWarning`,
`UnresolvedCellWarning`). The warning classes are dw2d's own, distinct from dw3d's classes of the same names: neither
package imports the other at module level, so a filter on `dw3d.EnclosedMediumWarning` does not catch dw2d's (filter
on both, or on their base class `UserWarning`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class EnclosedMediumWarning(UserWarning):
    """A pocket of label 0 enclosed by cells (it touches no side of the image) is meshed as the outer medium.

    dw2d has no enclosed-medium region yet (dw3d labels such a pocket as its own medium region): the pocket's
    triangles take label 0, so the mesh treats it as contact with the outer medium. The warning names each pocket.
    """


class UnresolvedCellWarning(UserWarning):
    """A label of the mask is absent from the finished mesh: the reconstruction did not resolve that cell.

    Usually a cell narrower than the point placement's `min_distance`, which no triangle of the watershed reached.
    The warning names each such label; `MeshReconstructionAlgorithm.unresolved_cells` lists them.
    """


class ThinMediumMarginWarning(UserWarning):
    """A cell's edges lie on a side of the image that its pixels do not reach, and padding could not prevent it.

    Emitted once per reconstruction, naming each such cell, the side, the cell's medium margin to it and why the side
    was not padded: a cell reaches it (it is a crop), padding is switched off, or the margin is already
    `thin_margin_bound(min_distance)` or more. The cell may have taken some of the medium between it and the side.
    The same name and meaning as dw3d's warning; see `dw2d.thin_medium_margins`.
    """


@dataclass(frozen=True)
class JunctionRelocationReport:
    """What one 2D junction relocation did (`dw2d.relocate_junction_vertices`).

    Attributes:
        mask_junctions: junctions the mask gives (connected groups of pixel corners where three labels meet).
        mesh_junctions: mesh vertices where exactly three materials meet.
        planned: mesh junctions paired with a mask junction within the gate.
        fixed: planned moves dropped because the vertex is fixed (on the image border).
        moved: moves applied.
        refused: guard name -> number of moves that guard refused.
        unmatched_triples: material triples whose mesh and mask junction counts differ (left untouched).
        crossings_before, crossings_after: proper crossings between edges before and after (never more after).
        contacts_before, contacts_after: edge contacts before and after (never more after).
        median_displacement_px: median distance travelled by the applied moves, in pixels, or `None` if none moved.
    """

    mask_junctions: int
    mesh_junctions: int
    planned: int
    fixed: int
    moved: int
    refused: dict[str, int] = field(default_factory=dict)
    unmatched_triples: int = 0
    crossings_before: int = 0
    crossings_after: int = 0
    contacts_before: int = 0
    contacts_after: int = 0
    median_displacement_px: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The report as a plain dictionary (for JSON records)."""
        return asdict(self)


@dataclass(frozen=True)
class InterfaceProjectionReport:
    """What one 2D interface projection did (`dw2d.project_interface_vertices`).

    Attributes:
        candidates: interface vertices eligible for a move (one interface, not a junction, not on the image border).
        moved: moves applied.
        not_converged: candidates whose Newton iteration did not converge (left in place).
        refused: guard name -> number of moves that guard refused.
        crossings_before, crossings_after: proper crossings between edges before and after.
        contacts_before, contacts_after: edge contacts before and after.
        median_displacement_px: median distance travelled by the applied moves, in pixels, or `None` if none moved.
    """

    candidates: int
    moved: int
    not_converged: int
    refused: dict[str, int] = field(default_factory=dict)
    crossings_before: int = 0
    crossings_after: int = 0
    contacts_before: int = 0
    contacts_after: int = 0
    median_displacement_px: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The report as a plain dictionary (for JSON records)."""
        return asdict(self)


@dataclass(frozen=True)
class EnclosedMediumPocket:
    """One 4-connected piece of label 0 that touches no side of the image.

    Attributes:
        pixels: its number of pixels.
        first_pixel: its first pixel in raster order, `(row, column)`.
        lining_labels: the labels 4-adjacent to it, ascending.
    """

    pixels: int
    first_pixel: tuple[int, int]
    lining_labels: tuple[int, ...]
