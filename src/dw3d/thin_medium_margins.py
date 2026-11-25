"""A thin margin of medium between a cell and the image box: why the cell took it, and the padding that prevents it.

The defect
----------
Before 0.8.0, a cell separated from a face of the image box by a few voxels of medium could take that medium: at the
default `min_distance` of 3, a 12-voxel cube 4 voxels from every face meshed to 2.98 times its voxel count, and the
two boxes of a 24-voxel smoke mask to 2.46 times. The mechanism, measured on cubes of side 8 to 32:

1. The distance field is 0 on the image box's outer layer of voxels (`dw3d.edt._pad_mask` makes the box a boundary),
   which is *below* the level of the membranes (`M - 1`, 0.732 on these masks).
2. A minimum is placed only where no lower voxel lies within `min_distance` (`skimage.feature.peak_local_max`'s
   window). So the membrane between the cell and a face loses its minima to the box: at a margin of 2 or 3 voxels it
   carries none at all, on every cube measured.
3. With no point on that membrane, the tetrahedra joining the cell's interior point to the box span the medium, and
   the seeded watershed hands them to the cell through faces scoring like the cell's interior. Even a labelling by
   tetrahedron centroid would give the cell 1.26 to 1.47 times its voxels: the tetrahedralisation is wrong there, not
   only its labelling.

The range grows with `min_distance`: the defect reached margins of 5 to 7 voxels at `min_distance` 3, 7 to 8 at 4 and
8 to 10 at 5, and was gone from `2 * min_distance + 2` voxels on (8, 10, 12), on cubes of side 12 to 32. Its signature
was exact on those cubes: every cube whose mesh volume exceeded its voxel count by more than 15 % had triangles of the
cell on a face of the image box that the cell's voxels do not reach (*unsupported border triangles*). Real cells are
rounded and come close to the box at one place only; of 684 masks (47 benchmark tissues, 141 of the resolution sweep,
492 frames of a real embryo series, the four test images), 6 had unsupported border triangles (1 to 5 each), and the
362 cells within 7 voxels of a face kept a mesh volume of 0.85 to 1.45 times their voxel count, inside the agreement
band of `dw3d.cell_volume_agreement`.

The fix
-------
`needs_padding` decides it after mesh surgery, before anything is reported: when a vertex of the mesh lies on a face
of the image box that no cell of the mask reaches, the reconstruction is run again on the mask padded with label 0 on
**every** face that no cell reaches and whose margin (its smallest gap to a cell) is below
`thin_margin_bound(min_distance)`, each up to that bound (`padding_of_thin_margins`), and the finished mesh is
translated back into the mask's frame. Every cell then sits at least `2 * min_distance + 2` voxels from those faces, as
on a mask imaged with a wider margin. The mask's own voxels keep the dither they have unpadded (the default placement
adds `U(0, 1e-5)` from `RandomState(42)` to the distance field; drawn on the padded array, every voxel would get another
value, and the whole mesh would be a new draw): `dither_padding` of `dw3d.points_on_edt._dithered`. So only the mesh
within the padded faces' reach changes: on the 18 padded masks of the lab's data, 0 to 28 line-mean wedges of 3 to 123
changed on 13 resolution-sweep masks and 4 to 18 of about 700 on 5 embryo frames, where a dither drawn on the padded
array changed every wedge. All thin faces are padded, not only the one that triggered: the tetrahedralisation
is global, and padding one face of a cube 4 voxels from every face moved the defect to the others (1.13 to 2.68 times
the voxel count). A mask whose first mesh has no vertex on a face no cell reaches is meshed exactly as before. A face
that a cell reaches is a crop (`dw3d.crop_faces`) and is never padded: padding it with medium would close the cut cells
there. A cell left with unsupported border triangles, on such a face or on a face whose margin is already at the
bound, is named by `ThinMediumMarginWarning`.

On a padded face the mesh is that of the same tissue imaged with a wider margin, so the cell's surface may reach a few
voxels into the medium there, and past the image box when the margin is thinner than that reach; before, the box
clipped it and the cell took the medium instead. The overshoot is at most the face's padding (the mesh lies inside the
padded box), and was measured at: 7 voxels on a 16-voxel cube 1 voxel from every face, and up to 4 on cubes of side 8
to 24 at margins of 1 to 5 voxels; up to 4 voxels on foambryo's test masks (four cells in a ball 4 voxels from the
faces, with cubic voxels); 0 to 2 voxels on the 18 padded masks of the lab's data (one vertex by 1
voxel on one resolution-sweep mask, five vertices by up to 2 voxels on one embryo frame, none on the others). No vertex
passes a crop face, which is never padded.

What the padding does not do: it does not change how the watershed treats small cells. On a ball of four cells cut by
the image box, at `min_distance` 3, the cell that the watershed empties (`UnresolvedCellWarning`, cause "removed by
mesh surgery") is kept only as a fragment of 4 triangles, 0.010 of its voxel volume inside the box of voxel centres; and
a cell with one or two interior points (a cube of 8 or 12 voxels, or a cell of that ball) can fall below half its voxel
count on some dither draws, with or without padding (2 of 20 seeds for the 8-voxel cube 8 voxels from every face).

Where it is not applied by default: a mask that `construct_mesh_in_physical_units` resamples (voxels longer along one
axis). On foambryo's four-cell ball 4 voxels from the faces, with z twice as long, over dither seeds 42-141, a cell kept
less than half its voxel volume on 20 of 100 seeds padded against 3 of 100 unpadded (McNemar exact p = 1.5e-5; with
cubic voxels 13 against 10, p = 0.51). The cell is lost in the watershed: the cell facing a padded face hands most of
its interior to the medium through a face straddling its outer membrane, before that interior reaches its seed; the
distance field there is the unpadded one, the tetrahedralisation near the padded faces is not, and why this happens
more often on resampled masks was not found. Such a mask keeps 0.7's thin-margin meshes, named by
`ThinMediumMarginWarning`; `pad_thin_medium_margins=True` in that call pads it.

What the trigger misses: it reads the first mesh's vertices, so a cell that collapses near a face without reaching it
is not padded. On cubes of side 6 to 12 voxels 1 to 5 voxels from one or every face, `min_distance` 3 to 5, ten dither
seeds (1 200 runs), 25 collapsed to 0.00-0.49 of their voxels with no vertex on a face; padded, they give 0.58-0.80,
and the same cubes 16 voxels from every face 0.51-0.71. dw2d decides from the mask instead (`dw2d.thin_medium_margins`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dw3d.crop_faces import IMAGE_FACES, CropFaceReport


class ThinMediumMarginWarning(UserWarning):
    """A cell's mesh lies on a face of the image box that its voxels do not reach, and padding could not prevent it.

    Emitted once per reconstruction, naming each such cell, the face, the cell's medium margin to it (the number of
    voxels of label 0 between the cell and the face) and why the face was not padded: a cell reaches it (it is a
    crop), padding is switched off, or the margin is already `thin_margin_bound(min_distance)` or more. The cell may
    have taken some of the medium between it and the face; compare its volume with its voxel count
    (`cell_volume_report`). See `dw3d.thin_medium_margins`.
    """


def thin_margin_bound(min_distance: int | None) -> int:
    """The medium margin, in voxels, from which a cell no longer takes the medium between it and the image box.

    Args:
        min_distance: the point placement's `min_distance`; `None` (a placement that does not carry one) counts as 3,
            the default.

    Returns:
        int: `2 * min_distance + 2` (8 at the default 3), the margin from which the defect was gone on every cube
            measured at `min_distance` 3, 4 and 5 (see the module docstring).
    """
    return 2 * (3 if min_distance is None else int(min_distance)) + 2


def face_margins(segmented_image: NDArray[np.integer]) -> tuple[int | None, ...]:
    """The medium margin of each face of the image box: the voxels of label 0 between it and the nearest other label.

    Args:
        segmented_image: the 3-D mask; label 0 is the medium.

    Returns:
        tuple[int | None, ...]: for each face of `IMAGE_FACES`, in that order, the smallest number of label-0 voxels
            between the face and a voxel of another label along the axis normal to it: 0 when a cell reaches the face,
            `None` when the mask has no voxel other than 0.
    """
    occupied = np.asarray(segmented_image) != 0
    out: list[int | None] = []
    for axis in range(3):
        profile = occupied.any(axis=tuple(a for a in range(3) if a != axis))
        filled = np.flatnonzero(profile)
        if not len(filled):
            out += [None, None]
            continue
        out += [int(filled[0]), int(len(profile) - 1 - filled[-1])]
    return tuple(out)


def cell_margin_to_face(segmented_image: NDArray[np.integer], cell: int, face: str) -> int:
    """The number of label-0 voxels between a cell and one face of the image box, along the face's normal.

    Args:
        segmented_image: the 3-D mask.
        cell: the cell's label.
        face: one of `IMAGE_FACES`.

    Returns:
        int: the smallest index of the cell's voxels counted from that face (0 when the cell reaches it); -1 when the
            mask has no voxel of that label.
    """
    axis, side = divmod(IMAGE_FACES.index(face), 2)
    coordinates = np.flatnonzero(np.asarray(segmented_image == cell).any(axis=tuple(a for a in range(3) if a != axis)))
    if not len(coordinates):
        return -1
    n = np.shape(segmented_image)[axis]
    return int(coordinates[0]) if side == 0 else int(n - 1 - coordinates[-1])


def unsupported_border_triangles(report: CropFaceReport) -> NDArray[np.bool_]:
    """Which triangles lie on a face of the image box that their cell's voxels do not reach.

    Args:
        report: the crop-face report of the mesh (`dw3d.crop_faces.classify_crop_faces`).

    Returns:
        NDArray[np.bool_]: `(n_triangles,)`, `True` on a border triangle on a single image face whose cell has no voxel
            on that face.
    """
    return (report.image_face_of_triangle >= 0) & ~report.cell_reaches_image_face


def needs_padding(
    points: NDArray[np.floating],
    triangles: NDArray[np.integer],
    cells_on_faces: tuple[tuple[int, ...], ...],
    image_shape: tuple[int, ...],
) -> bool:
    """Whether a vertex of the mesh lies on a face of the image box that no cell of the mask reaches.

    Every triangle carries at least one cell, so such a vertex is a cell's surface reaching the box across medium:
    the signature of the thin-margin defect (see the module docstring).

    Args:
        points: mesh points in voxel units of the mask, **before** junction relocation (on the lattice, so a vertex on
            an image face has exactly the coordinate `0` or `shape[k] - 1`).
        triangles: `(n, 3)` mesh triangles.
        cells_on_faces: `dw3d.crop_faces.cells_on_each_image_face` of the mask.
        image_shape: the mask's shape.

    Returns:
        bool: `True` when some face with no cell on it carries a vertex of a triangle.
    """
    if not len(triangles):
        return False
    used = np.asarray(points)[np.unique(np.asarray(triangles))]
    for index, cells in enumerate(cells_on_faces):
        if cells:
            continue
        axis, side = divmod(index, 2)
        plane = 0 if side == 0 else image_shape[axis] - 1
        if np.any(used[:, axis] == plane):
            return True
    return False


def padding_of_thin_margins(
    margins: tuple[int | None, ...],
    cells_on_faces: tuple[tuple[int, ...], ...],
    bound: int,
) -> tuple[tuple[int, int], ...] | None:
    """The label-0 padding that brings every face no cell reaches to a margin of at least `bound` voxels.

    Args:
        margins: `face_margins` of the mask.
        cells_on_faces: `dw3d.crop_faces.cells_on_each_image_face` of the mask; a face with a cell on it is a crop and
            is never padded.
        bound: `thin_margin_bound` of the placement's `min_distance`.

    Returns:
        tuple[tuple[int, int], ...] | None: `numpy.pad` widths, `((low, high),) * 3` along axes 0, 1, 2, or `None` when
            no face is to be padded.
    """
    widths = [[0, 0], [0, 0], [0, 0]]
    for index, margin in enumerate(margins):
        if cells_on_faces[index] or margin is None or margin >= bound:
            continue
        axis, side = divmod(index, 2)
        widths[axis][side] = bound - margin
    if not any(any(pair) for pair in widths):
        return None
    return tuple((low, high) for low, high in widths)


def warning_message(cells: list[tuple]) -> str:
    """The text of `ThinMediumMarginWarning`: one clause per `(cell, face, margin, why the face was not padded[, ratio])`.

    `ratio`, when given, is the cell's mesh volume over its voxel count (as `cell_volume_report` computes it), so that
    a reader can tell a cell that took medium (a ratio above 1) from one whose volume is normal.
    """
    clauses = "; ".join(
        f"cell {cell} lies on the image face {face}, {margin} voxel{'s' if margin != 1 else ''} of medium away from "
        f"its voxels ({why})" + (f", mesh volume {rest[0]:.2f} times its voxel count" if rest else "")
        for cell, face, margin, why, *rest in cells
    )
    return (
        f"{clauses}. The mesh of such a cell may include medium between it and the face, which makes its volume "
        "larger than its voxel count (cell_volume_report lists the cells outside [0.5, 1.5]). See "
        "dw3d.thin_medium_margins."
    )
