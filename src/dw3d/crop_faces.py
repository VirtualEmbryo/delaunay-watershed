"""Which faces of a reconstructed mesh are crop faces, and which cells the crop cuts.

The problem
-----------
A segmentation mask whose cells touch a face of the image box was cropped: the tissue goes on
beyond the image, and nothing beyond it was observed. The reconstruction closes every such cell
with triangles lying on the image face, the *border triangles*: faces of the tesselation's hull,
labelled `(0, c)`, with the cell `c` on one side and nothing on the other. They are not contact
with the medium. A tension or a contact angle read on them measures the crop, not the tissue: on
plane crops of the benchmark tissues, reading them as medium contact puts 13 % to 40 % error on
the tensions of the interfaces that touch the crop (10 % to 40 % of the tissue removed), against
9 % to 12 % when they are kept out of every equation.

So the reconstruction records them. A **crop face** is a border triangle on an image face that is
not declared physical, **and that its cell reaches in the mask**. A **cut cell** is a cell with a
crop face, or with a vertex on one: its volume and some of its interfaces were not observed, and its
pressure is not a measurement.

The reconstruction also closes a cell on the image box where the mask's cell stops short of the
image face: with no point in the few voxels of medium between the cell and the face, the
tetrahedra that reach the hull there take the cell's label. Such a border triangle is not a crop,
since the cell was observed to end inside the image, and it stays contact with the medium, as on
a mesh without crop faces. On a mask none of whose cells reaches an image face, no triangle is a
crop face and no cell is cut, whatever the reconstruction put on the hull.

Conventions
-----------
* A border triangle is found from the tesselation itself, not from coordinates: its three
  tesselation vertices form a *lone face* of the tesselation (a face with one tetrahedron), which
  lies on the hull, and the hull is the image box (the point placements put the box's eight
  corners in the point set). Coordinates would miss some: junction relocation moves a few
  border-triangle vertices off the image face (23 of 500 on one benchmark mask cut at its
  mid-plane).
* The image face a border triangle lies on is read from its vertices' coordinates **before**
  relocation, in voxel units of the mask: all three have coordinate `0` (`"axis<k>_low"`) or
  `shape[k] - 1` (`"axis<k>_high"`) along the same axis `k`. A border triangle on no single image
  face (none was seen) is a crop face whatever is declared physical.
* A cell **reaches** an image face when at least one voxel of the mask on that face (index `0` or
  `shape[k] - 1` along axis `k`) carries its label. The test is per cell and per image face, with
  no tolerance: a border triangle is a crop face when its cell has any voxel on the image face the
  triangle lies on, wherever that voxel is on the face. A border triangle on no single image face
  is a crop face when its cell reaches an image face, not declared physical, on which one of its
  vertices lies.
* An image face can be declared **physical** (`physical_image_faces`), for example a coverslip
  the tissue rests on: its border triangles stay contact with the medium, as before this module
  existed, and are not crop faces. The default declares none, because the medium beyond the image
  box is never observed.
* The mesh is not changed: crop faces keep their labels `(0, c)` and their place in the triangle
  array. A crop face is a row of a boolean mask beside the labels, never a reserved label: a
  label for the crop puts four regions around a three-face junction edge wherever the medium
  reaches the image face, which no reader of three-region junctions accepts.

Limits of the test, which the report lets a caller check: a cell that reaches an image face at one
place is cut, and every border triangle it has on that face is a crop face, even one far from the
place where it reaches it; and a cell that reaches a crop image face with too few voxels to get a
border triangle is not marked cut (`CropFaceReport.cells_reaching_a_crop_without_a_crop_face`).

On a mask that touches no image face there are no crop faces and no cut cells, and nothing else in
the reconstruction depends on this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: The six faces of the image box, by axis and side, in this order.
IMAGE_FACES = ("axis0_low", "axis0_high", "axis1_low", "axis1_high", "axis2_low", "axis2_high")

#: `image_face_of_triangle` value of a triangle that is not a border triangle.
NOT_A_BORDER_TRIANGLE = -1

#: `image_face_of_triangle` value of a border triangle that lies on no single image face.
ON_NO_SINGLE_IMAGE_FACE = -2


def normalise_physical_image_faces(faces: Iterable[str] | str | None) -> tuple[str, ...]:
    """Check a declaration of physical image faces and put it in `IMAGE_FACES` order.

    Args:
        faces: names from `IMAGE_FACES` (for example `("axis0_low",)` for a coverslip under the
            tissue at the first index of axis 0), a single name, or `None` / empty for none.

    Returns:
        tuple[str, ...]: the declared faces, each once, in `IMAGE_FACES` order.

    Raises:
        ValueError: if a name is not one of `IMAGE_FACES`.
    """
    if faces is None:
        return ()
    if isinstance(faces, str):
        faces = (faces,)
    declared = set(faces)
    unknown = sorted(declared - set(IMAGE_FACES))
    if unknown:
        message = f"unknown image face(s) {unknown}; an image face is one of {IMAGE_FACES}"
        raise ValueError(message)
    return tuple(face for face in IMAGE_FACES if face in declared)


@dataclass(frozen=True)
class CropFaceReport:
    """Where a reconstructed mesh meets the image box, and what that makes of its cells.

    Attributes:
        crop_faces: `(n_triangles,)` bool, aligned with the mesh's triangle rows: `True` on every
            border triangle that lies on an image face not declared physical and that its cell
            reaches in the mask.
        image_face_of_triangle: `(n_triangles,)` int: the index in `IMAGE_FACES` of the image face
            a border triangle lies on, `ON_NO_SINGLE_IMAGE_FACE` for a border triangle on none,
            `NOT_A_BORDER_TRIANGLE` elsewhere.
        cell_reaches_image_face: `(n_triangles,)` bool: `True` on a border triangle whose cell has
            a voxel of the mask on the image face the triangle lies on (for a triangle on no single
            image face: on an image face one of its vertices lies on), physical or not.
        cut_cells: labels of the cells with a crop face or with a vertex on one, sorted.
        physical_image_faces: the image faces declared physical, in `IMAGE_FACES` order.
        image_shape: the mask's shape, which sets where the image faces are.
        cells_on_image_faces: for each face of `IMAGE_FACES`, in that order, the labels of the
            cells with at least one voxel of the mask on it, sorted.
    """

    crop_faces: NDArray[np.bool_]
    image_face_of_triangle: NDArray[np.int64]
    cell_reaches_image_face: NDArray[np.bool_]
    cut_cells: tuple[int, ...]
    physical_image_faces: tuple[str, ...]
    image_shape: tuple[int, ...]
    cells_on_image_faces: tuple[tuple[int, ...], ...]

    @property
    def n_border_triangles(self) -> int:
        """Border triangles, crop faces and border triangles on physical image faces together."""
        return int(np.count_nonzero(self.image_face_of_triangle != NOT_A_BORDER_TRIANGLE))

    @property
    def n_crop_faces(self) -> int:
        """Crop faces."""
        return int(np.count_nonzero(self.crop_faces))

    @property
    def n_border_triangles_the_cell_does_not_reach(self) -> int:
        """Border triangles on an image face their cell does not reach in the mask: medium contact."""
        border = self.image_face_of_triangle != NOT_A_BORDER_TRIANGLE
        return int(np.count_nonzero(border & ~self.cell_reaches_image_face))

    @property
    def meets_the_image_box(self) -> bool:
        """Whether a border triangle lies on an image face its cell reaches, crop or physical.

        This is when the crop record of a file says something: `save_to_rec_mesh` and
        `save_to_vtk_mesh` write it exactly then, and otherwise write the file as before crop faces
        were recorded.
        """
        return bool(np.any(self.cell_reaches_image_face))

    @property
    def cells_reaching_a_crop_without_a_crop_face(self) -> tuple[int, ...]:
        """Cells that reach a crop image face in the mask but are not cut cells.

        Such a cell continues beyond the image, but the reconstruction put no border triangle of it
        on that face (it reaches the face with too few voxels), so nothing marks it cut and its
        pressure is reported. Empty on every mask measured so far is not a guarantee.
        """
        crop_faces = [index for index, face in enumerate(IMAGE_FACES) if face not in self.physical_image_faces]
        reaching = {cell for index in crop_faces for cell in self.cells_on_image_faces[index]}
        return tuple(sorted(reaching - set(self.cut_cells)))

    def as_dict(self) -> dict:
        """JSON-serialisable summary: counts per image face, the crop face count and the cut cells."""
        per_face = {
            face: int(np.count_nonzero(self.image_face_of_triangle == index)) for index, face in enumerate(IMAGE_FACES)
        }
        return {
            "n_border_triangles": self.n_border_triangles,
            "n_crop_faces": self.n_crop_faces,
            "border_triangles_per_image_face": per_face,
            "border_triangles_on_no_single_image_face": int(
                np.count_nonzero(self.image_face_of_triangle == ON_NO_SINGLE_IMAGE_FACE),
            ),
            "border_triangles_the_cell_does_not_reach": self.n_border_triangles_the_cell_does_not_reach,
            "physical_image_faces": list(self.physical_image_faces),
            "cut_cells": [int(c) for c in self.cut_cells],
            "cells_reaching_a_crop_without_a_crop_face": [
                int(c) for c in self.cells_reaching_a_crop_without_a_crop_face
            ],
            "cells_on_image_faces": {
                face: [int(c) for c in cells]
                for face, cells in zip(IMAGE_FACES, self.cells_on_image_faces, strict=True)
            },
            "image_shape": [int(s) for s in self.image_shape],
        }


def cells_on_each_image_face(segmented_image: NDArray[np.integer]) -> tuple[tuple[int, ...], ...]:
    """The cells with at least one voxel on each face of the image box.

    Args:
        segmented_image: the 3-D mask; label 0 is the medium.

    Returns:
        tuple[tuple[int, ...], ...]: for each face of `IMAGE_FACES`, in that order, the labels other
        than 0 found on the slice at index `0` (`"axis<k>_low"`) or `shape[k] - 1`
        (`"axis<k>_high"`) along axis `k`, sorted.
    """
    image = np.asarray(segmented_image)
    out = []
    for index in range(len(IMAGE_FACES)):
        axis, side = divmod(index, 2)
        face = np.take(image, 0 if side == 0 else image.shape[axis] - 1, axis=axis)
        out.append(tuple(int(c) for c in np.unique(face) if c != 0))
    return tuple(out)


def find_border_triangles(
    triangles: NDArray[np.integer],
    tesselation_ids: ArrayLike,
    lone_faces: ArrayLike,
) -> NDArray[np.bool_]:
    """Which mesh triangles are faces of the tesselation's hull.

    Args:
        triangles: `(n, 3)` mesh triangles, as indices into the mesh's points.
        tesselation_ids: `tesselation_ids[i]` is the tesselation index of mesh point `i` (the
            algorithm's `_surface_point_ids`).
        lone_faces: `(m, 3)` the tesselation's faces with one tetrahedron, as tesselation indices.

    Returns:
        NDArray[np.bool_]: `(n,)`, `True` where the triangle's three tesselation vertices form a
        lone face, in any order.
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    lone_array = np.asarray(lone_faces, dtype=np.int64)
    if len(triangles) == 0 or len(lone_array) == 0:
        return np.zeros(len(triangles), dtype=bool)
    as_tesselation = np.sort(np.asarray(tesselation_ids, dtype=np.int64)[triangles], axis=1)
    lone = np.sort(lone_array, axis=1)
    lone_set = {tuple(row) for row in lone.tolist()}
    return np.fromiter((tuple(row) in lone_set for row in as_tesselation.tolist()), dtype=bool, count=len(triangles))


def image_face_of_border_triangles(
    points: NDArray[np.floating],
    triangles: NDArray[np.integer],
    is_border: NDArray[np.bool_],
    image_shape: tuple[int, ...],
) -> NDArray[np.int64]:
    """The image face each border triangle lies on, from its vertices' voxel coordinates.

    Args:
        points: mesh points in voxel units of the mask, **before** junction relocation.
        triangles: `(n, 3)` mesh triangles.
        is_border: `(n,)` which triangles are border triangles (`find_border_triangles`).
        image_shape: the mask's shape.

    Returns:
        NDArray[np.int64]: `(n,)` index into `IMAGE_FACES`, `ON_NO_SINGLE_IMAGE_FACE` for a border
        triangle whose three vertices share no image face, `NOT_A_BORDER_TRIANGLE` elsewhere.
    """
    out = np.full(len(triangles), NOT_A_BORDER_TRIANGLE, dtype=np.int64)
    rows = np.flatnonzero(is_border)
    if len(rows) == 0:
        return out
    corners = np.asarray(points, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)[rows]]  # (k, 3, 3)
    out[rows] = ON_NO_SINGLE_IMAGE_FACE
    # Assigned in reverse order so that, for a degenerate triangle on a box edge, the first face
    # in `IMAGE_FACES` order wins; a non-degenerate triangle lies on at most one plane.
    for index in reversed(range(len(IMAGE_FACES))):
        axis, side = divmod(index, 2)
        level = 0.0 if side == 0 else float(image_shape[axis] - 1)
        on_face = np.all(corners[:, :, axis] == level, axis=1)
        out[rows[on_face]] = index
    return out


def border_triangles_the_cell_reaches(
    points: NDArray[np.floating],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
    image_face_of_triangle: NDArray[np.integer],
    cells_on_image_faces: tuple[tuple[int, ...], ...],
    image_shape: tuple[int, ...],
    faces: tuple[int, ...] = tuple(range(len(IMAGE_FACES))),
) -> NDArray[np.bool_]:
    """Border triangles whose cell has a voxel of the mask on the image face the triangle lies on.

    Args:
        points: mesh points in voxel units of the mask, before junction relocation.
        triangles: `(n, 3)` mesh triangles.
        labels: `(n, 2)` the two regions of each triangle; a border triangle is `(0, c)`.
        image_face_of_triangle: `(n,)` from `image_face_of_border_triangles`.
        cells_on_image_faces: from `cells_on_each_image_face`.
        image_shape: the mask's shape.
        faces: the indices in `IMAGE_FACES` that count, for a border triangle on no single image
            face: one of its vertices must lie on one of them, and its cell reach it. Defaults to all.

    Returns:
        NDArray[np.bool_]: `(n,)`, `False` on every triangle that is not a border triangle.
    """
    face_of = np.asarray(image_face_of_triangle, dtype=np.int64)
    cell = np.asarray(labels, dtype=np.int64).max(axis=1)
    out = np.zeros(len(face_of), dtype=bool)
    for index, cells in enumerate(cells_on_image_faces):
        rows = face_of == index
        out[rows] = np.isin(cell[rows], np.asarray(cells, dtype=np.int64))
    nowhere = np.flatnonzero(face_of == ON_NO_SINGLE_IMAGE_FACE)
    if len(nowhere):
        corners = np.asarray(points, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)[nowhere]]
        for index in faces:
            axis, side = divmod(index, 2)
            level = 0.0 if side == 0 else float(image_shape[axis] - 1)
            touches = np.any(corners[:, :, axis] == level, axis=1)
            reaches = np.isin(cell[nowhere], np.asarray(cells_on_image_faces[index], dtype=np.int64))
            out[nowhere[touches & reaches]] = True
    return out


def cells_touching_faces(
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
    faces: NDArray[np.bool_],
) -> tuple[int, ...]:
    """The cells (labels other than 0) carried by a triangle that has a vertex on one of `faces`.

    Args:
        triangles: `(n, 3)` mesh triangles.
        labels: `(n, 2)` the two regions of each triangle.
        faces: `(n,)` bool, the faces in question (the crop faces).

    Returns:
        tuple[int, ...]: sorted labels.
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    faces = np.asarray(faces, dtype=bool)
    if not faces.any():
        return ()
    vertices = np.unique(triangles[faces])
    touching = np.isin(triangles, vertices).any(axis=1)
    cells = np.unique(np.asarray(labels, dtype=np.int64)[touching])
    return tuple(int(c) for c in cells if c != 0)


def classify_crop_faces(
    points: NDArray[np.floating],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
    tesselation_ids: ArrayLike,
    lone_faces: ArrayLike,
    segmented_image: NDArray[np.integer],
    physical_image_faces: Iterable[str] | str | None = (),
) -> CropFaceReport:
    """Mark the crop faces and the cut cells of a freshly reconstructed mesh.

    Args:
        points: mesh points in voxel units of the mask, before junction relocation.
        triangles: `(n, 3)` mesh triangles.
        labels: `(n, 2)` the two regions of each triangle; a border triangle is `(0, c)`.
        tesselation_ids: tesselation index of each mesh point.
        lone_faces: the tesselation's faces with one tetrahedron.
        segmented_image: the mask the mesh was reconstructed from: its shape sets where the image
            faces are, and its voxels on them which cells reach each one.
        physical_image_faces: image faces whose border triangles are contact with the medium,
            not crop faces. Defaults to none.

    Returns:
        CropFaceReport: the crop-face mask, the image face of each border triangle, the cut cells.

    Raises:
        ValueError: if `physical_image_faces` names an unknown face, or if a border triangle does
            not carry the medium label 0 (the extraction gives every one `(0, c)`).
    """
    physical = normalise_physical_image_faces(physical_image_faces)
    is_border = find_border_triangles(triangles, tesselation_ids, lone_faces)
    labels = np.asarray(labels, dtype=np.int64)
    if is_border.any() and not np.all(labels[is_border].min(axis=1) == 0):
        message = "a border triangle does not carry the medium label 0"
        raise ValueError(message)
    image_shape = tuple(int(s) for s in np.shape(segmented_image))
    face_of = image_face_of_border_triangles(points, triangles, is_border, image_shape)
    cells_on_faces = cells_on_each_image_face(segmented_image)
    reaches = border_triangles_the_cell_reaches(points, triangles, labels, face_of, cells_on_faces, image_shape)
    physical_indices = [IMAGE_FACES.index(face) for face in physical]
    crop_indices = tuple(index for index in range(len(IMAGE_FACES)) if index not in physical_indices)
    reaches_a_crop = border_triangles_the_cell_reaches(
        points,
        triangles,
        labels,
        face_of,
        cells_on_faces,
        image_shape,
        faces=crop_indices,
    )
    crop = is_border & reaches_a_crop & ~np.isin(face_of, physical_indices)
    return CropFaceReport(
        crop_faces=crop,
        image_face_of_triangle=face_of,
        cell_reaches_image_face=reaches,
        cut_cells=cells_touching_faces(triangles, labels, crop),
        physical_image_faces=physical,
        image_shape=image_shape,
        cells_on_image_faces=cells_on_faces,
    )
