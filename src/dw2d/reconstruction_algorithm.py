"""dw2d's mesh reconstruction algorithm, with the names, arguments and return layout of dw3d's.

Configure, then run: `dw2d.get_default_mesh_reconstruction_algorithm` or `dw2d.MeshReconstructionAlgorithmFactory`
build a `MeshReconstructionAlgorithm`; `construct_mesh_from_segmentation_mask(mask)` runs it and returns
`(points, segments, labels)`; one object can mesh several masks, each reconstruction replacing the last mesh and every
report about it. The computation is dw2d's (the engine behind the deprecated `geometry_reconstruction_2d`), unchanged:
the same mask and options give the same arrays, bit for bit.

Conventions, stated once here and in dw3d's package docstring:

* The mask is a 2-D integer array; label `0` is the medium, every other label a cell. `points` are in the mask's index
  order, `(row, column)`, in pixels: the centre of pixel `(i, j)` is the point `(i, j)`.
* `labels[n] = (a, b)` with `a < b` are the two materials edge `n` separates. Orientation, dw3d's rule in index order
  (since 0.8.0): each edge is the positively oriented boundary of `a`. Material `a` lies on the left of
  `points[segments[n, 0]] -> points[segments[n, 1]]` in the `(row, column)` frame, so the edges of each material, taken
  as stored where it is `a` and reversed where it is `b`, run counter-clockwise around it in `(row, column)`, and the
  edge's normal `(dc, -dr)` points out of `a` into `b`, as a dw3d triangle's right-hand normal does. See the user
  guide's page on orientation, which states the rule and the test that decides it. Until 0.8.0 each edge was stored
  the other way round; only the order of the two vertices within each edge changed.
* `min_distance` counts pixels; its default is 4 (dw3d's is 3 voxels).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from dw2d.reports import (
    EnclosedMediumPocket,
    EnclosedMediumWarning,
    InterfaceProjectionReport,
    JunctionRelocationReport,
    ThinMediumMarginWarning,
    UnresolvedCellWarning,
)
from dw2d.thin_medium_margins import (
    IMAGE_SIDES,
    cells_on_each_image_side,
    padding_of_thin_margins,
    side_margins,
    thin_margin_bound,
    unsupported_border_edges,
)

#: dw2d's default `min_distance`, in pixels (dw3d's is 3 voxels).
DEFAULT_MIN_DISTANCE = 4


def checked_two_dimensional_mask(segmented_image: object) -> NDArray:
    """The mask as an array, after the input checks every dw2d entry point runs.

    Raises:
        ValueError: if the mask is not 2-D. A 3-D mask with an axis of length 1 is refused with a message suggesting
            `mask[0]` or `numpy.squeeze` (never squeezed silently); any other 3-D mask is sent to dw3d.

    Warns:
        UserWarning: when the mask's dtype is not an integer (or boolean) type. The mask is meshed anyway, as before:
            refusing it would break callers; its labels are read as the numbers they hold.
    """
    mask = np.asarray(segmented_image)
    if mask.ndim == 3 and 1 in mask.shape:
        error = (
            f"the mask has shape {mask.shape}, a 3-D array with an axis of length 1; dw2d meshes 2-D masks. "
            "Pass the plane itself, for example mask[0] or numpy.squeeze(mask)."
        )
        raise ValueError(error)
    if mask.ndim != 2:
        error = f"dw2d meshes 2-D masks, got an array of shape {mask.shape}" + (
            " (a 3-D mask is meshed by dw3d, or by dw3d's algorithm, which hands 2-D masks to dw2d)" if mask.ndim == 3 else ""
        )
        raise ValueError(error)
    if not (np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.bool_)):
        warnings.warn(
            f"the mask's dtype is {mask.dtype}, not an integer type: its values are read as labels as they are. "
            "Convert it (mask.astype(int)) if its values are whole numbers.",
            UserWarning,
            stacklevel=3,
        )
    return mask


def _enclosed_pockets(mask: NDArray) -> tuple[EnclosedMediumPocket, ...]:
    """The 4-connected pieces of label 0 that touch no side of the image, in raster order of their first pixel."""
    medium = mask == 0
    pieces, n = ndimage.label(medium)  # 4-connectivity, the 2-D analogue of dw3d's 6-connectivity
    if n == 0:
        return ()
    border = np.unique(np.concatenate([pieces[0], pieces[-1], pieces[:, 0], pieces[:, -1]]))
    enclosed = [k for k in range(1, n + 1) if k not in set(border.tolist())]
    out = []
    for k in enclosed:
        where = pieces == k
        first = np.argwhere(where)[0]
        ring = ndimage.binary_dilation(where) & ~where
        out.append(
            EnclosedMediumPocket(
                pixels=int(where.sum()),
                first_pixel=(int(first[0]), int(first[1])),
                lining_labels=tuple(int(v) for v in np.unique(mask[ring])),
            ),
        )
    return tuple(sorted(out, key=lambda p: p.first_pixel))



def _cell_areas(points: NDArray, segments: NDArray, labels: NDArray) -> dict[int, float]:
    """The area of each cell's mesh, in square pixels, by the shoelace sum over its edges.

    Under the orientation rule, `labels[k, 0]` lies on the left of the edge `segments[k, 0] -> segments[k, 1]` in
    `(row, column)`: each edge counts for its first label as stored and for its second reversed. A cell whose edges
    do not close (it should not happen) gets the sum of what it has.
    """
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(segments):
        return {}
    start, end = points[segments[:, 0]], points[segments[:, 1]]
    cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
    areas: dict[int, float] = {}
    for column, sign in ((0, 0.5), (1, -0.5)):
        for label, value in zip(labels[:, column], cross, strict=True):
            areas[int(label)] = areas.get(int(label), 0.0) + sign * float(value)
    return areas

class MeshReconstructionAlgorithm:
    """Algorithm that builds a 2-D multimaterial polyline mesh from a 2-D segmentation mask.

    Build one with `dw2d.get_default_mesh_reconstruction_algorithm` or `dw2d.MeshReconstructionAlgorithmFactory`. The
    mesh is `(points, segments, labels)`: `points` `(n, 2)` float64 in the mask's `(row, column)` order, in pixels unless
    built by `construct_mesh_in_physical_units`; `segments` `(m, 2)` int64 vertex indices; `labels` `(m, 2)` int64, the
    two materials each edge separates, smaller first (orientation: the module docstring).

    Attributes:
        print_info (bool): whether to print the size of the triangulation's dual graph.
        min_distance (int): the point placement's minimum distance between points, in pixels.
        relocate_junctions (bool): move junction vertices onto the mask's own junctions after meshing (default True).
        canonical_triangle_order (bool): a machine-independent triangulation and tie order (default True; dw3d's
            analogue is `canonical_tetrahedron_order`).
        interface_projection (bool): project interface vertices onto the mask's interface locus (default False;
            requires `relocate_junctions`).
        pinch_vertex_repair (bool): absorb label fragments and repair pinch vertices after the watershed (default True).
        near_tie_sliver_relabelling (bool | None): relabel near-tie slivers (default None: on without relocation, off
            with it).
        label_expansion (int): grow the labels by this many pixels before meshing (`skimage.segmentation.expand_labels`;
            default 0), for masks whose cells do not touch.
        score_samples (int): points sampled along each triangle edge to score it (default 10).
        pad_thin_medium_margins (bool): mesh the mask padded with label 0 on every side of the image that no cell
            reaches and whose margin is below `2 * min_distance + 2` pixels, up to that margin, and return the mesh in
            the mask's frame (default True since 0.8.0; `False` gives the meshes of earlier versions). A mask whose
            cells reach every side or stay that far from it is meshed as before, bit for bit. See
            `dw2d.thin_medium_margins`; `medium_margin_padding` says which sides were padded.
    """

    def __init__(
        self,
        print_info: bool = False,
        min_distance: int = DEFAULT_MIN_DISTANCE,
        *,
        relocate_junctions: bool = True,
        canonical_triangle_order: bool = True,
        interface_projection: bool = False,
        pinch_vertex_repair: bool = True,
        near_tie_sliver_relabelling: bool | None = None,
        label_expansion: int = 0,
        score_samples: int = 10,
        pad_thin_medium_margins: bool = True,
    ) -> None:
        """Set the options; see the class docstring. Every option is also an attribute that can be set before a run."""
        self.print_info = bool(print_info)
        self.min_distance = int(min_distance)
        self.relocate_junctions = bool(relocate_junctions)
        self.canonical_triangle_order = bool(canonical_triangle_order)
        self.interface_projection = bool(interface_projection)
        self.pinch_vertex_repair = bool(pinch_vertex_repair)
        self.near_tie_sliver_relabelling = near_tie_sliver_relabelling
        self.label_expansion = int(label_expansion)
        self.score_samples = int(score_samples)
        self.pad_thin_medium_margins = bool(pad_thin_medium_margins)
        self._engine = None
        self._mask_shape: tuple[int, ...] = ()
        self._padding_low = np.zeros(2, dtype=np.int64)
        self._medium_margin_padding: dict[str, int] | None = None
        self._physical_frame: dict | None = None
        self._unresolved_cells: tuple[int, ...] = ()
        self._enclosed_pockets: tuple[EnclosedMediumPocket, ...] = ()

    # ------------------------------------------------------------------------------------------- reconstruction
    def construct_mesh_from_segmentation_mask(
        self,
        segmented_image: NDArray[np.integer],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """Build a 2-D mesh from a segmentation mask.

        After the call, `last_constructed_mesh`, `image_border_vertices`, `cut_cells`, `relocation_report`,
        `interface_projection_report`, `unresolved_cells` and `enclosed_medium_pockets` describe this mesh, and
        `save_to_rec_mesh` writes no physical frame.

        Args:
            segmented_image: the 2-D integer label mask; label 0 is the medium, every other label a cell.

        Returns:
            tuple: `(points, segments, labels)`: `(n, 2)` float64 in pixels, `(row, column)`; `(m, 2)` int64;
                `(m, 2)` int64 (smaller label first).

        Raises:
            ValueError: for a mask that is not 2-D (see `checked_two_dimensional_mask`), or for
                `interface_projection` without `relocate_junctions`.

        Warns:
            UserWarning: for a mask whose dtype is not an integer type.
            EnclosedMediumWarning: once, when a piece of label 0 touches no side of the image (it is meshed as the
                outer medium), naming each such pocket.
            UnresolvedCellWarning: once, when a label of the mask is absent from the mesh, naming each. Since 0.8.0
                this includes a mask whose mesh has no edge at all (returned empty, `(0, 2)` segments); before, it
                raised IndexError.
            ThinMediumMarginWarning: once, when a cell's edges lie on a side of the image its pixels do not reach and
                that could not be padded (see `pad_thin_medium_margins`), naming each such cell, the side and its
                margin.

        A mask with a side that no cell reaches and whose margin is below `2 * min_distance + 2` pixels is meshed padded
        with label 0 on such sides (`pad_thin_medium_margins`, `medium_margin_padding`); the mesh and the compressed
        segmentation are in the mask's frame either way.
        """
        mask = checked_two_dimensional_mask(segmented_image)
        self._physical_frame = None
        self._mask_shape = tuple(int(n) for n in mask.shape)
        self._padding_low = np.zeros(2, dtype=np.int64)
        self._medium_margin_padding = {}
        widths = self._padding_of_thin_medium_margins(mask)
        if widths is None:
            self._run_the_engine(mask)
        else:
            self._run_the_engine(np.pad(mask, widths, constant_values=0))
            self._padding_low = np.array([widths[0][0], widths[1][0]], dtype=np.int64)
            self._medium_margin_padding = {
                IMAGE_SIDES[2 * axis + side]: int(width)
                for axis, pair in enumerate(widths)
                for side, width in enumerate(pair)
                if width
            }
            if self.print_info:
                print(f"Thin medium margins: the mask was padded with medium on {self._medium_margin_padding} (pixels)")
        meshed = self._the_engine().labels
        left_on_the_border = unsupported_border_edges(self._points, self._segments, self._labels, meshed)
        if widths is not None:
            self._points = self._points - self._padding_low.astype(self._points.dtype)
        self._report_absent_cells_and_enclosed_media(self._meshed_mask())
        self._name_the_cells_left_on_the_image_border(left_on_the_border)
        return self.last_constructed_mesh

    def _run_the_engine(self, mask: NDArray) -> None:
        """Mesh `mask` with dw2d's engine (`_Reconstruction2D`) and keep its mesh."""
        from dw2d.functions import _Reconstruction2D

        self._engine = _Reconstruction2D(
            mask,
            min_dist=self.min_distance,
            interpolation_points=self.score_samples,
            expansion_labels=self.label_expansion,
            deterministic=self.canonical_triangle_order,
            relocate_junctions=self.relocate_junctions,
            project_interface_vertices=self.interface_projection,
            repair_pinch_vertices=self.pinch_vertex_repair,
            relabel_near_tie_slivers=self.near_tie_sliver_relabelling,
            print_info=self.print_info,
        )
        verts, edges = self._engine.return_mesh()
        self._points = np.asarray(verts)
        self._segments = np.ascontiguousarray(edges[:, :2])
        self._labels = np.ascontiguousarray(edges[:, 2:])

    def _the_engine(self):  # noqa: ANN202 -- dw2d.functions._Reconstruction2D, imported lazily
        """The engine of the last reconstruction (raises before any)."""
        if self._engine is None:
            error = "no mesh yet: call construct_mesh_from_segmentation_mask first"
            raise RuntimeError(error)
        return self._engine

    def _meshed_mask(self) -> NDArray:
        """The mask the engine meshed (after any label expansion), in the frame of the mask given (unpadded)."""
        labels = np.asarray(self._the_engine().labels)
        low = self._padding_low
        return labels[low[0] : low[0] + self._mask_shape[0], low[1] : low[1] + self._mask_shape[1]]

    def _padding_of_thin_medium_margins(self, mask: NDArray) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """The label-0 padding that keeps the cells off image sides they do not reach, or `None` (module docstring).

        Read on the mask the engine meshes, after its label expansion (`label_expansion`).
        """
        if not self.pad_thin_medium_margins:
            return None
        meshed = mask
        if self.label_expansion:
            from skimage.segmentation import expand_labels

            meshed = expand_labels(mask, self.label_expansion)
        return padding_of_thin_margins(
            side_margins(meshed), cells_on_each_image_side(meshed), thin_margin_bound(self.min_distance),
        )

    def _name_the_cells_left_on_the_image_border(self, left: list[tuple[int, str]]) -> None:
        """Warn once, naming each cell with edges on an image side its pixels do not reach (after any padding)."""
        if not left:
            return
        mask = self._meshed_mask()
        reached = cells_on_each_image_side(mask)
        bound = thin_margin_bound(self.min_distance)
        padding = self._medium_margin_padding or {}
        areas = _cell_areas(self._points, self._segments, self._labels)
        clauses = []
        for cell, side in left:
            index = IMAGE_SIDES.index(side)
            axis, high = divmod(index, 2)
            where = np.flatnonzero((mask == cell).any(axis=1 - axis))
            margin = int(where[0]) if not high else int(mask.shape[axis] - 1 - where[-1])
            if reached[index]:
                why = "not padded: cells reach that side, which is a crop"
            elif not self.pad_thin_medium_margins:
                why = "not padded: pad_thin_medium_margins is off"
            elif side in padding:
                why = f"after padding that side by {padding[side]} pixels"
            else:
                why = f"not padded: the side's margin is already {bound} pixels or more"
            pixels = int(np.count_nonzero(mask == cell))
            ratio = areas.get(cell, 0.0) / pixels if pixels else float("nan")
            clauses.append(
                f"cell {cell} lies on the image side {side}, {margin} pixel{'s' if margin != 1 else ''} of medium away "
                f"from its pixels ({why}), mesh area {ratio:.2f} times its pixel count",
            )
        warnings.warn(
            f"{'; '.join(clauses)}. The mesh of such a cell may include medium between it and the side, which makes "
            "its area larger than its pixel count. See dw2d.thin_medium_margins.",
            ThinMediumMarginWarning,
            stacklevel=3,
        )

    @property
    def medium_margin_padding(self) -> dict[str, int] | None:
        """The label-0 padding the last reconstruction added to the mask's sides, in pixels, by side name.

        Empty when the mask was meshed as given (the common case); `None` before any reconstruction. See
        `pad_thin_medium_margins` and `dw2d.thin_medium_margins`.
        """
        return self._medium_margin_padding

    def construct_mesh_in_physical_units(
        self,
        segmented_image: NDArray[np.integer],
        spacing: tuple[float, float],
        *,
        spacing_unit: str | None = None,
        axis_order: str = "yx",
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """Build a mesh in physical units from a mask of square pixels.

        The mask is meshed on its own pixel grid (`construct_mesh_from_segmentation_mask`, `min_distance` in pixels),
        and the points are multiplied by the pixel size: a point at pixel index `i` is at `i * h`, so the centre of the
        first pixel is the origin. Only square pixels are accepted in 0.8.0 (dw3d resamples anisotropic voxels to cubic
        ones; the accuracy of the same route in 2D has not been measured).

        After this call `save_to_rec_mesh` records `coordinate_frame = physical`, `axis_order`, `spacing`,
        `spacing_unit` (when given) and `source_shape` in the file's appendix. Any later reconstruction clears them.

        Args:
            segmented_image: the 2-D integer label mask.
            spacing: the pixel size along the mask's axes 0 and 1, `(dy, dx)`; the two must be equal (relative
                tolerance 1e-6).
            spacing_unit: the unit of `spacing` (e.g. `"um"`), recorded, not used.
            axis_order: what the mask's axes are, recorded, not used. Defaults to `"yx"`.

        Returns:
            tuple: `(points, segments, labels)` as `construct_mesh_from_segmentation_mask` returns them, the points in
                the unit of `spacing`. For a spacing of `(1, 1)` they are bitwise the pixel ones.

        Raises:
            ValueError: for a spacing that is not two positive finite numbers, or whose two values differ.
        """
        values = np.asarray(spacing, dtype=np.float64).ravel()
        if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            error = f"spacing must be two positive finite numbers (dy, dx), got {spacing!r}"
            raise ValueError(error)
        if abs(values[0] - values[1]) > 1e-6 * max(values):
            error = (
                f"spacing {tuple(values.tolist())}: only square pixels are supported in 2D (dy == dx); anisotropic "
                "pixels are not yet measured"
            )
            raise ValueError(error)
        h = float(values.min())
        self.construct_mesh_from_segmentation_mask(segmented_image)
        self._points = self._points * h
        frame = {
            "coordinate_frame": "physical",
            "axis_order": axis_order,
            "spacing": (float(values[0]), float(values[1])),
            "source_shape": tuple(int(n) for n in np.shape(segmented_image)),
        }
        if spacing_unit is not None:
            frame["spacing_unit"] = spacing_unit
        self._physical_frame = frame
        return self.last_constructed_mesh

    def _report_absent_cells_and_enclosed_media(self, mask: NDArray) -> None:
        present = set(np.unique(self._labels).tolist())
        cells = [int(v) for v in np.unique(mask) if v != 0]
        self._unresolved_cells = tuple(v for v in cells if v not in present)
        if self._unresolved_cells:
            warnings.warn(
                f"{len(self._unresolved_cells)} label(s) of the mask are absent from the mesh: "
                f"{', '.join(map(str, self._unresolved_cells))}. A cell narrower than min_distance "
                f"({self.min_distance} px) may be reached by no triangle; try a smaller min_distance.",
                UnresolvedCellWarning,
                stacklevel=3,
            )
        self._enclosed_pockets = _enclosed_pockets(mask)
        if self._enclosed_pockets:
            described = "; ".join(
                f"{p.pixels} px at (row, column) {p.first_pixel}, lined by label(s) {', '.join(map(str, p.lining_labels))}"
                for p in self._enclosed_pockets
            )
            warnings.warn(
                f"{len(self._enclosed_pockets)} piece(s) of label 0 touch no side of the image ({described}). dw2d has "
                "no enclosed-medium region: such a pocket is meshed as label 0, i.e. as contact with the outer medium.",
                EnclosedMediumWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------------------------------------ results
    def _require_mesh(self) -> None:
        if self._engine is None:
            error = "no mesh yet: call construct_mesh_from_segmentation_mask first"
            raise RuntimeError(error)

    @property
    def last_constructed_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """The last mesh built, `(points, segments, labels)`. The algorithm's own arrays, not copies."""
        self._require_mesh()
        return self._points, self._segments, self._labels

    @property
    def image_border_vertices(self) -> NDArray[np.int64]:
        """The indices of the mesh vertices on the image border, sorted.

        Taken from the triangulation's structure (the endpoints of its hull edges) before any vertex moves; the
        junction relocation and the projection never move them. foambryo's 2D inference excludes the junctions there,
        where a cell-cell interface meets the image border at a straight 180-degree wedge that is not a contact angle.
        """
        self._require_mesh()
        return np.unique(np.asarray(self._engine.image_border_vertices, dtype=np.int64))

    @property
    def cut_cells(self) -> tuple[int, ...]:
        """The cells (non-zero labels) with a vertex on the image border: cells the image cuts, ascending.

        dw3d's `cut_cells` names the cells with a crop face; in 2D a cell is cut where its boundary reaches the image
        border. Their pressures are not withheld by foambryo in 2D: Laplace's law reads them from the interfaces
        inside the image.
        """
        self._require_mesh()
        border = np.zeros(len(self._points), dtype=bool)
        border[self.image_border_vertices] = True
        touching = border[self._segments].any(axis=1)
        return tuple(int(v) for v in np.unique(self._labels[touching]) if v != 0)

    @property
    def medium_labels(self) -> tuple[int, ...]:
        """The labels that are a medium: `(0,)` (dw2d has no enclosed-medium region)."""
        return (0,)

    @property
    def unresolved_cells(self) -> tuple[int, ...]:
        """The labels of the last mask that are absent from its mesh, ascending (see `UnresolvedCellWarning`)."""
        self._require_mesh()
        return self._unresolved_cells

    @property
    def enclosed_medium_pockets(self) -> tuple[EnclosedMediumPocket, ...]:
        """The pieces of label 0 of the last mask that touch no side of the image (see `EnclosedMediumWarning`)."""
        self._require_mesh()
        return self._enclosed_pockets

    @property
    def relocation_report(self) -> JunctionRelocationReport | None:
        """What the junction relocation did on the last mesh, or `None` when it was off."""
        self._require_mesh()
        report = self._engine.junction_relocation_report
        return None if report is None else JunctionRelocationReport(**report)

    @property
    def interface_projection_report(self) -> InterfaceProjectionReport | None:
        """What the interface projection did on the last mesh, or `None` when it was off."""
        self._require_mesh()
        report = self._engine.locus_projection_report
        return None if report is None else InterfaceProjectionReport(**report)

    @property
    def last_constructed_dcel(self):  # noqa: ANN201 -- dw2d.DCEL_Data
        """The last mesh as dw2d's half-edge structure, `dw2d.DCEL_Data`, which measures angles, lengths, areas and
        curvatures; it carries `image_border_vertices`. Built from the points as they are now (physical if scaled).
        """
        from dw2d.Dcel import DCEL_Data

        self._require_mesh()
        if not len(self._segments):
            error = "the mesh has no edge (see the UnresolvedCellWarning of the reconstruction): no half-edge structure"
            raise ValueError(error)
        edges = np.hstack((self._segments, self._labels))
        return DCEL_Data(
            self._points.copy(),
            edges,
            allow_several_loops=self._engine.surgery_report is not None,
            image_border_vertices=self._engine.image_border_vertices,
        )

    # ------------------------------------------------------------------------------- compressed segmentation
    def compress_segmentation_mask(self, segmented_image: NDArray[np.integer]) -> dict[str, Any]:
        """Mesh a mask and return its compressed segmentation (see `last_compressed_segmentation`)."""
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return self.last_compressed_segmentation

    @property
    def last_compressed_segmentation(self) -> dict[str, Any]:
        """The last mesh, the watershed seeds and the mask's shape, as a dictionary.

        Keys: `points`, `segments`, `labels`, `seeds` (one pixel index per label of the mask, in ascending label
        order), `image_shape` and `image_border_vertices`. `dw2d.reconstruct_mask_from_dict` turns it back into an
        approximate mask; it expects the points in pixels, so read this before any scaling (after
        `construct_mesh_in_physical_units`, divide the points by the pixel size first).
        """
        self._require_mesh()
        return {
            "points": self._points,
            "segments": self._segments,
            "labels": self._labels,
            "seeds": self._seeds_in_the_mask_frame(),
            "image_shape": np.array(self._mask_shape),
            "image_border_vertices": self.image_border_vertices,
        }

    def _seeds_in_the_mask_frame(self) -> NDArray:
        """The watershed seeds in the mask's frame.

        Moved back by any padding; a seed of label 0 in the padding goes to the nearest pixel of the mask, which is
        label 0 (a padded side is one no cell reaches).
        """
        engine = self._the_engine()
        seeds = np.asarray(engine.seeds_coords)
        if not self._padding_low.any() and self._mask_shape == tuple(np.shape(engine.labels)):
            return seeds
        moved = seeds.astype(np.int64) - self._padding_low
        moved = np.clip(moved, 0, np.array(self._mask_shape, dtype=np.int64) - 1)
        return moved.astype(seeds.dtype)

    def both_construct_mesh_and_compressed_segmentation(
        self,
        segmented_image: NDArray[np.integer],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
        """Mesh a mask; return `(points, segments, labels, compressed_segmentation)`."""
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return *self.last_constructed_mesh, self.last_compressed_segmentation

    # ----------------------------------------------------------------------------------------------- saving
    def rec_metadata(self) -> dict[str, Any]:
        """The appendix fields `save_to_rec_mesh` writes: `dimension = 2`, `axis_order`, `orientation` (the rule the
        edges follow, `dw2d.io.ORIENTATION_RECORDED`), `source_shape` (the label image's shape), the image border
        vertices, the unresolved cells (when any) and, after `construct_mesh_in_physical_units`, the physical frame.
        """
        from dw2d.io import ORIENTATION_RECORDED

        self._require_mesh()
        metadata: dict[str, Any] = {
            "dimension": 2,
            "axis_order": "yx",
            "orientation": ORIENTATION_RECORDED,
            "source_shape": tuple(self._mask_shape),
        }
        metadata.update(self._physical_frame or {})
        metadata["image_border_vertices"] = tuple(int(v) for v in self.image_border_vertices)
        if self._unresolved_cells:
            metadata["unresolved_regions"] = self._unresolved_cells
        return metadata

    def save_to_rec_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the last mesh in the 2-D `.rec` format, with its current points and a metadata appendix.

        The appendix (version 1, as dw3d writes it) records `dimension = 2`, so that every reader can tell the file
        from a 3-D one, the orientation rule of the edges (`orientation`), and the other fields of `rec_metadata`. A
        reader that ignores the appendix reads the mesh unchanged.

        Args:
            filename: where to write the file (`.rec`).
            binary_mode: write the binary variant of the format. Defaults to False (text), as dw3d.
        """
        from dw2d.io import save_rec

        save_rec(filename, self._points, self._segments, self._labels, binary_mode, metadata=self.rec_metadata())

    # ----------------------------------------------------------------------------------------------- plots
    def simple_plot(self) -> None:
        """Draw the labels, the polylines over the distance map, and the labelled triangles (matplotlib)."""
        self._require_mesh()
        self._engine.simple_plot()

    def plot_polylines_and_seeds(self) -> None:
        """Draw the polylines and the watershed seeds (matplotlib)."""
        self._require_mesh()
        self._engine.plot_polylines_and_seeds()


def save_mesh_to_rec_mesh(
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
    filename: str | Path,
    binary_mode: bool = False,
) -> None:
    """Save a 2-D mesh in the `.rec` format, with an appendix recording `dimension = 2` (as dw3d's function of this name).

    Args:
        points: `(n, 2)` points, written as given.
        segments: `(m, 2)` vertex indices.
        labels: `(m, 2)` the two materials each edge separates.
        filename: where to write the file.
        binary_mode: write the binary variant. Defaults to False.
    """
    from dw2d.io import save_rec

    save_rec(filename, points, segments, labels, binary_mode, metadata={"dimension": 2, "axis_order": "yx"})


def save_compressed_segmentation(filename: str | Path, compressed_segmentation: dict[str, Any]) -> None:
    """Save a compressed segmentation with `numpy.save` (a pickled dictionary; numpy adds `.npy`). Read it back with
    `dw2d.reconstruct_mask_from_saved_file_dict`.
    """
    np.save(filename, compressed_segmentation)

