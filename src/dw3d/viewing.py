"""Optional Polyscope viewer for a reconstructed mesh, for inspection and debugging.

Install it with the `viewing` extra (`pip install "delaunay-watershed-3d[viewing]"`), which brings
Polyscope and matplotlib. Calling the viewer opens a Polyscope window.

The window draws the tissue, not its mirror image. The mesh's points are in the mask's index order, `(z, y, x)`
for a microscope stack, and Polyscope reads the columns it is handed as its own `(x, y, z)`. The viewer hands it the
columns reversed, `(x, y, z)` = columns `(2, 1, 0)`, the mask's column, row and slice, and each triangle `(a, b, c)`
as `(a, c, b)`, so that it still faces the same side (`_drawn_points`, `_drawn_triangles`). Exchanging two axes is a
reflection, which reverses every triangle's orientation; reversing the triangles undoes that. Up to 0.7.0 the
columns were handed as they are: every view was the mirror image of the tissue. Only what is drawn is reordered;
the stored mesh and the files written (`.rec`, `.vtk`) keep the mask's order.

**A 2-D mesh** (dw2d's `(points (n, 2), segments (m, 2), labels (m, 2))`, or the algorithm after a 2-D mask) is drawn
in a 2-D window by the same function. Each cell's boundary, the 2-D counterpart of the closed surface the 3-D view draws
for it, is one curve network in the cell's colour, the colour the 3-D view gives the same label (the same generator,
seeded as there). The cells are not filled: no polygon triangulation is needed, and none is done. Polyscope's `(x, y)`
is the mesh's `(column, -row)` (`_drawn_planar_points`): a quarter-turn rotation, not a reflection, so the window shows
the image as napari and matplotlib's `imshow` do, row 0 at the top and column 0 at the left. The camera looks straight
down on the plane, with planar navigation (pan and zoom, no rotation), restored to the previous style when the
function returns. An image or a mask given as `image` is drawn under the curves, one texel per pixel, pixel
`(row, column)` centred on the mesh point `(row, column)` (times the pixel size of a mesh built in physical units).

Matthieu Perez, 2024.
"""

from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

if TYPE_CHECKING:
    from dw3d import MeshReconstructionAlgorithm

import numpy as np

#: The 2-D window's image lies this fraction of the tissue's extent below the plane of the curves.
_IMAGE_DEPTH = 1e-3
#: The camera of the 2-D window frames the tissue with this margin around its largest side.
_FRAME_MARGIN = 1.15


def plot_cells_polyscope(
    reconstruct: "MeshReconstructionAlgorithm | tuple[NDArray, NDArray, NDArray]",
    anisotropy_factor: float = 1.0,
    clean_before: bool = True,
    clean_after: bool = True,
    transparency: bool = False,
    show: bool = True,
    view: str = "Simple",
    scattering_coeff: float = 0.5,
    *,
    image: NDArray | None = None,
) -> None:
    """Plot separated cells of the mesh in a polyscope viewer, in 3D or in 2D: the mesh decides.

    Each cell (every label but the medium 0) is shown as its own surface, in a random colour
    (fixed seed). Polyscope's x, y, z are the mask's column, row and slice (the points' columns 2, 1, 0): the
    tissue is drawn as it is, not its mirror image (see the module's docstring).

    A 2-D mesh (points `(n, 2)`, `(row, column)`) opens a 2-D window instead: each cell's boundary as a curve network
    in the colour the 3-D view gives that label, `(x, y) = (column, -row)` (rows downward, as in napari and
    matplotlib), a camera straight down, `image` under the curves when given (the module's docstring).

    Args:
        reconstruct (MeshReconstructionAlgorithm | tuple): the algorithm whose last constructed mesh to show (dw3d's,
            after a 3-D or a 2-D mask, or dw2d's), or a mesh as three arrays: `(points, triangles, labels)` in 3D,
            dw2d's `(points, segments, labels)` in 2D.
        anisotropy_factor (float, optional): Multiply column 0 of the points, the mask's axis 0 (z for a
            `(z, y, x)` stack), by this factor, for display only: a scaled copy is shown, and the reconstruction's
            own mesh is left unchanged. Defaults to 1.0. To build a mesh in physical units from a mask whose voxels
            are longer along z, use `MeshReconstructionAlgorithm.construct_mesh_in_physical_units` instead. 3D only:
            2-D meshes have square pixels, and any other value raises `ValueError` for a 2-D mesh.
        clean_before (bool, optional): Clean viewer before the function. Defaults to True.
        clean_after (bool, optional): Clean viewer after the function. Defaults to True.
        transparency (bool, optional): Allow transparency in viewer. Defaults to False.
        show (bool, optional): Show viewer. Defaults to True.
        view (str, optional): "Simple" or "Scattered", scatter different cells or not. Defaults to "Simple".
        scattering_coeff (float, optional): If "Scattered" mode, Coefficient of scattering. Defaults to 0.5.
        image (NDArray | None, optional): 2D only: an image drawn under the mesh, on the mask's pixel grid: `(rows,
            columns)` grey (scaled between its 0.5 % and 99.5 % quantiles; a label mask is drawn this way too) or
            `(rows, columns, 3 or 4)` RGB(A) (floats in [0, 1] or integers in [0, 255]). Not drawn in the scattered
            view. Defaults to None. Keyword-only.

    Raises:
        ImportError: if Polyscope is not installed (the `viewing` extra).
        ValueError: for an option that has no meaning in the mesh's dimension (`anisotropy_factor` other than 1 with
            a 2-D mesh, `image` with a 3-D mesh), an `image` that is neither grey nor RGB(A), an unknown `view`, or a
            mesh whose points are neither `(n, 3)` nor `(n, 2)`.
    """
    points, triangles, labels = _mesh_of(reconstruct)
    _check_view_and_image(points.shape[1], view, image)
    if points.shape[1] == 2:
        _plot_planar_cells_polyscope(
            points,
            triangles,
            labels,
            anisotropy_factor=anisotropy_factor,
            clean_before=clean_before,
            clean_after=clean_after,
            transparency=transparency,
            show=show,
            view=view,
            scattering_coeff=scattering_coeff,
            image=image,
            pixel_size=_pixel_size_of(reconstruct),
        )
        return

    import polyscope as ps

    points = points * np.array([anisotropy_factor, 1.0, 1.0])  # a copy: never scale the stored mesh
    drawn = _drawn_points(points)

    clusters = _separate_faces_dict(triangles, labels)
    del clusters[0]  # we don't want to see the exterior

    rng = np.random.default_rng(1)
    color_cells = {key: rng.random(3) for key in clusters}
    ps.init()

    if clean_before:
        ps.remove_all_structures()

    if view == "Simple":
        for key in clusters:
            cluster = clusters[key]
            ps.register_surface_mesh(
                "Cell " + str(key),
                drawn,
                _drawn_triangles(cluster),
                color=color_cells[key][:3],
                smooth_shade=False,
            )
    elif view == "Scattered":
        centroid_mesh = np.mean(points[triangles.astype(int)].reshape(-1, 3), axis=0)
        for key in clusters:
            cluster = clusters[key]
            centroid_vert = np.mean(points[cluster].reshape(-1, 3), axis=0)
            _ = ps.register_surface_mesh(
                "Cell " + str(key),
                _drawn_points(points - (centroid_mesh - centroid_vert) * (scattering_coeff)),
                _drawn_triangles(cluster),
                color=color_cells[key][:3],
            )

    ps.set_ground_plane_mode("none")
    if transparency:
        ps.set_transparency_mode("simple")
    else:
        ps.set_transparency_mode("none")

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()


def _mesh_of(reconstruct: Any) -> tuple[NDArray, NDArray, NDArray]:  # noqa: ANN401 - an algorithm or three arrays
    """The `(points, elements, labels)` of an algorithm's last constructed mesh, or of a mesh given as three arrays.

    Raises:
        ValueError: if the points are neither `(n, 3)` nor `(n, 2)`, or the elements and labels do not match them.
    """
    if hasattr(reconstruct, "last_constructed_mesh"):
        points, elements, labels = reconstruct.last_constructed_mesh
    elif isinstance(reconstruct, (tuple, list)) and len(reconstruct) == 3:
        points, elements, labels = reconstruct
    else:
        message = (
            "expected a mesh reconstruction algorithm (dw3d's or dw2d's) or a mesh as three arrays (points, triangles "
            f"or segments, labels), got {type(reconstruct).__name__}"
        )
        raise ValueError(message)
    points, elements, labels = np.asarray(points), np.asarray(elements), np.asarray(labels)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        message = f"the points must be (n, 3) for a 3-D mesh or (n, 2) for a 2-D one, got {points.shape}"
        raise ValueError(message)
    corners = points.shape[1]
    if elements.ndim != 2 or elements.shape[1] != corners or labels.shape != (len(elements), 2):
        what = "triangles (m, 3)" if corners == 3 else "segments (m, 2)"
        message = (
            f"a {corners}-D mesh has {what} and labels (m, 2), got elements {elements.shape} and labels {labels.shape}"
        )
        raise ValueError(message)
    return points, elements, labels


def _check_view_and_image(dimension: int, view: str, image: NDArray | None) -> None:
    """Refuse an unknown `view`, and an `image` under a 3-D mesh."""
    if view not in ("Simple", "Scattered"):
        message = f'view must be "Simple" or "Scattered", got {view!r}'
        raise ValueError(message)
    if dimension == 3 and image is not None:
        message = "image is drawn under a 2-D mesh only: this mesh is 3-D"
        raise ValueError(message)


def _pixel_size_of(reconstruct: Any) -> float:  # noqa: ANN401 - an algorithm or three arrays
    """The pixel size of a 2-D mesh built in physical units by the algorithm (dw3d's or dw2d's), else 1."""
    algorithm = getattr(reconstruct, "two_dimensional_algorithm", None) or reconstruct
    frame = getattr(algorithm, "_physical_frame", None)
    if isinstance(frame, dict) and frame.get("spacing") is not None:
        return float(np.min(frame["spacing"]))
    return 1.0


def _plot_planar_cells_polyscope(
    points: NDArray[np.float64],
    segments: NDArray,
    labels: NDArray,
    *,
    anisotropy_factor: float,
    clean_before: bool,
    clean_after: bool,
    transparency: bool,
    show: bool,
    view: str,
    scattering_coeff: float,
    image: NDArray | None,
    pixel_size: float,
) -> None:
    """The 2-D window of `plot_cells_polyscope` (the module's docstring); the options are that function's."""
    if anisotropy_factor != 1.0:
        message = (
            f"anisotropy_factor={anisotropy_factor!r} has no meaning for a 2-D mesh: 2-D meshes have square pixels "
            "(dw2d refuses anisotropic ones); leave it at 1.0"
        )
        raise ValueError(message)
    rgb = _image_rgb(image)
    import polyscope as ps

    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments)
    clusters = _separate_faces_dict(segments, labels)
    clusters.pop(0, None)  # we don't want to see the exterior

    rng = np.random.default_rng(1)  # the colours of the 3-D view, label by label
    color_cells = {key: rng.random(3) for key in clusters}
    ps.init()
    previous_navigation = _enter_planar_navigation(ps)
    try:
        if clean_before:
            ps.remove_all_structures()

        extent = _planar_extent(points[np.unique(segments)] if len(segments) else points)
        moves = dict.fromkeys(clusters, np.zeros(2))
        if view == "Scattered":
            centroid_mesh = np.mean(points[segments.astype(int)].reshape(-1, 2), axis=0)
            for key, cluster in clusters.items():
                centroid_vert = np.mean(points[cluster.astype(int)].reshape(-1, 2), axis=0)
                moves[key] = -(centroid_mesh - centroid_vert) * scattering_coeff
        if rgb is not None and view == "Simple":
            _register_image(ps, rgb, pixel_size, depth=-_IMAGE_DEPTH * extent)
        for key, cluster in clusters.items():
            nodes, edges = _compact(points, cluster)
            ps.register_curve_network(
                "Cell " + str(key),
                _drawn_planar_points(nodes + moves[key]),
                edges,
                color=color_cells[key][:3],
            )

        if transparency:
            ps.set_transparency_mode("simple")
        else:
            ps.set_transparency_mode("none")
        _look_straight_down(ps, points, segments, rgb, pixel_size)

        if show:
            ps.show()

        if clean_after:
            ps.remove_all_structures()
    finally:
        _leave_planar_navigation(ps, previous_navigation)


def _drawn_planar_points(points: NDArray[np.float64], depth: float = 0.0) -> NDArray[np.float64]:
    """2-D points `(row, column)` as Polyscope draws them: `(x, y, z) = (column, -row, depth)` (a copy).

    `(row, column) -> (column, -row)` is a quarter-turn rotation (determinant +1), not a reflection: the window shows
    the image as napari and matplotlib's `imshow` do, row 0 at the top and column 0 at the left.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([p[:, 1], -p[:, 0], np.full(len(p), float(depth))])


def _compact(points: NDArray[np.float64], segments: NDArray[np.integer]) -> tuple[NDArray[np.float64], NDArray]:
    """The points a set of segments uses, and the segments re-indexed on them (a curve network draws every node)."""
    used, inverse = np.unique(np.asarray(segments, dtype=np.int64).reshape(-1), return_inverse=True)
    return points[used], inverse.reshape(-1, 2)


def _planar_extent(points: NDArray[np.float64]) -> float:
    """The largest side of the points' bounding box (1 when they are a single point or none)."""
    if not len(points):
        return 1.0
    extent = float(np.max(np.ptp(points, axis=0)))
    return extent if np.isfinite(extent) and extent > 0 else 1.0


def _image_rgb(image: NDArray | None) -> NDArray[np.float64] | None:
    """An image to draw under a 2-D mesh, as `(rows, columns, 3)` RGB in [0, 1], or None.

    Raises:
        ValueError: if the array is neither `(rows, columns)` nor `(rows, columns, 3 or 4)`.
    """
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        values = image.astype(np.float64)
        low, high = np.quantile(values, [0.005, 0.995]) if values.size else (0.0, 1.0)
        grey = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
        return np.repeat(grey[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = image[..., :3]
        if np.issubdtype(rgb.dtype, np.integer):
            return rgb.astype(np.float64) / 255.0
        return np.clip(rgb.astype(np.float64), 0.0, 1.0)
    message = (
        f"image must be (rows, columns) grey or (rows, columns, 3 or 4) RGB(A) under a 2-D mesh, not {image.shape}"
    )
    raise ValueError(message)


def _register_image(ps: Any, rgb: NDArray[np.float64], pixel_size: float, depth: float) -> None:  # noqa: ANN401
    """The image as a textured quad, one texel per pixel, pixel `(row, column)` centred on the point `(row, column)`."""
    rows, columns = rgb.shape[:2]
    corners = np.array([[-0.5, -0.5], [-0.5, columns - 0.5], [rows - 0.5, columns - 0.5], [rows - 0.5, -0.5]])
    uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    # The corners run counter-clockwise on screen once drawn: the quad faces the camera.
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    quad = ps.register_surface_mesh("Image", _drawn_planar_points(corners * pixel_size, depth), faces)
    quad.add_parameterization_quantity("pixel coordinates", uv, defined_on="vertices")
    quad.add_color_quantity(
        "pixels",
        rgb,
        defined_on="texture",
        param_name="pixel coordinates",
        image_origin="upper_left",
        enabled=True,
        filter_mode="nearest",
    )
    quad.set_material("flat")
    quad.set_back_face_policy("identical")


def _look_straight_down(
    ps: Any,  # noqa: ANN401 - polyscope
    points: NDArray[np.float64],
    segments: NDArray[np.integer],
    rgb: NDArray[np.float64] | None,
    pixel_size: float,
) -> None:
    """Frame the tissue (and its image) with the camera straight above the plane, rows downward."""
    boxes = [points[np.unique(segments)] if len(segments) else points]
    if rgb is not None:
        rows, columns = rgb.shape[:2]
        boxes.append(np.array([[-0.5, -0.5], [rows - 0.5, columns - 0.5]]) * pixel_size)
    drawn = _drawn_planar_points(np.vstack(boxes))
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    centre = (low + high) / 2
    width, height = ps.get_window_size()
    aspect = width / height if height else 1.0
    half = 0.5 * max(high[1] - low[1], (high[0] - low[0]) / aspect, 1e-9)
    distance = _FRAME_MARGIN * half / np.tan(np.radians(ps.get_vertical_fov_degrees()) / 2)
    ps.look_at((float(centre[0]), float(centre[1]), float(distance)), (float(centre[0]), float(centre[1]), 0.0))


def _enter_planar_navigation(ps: Any) -> str | None:  # noqa: ANN401 - polyscope
    """Planar navigation (pan and zoom, no rotation) and no ground plane; returns the navigation style to restore."""
    try:
        previous = ps.get_navigation_style()
    except Exception:  # an older Polyscope without the getter: restore the default
        previous = None
    ps.set_navigation_style("planar")
    ps.set_ground_plane_mode("none")
    return previous


def _leave_planar_navigation(ps: Any, previous: str | None) -> None:  # noqa: ANN401 - polyscope
    """Restore the navigation style a later 3-D window of the same process expects."""
    ps.set_navigation_style(previous if previous is not None else "turntable")


def _drawn_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Points in the mask's order `(z, y, x)` as Polyscope draws them, `(x, y, z)`: the columns `(2, 1, 0)` (a copy)."""
    return np.asarray(points)[:, [2, 1, 0]]


def _drawn_triangles(triangles: NDArray[np.uint]) -> NDArray[np.uint]:
    """Triangles `(a, b, c)` as handed to Polyscope, `(a, c, b)`, so that each faces the same side once drawn (a copy).

    With the points' columns reversed (`_drawn_points`, a reflection), the normal of `(a, b, c)` computed from the
    drawn points would be the mirror of the mesh's normal turned the other way; reversing the corners restores it.
    """
    return np.asarray(triangles)[:, [0, 2, 1]]


def _separate_faces_dict(triangles: NDArray[np.uint], labels: NDArray[np.uint]) -> dict[int, NDArray[np.uint]]:
    """Construct a dictionnary that maps a region id to the array of triangles forming this region."""
    nb_regions = int(np.amax(labels) + 1)

    occupancy = np.zeros(nb_regions, dtype=np.int64)
    triangles_of_region: dict[int, list[int]] = {}
    for triangle, label in zip(triangles, labels, strict=True):
        region1, region2 = label
        if region1 >= 0:
            if occupancy[region1] == 0:
                triangles_of_region[region1] = [triangle]
                occupancy[region1] += 1
            else:
                triangles_of_region[region1].append(triangle)

        if region2 >= 0:
            if occupancy[region2] == 0:
                triangles_of_region[region2] = [triangle]
                occupancy[region2] += 1
            else:
                triangles_of_region[region2].append(triangle)

    faces_separated: dict[int, NDArray[np.uint]] = {}
    for i in sorted(triangles_of_region.keys()):
        faces_separated[i] = np.array(triangles_of_region[i])

    return faces_separated


def _renormalize_verts(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong]]:
    """Take a mesh made from points and triangles and remove points not indexed in triangles. Re-index triangles.

    Return the filtered points and reindexed triangles.
    """
    used_points_id = np.unique(triangles)
    used_points = np.copy(points[used_points_id])
    idx_mapping = np.arange(len(used_points))
    mapping = dict(zip(used_points_id, idx_mapping, strict=True))

    reindexed_triangles = np.fromiter(
        (mapping[xi] for xi in triangles.reshape(-1)),
        dtype=np.ulonglong,
        count=3 * len(triangles),
    ).reshape((-1, 3))

    return (used_points, reindexed_triangles)
