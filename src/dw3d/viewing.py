"""Optional Polyscope viewer for a reconstructed mesh, for inspection and debugging.

Install it with the `viewing` extra (`pip install "delaunay-watershed-3d[viewing]"`), which brings
Polyscope and matplotlib. Calling the viewer opens a Polyscope window.

Matthieu Perez, 2024.
"""

from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    from dw3d import MeshReconstructionAlgorithm

import numpy as np


def plot_cells_polyscope(
    reconstruct: "MeshReconstructionAlgorithm",
    anisotropy_factor: float = 1.0,
    clean_before: bool = True,
    clean_after: bool = True,
    transparency: bool = False,
    show: bool = True,
    view: str = "Simple",
    scattering_coeff: float = 0.5,
) -> None:
    """Plot separated cells of the mesh in a polyscope viewer.

    Each cell (every label but the medium 0) is shown as its own surface, in a random colour
    (fixed seed). The points are shown in their own frame, in the mask's index order.

    Args:
        reconstruct (MeshReconstructionAlgorithm): the algorithm whose last constructed mesh to show.
        anisotropy_factor (float, optional): Multiply column 0 of the points, the mask's axis 0 (z for a
            `(z, y, x)` stack), by this factor, for display only: a scaled copy is shown, and the reconstruction's
            own mesh is left unchanged. Defaults to 1.0. To build a mesh in physical units from a mask whose voxels
            are longer along z, use `MeshReconstructionAlgorithm.construct_mesh_in_physical_units` instead.
        clean_before (bool, optional): Clean viewer before the function. Defaults to True.
        clean_after (bool, optional): Clean viewer after the function. Defaults to True.
        transparency (bool, optional): Allow transparency in viewer. Defaults to False.
        show (bool, optional): Show viewer. Defaults to True.
        view (str, optional): "Simple" or "Scattered", scatter different cells or not. Defaults to "Simple".
        scattering_coeff (float, optional): If "Scattered" mode, Coefficient of scattering. Defaults to 0.5.

    Raises:
        ImportError: if Polyscope is not installed (the `viewing` extra).
    """
    import polyscope as ps

    points, triangles, labels = reconstruct.last_constructed_mesh
    points = points * np.array([anisotropy_factor, 1.0, 1.0])  # a copy: never scale the stored mesh

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
                points,
                np.array(cluster),
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
                points - (centroid_mesh - centroid_vert) * (scattering_coeff),
                cluster,
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
