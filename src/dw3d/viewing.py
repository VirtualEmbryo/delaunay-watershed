"""This module gathers all optional viewing utilities for debugging.

Install them with the [viewing] option.

Matthieu Perez, 2024.
"""
from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    import napari

    from dw3d import MeshReconstructionAlgorithm

import numpy as np


def plot_in_napari(
    reconstruct: "MeshReconstructionAlgorithm",
    add_mesh: bool = True,
    original_image: NDArray[np.uint] | None = None,
) -> "napari.Viewer":
    """Plot results in Napari.

    Args:
        reconstruct (GeometryReconstruction3D): The reconstruction algorithm object.
        add_mesh (bool, optional): Whether to show the mesh in Napari (experimental). Defaults to True.
        original_image (NDArray[np.uint] | None, optional): Original microscopy image (not its mask). Defaults to None.
    """
    import matplotlib.pyplot as plt
    import napari

    v = napari.view_image(reconstruct._segmented_image, name="Labels")
    v.add_image(reconstruct._edt_image, name="Distance Transform")
    if original_image is not None:
        v.add_image(original_image, name="Original Image")
    if not add_mesh:
        rng = np.random.default_rng()
        v.add_points(
            reconstruct._seeds_coords,
            name="Watershed seeds",
            n_dimensional=True,
            face_color=rng.random((len(reconstruct._seeds_coords), 3)),
            size=10,
        )
    v.add_points(
        reconstruct._tesselation_graph.vertices,
        name="triangulation_points",
        n_dimensional=False,
        face_color="red",
        size=1,
    )

    # colors = ["blue", "yellow", "green"]
    # for i in range(len(self.nodes_centroids)):
    #     print("centroids for label", i)
    #     v.add_points(
    #         self.nodes_centroids[i],
    #         name=f"centroids {i}",
    #         n_dimensional=False,
    #         face_color=colors[i],
    #         size=1,
    #         text=[
    #             str(self.nodes_centroids[i][k])
    #             for k in range(len(self.nodes_centroids[i]))
    #         ],
    #     )

    if add_mesh:
        points, trianges, labels = reconstruct.last_constructed_mesh
        clusters = _separate_faces_dict(trianges, labels)
        maxkey = np.amax(trianges)
        all_verts = []
        all_faces = []
        all_labels = []
        offset = 0

        for key in sorted(clusters.keys()):
            if key == 0:
                continue
            faces = np.array(clusters[key])

            vn, fn = _renormalize_verts(points, faces)
            ln = np.ones(len(vn)) * key / maxkey

            all_verts.append(vn.copy())
            all_faces.append(fn.copy() + offset)
            all_labels.append(ln.copy())
            offset += len(vn)
        all_verts.append(np.array([np.mean(np.vstack(all_verts), axis=0)]))
        all_labels.append(np.array([0]))
        all_verts = np.vstack(all_verts)
        all_faces = np.vstack(all_faces)
        all_labels = np.hstack(all_labels)
        v.add_points(
            reconstruct._seeds_coords,
            name="Watershed seeds",
            n_dimensional=True,
            face_color=np.array(plt.cm.viridis(np.array(sorted(clusters.keys())) / maxkey))[:, :3],
            size=10,
        )

        v.add_surface((all_verts, all_faces, all_labels), colormap="viridis")

    return v


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

    Args:
        reconstruct (GeometryReconstruction3D): geometric reconstruction object containing the mesh to show.
        anisotropy_factor (float, optional): Multiply the x-axis of points by this factor. Defaults to 1.0.
        clean_before (bool, optional): Clean viewer before the function. Defaults to True.
        clean_after (bool, optional): Clean viewer after the function. Defaults to True.
        transparency (bool, optional): Allow transparency in viewer. Defaults to False.
        show (bool, optional): Show viewer. Defaults to True.
        view (str, optional): "Simple" or "Scattered", scatter different cells or not. Defaults to "Simple".
        scattering_coeff (float, optional): If "Scattered" mode, Coefficient of scattering. Defaults to 0.5.
    """
    import polyscope as ps

    points, triangles, labels = reconstruct.last_constructed_mesh
    points[:, 0] *= anisotropy_factor

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
