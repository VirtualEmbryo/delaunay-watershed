"""This module gathers all optional viewing utilities for debugging.

Install them with the [viewing] option.

Matthieu Perez, 2024.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

    from dw3d import GeometryReconstruction3D

import numpy as np

from dw3d.mesh_utilities import renormalize_verts, separate_faces_dict


def plot_in_napari(reconstruct: "GeometryReconstruction3D", add_mesh: bool = True) -> "napari.Viewer":
    """Plot results in Napari."""
    import matplotlib.pyplot as plt
    import napari

    v = napari.view_image(reconstruct.segmented_image, name="Labels")
    v.add_image(reconstruct.edt_image, name="Distance Transform")
    if reconstruct.original_image is not None:
        v.add_image(reconstruct.original_image, name="Original Image")
    if not add_mesh:
        rng = np.random.default_rng()
        v.add_points(
            reconstruct.seeds_coords,
            name="Watershed seeds",
            n_dimensional=True,
            face_color=rng.random((len(reconstruct.seeds_coords), 3)),
            size=10,
        )
    v.add_points(
        reconstruct.tesselation_graph.vertices,
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
        points, trianges, labels = reconstruct.return_mesh()
        clusters = separate_faces_dict(trianges, labels)
        maxkey = np.amax(trianges)
        all_verts = []
        all_faces = []
        all_labels = []
        offset = 0

        for key in sorted(clusters.keys()):
            if key == 0:
                continue
            faces = np.array(clusters[key])

            vn, fn = renormalize_verts(points, faces)
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
            reconstruct.seeds_coords,
            name="Watershed seeds",
            n_dimensional=True,
            face_color=np.array(plt.cm.viridis(np.array(sorted(clusters.keys())) / maxkey))[:, :3],
            size=10,
        )

        v.add_surface((all_verts, all_faces, all_labels), colormap="viridis")

    return v


def plot_cells_polyscope(
    reconstruct: "GeometryReconstruction3D",
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

    points, triangles, labels = reconstruct.return_mesh()
    points[:, 0] *= anisotropy_factor

    clusters = separate_faces_dict(triangles, labels)
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
