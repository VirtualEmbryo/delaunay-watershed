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

    v = napari.view_image(reconstruct.labels, name="Labels")
    v.add_image(reconstruct.EDT, name="Distance Transform")
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
        reconstruct.tri.points,
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
        points, trianges_and_labels = reconstruct.return_mesh()
        clusters = separate_faces_dict(trianges_and_labels)
        maxkey = np.amax(trianges_and_labels[:, 3:])
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
