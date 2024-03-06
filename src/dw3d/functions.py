"""Main module with GeometryReconstruction3D, the class allowing the construction of a mesh from a segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from pathlib import Path
from time import time

import numpy as np
from numpy.typing import NDArray
from skimage.segmentation import expand_labels

from dw3d.dcel import DcelData
from dw3d.geometric_utilities import build_triangulation, interpolate_image
from dw3d.graph_functions import DelaunayGraph
from dw3d.mesh_utilities import (
    clean_mesh_from_seg,
    compute_seeds_idx_from_voxel_coords,
    retrieve_border_tetra_with_index_map,
    write_mesh_bin,
    write_mesh_text,
)
from dw3d.networkx_functions import seeded_watershed_map


class GeometryReconstruction3D:
    """Important class which compute a mesh from a segmented image."""

    def __init__(
        self,
        labels: NDArray[np.uint8],
        min_dist: int = 5,
        expansion_labels: int = 0,
        original_image: NDArray[np.uint8] | None = None,
        print_info: bool = False,
    ) -> None:
        """Prepare the data.

        Args:
            labels (NDArray[np.uint8]): Segmented image.
            min_dist (int, optional): Minimum distance (in pixels) between 2 points in the final mesh. Defaults to 5.
            expansion_labels (int, optional): distance in pixel to grow each non-zero labels
              of the segmented image. Defaults to 0.
            original_image (NDArray[np.uint8] | None, optional): Original segmented image, kept for plotting purposes,
                if the labels are changed by the expansion_labels argument. Defaults to None.
            print_info (bool, optional): Verbosity flag. Defaults to False.
        """
        self.original_image = original_image
        if expansion_labels > 0:
            self.labels = expand_labels(labels, expansion_labels)
        else:
            self.labels = labels

        self.seeds_coords, self.seeds_indices, self.tri, self.EDT = build_triangulation(
            self.labels,
            min_distance=min_dist,
            prints=print_info,
        )

        labels = interpolate_image(self.labels)
        edt = interpolate_image(self.EDT)
        self.delaunay_graph = DelaunayGraph(self.tri, edt, labels, print_info=print_info)
        self._build_graph()

        # try Matthieu Perez: use centroid & labels to create map_label_to_nodes_ids instead of watershed ?
        # self.label_nodes()

        self._watershed_seeded(print_info=print_info)

    # def label_nodes(self):
    #     centroids = self.Delaunay_Graph.compute_nodes_centroids() + 0.5
    #     labels = interpolate_image(self.labels)
    #     self.nodes_centroids = []
    #     pixels_centroids = np.round(centroids).astype(np.int64)
    #     xc = np.minimum(156, pixels_centroids[:, 0])
    #     yc = np.minimum(199, pixels_centroids[:, 1])
    #     zc = np.minimum(156, pixels_centroids[:, 2])
    #     node_to_region = self.labels[xc, yc, zc]
    #     # node_to_region = np.round(labels((centroids))).astype(np.int64)
    #     values = np.unique(node_to_region)
    #     self.map_label_to_nodes_ids = {}
    #     for value in values:
    #         self.map_label_to_nodes_ids[value] = np.argwhere(node_to_region == value).reshape(-1)
    #         self.nodes_centroids.append(centroids[self.map_label_to_nodes_ids[value]])

    def _build_graph(self) -> None:
        """Build the networkx graph for watershed, from a DelaunayGraph."""
        self.nx_graph = self.delaunay_graph.networkx_graph_weights_and_borders()

    def _watershed_seeded(self, print_info: bool = True) -> None:
        """Perform watershed algorithm to label tetrahedrons of the Delaunay networkX graph."""
        t1 = time()
        seeds_nodes = compute_seeds_idx_from_voxel_coords(
            self.EDT,
            self.delaunay_graph.compute_nodes_centroids(),
            self.seeds_coords,
        )
        zero_nodes = self.delaunay_graph.compute_zero_nodes()
        self.map_label_to_nodes_ids = seeded_watershed_map(self.nx_graph, seeds_nodes, self.seeds_indices, zero_nodes)

        t2 = time()
        if print_info:
            print("Watershed done in ", np.round(t2 - t1, 3))

    def retrieve_clusters(self) -> list[list[list[int]]]:
        """Give a list that maps region number to list of triangles."""
        return retrieve_border_tetra_with_index_map(self.delaunay_graph, self.map_label_to_nodes_ids)

    def return_dcel(self) -> DcelData:
        """Get a DcelData mesh from segmented image."""
        points, triangles_and_labels = self.return_mesh()
        return DcelData(points, triangles_and_labels)

    def return_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.uint]]:
        """Get a couple of (points, triangles_and_labels) describing the mesh obtained from segmented image."""
        return clean_mesh_from_seg(self)

    def export_mesh(self, filename: str | Path, mode: str = "bin") -> None:
        """Save the output mesh on disk."""
        points, triangles_and_labels = self.return_mesh()
        if mode == "txt":
            write_mesh_text(filename, points, triangles_and_labels)
        elif mode == "bin":
            write_mesh_bin(filename, points, triangles_and_labels)
        else:
            print("Please choose a valid format")

    def export_segmentation(self, filename: str | Path) -> None:
        """Export mesh, seeds coordinates and image shape in numpy files."""
        points, triangles_and_labels = self.return_mesh()
        seeds = self.seeds_coords
        image_shape = np.array(self.labels.shape)
        mesh_dict = {
            "Verts": points,
            "Faces": triangles_and_labels,
            "seeds": seeds,
            "image_shape": image_shape,
        }
        np.save(filename, mesh_dict)
