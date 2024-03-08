"""Main module with GeometryReconstruction3D, the class allowing the construction of a mesh from a segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from pathlib import Path
from time import time

import numpy as np
from numpy.typing import NDArray
from skimage.segmentation import expand_labels

from dw3d.edt import compute_edt_base
from dw3d.graph_functions import TesselationGraph
from dw3d.mesh_utilities import (
    compute_seeds_idx_from_voxel_coords,
    labeled_mesh_from_labeled_graph,
    write_mesh_bin,
    write_mesh_text,
)
from dw3d.networkx_functions import seeded_watershed_map
from dw3d.segmentation import extract_seed_coords_and_indices
from dw3d.iorec import save_rec

# def mesh_from_segmentation()


class GeometryReconstruction3D:
    """Build a mesh from a segmented image."""

    def __init__(
        self,
        segmented_image: NDArray[np.uint],
        min_dist: int = 5,
        expansion_labels: int = 0,
        original_image: NDArray[np.uint] | None = None,
        print_info: bool = False,
    ) -> None:
        """Prepare the data.

        Args:
            segmented_image (NDArray[np.uint]): Segmented image.
            min_dist (int, optional): Minimum distance (in pixels) between 2 points in the final mesh. Defaults to 5.
            expansion_labels (int, optional): distance in pixel to grow each non-zero labels
              of the segmented image. Defaults to 0.
            original_image (NDArray[np.uint] | None, optional): Original segmented image, kept for plotting purposes,
                if the labels are changed by the expansion_labels argument. Defaults to None.
            print_info (bool, optional): Verbosity flag. Defaults to False.
        """
        self.original_image = original_image
        if expansion_labels > 0:
            self.segmented_image = expand_labels(segmented_image, expansion_labels)
        else:
            self.segmented_image = segmented_image

        # needed here: segmented image only
        self.edt_image = compute_edt_base(self.segmented_image, print_info=print_info)
        # needed here: edt_image, min_dist

        self.tesselation_graph = TesselationGraph(
            self.edt_image,
            min_distance=min_dist,
            print_info=print_info,
        )

        # needed there : edt_image, tesselation_graph, seeds_coords, segmented_image if zero_nodes, seeds_indices
        self.seeds_coords, self.seeds_indices = extract_seed_coords_and_indices(self.segmented_image, self.edt_image)
        self._watershed_seeded(print_info=print_info)

    def _watershed_seeded(self, print_info: bool = True) -> None:
        """Perform watershed algorithm to label tetrahedrons of the tesselation networkX graph."""
        t1 = time()
        seeds_nodes = compute_seeds_idx_from_voxel_coords(
            self.edt_image,
            self.tesselation_graph.compute_nodes_centroids(),
            self.seeds_coords,
        )
        zero_nodes = self.tesselation_graph.compute_zero_nodes(self.segmented_image)

        nx_graph = self.tesselation_graph.networkx_graph_weights_and_borders()
        self.map_label_to_nodes_ids = seeded_watershed_map(nx_graph, seeds_nodes, self.seeds_indices, zero_nodes)

        t2 = time()
        if print_info:
            print("Watershed done in ", np.round(t2 - t1, 3))

    def return_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Get points, triangles and labels (materials) describing the mesh obtained from segmented image."""
        return labeled_mesh_from_labeled_graph(self.tesselation_graph, self.map_label_to_nodes_ids)

    def export_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the output mesh on disk."""
        points, triangles, labels = self.return_mesh()
        save_rec(filename, points, triangles, labels, binary_mode)

    def export_segmentation(self, filename: str | Path) -> None:
        """Export mesh, seeds coordinates and image shape in numpy files."""
        points, triangles, labels = self.return_mesh()
        triangles_and_labels = np.hstack((triangles, labels))
        seeds = self.seeds_coords
        image_shape = np.array(self.segmented_image.shape)
        mesh_dict = {
            "Verts": points,
            "Faces": triangles_and_labels,
            "seeds": seeds,
            "image_shape": image_shape,
        }
        np.save(filename, mesh_dict)
