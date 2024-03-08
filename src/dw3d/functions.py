"""Main module with GeometryReconstruction3D, the class allowing the construction of a mesh from a segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from pathlib import Path
from time import time

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from skimage.segmentation import expand_labels

from dw3d.edt import compute_edt_base
from dw3d.geometric_utilities import build_triangulation
from dw3d.graph_functions import DelaunayGraph
from dw3d.mesh_utilities import (
    clean_mesh_from_seg,
    compute_seeds_idx_from_voxel_coords,
    write_mesh_bin,
    write_mesh_text,
)
from dw3d.networkx_functions import seeded_watershed_map
from dw3d.segmentation import extract_seed_coords_and_indices


class GeometryReconstruction3D:
    """Important class which compute a mesh from a segmented image."""

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

        self.edt_image = compute_edt_base(self.segmented_image, print_info=print_info)
        self.seeds_coords, self.seeds_indices = extract_seed_coords_and_indices(self.segmented_image, self.edt_image)
        self.tri = build_triangulation(
            self.edt_image,
            min_distance=min_dist,
            prints=print_info,
        )

        self.delaunay_graph = DelaunayGraph(
            self.tri,
            _interpolate_image(self.edt_image),
            _interpolate_image(self.segmented_image),
            print_info=print_info,
        )
        # Build the networkx graph for watershed, from a DelaunayGraph.
        self.nx_graph = self.delaunay_graph.networkx_graph_weights_and_borders()

        self._watershed_seeded(print_info=print_info)

    def _watershed_seeded(self, print_info: bool = True) -> None:
        """Perform watershed algorithm to label tetrahedrons of the Delaunay networkX graph."""
        t1 = time()
        seeds_nodes = compute_seeds_idx_from_voxel_coords(
            self.edt_image,
            self.delaunay_graph.compute_nodes_centroids(),
            self.seeds_coords,
        )
        zero_nodes = self.delaunay_graph.compute_zero_nodes()
        self.map_label_to_nodes_ids = seeded_watershed_map(self.nx_graph, seeds_nodes, self.seeds_indices, zero_nodes)

        t2 = time()
        if print_info:
            print("Watershed done in ", np.round(t2 - t1, 3))

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
        image_shape = np.array(self.segmented_image.shape)
        mesh_dict = {
            "Verts": points,
            "Faces": triangles_and_labels,
            "seeds": seeds,
            "image_shape": image_shape,
        }
        np.save(filename, mesh_dict)


def _interpolate_image(image: NDArray[np.uint8]) -> RegularGridInterpolator:
    """Return an interpolated image, a function with values based on pixels."""
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    image_interp = RegularGridInterpolator((x, y, z), image)
    return image_interp
