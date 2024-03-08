"""Main module with GeometryReconstruction3D, the class allowing the construction of a mesh from a segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from pathlib import Path
from time import time

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ckdtree
from skimage.segmentation import expand_labels

from dw3d.edt import compute_edt_base
from dw3d.iorec import save_rec
from dw3d.mesh_utilities import (
    center_around_origin,
    labeled_mesh_from_labeled_graph,
    set_pixel_size,
    set_points_min_max,
)
from dw3d.segmentation import extract_seed_coords_and_indices
from dw3d.tesselation_graph import TesselationGraph
from dw3d.watershed import seeded_watershed_map

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

        self.points, self.triangles, self.labels = labeled_mesh_from_labeled_graph(
            self.tesselation_graph,
            self.map_label_to_nodes_ids,
        )

    def _watershed_seeded(self, print_info: bool = True) -> None:
        """Perform watershed algorithm to label tetrahedrons of the tesselation networkX graph."""
        t1 = time()
        seeds_nodes = _compute_seeds_idx_from_voxel_coords(
            self.edt_image,
            self.tesselation_graph.compute_nodes_centroids(),
            self.seeds_coords,
        )
        zero_nodes = self.tesselation_graph.compute_zero_nodes(self.segmented_image)

        nx_graph = self.tesselation_graph.to_networkx_graph()
        self.map_label_to_nodes_ids = seeded_watershed_map(nx_graph, seeds_nodes, self.seeds_indices, zero_nodes)

        t2 = time()
        if print_info:
            print("Watershed done in ", np.round(t2 - t1, 3))

    @property
    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Get points, triangles and labels (materials) describing the mesh obtained from segmented image."""
        return self.points, self.triangles, self.labels

    def center_around_zero(self) -> None:
        """Center the mesh around 0."""
        self.points = center_around_origin(self.points)

    def set_pixel_size(self, xy_pixel_size: float, z_pixel_size: float) -> None:
        """Scales the mesh from pixel coordinates to real coordinates, given microscope's xy and z pixel size."""
        self.points = set_pixel_size(self.points, xy_pixel_size, z_pixel_size)

    def set_global_min_max(self, global_min: float, global_max: float) -> None:
        """Scales homogeneously the mesh such that its min & max values are global_min and global_max."""
        self.points = set_points_min_max(self.points, global_min, global_max)

    def export_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the output mesh on disk."""
        save_rec(filename, self.points, self.triangles, self.labels, binary_mode)

    def export_segmentation(self, filename: str | Path) -> None:
        """Export mesh, seeds coordinates and image shape in numpy files."""
        triangles_and_labels = np.hstack((self.triangles, self.labels))
        seeds = self.seeds_coords
        image_shape = np.array(self.segmented_image.shape)
        mesh_dict = {
            "Verts": self.points,
            "Faces": triangles_and_labels,
            "seeds": seeds,
            "image_shape": image_shape,
        }
        np.save(filename, mesh_dict)


def _compute_seeds_idx_from_voxel_coords(
    edt: NDArray,
    centroids: NDArray[np.float64],
    seed_pixel_coords: NDArray[np.uint],
) -> NDArray[np.uint]:
    """Compute the seeds used for watershed."""
    nx, ny, nz = edt.shape
    points = _pixels_coords(nx, ny, nz)
    anchors = seed_pixel_coords[:, 0] * ny * nz + seed_pixel_coords[:, 1] * nz + seed_pixel_coords[:, 2]

    p = points[anchors]

    tree = ckdtree.cKDTree(centroids)
    _, idx_seeds = tree.query(p)
    return idx_seeds  # "seed" nodes ids


def _pixels_coords(nx: int, ny: int, nz: int) -> NDArray[np.int64]:
    """Create all pixels coordinates for an image of size nx*ny*nz."""
    xv = np.linspace(0, nx - 1, nx)
    yv = np.linspace(0, ny - 1, ny)
    zv = np.linspace(0, nz - 1, nz)
    xvv, yvv, zvv = np.meshgrid(xv, yv, zv)
    xvv = np.transpose(xvv, (1, 0, 2)).flatten()
    yvv = np.transpose(yvv, (1, 0, 2)).flatten()
    zvv = zvv.flatten()
    points = np.vstack([xvv, yvv, zvv]).transpose().astype(int)
    return points
