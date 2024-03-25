"""Module defining MeshReconstructionAlgorithm, the class allowing the construction of a mesh from a segmented image.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ckdtree

from dw3d.io import save_rec, save_vtk
from dw3d.mesh_utilities import (
    center_around_origin,
    filter_unused_points,
    labeled_mesh_from_labeled_graph,
    set_pixel_size,
    set_points_min_max,
)
from dw3d.segmentation import extract_seed_coords_and_indices
from dw3d.tesselation_graph import TesselationGraph
from dw3d.watershed import seeded_watershed_map

# segmentation mask -> EDT image
EdtCreationFunction = Callable[[NDArray[np.uint]], NDArray[np.float64]]
# EDT image -> array of pixel coordinates
PointPlacingFunction = Callable[[NDArray[np.float64]], NDArray[np.uint]]
# array of pixel coordinates -> tesselation: array of points coordinates + array of tetrahedrons as points indices
TesselationCreationFunction = Callable[[NDArray[np.uint]], tuple[NDArray[np.float64], NDArray[np.int64]]]
# EDT image, array of points + triangle faces (from tesselation) -> Array of scores
ScoreComputationFunction = Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]], NDArray[np.float64]]
# Watershed : one version only
# Mesh recreation : one version only
# Mesh surgery : one bool to perform surgery ?


class MeshReconstructionAlgorithm:
    """Algoithm to build a mesh from a segmented image."""

    def __init__(
        self,
        print_info: bool,
        edt_creation_function: EdtCreationFunction,
        point_placing_function: PointPlacingFunction,
        tesselation_creation_function: TesselationCreationFunction,
        score_computation_function: ScoreComputationFunction,
        perform_mesh_postprocess_surgery: bool,
    ) -> None:
        """Set the methods used by this algorithm to reconstruct a mesh from a segmentation mask.

        Args:
            print_info (bool): Whether to show details about the computation during the algorithm's execution.
            edt_creation_function (EdtCreationFunction): Function used to compute an EDT from the segmentation mask.
            point_placing_function (PointPlacingFunction): Function used to place points on the EDT for a tesselation.
            tesselation_creation_function (TesselationCreationFunction): Function used to tesselate the points.
            score_computation_function (ScoreComputationFunction): Function used to compute the scores for Watershed.
            perform_mesh_postprocess_surgery (bool): Whether to try to detect and fix problems on the output mesh.
        """
        self.print_info = print_info
        self.edt_creation_function = edt_creation_function
        self.point_placing_function = point_placing_function
        self.tesselation_creation_function = tesselation_creation_function
        self.score_computation_function = score_computation_function
        self.perform_mesh_postprocess_surgery = perform_mesh_postprocess_surgery

        self._first_computation_done = False

    def construct_mesh_from_segmentation_mask(
        self,
        segmented_image: NDArray[np.uint],
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Build a 3D mesh from a segmentation mask.

        Args:
            segmented_image (NDArray[np.uint]): Segmentation mask input.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
               - mesh points (geometry)
               - mesh triangles (topology)
               - labels (materials) on each side of the triangles. 0 is exterior.
        """
        self._segmented_image = segmented_image

        self._edt_image = self.edt_creation_function(self._segmented_image)

        points_for_tesselation = self.point_placing_function(self._edt_image)

        t_init_tesselation = perf_counter()
        tesselation_points, tesselation_tetrahedrons = self.tesselation_creation_function(points_for_tesselation)
        if self.print_info:
            print(f"Delaunay Tesselation built in {perf_counter() - t_init_tesselation:.2} seconds")

        self._tesselation_graph = TesselationGraph(
            tesselation_points,
            tesselation_tetrahedrons,
            self.score_computation_function,
            self._edt_image,
            print_info=self.print_info,
        )

        # Here we use again the segmented image to get seeds, but another input to create a mesh could be
        # the EDT directly + those seeds obtained with another method ?
        # (note we also use the segmented image to find zero nodes later but I argue that it's useless)
        self._seeds_coords, self._seeds_indices = extract_seed_coords_and_indices(
            self._segmented_image,
            self._edt_image,
        )
        self._watershed_seeded()

        self._points, self._triangles, self._labels = labeled_mesh_from_labeled_graph(
            self._tesselation_graph,
            self._map_label_to_nodes_ids,
        )

        self._mesh_surgery()

        self._first_computation_done = True
        return self.last_constructed_mesh

    def _watershed_seeded(self) -> None:
        """Perform watershed algorithm to label tetrahedrons of the tesselation networkX graph."""
        t1 = perf_counter()
        seeds_nodes = _compute_seeds_idx_from_voxel_coords(
            self._edt_image,
            self._tesselation_graph.compute_nodes_centroids(),
            self._seeds_coords,
        )
        zero_nodes = self._tesselation_graph.compute_zero_nodes(self._segmented_image)

        nx_graph = self._tesselation_graph.to_networkx_graph()
        self._map_label_to_nodes_ids, self._map_node_id_to_label = seeded_watershed_map(
            nx_graph,
            seeds_nodes,
            self._seeds_indices,
            zero_nodes,
        )

        if self.print_info:
            print(f"Watershed done in {perf_counter() - t1:.3} seconds.")

    def _mesh_surgery(self, max_iter: int = 3) -> None:
        """Try to detect and fix mesh problems while we have all the Watershed data."""
        # Optional part
        if self.perform_mesh_postprocess_surgery:
            abnormal_edges = self._find_abnormal_non_manifold_edges()

            for abnormal_edge in abnormal_edges:
                # find ordered cycle of tetrahedrons around edge (& labels)
                tetra_cycle = self._find_tetra_cycle_around_edge(abnormal_edge)
                label_cycle = self._map_node_id_to_label[tetra_cycle]
                # find where we can switch label and switch
                current_nb_iter = 0
                candidates_id = _find_candidate_for_label_switching(label_cycle)

                while len(candidates_id) > 0 and current_nb_iter < max_iter:
                    candidate = candidates_id[0]
                    tetra_to_change = tetra_cycle[candidate]
                    old_label = self._map_node_id_to_label[tetra_to_change]
                    new_label = self._map_node_id_to_label[tetra_cycle[candidate - 1]]

                    self._map_node_id_to_label[tetra_to_change] = new_label

                    self._map_label_to_nodes_ids[old_label].remove(tetra_to_change)
                    self._map_label_to_nodes_ids[new_label].append(tetra_to_change)

                    label_cycle = self._map_node_id_to_label[tetra_cycle]
                    candidates_id = _find_candidate_for_label_switching(label_cycle)
                    current_nb_iter += 1

                # sometimes there's more than one candidates and it's not obvious to me which one to choose...
                # so I arbitrarily choose the first for now.

                # switch label => what does it change for the mesh ? do we recompute everything ? that the simple way
                self._points, self._triangles, self._labels = labeled_mesh_from_labeled_graph(
                    self._tesselation_graph,
                    self._map_label_to_nodes_ids,
                )

        # Always do this part
        # filter unused points
        self._points, self._triangles = filter_unused_points(self._points, self._triangles)

    def _find_abnormal_non_manifold_edges(self) -> NDArray[np.uint]:
        """Find edges in mesh that appear 4 or more times but are not at the intersection of 4 or more cells.

        Returns:
            NDArray[np.uint]: an array of those edges (id_p1, id_p2)
        """
        edges = np.vstack((self._triangles[:, [0, 1]], self._triangles[:, [0, 2]], self._triangles[:, [1, 2]]))
        edges = np.sort(edges, axis=1)
        edges_key = edges[:, 0] * len(self._points) + edges[:, 1]
        _, index_first_occurence, index_counts = np.unique(
            edges_key,
            return_index=True,
            return_counts=True,
        )
        more_than_trijunctions_id = index_counts > 3
        suspicious_edges = edges[index_first_occurence[more_than_trijunctions_id]]
        suspicious_counts = index_counts[more_than_trijunctions_id]
        abnormal_mask = np.ones(len(suspicious_counts), dtype=np.bool_)

        for i, edge in enumerate(suspicious_edges):
            if self._number_of_adjacent_cells_of_edge(edge) == suspicious_counts[i]:
                abnormal_mask[i] = False

        return suspicious_edges[abnormal_mask]

    def _find_tetra_cycle_around_edge(self, edge: tuple[int, int]) -> list[int]:
        """Find a cycle of adjacent tetrahedrons around the edge."""
        pid1, pid2 = edge

        # Find all tetrahedrons with this edge
        adjacent_tetrahedrons = list(
            np.where(
                np.logical_and(
                    (self._tesselation_graph.tetrahedrons == pid1).any(axis=1),
                    (self._tesselation_graph.tetrahedrons == pid2).any(axis=1),
                ),
            )[0],
        )

        # Now let's find a cycle
        ordered_tetrahedrons = [adjacent_tetrahedrons[0]]
        del adjacent_tetrahedrons[0]

        # We find a cycle by finding tetrahedrons sharing a triangle face
        nb_tetra = len(adjacent_tetrahedrons)
        selected_id = 0
        for _ in range(nb_tetra):
            last_tetra_in_cycle = ordered_tetrahedrons[-1]
            last_triangles = set(self._tesselation_graph.faces_of_nodes[last_tetra_in_cycle])

            to_remove = -1
            for i, tet_id in enumerate(adjacent_tetrahedrons):
                selected_id = tet_id
                tet_triangles = self._tesselation_graph.faces_of_nodes[tet_id]
                if len(last_triangles.intersection(tet_triangles)) > 0:
                    to_remove = i
                    break

            ordered_tetrahedrons.append(selected_id)
            del adjacent_tetrahedrons[to_remove]

        return ordered_tetrahedrons

    def _number_of_adjacent_cells_of_edge(self, edge: tuple[int, int]) -> int:
        """Return the number of adjacent cells of an edge."""
        pid1, pid2 = edge

        adjacent_triangles_id = np.where(
            np.logical_and(
                (self._triangles == pid1).any(axis=1),
                (self._triangles == pid2).any(axis=1),
            ),
        )[0]

        adjacent_labels = self._labels[adjacent_triangles_id]

        return len(np.unique(adjacent_labels))

    @property
    def last_constructed_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Get points, triangles and labels (materials) describing the mesh obtained from segmented image."""
        return self._points, self._triangles, self._labels

    def compress_segmentation_mask(self, segmented_image: NDArray[np.uint]) -> dict[str]:
        """Compress a segmentation mask using a 3D mesh reconstruction.

        Args:
            segmented_image (NDArray[np.uint]): Segmentation mask input.

        Returns:
            dict[str]:
               - dictionary of the compressed segmentation that can be reconstructed with this package.
        """
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return self.last_compressed_segmentation

    @property
    def last_compressed_segmentation(self) -> dict[str]:
        """Export mesh, seeds coordinates and image shape in a dictionary that can be saved with numpy.save()."""
        return {
            "points": self._points,
            "triangles": self._triangles,
            "seeds": self._seeds_coords,
            "image_shape": self._segmented_image.shape,
        }

    def both_construct_mesh_and_compressed_segmentation(
        self,
        segmented_image: NDArray[np.uint],
    ) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
        """Construct and return a 3D mesh from segmentation and obtain the dictionary of the compressed segmentation.

        Args:
            segmented_image (NDArray[np.uint]): Segmentation mask input.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
               - mesh points (geometry)
               - mesh triangles (topology)
               - labels (materials) on each side of the triangles. 0 is exterior.
               - dictionary of the compressed segmentation that can be reconstructed with this package.
        """
        self.construct_mesh_from_segmentation_mask(segmented_image)
        return *self.last_constructed_mesh, self.last_compressed_segmentation

    def center_around_zero(self) -> None:
        """Center the mesh around 0."""
        self._points = center_around_origin(self._points)

    def set_pixel_size(self, xy_pixel_size: float, z_pixel_size: float) -> None:
        """Scales the mesh from pixel coordinates to real coordinates, given microscope's xy and z pixel size."""
        self._points = set_pixel_size(self._points, xy_pixel_size, z_pixel_size)

    def set_global_min_max(self, global_min: float, global_max: float) -> None:
        """Scales homogeneously the mesh such that its min & max values are global_min and global_max."""
        self._points = set_points_min_max(self._points, global_min, global_max)

    def save_to_rec_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the output mesh on disk in the rec format."""
        save_rec(filename, self._points, self._triangles, self._labels, binary_mode)

    def save_to_vtk_mesh(self, filename: str | Path, binary_mode: bool = False) -> None:
        """Save the output mesh on disk in the vtk format."""
        save_vtk(filename, self._points, self._triangles, self._labels, binary_mode)


def save_compressed_segmentation(filename: str | Path, compressed_segmentation: dict[str]) -> None:
    """Save a compressed segmentation on disk with numpy.save."""
    np.save(filename, compressed_segmentation)


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


def _find_candidate_for_label_switching(label_cycle: list[int]) -> list[int]:
    candidates = []
    nb_labels = len(label_cycle)
    for current_id in range(nb_labels):
        previous_id = (current_id - 1) % nb_labels
        if label_cycle[previous_id] == label_cycle[current_id]:
            continue
        next_id = (current_id + 1) % nb_labels
        if label_cycle[next_id] == label_cycle[current_id]:
            continue
        # here current id != prev and next
        if label_cycle[next_id] == label_cycle[previous_id]:
            # and they are the same !
            candidates.append(current_id)

    return candidates
