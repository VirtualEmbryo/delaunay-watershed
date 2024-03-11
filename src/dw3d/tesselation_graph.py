"""Module defining TesselationGraph, allowing to export a NetworkX graph with scores for the Watershed algorithm.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from time import time
from typing import TYPE_CHECKING

import networkx
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from dw3d.reconstruction_algorithm import ScoreComputationFunction


class TesselationGraph:
    """Graph computed from a tesselation to compute scores for a Watershed algorithm to label the tetrahedrons."""

    def __init__(
        self,
        tesselation_points: NDArray[np.float64],
        tesselation_tetrahedrons: NDArray[np.int64],
        score_computation_function: "ScoreComputationFunction",
        edt_image: NDArray[np.float64],
        print_info: bool = False,
    ) -> None:
        """Create Tesselation graph from EDT image. Compute scores on triangles for watershed."""
        self.vertices = tesselation_points
        self.nodes = tesselation_tetrahedrons
        self.n_simplices = len(self.nodes)

        t1 = time()

        edges_table = self._construct_edges_table()
        self._construct_edges(edges_table)

        self.scores = score_computation_function(edt_image, self.vertices, self.triangle_faces)

        t2 = time()
        if print_info:
            print("Graph build in ", np.round(t2 - t1, 3))

    def _construct_edges_table(self) -> NDArray[np.int64]:
        """Get an ordered array of triangle faces from tetrahedrons."""
        tetrahedrons = np.sort(self.nodes, axis=1)
        self.tetrahedrons = tetrahedrons.copy()
        tetrahedrons += 1  # We shift to get the right keys
        faces_table = np.array(_give_faces_table(tetrahedrons), dtype=np.int64)
        key_multiplier = _find_key_multiplier(max(len(self.vertices), len(self.nodes)))
        keys = (
            faces_table[:, 0] * (key_multiplier**3)
            + faces_table[:, 1] * (key_multiplier**2)
            + faces_table[:, 2] * (key_multiplier**1)
            + faces_table[:, 3] * (key_multiplier**0)
        )
        edges_table: NDArray[np.uint] = faces_table[np.argsort(keys)]  # .tolist()

        return edges_table

    def _construct_edges(self, edges_table: NDArray[np.int64]) -> None:
        """Build adjacency maps between tetrahedrons (nodes) and triangles faces ("edges")."""
        index = 0
        n = len(edges_table)

        self.triangle_faces = []
        self.nodes_linked_by_faces = []
        self.nodes_on_the_border = np.zeros(len(self.nodes))
        self.faces_of_nodes = {}
        self.lone_faces = []
        self.nodes_linked_by_lone_faces = []
        while index < n - 1:
            if (
                edges_table[index][0] == edges_table[index + 1][0]
                and edges_table[index][1] == edges_table[index + 1][1]
                and edges_table[index][2] == edges_table[index + 1][2]
            ):
                a, b = edges_table[index][3], edges_table[index + 1][3]
                self.triangle_faces.append(edges_table[index][:-1] - 1)  # We correct the previous shift
                self.nodes_linked_by_faces.append([a, b])
                self.faces_of_nodes[a] = [*self.faces_of_nodes.get(a, []), len(self.triangle_faces) - 1]
                self.faces_of_nodes[b] = [*self.faces_of_nodes.get(b, []), len(self.triangle_faces) - 1]
                index += 2
            else:
                self.nodes_on_the_border[edges_table[index][3]] = 1
                self.lone_faces.append(edges_table[index][:-1] - 1)
                self.nodes_linked_by_lone_faces.append(edges_table[index][3])
                index += 1

        self.triangle_faces: NDArray[np.uint] = np.array(self.triangle_faces, dtype=np.uint)
        self.nodes_linked_by_faces = np.array(self.nodes_linked_by_faces)

        self.lone_faces = np.array(self.lone_faces)
        self.nodes_linked_by_lone_faces = np.array(self.nodes_linked_by_lone_faces)

    def _compute_volumes(self) -> NDArray[np.float64]:
        """Get volume of all tetrahedrons of the tesselation."""
        positions = self.vertices[self.tetrahedrons]
        vects = positions[:, [0, 0, 0]] - positions[:, [1, 2, 3]]
        volumes = np.abs(np.linalg.det(vects)) / 6
        return volumes

    def _compute_areas(self) -> NDArray[np.float64]:
        """Get the area of all triangles faces of the tesselation."""
        # Triangles[i] = 3*2 array of 3 points of the plane
        # Triangles = self.Vertices[self.Faces]
        positions = self.vertices[self.triangle_faces]
        sides = positions - positions[:, [2, 0, 1]]
        lengths_sides = np.linalg.norm(sides, axis=2)
        half_perimeters = np.sum(lengths_sides, axis=1) / 2

        diffs = np.array([half_perimeters] * 3).transpose() - lengths_sides
        areas = (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)
        return areas

    def compute_nodes_centroids(self) -> NDArray[np.float64]:
        """Compute tesselation's tetrahedrons' centroid point."""
        return np.mean(self.vertices[self.nodes], axis=1)

    def compute_zero_nodes(self, segmented_image: NDArray[np.uint]) -> NDArray[np.uint]:
        """Get index of tetrahedrons with centroids on the part where segmented image is 0."""
        centroids = self.compute_nodes_centroids()
        segmented_image = _interpolate_image(segmented_image)
        bools = segmented_image(centroids) == 0
        ints = np.arange(len(centroids))[bools]
        return ints

    def to_networkx_graph(self) -> networkx.Graph:
        """Compute a NetworkX graph with nodes = tetrahedrons, edges = triangle faces and data associated.

        Data on nodes = volumes, Data on edges = scores and area.
        """
        self.volumes = self._compute_volumes()  # Number of nodes (Tetrahedras)
        self.areas = self._compute_areas()  # Number of edges (Faces)

        nx_graph = networkx.Graph()
        nt = len(self.volumes)
        node_data_dicts = [{"volume": x} for x in self.volumes]
        nx_graph.add_nodes_from(zip(np.arange(nt), node_data_dicts, strict=False))

        network_edges = np.array(
            [
                (
                    self.nodes_linked_by_faces[idx][0],
                    self.nodes_linked_by_faces[idx][1],
                    {"score": self.scores[idx], "area": self.areas[idx]},
                )
                for idx in np.arange(len(self.triangle_faces))
            ],
        )

        nx_graph.add_edges_from(network_edges)

        return nx_graph


def _interpolate_image(image: NDArray[np.uint8]) -> RegularGridInterpolator:
    """Return an interpolated image, a function with values based on pixels."""
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    z = np.linspace(0, image.shape[2] - 1, image.shape[2])
    image_interp = RegularGridInterpolator((x, y, z), image)
    return image_interp


def _give_faces_table(tetrahedrons: NDArray[np.uint]) -> list[list[int]]:
    """Give all triangle faces of a list of tetrahedrons."""
    faces_table = []
    for i, tet in enumerate(tetrahedrons):
        a, b, c, d = tet
        faces_table.append([a, b, c, i])
        faces_table.append([a, b, d, i])
        faces_table.append([a, c, d, i])
        faces_table.append([b, c, d, i])
    return faces_table


def _find_key_multiplier(num_points: int) -> int:
    key_multiplier = 1
    while num_points // key_multiplier != 0:
        key_multiplier *= 10
    return key_multiplier
