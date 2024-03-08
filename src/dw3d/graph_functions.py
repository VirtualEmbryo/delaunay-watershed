"""Module defining TesselationGraph, allowing to export a NetworkX graph with scores for the Watershed algorithm.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from time import time

import networkx
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from dw3d.geometric_utilities import tesselation_from_edt


def give_faces_table(tetrahedrons: NDArray[np.uint]) -> list[list[int]]:
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


def _faces_score_from_sampling(
    triangle_faces: NDArray[np.uint],
    vertices: NDArray[np.float64],
    f: RegularGridInterpolator,
) -> NDArray[np.float64]:
    alpha = np.linspace(0, 1, 5)[1:-1]
    beta = np.linspace(0, 1, 5)[1:-1]
    gamma = np.linspace(0, 1, 5)[1:-1]

    vertices = vertices.copy()
    scale = np.amax(vertices, axis=0) / 2
    vertices -= scale
    vertices *= 1 - 1e-4
    vertices += scale * (1 - 1e-4)
    v = vertices[triangle_faces]

    v1 = v[:, 0]
    v2 = v[:, 1]
    v3 = v[:, 2]
    count = 0
    count_bad = 0
    score_faces = np.zeros(len(triangle_faces), dtype=np.float64)
    for a in alpha:
        for b in beta:
            for c in gamma:
                try:
                    s = a + b + c
                    l1 = a / s
                    l2 = b / s
                    l3 = c / s

                    score_faces += np.array(f(v1 * l1 + v2 * l2 + v3 * l3))

                    # Test Matthieu Perez: take score max (seems to improve a bit the results)
                    # Score_Faces = np.maximum(
                    #     Score_Faces, f(V1 * l1 + V2 * l2 + V3 * l3)
                    # )
                    count += 1
                except:  # except what ?? what can happen badly ??  # noqa: E722
                    count_bad += 1
    score_faces /= count
    return score_faces


class TesselationGraph:
    """Graph computed from a tesselation to compute scores for a Watershed algorithm to label the tetrahedrons."""

    def __init__(
        self,
        edt_image: NDArray[np.float64],
        min_distance: int = 5,
        print_info: bool = False,
    ) -> None:
        """Create Tesselation graph from EDT image. Compute scores on triangles for watershed."""
        tri = tesselation_from_edt(
            edt_image,
            min_distance=min_distance,
            print_info=print_info,
        )
        t1 = time()
        self.nodes = tri.simplices
        self.vertices = tri.points
        self.tri = tri
        self.n_simplices = len(tri.simplices)

        edges_table = self._construct_edges_table()
        self._construct_edges(edges_table)
        self._compute_scores(edt_image)
        t2 = time()
        if print_info:
            print("Graph build in ", np.round(t2 - t1, 3))

    def _construct_edges_table(self) -> NDArray[np.int64]:
        """Get an ordered array of triangle faces from tetrahedrons."""
        tetrahedrons = np.sort(self.tri.simplices, axis=1)
        self.tetrahedrons = tetrahedrons.copy()
        tetrahedrons += 1  # We shift to get the right keys
        faces_table = np.array(give_faces_table(tetrahedrons), dtype=np.int64)
        key_multiplier = _find_key_multiplier(max(len(self.tri.points), len(self.tri.simplices)))
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

    def _compute_scores(self, edt: NDArray[np.float64]) -> None:
        # Remember : each edge is a face !
        edt = _interpolate_image(edt)
        self.scores = _faces_score_from_sampling(self.triangle_faces, self.vertices, edt)

    def compute_volumes(self) -> NDArray[np.float64]:
        """Get volume of all tetrahedrons of the tesselation."""
        positions = self.vertices[self.tetrahedrons]
        vects = positions[:, [0, 0, 0]] - positions[:, [1, 2, 3]]
        volumes = np.abs(np.linalg.det(vects)) / 6
        return volumes

    def compute_areas(self) -> NDArray[np.float64]:
        """Get the are of all triangles faces of the tesselation."""
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

    def networkx_graph_weights_and_borders(self) -> networkx.Graph:
        """Compute a NetworkX graph with nodes = tetrahedrons, edges = triangle faces and data associated.

        Data on nodes = volumes, Data on edges = scores and area.
        """
        self.Volumes = self.compute_volumes()  # Number of nodes (Tetrahedras)
        self.Areas = self.compute_areas()  # Number of edges (Faces)

        nx_graph = networkx.Graph()
        nt = len(self.Volumes)
        node_data_dicts = [{"volume": x} for x in self.Volumes]
        nx_graph.add_nodes_from(zip(np.arange(nt), node_data_dicts, strict=False))

        network_edges = np.array(
            [
                (
                    self.nodes_linked_by_faces[idx][0],
                    self.nodes_linked_by_faces[idx][1],
                    {"score": self.scores[idx], "area": self.Areas[idx]},
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
