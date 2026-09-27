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
        indices_of_sorted_maxes: NDArray[np.uint],
        score_computation_function: "ScoreComputationFunction",
        edt_image: NDArray[np.float64],
        print_info: bool = False,
    ) -> None:
        """Create Tesselation graph from EDT image. Compute scores on triangles for watershed.

        `indices_of_sorted_maxes` is not used to build the graph. It is stored so that
        callers (benchmarks, later point-placement schemes) can tell which tesselation
        vertices are interior EDT maxima without recomputing the point-placement stage.
        It used to be consumed by `_improve_tesselation`, which was dead and known
        broken and was deleted as dead code; the code is
        preserved on branch `archive/improve-tesselation`.
        """
        self.vertices = tesselation_points
        self.nodes = tesselation_tetrahedrons
        self.n_simplices = len(self.nodes)
        self.indices_of_sorted_maxes = indices_of_sorted_maxes

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

        # sort faces (3 triangles point indices + tetra id) lexicographically.
        # i.e. first point index is sorted, then second, then third, then tetra id.
        # Note : a triangle separating two tetrahedrons will appear twice in a row (which is what we want)
        lex_keys = (faces_table[:, 3], faces_table[:, 2], faces_table[:, 1], faces_table[:, 0])
        edges_table: NDArray[np.uint] = faces_table[np.lexsort(lex_keys)]
        return edges_table

    def _construct_edges(self, edges_table: NDArray[np.int64]) -> None:
        """Build adjacency maps between tetrahedrons (nodes) and triangles faces ("edges").

        `edges_table` is lexicographically sorted, so the two copies of a triangle shared
        by two tetrahedrons are adjacent rows. Every row is therefore either the first of
        such a pair (a shared face) or unpaired (a "lone" face on the tesselation
        boundary), which gives the invariant
        `2 * len(triangle_faces) + len(lone_faces) == len(edges_table)`.

        The loop bound is `index < n`, not `index < n - 1`: the latter never examined the
        last row, so a *trailing lone face* was silently dropped and its tetrahedron not
        marked in `nodes_on_the_border`. On all 8
        golden-master configurations the last row happens to be the second half of a
        matched pair, so the bug does not fire there and this fix changes no output; it
        fires whenever the lexicographically largest face is a boundary face. See
        `tests/test_construct_edges.py`.
        """
        index = 0
        n = len(edges_table)

        self.triangle_faces = []
        self.nodes_linked_by_faces = []
        self.nodes_on_the_border = np.zeros(len(self.nodes))
        self.faces_of_nodes = {i: [] for i in range(len(self.nodes))}
        self.lone_faces = []
        self.nodes_linked_by_lone_faces = []
        while index < n:
            if index + 1 < n and (
                edges_table[index][0] == edges_table[index + 1][0]
                and edges_table[index][1] == edges_table[index + 1][1]
                and edges_table[index][2] == edges_table[index + 1][2]
            ):
                # same triangle,two tetraedron indices
                a, b = edges_table[index][3], edges_table[index + 1][3]
                self.triangle_faces.append(edges_table[index][:-1] - 1)  # We correct the previous shift
                self.nodes_linked_by_faces.append([a, b])

                self.faces_of_nodes[a].append(len(self.triangle_faces) - 1)
                self.faces_of_nodes[b].append(len(self.triangle_faces) - 1)
                # self.faces_of_nodes[a] = [*self.faces_of_nodes.get(a, []), len(self.triangle_faces) - 1]
                # self.faces_of_nodes[b] = [*self.faces_of_nodes.get(b, []), len(self.triangle_faces) - 1]
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
        return np.mean(self.vertices[self.tetrahedrons], axis=1)

    def compute_zero_nodes(self, segmented_image: NDArray[np.uint]) -> NDArray[np.uint]:
        """Get index of tetrahedrons with centroids on the part where segmented image is 0.

        These are the "background" tetrahedrons. **Nothing in the pipeline currently calls
        this.** It was called by `MeshReconstructionAlgorithm._watershed_seeded` and passed
        to `seeded_watershed_map`, which guarded its use behind an inverted condition and
        so never applied it. The defect-cleanup work
        deleted the dead guard and the now-pointless call, preserving the de-facto behaviour
        (background-tetrahedron forcing *inactive*).

        The method itself is correct and kept deliberately: whether the watershed *should*
        force these tetrahedrons to label 0 is an open question, not a settled one, and
        answering it changes the labelling. See `seeded_watershed_map`'s docstring for what
        re-enabling would take.
        """
        centroids = self.compute_nodes_centroids()
        segmented_image = _interpolate_image(segmented_image)
        bools = segmented_image(centroids) == 0
        ints = np.arange(len(centroids))[bools]
        return ints

    def find_tetra_cycle_around_edge(self, edge: tuple[int, int]) -> list[int]:
        """Find a cycle of adjacent tetrahedrons around the edge."""
        pid1, pid2 = edge

        # Find all tetrahedrons with this edge
        adjacent_tetrahedrons = list(
            np.where(
                np.logical_and(
                    (self.tetrahedrons == pid1).any(axis=1),
                    (self.tetrahedrons == pid2).any(axis=1),
                ),
            )[0],
        )

        # Now let's find a cycle
        ordered_tetrahedrons = [adjacent_tetrahedrons[0]]
        del adjacent_tetrahedrons[0]
        ordered_triangles = []  # transition between each tetra

        # We find a cycle by finding tetrahedrons sharing a triangle face
        nb_tetra = len(adjacent_tetrahedrons)
        selected_id = 0
        for _ in range(nb_tetra):
            last_tetra_in_cycle = ordered_tetrahedrons[-1]
            last_triangles = set(self.faces_of_nodes[last_tetra_in_cycle])

            to_remove = -1
            for i, tet_id in enumerate(adjacent_tetrahedrons):
                selected_id = tet_id
                tet_triangles = self.faces_of_nodes[tet_id]
                common_triangles = last_triangles.intersection(tet_triangles)  # normally 0 or 1
                if len(common_triangles) > 0:
                    ordered_triangles.append(next(iter(common_triangles)))  # first triangle in intersection (only one)
                    to_remove = i
                    break

            ordered_tetrahedrons.append(selected_id)
            del adjacent_tetrahedrons[to_remove]

        # tetrahedrons are sorted
        first_triangles = self.faces_of_nodes[ordered_tetrahedrons[0]]
        last_triangles = set(self.faces_of_nodes[ordered_tetrahedrons[-1]])
        common_triangles = last_triangles.intersection(first_triangles)  # normally 0 or 1
        if len(common_triangles) > 0:
            ordered_triangles.append(next(iter(common_triangles)))  # first triangle in intersection (only one)

        return ordered_tetrahedrons

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


def intersect_line_triangle(l1: NDArray, l2: NDArray, t1: NDArray, t2: NDArray, t3: NDArray) -> NDArray | None:
    """Get the intersection point of a line segment and a triangle, if it exists and is unique. Otherwise return None.

    Args:
        l1 (NDArray): First 3D point of the line segment.
        l2 (NDArray): Second 3D point of the line segment.
        t1 (NDArray): First 3D point of the triangle.
        t2 (NDArray): Second 3D point of the triangle.
        t3 (NDArray): Third 3D point of the triangle.

    Returns:
        NDArray | None: Intersection point between triangle and line segment if it exists and is unique. Otherwise None.
    """
    # Thanks to Bruno Levy https://stackoverflow.com/a/42752998

    def sign_of_tetra_volume(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> int:
        """Get the sign of the volume of the tetraedron defined by the 4 points a, b, c, d.

        Return 1 if the volume is positive, -1 if negative, 0 if degenerate/flat.
        Designed for 3D points.
        """
        return np.sign(np.dot(np.cross(b - a, c - a), d - a))

    # unnecessary first test (in our case) because we know that l1 and l2 are on two different sides of the triangle.
    # Under the assumption that the first tesselation is correct and that our modifications of it are also correct!
    # ---
    # Check 1 : check that l1 and l2 are on two different sides of the triangle
    # and also that neither l1 or l2 are on the plane defined by the triangle.
    # Therefore there will be at most a unique intersection point.
    if sign_of_tetra_volume(l1, t1, t2, t3) * sign_of_tetra_volume(l2, t1, t2, t3) == -1:
        s3 = sign_of_tetra_volume(l1, l2, t1, t2)
        s4 = sign_of_tetra_volume(l1, l2, t2, t3)
        # Check 2 & 3: all of the following tetraedrons have the same signed volume
        if s3 == s4:
            s5 = sign_of_tetra_volume(l1, l2, t3, t1)
            if s4 == s5:
                # If yes, it means that there is a unique intersection point inside the triangle. Let's find it.
                n = np.cross(t2 - t1, t3 - t1)
                t = np.dot(t1 - l1, n) / np.dot(l2 - l1, n)
                return l1 + t * (l2 - l1)
    # All other cases : no unique, well-defined intersection point : we return None.
    return None


if __name__ == "__main__":
    # Define triangle
    t1 = np.array([0, 0, 0])
    t2 = np.array([1, 0, 0])
    t3 = np.array([0, 1, 0])
    # Exactly over a point : No intersection
    l1 = np.array([0, 0, 1])
    l2 = np.array([0, 0, -1])
    assert np.array_equal(None, intersect_line_triangle(l1, l2, t1, t2, t3))
    # Exactly over a segment of the triangle : No intersection
    l1 = np.array([0.5, 0, 1])
    l2 = np.array([0.5, 0, -1])
    assert np.array_equal(None, intersect_line_triangle(l1, l2, t1, t2, t3))
    # We also check for same side points
    l1 = np.array([0.25, 0.25, 1])
    l2 = np.array([0.26, 0.24, 1.1])
    assert np.array_equal(None, intersect_line_triangle(l1, l2, t1, t2, t3))
    # Or coplanar points
    l1 = np.array([0.25, 0.25, 0])
    l2 = np.array([0.25, 0.25, -1])
    assert np.array_equal(None, intersect_line_triangle(l1, l2, t1, t2, t3))
    # Inside the triangle : intersection
    l1 = np.array([0.25, 0.25, 1])
    l2 = np.array([0.25, 0.25, -1])
    assert np.array_equal(np.array([0.25, 0.25, 0]), intersect_line_triangle(l1, l2, t1, t2, t3))
    l1 = np.array([0.26, 0.23, 1])
    l2 = np.array([0.24, 0.27, -1])
    assert np.array_equal(np.array([0.25, 0.25, 0]), intersect_line_triangle(l1, l2, t1, t2, t3))
