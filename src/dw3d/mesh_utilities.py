"""Module for Mesh creation and cleaning.

Sacha Ichbiah 2021.
Matthieu Perez 2024.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dw3d.tesselation_graph import TesselationGraph


#####
#####
# Mesh creation
#####
#####


def labeled_mesh_from_labeled_graph(
    tesselation_graph: "TesselationGraph",
    map_label_to_nodes_ids: dict[int, list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]:
    """Extract a labeled mesh from a labeled tesselation graph.

    Args:
        tesselation_graph (TesselationGraph): TesselationGraph object
        map_label_to_nodes_ids (dict[int, list[int]]): Labels on this graph.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint]]: points, triangles, and labels (materials)
    """
    # Take a Segmentation class as entry

    (
        points,
        triangles,
        labels,
        nodes_idx_in_graph_linked_to_triangle,
    ) = _retrieve_mesh_multimaterial_multitracker_format(
        tesselation_graph,
        map_label_to_nodes_ids,
    )

    # sort label (and nodes)
    for i, label in enumerate(labels):
        if label[0] > label[1]:  # if label0 > label1 we swap them
            labels[i] = labels[i, [1, 0]]
            nodes_idx_in_graph_linked_to_triangle[i] = nodes_idx_in_graph_linked_to_triangle[i][[1, 0]]
    # reorient triangles to have coherent normals (for plotting mostly)
    triangles = _reorient_triangles(
        triangles,
        tesselation_graph.vertices,
        tesselation_graph.tetrahedrons,
        nodes_idx_in_graph_linked_to_triangle,
    )

    return (points, triangles, labels)


def _retrieve_mesh_multimaterial_multitracker_format(
    tesselation_graph: "TesselationGraph",
    map_label_to_nodes: dict[int, list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.uint], NDArray[np.uint], NDArray[np.uint]]:
    """Extract multi-material mesh from the tesselation graph with every nodes (tetrahedrons) marked with a material.

    The extracted surface mesh is composed of all triangles that are faces of 2 tetrahedrons with different materials.
    Note that there will be a lot of unused points in this mesh. A filtering step will be necessary.

    Args:
        tesselation_graph (TesselationGraph): the tesselation graph.
        map_label_to_nodes (dict[int, list[int]]): The map that links materials to tetrahedrons id.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.uint], list[int], NDArray[np.uint]]:
            - an array of vertices of the mesh. Note that there are some unused points, to be filtered later.
            - an array of triangles (id p1, id p2, id p3)
            - an array of labels (material 1, material 2)
            - the list of ids of nodes (tetrahedrons) in graph.nodes that are linked to selected triangles.
    """
    # faces = triangles ; nodes = tetras, with map linking regions to list of tetra/node ids
    map_nodes_to_labels: dict[int, int] = {}
    for key in map_label_to_nodes:
        for node_idx in map_label_to_nodes[key]:
            map_nodes_to_labels[node_idx] = key
    triangles: list[list[int]] = []  # p1, p2, p3, selected faces
    labels: list[list[int]] = []  # l1, l2 of selected faces
    nodes_linked_by_face: list[list[int]] = []  # list nodes ids linked to triangles from selected graph faces

    for idx, face in enumerate(tesselation_graph.triangle_faces):
        # faces are triangles, nodes linked are the 2 adjacent tetrahedrons (nodes).
        nodes_linked = tesselation_graph.nodes_linked_by_faces[idx]

        # corresponding regions
        cluster_1 = map_nodes_to_labels[nodes_linked[0]]
        cluster_2 = map_nodes_to_labels[nodes_linked[1]]
        cells = [cluster_1, cluster_2]

        if cluster_1 != cluster_2:
            # some faces belong to 2 tetras of two different regions : those are the triangles in final mesh !
            triangles.append([face[0], face[1], face[2]])  # tri p1, p2, p3, label l1, l2
            labels.append([cells[0], cells[1]])
            nodes_linked_by_face.append(nodes_linked)  # 2 tetras

    # Matthieu Perez:
    # Apparently, there are triangles on only one tetra in the tesselation graph,
    # and sometimes they might belong to the mesh ? If segmented cell touches the border of the image ?
    for idx in range(len(tesselation_graph.lone_faces)):
        face = tesselation_graph.lone_faces[idx]
        node_linked = tesselation_graph.nodes_linked_by_lone_faces[idx]
        cluster_1 = map_nodes_to_labels[node_linked]
        # We incorporate all these edges because they are border edges
        if cluster_1 != 0:
            cells = [0, cluster_1]
            triangles.append([face[0], face[1], face[2]])
            labels.append([cells[0], cells[1]])
            nodes_linked_by_face.append(nodes_linked)

    # Note that the extraction might lead to a mesh that is non-manifold where it is not expected to,
    # if the labeling of nodes is not perfect. Some kind of "mesh surgery" might be necessary to improve
    # extracted mesh quality.
    return (
        tesselation_graph.vertices,
        np.array(triangles, dtype=np.uint),
        np.array(labels, dtype=np.uint),
        np.array(nodes_linked_by_face, dtype=np.uint),
    )


###############
# Mesh Cleaning
###############
def set_points_min_max(
    points: NDArray[np.float64],
    global_min: float,
    global_max: float,
) -> NDArray[np.float64]:
    """Return a homogeneously scaled points array such that its min & max values are global_min and global_max."""
    current_min = points.min()
    current_max = points.max()
    new_points = (
        (np.copy(points) - current_min) / (current_max - current_min) * (global_max - global_min)
    ) + global_min

    return new_points


def set_pixel_size(
    points: NDArray[np.float64],
    xy_pixel_size: float,
    z_pixel_size: float,
) -> NDArray[np.float64]:
    """Return a scaled points array from pixel coordinates space to real coordinates.

    Given microscope's xy and z pixel size.
    """
    new_points = np.copy(points)
    new_points[:, :2] *= xy_pixel_size
    new_points[:, 2] *= z_pixel_size

    return new_points


def center_around_origin(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a centered points array around 0."""
    current_min = points.min(axis=0)
    current_max = points.max(axis=0)

    return np.copy(points) - (current_max + current_min) / 2.0


def _reorient_triangles(
    triangles: NDArray[np.uint],
    vertices: NDArray[np.float64],
    tetrahedrons: NDArray[np.intc],
    nodes_linked: NDArray[np.uint],
) -> NDArray[np.uint]:
    """Swap point order in triangles such that all normals points in the same direction.

    Args:
        triangles (NDArray[np.uint]): Triangles of a multimaterial mesh.
        vertices (NDArray[np.float64]): Vertices of a 3D tesselation (geometry).
        tetrahedrons (NDArray[np.intc]): Tetrahedrons of the same 3D tesselation (topology).
        nodes_linked (NDArray[np.uint]): Tetrahedrons id on the tesselation linked to each triangles of the mesh.

    Returns:
        NDArray[np.uint]: reoriented triangles.
    """
    # Thumb rule for all the faces

    normals = _compute_normal_faces(vertices, triangles)

    points = vertices[triangles]
    centroids_faces = np.mean(points, axis=1)  # center of tirangles
    centroids_nodes = np.mean(
        vertices[tetrahedrons[nodes_linked[:, 0]]],
        axis=1,
    )  # center of "first" adjacent tetrahedron in Tesselation Graph

    vectors = centroids_nodes - centroids_faces

    dot_product = np.sum(np.multiply(vectors, normals), axis=1)
    normals_sign = np.sign(dot_product)

    # Reorientation according to the normal sign
    reoriented_triangles = triangles.copy()

    # Matthieu Perez: one liner is quicker when there's more than 100 faces (ie. always)
    reoriented_triangles[normals_sign > 0] = reoriented_triangles[normals_sign > 0][:, [0, 2, 1]]
    return reoriented_triangles


def _compute_normal_faces(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> NDArray[np.float64]:
    """Return the normalized normals for each triangles."""
    positions = points[triangles]
    sides_1 = positions[:, 1] - positions[:, 0]
    sides_2 = positions[:, 2] - positions[:, 1]
    normals = np.cross(sides_1, sides_2, axis=1)
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.array([norms] * 3).transpose()
    return normals


def filter_unused_points(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong]]:
    """Take a mesh made from points and triangles and remove points not indexed in triangles. Re-index triangles.

    Return the filtered points and reindexed triangles.
    """
    used_points_id = np.unique(triangles)
    used_points = np.copy(points[used_points_id])
    idx_mapping = np.arange(len(used_points))
    mapping = dict(zip(used_points_id, idx_mapping, strict=True))

    reindexed_triangles = np.fromiter(
        (mapping[xi] for xi in triangles.reshape(-1)),
        dtype=np.ulonglong,
        count=3 * len(triangles),
    ).reshape((-1, 3))

    return (used_points, reindexed_triangles)
