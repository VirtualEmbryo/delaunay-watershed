"""Module for Mesh writing, mesh cleaning, mesh plotting.

Sacha Ichbiah 2021.
Matthieu Perez 2024.
"""
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ckdtree

if TYPE_CHECKING:
    from dw3d.graph_functions import TesselationGraph
#####
#####
# I/O TOOLS
#####
#####


def separate_faces_dict(triangles: NDArray[np.uint], labels: NDArray[np.uint]) -> dict[int, NDArray[np.uint]]:
    """Construct a dictionnary that maps a region id to the array of triangles forming this region."""
    nb_regions = np.amax(labels) + 1

    occupancy = np.zeros(nb_regions, dtype=np.int64)
    triangles_of_region: dict[int, list[int]] = {}
    for triangle, label in zip(triangles, labels, strict=True):
        region1, region2 = label
        if region1 >= 0:
            if occupancy[region1] == 0:
                triangles_of_region[region1] = [triangle]
                occupancy[region1] += 1
            else:
                triangles_of_region[region1].append(triangle)

        if region2 >= 0:
            if occupancy[region2] == 0:
                triangles_of_region[region2] = [triangle]
                occupancy[region2] += 1
            else:
                triangles_of_region[region2].append(triangle)

    faces_separated: dict[int, NDArray[np.uint]] = {}
    for i in sorted(triangles_of_region.keys()):
        faces_separated[i] = np.array(triangles_of_region[i])

    return faces_separated


def write_mesh_bin(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles_and_labels: NDArray[np.ulonglong],
) -> None:
    """Save bin .rec mesh."""
    assert len(triangles_and_labels[0]) == 5
    assert len(points[0]) == 3
    strfile = struct.pack("Q", len(points))
    strfile += points.flatten().astype(np.float64).tobytes()
    strfile += struct.pack("Q", len(triangles_and_labels))
    dt = np.dtype([("triangles", np.uint64, (3,)), ("labels", np.int32, (2,))])
    triangles = triangles_and_labels[:, :3].astype(np.uint64)
    labels = triangles_and_labels[:, 3:].astype(np.int32)

    def func(i: int) -> tuple[int, int]:
        return (triangles[i], labels[i])

    t = np.array(list(map(func, np.arange(len(triangles_and_labels)))), dtype=dt)
    strfile += t.tobytes()
    with Path(filename).open("wb") as file:
        file.write(strfile)


def write_mesh_text(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles_and_labels: NDArray[np.ulonglong],
) -> None:
    """Save text .rec mesh."""
    with Path(filename).open("w") as file:
        file.write(str(len(points)) + "\n")
        for i in range(len(points)):
            file.write(f"{points[i][0]:.5f} {points[i][1]:.5f} {points[i][2]:.5f}" + "\n")
        file.write(str(len(triangles_and_labels)) + "\n")
        for i in range(len(triangles_and_labels)):
            content = f"{triangles_and_labels[i][0]} {triangles_and_labels[i][1]} {triangles_and_labels[i][2]} "
            content += f"{triangles_and_labels[i][3]} {triangles_and_labels[i][4]}\n"
            file.write(content)


#####
#####
# Mesh cleaning
#####
#####


def retrieve_mesh_multimaterial_multitracker_format(
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
    ) = retrieve_mesh_multimaterial_multitracker_format(
        tesselation_graph,
        map_label_to_nodes_ids,
    )
    vertices = points.copy()

    for i, label in enumerate(labels):
        if label[0] > label[1]:  # if label0 > label1 we swap them
            labels[i] = labels[i, [1, 0]]
            nodes_idx_in_graph_linked_to_triangle[i] = nodes_idx_in_graph_linked_to_triangle[i][[1, 0]]

    triangles = reorient_triangles(
        triangles,
        tesselation_graph.vertices,
        tesselation_graph.tetrahedrons,
        nodes_idx_in_graph_linked_to_triangle,
    )

    return (vertices, triangles, labels)


def compute_normal_faces(
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


def reorient_triangles(
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

    normals = compute_normal_faces(vertices, triangles)

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


#####
#####
# MESH PLOTTING
#####
#####


def compute_seeds_idx_from_voxel_coords(
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


def renormalize_verts(
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
