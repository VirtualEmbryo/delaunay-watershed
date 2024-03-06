"""Module for Mesh writing, mesh cleaning, mesh plotting.

Sacha Ichbiah 2021.
Matthieu Perez 2024.
"""
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polyscope as ps
from numpy.typing import NDArray
from scipy.spatial import ckdtree

if TYPE_CHECKING:
    from dw3d.functions import GeometryReconstruction3D
    from dw3d.graph_functions import DelaunayGraph
#####
#####
# I/O TOOLS
#####
#####


def separate_faces_dict(triangles_and_labels: NDArray[np.uint]) -> dict[int, NDArray[np.uint]]:
    """Construct a dictionnary that maps a region id to the array of triangles forming this region."""
    nb_regions = np.amax(triangles_and_labels[:, [3, 4]]) + 1

    occupancy = np.zeros(nb_regions, dtype=np.int64)
    triangles_of_region: dict[int, list[int]] = {}
    for face in triangles_and_labels:
        triangle = face[:3]
        region1, region2 = face[3:]
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
    graph: "DelaunayGraph",
    map_label_to_nodes: dict[int, list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.uint], list[int], NDArray[np.uint]]:
    """Extract multi-material mesh from the Delaunay graph with every nodes (tetrahedrons) marked with a material.

    The extracted surface mesh is composed of all triangles that are faces of 2 tetrahedrons with different materials.
    Note that there will be a lot of unused points in this mesh. A filtering step will be necessary.

    Args:
        graph (Delaunay_Graph): the Delaunay graph.
        map_label_to_nodes (dict[int, list[int]]): The map that links materials to tetrahedrons id.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.uint], list[int], NDArray[np.uint]]:
            - an array of vertices of the mesh. Note that there are some unused points, to be filtered later.
            - an array of triangles and labels (id p1, id p2, id p3, material 1, material 2)
            - the list of ids of faces in graph.faces that made it to the array of triangles and labels.
            - the list of ids of nodes (tetrahedrons) in graph.nodes that are linked to selected triangles.
    """
    # faces = triangles ; nodes = tetras, with map linking regions to list of tetra/node ids
    map_nodes_to_labels: dict[int, int] = {}
    for key in map_label_to_nodes:
        for node_idx in map_label_to_nodes[key]:
            map_nodes_to_labels[node_idx] = key
    faces: list[list[int]] = []  # p1, p2, p3, l1, l2 of selected faces
    faces_idx: list[int] = []  # list ids of triangles from selected graph faces.
    nodes_linked_by_face: list[list[int]] = []  # list nodes ids linked to triangles from selected graph faces

    for idx, face in enumerate(graph.triangle_faces):
        # faces are triangles, nodes linked are the 2 adjacent tetrahedrons (nodes).
        nodes_linked = graph.nodes_linked_by_faces[idx]

        # corresponding regions
        cluster_1 = map_nodes_to_labels[nodes_linked[0]]
        cluster_2 = map_nodes_to_labels[nodes_linked[1]]
        cells = [cluster_1, cluster_2]

        if cluster_1 != cluster_2:
            # some faces belong to 2 tetras of two different regions : those are the triangles in final mesh !
            faces.append([face[0], face[1], face[2], cells[0], cells[1]])  # tri p1, p2, p3, label l1, l2
            faces_idx.append(idx)  # tri id
            nodes_linked_by_face.append(nodes_linked)  # 2 tetras

    # Matthieu Perez:
    # Apparently, there are triangles on only one tetra in the delaunay graph,
    # and sometimes they might belong to the mesh ? If segmented cell touches the border of the image ?
    for idx in range(len(graph.lone_faces)):
        face = graph.lone_faces[idx]
        node_linked = graph.nodes_linked_by_lone_faces[idx]
        cluster_1 = map_nodes_to_labels[node_linked]
        # We incorporate all these edges because they are border edges
        if cluster_1 != 0:
            cells = [0, cluster_1]
            faces.append([face[0], face[1], face[2], cells[0], cells[1]])
            faces_idx.append(idx)
            nodes_linked_by_face.append(nodes_linked)

    # Note that the extraction might lead to a mesh that is non-manifold where it is not expected to,
    # if the labeling of nodes is not perfect. Some kind of "mesh surgery" might be necessary to improve
    # extracted mesh quality.
    return (graph.vertices, np.array(faces, dtype=np.uint), faces_idx, np.array(nodes_linked_by_face, dtype=np.uint))


def clean_mesh_from_seg(
    geometry_reconstruction: "GeometryReconstruction3D",
) -> tuple[NDArray[np.float64], NDArray[np.uint]]:
    """Extract points and triangles_and_labels from a GeometryReconstruction3D object."""
    # Take a Segmentation class as entry

    (
        points,
        triangles_and_labels,
        _,
        nodes_idx_in_graph_linked_to_triangle,
    ) = retrieve_mesh_multimaterial_multitracker_format(
        geometry_reconstruction.delaunay_graph,
        geometry_reconstruction.map_label_to_nodes_ids,
    )
    vertices = points.copy()

    for i, f in enumerate(triangles_and_labels):
        if f[3] > f[4]:  # if label0 > label1 we swap them
            triangles_and_labels[i] = triangles_and_labels[
                i,
                [0, 1, 2, 4, 3],
            ]
            nodes_idx_in_graph_linked_to_triangle[i] = nodes_idx_in_graph_linked_to_triangle[i][[1, 0]]

    triangles_and_labels = reorient_faces(
        triangles_and_labels,
        geometry_reconstruction,
        nodes_idx_in_graph_linked_to_triangle,
    )

    # Automatic swap of all faces after reorientation ? I guess it's not the good norm
    for i in range(len(triangles_and_labels)):
        triangles_and_labels[i] = triangles_and_labels[i, [0, 2, 1, 3, 4]]

    return (vertices, triangles_and_labels)


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


def reorient_faces(
    triangles_and_labels: NDArray[np.uint],
    geometry_reconstruction: "GeometryReconstruction3D",
    nodes_linked: NDArray[np.uint],
) -> NDArray[np.uint]:
    """Swap point order in triangles such that all normals points in the same direction."""
    # Thumb rule for all the faces

    normals = compute_normal_faces(geometry_reconstruction.delaunay_graph.vertices, triangles_and_labels[:, :3])

    points = geometry_reconstruction.delaunay_graph.vertices[triangles_and_labels[:, :3]]
    centroids_faces = np.mean(points, axis=1)  # center of tirangles
    centroids_nodes = np.mean(
        geometry_reconstruction.delaunay_graph.vertices[
            geometry_reconstruction.delaunay_graph.tetrahedrons[nodes_linked[:, 0]]
        ],
        axis=1,
    )  # center of "first" adjacent tetrahedron in Delaunay Graph

    vectors = centroids_nodes - centroids_faces
    # Matthieu Perez : not necessary to normalize vectors
    # norms = np.linalg.norm(vectors, axis=1)
    # vectors[:, 0] /= norms
    # vectors[:, 1] /= norms
    # vectors[:, 2] /= norms

    dot_product = np.sum(np.multiply(vectors, normals), axis=1)
    normals_sign = np.sign(dot_product)

    # Reorientation according to the normal sign
    reoriented_faces = triangles_and_labels.copy()
    # for i, s in enumerate(normals_sign):
    #     if s < 0:
    #         reoriented_faces[i] = reoriented_faces[i][[0, 2, 1, 3, 4]]

    # Matthieu Perez: one liner is quicker when there's more than 100 faces (ie. always)
    reoriented_faces[normals_sign < 0] = reoriented_faces[normals_sign < 0][:, [0, 2, 1, 3, 4]]
    return reoriented_faces


#####
#####
# MESH PLOTTING
#####
#####


def retrieve_border_tetra_with_index_map(
    graph: "DelaunayGraph",
    map_label_to_nodes: dict[int, list[int]],
) -> list[list[list[int]]]:
    """Give a list that maps region number to list of triangles."""
    map_nodes_to_labels = {}
    for key in map_label_to_nodes:
        for node_idx in map_label_to_nodes[key]:
            map_nodes_to_labels[node_idx] = key

    clusters = [[] for _ in range(len(map_label_to_nodes))]
    # for _ in range(len(map_label_to_nodes)):
    #     clusters.append([])

    for idx in range(len(graph.triangle_faces)):
        nodes_linked = graph.nodes_linked_by_faces[idx]

        cluster_1 = map_nodes_to_labels.get(nodes_linked[0], -1)
        cluster_2 = map_nodes_to_labels.get(nodes_linked[1], -2)
        # if the two nodes of the edges belong to the same cluster we ignore them
        # otherwise we add them to the mesh
        if cluster_1 != cluster_2:
            face = list(graph.triangle_faces[idx])
            if cluster_1 >= 0:
                clusters[cluster_1].append(face)
            if cluster_2 >= 0:
                clusters[cluster_2].append(face)

    for idx in range(len(graph.lone_faces)):
        edge = graph.lone_faces[idx]
        node_linked = graph.nodes_linked_by_lone_faces[idx]
        cluster_1 = map_nodes_to_labels[node_linked]
        # We incorporate all these edges because they are border edges
        if cluster_1 != 0:
            v1, v2, v3 = edge[0], edge[1], edge[2]
            clusters[cluster_1].append([v1, v2, v3])
    return clusters


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
