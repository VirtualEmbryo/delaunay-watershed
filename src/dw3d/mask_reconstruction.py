"""Segmented image creation from generic surface mesh.

Sacha Ichbiah, 2021.
Matthieu Perez, 2024.
"""
# from Mesh_centering import center_verts
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
from trimesh import remesh


def _create_coords(nx: int, ny: int, nz: int) -> NDArray[np.float64]:
    """Create the grid coords of the pixels."""
    xv = np.linspace(0, 1, nx)
    yv = np.linspace(0, 1, ny)
    zv = np.linspace(0, 1, nz)
    xvv, yvv, zvv = np.meshgrid(xv, yv, zv)
    xvv = np.transpose(xvv, (1, 0, 2)).flatten()
    yvv = np.transpose(yvv, (1, 0, 2)).flatten()
    zvv = zvv.flatten()
    return np.vstack([xvv, yvv, zvv]).transpose()


def _create_mesh_semantic_masks(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
    image_shape: NDArray[np.uint],
) -> NDArray[np.float64]:
    """Create segmentation mask of the mesh membrane, not the interior."""
    verts, faces = points.copy()[:, [2, 1, 0]], triangles.copy()
    for i in range(3):
        verts[:, i] /= image_shape[2 - i]

    dmax = 1 / np.amax(image_shape)
    # print("start of the subidivision")
    verts, faces = _subdivide_mesh(verts, faces, max_edge=dmax / 2)
    # print("subdivision finished")
    membrane = _make_mask(verts, image_shape)
    return membrane


def _subdivide_mesh(
    verts: NDArray[np.float64],
    faces: NDArray[np.ulonglong],
    max_edge: float,
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong]]:
    """Subdivide the triangles of the mesh until every edge is smaller than given threshold."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1).max()
    max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0) * 2
    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    verts, faces = remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter)
    return (verts, faces)


def _make_mask(verts: NDArray[np.float64], image_shape: NDArray[np.uint]) -> NDArray[np.float64]:
    """Mark the pixels touching a vertex of the subdivided mesh."""
    nz, ny, nx = image_shape
    # nx,ny,nz = grid_size
    points = _create_coords(nx, ny, nz)
    tree = cKDTree(points)
    distances = tree.query(verts)

    _, idx = distances
    membrane = np.zeros(nx * ny * nz)
    membrane[idx] = 1
    membrane = membrane.reshape(nx, ny, nz)
    return membrane


def _create_mesh_instance_masks(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
    image_shape: NDArray[np.uint],
    seeds: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Reconstruct a segmentation image from vertices, faces, seeds coords and desired image shape."""
    semantic_mask = _create_mesh_semantic_masks(points, triangles, image_shape).transpose(2, 1, 0)
    distance = ndi.distance_transform_edt(1 - semantic_mask)
    markers = np.zeros(distance.shape, dtype=np.int_)
    for i in range(len(seeds)):
        markers[tuple(seeds[i].T)] = i + 1
    labels = watershed(-distance, markers)
    # labels=labels[::-1]
    return labels


def reconstruct_mask_from_dict(dict_mask: dict[str]) -> NDArray[np.uint8]:
    """Reconstruct a segmentation image from a saved dict of points, triangles, seeds coords and desired image shape."""
    points = dict_mask["points"]
    triangles = dict_mask["triangles"]
    seeds = dict_mask["seeds"]
    image_shape = dict_mask["image_shape"]

    labels: NDArray[np.uint8] = _create_mesh_instance_masks(points, triangles, image_shape, seeds) - 1
    return labels


def reconstruct_mask_from_saved_file_dict(filename_dict: str) -> NDArray[np.uint8]:
    """Reconstruct a segmentation image from a saved dict of points, triangles, seeds coords and desired image shape."""
    dict_mask: dict[str, NDArray] = np.load(filename_dict, allow_pickle=True).item()
    return reconstruct_mask_from_dict(dict_mask)
