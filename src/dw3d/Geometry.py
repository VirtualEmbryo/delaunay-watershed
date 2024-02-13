# Sacha Ichbiah, Sept 2021
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from dw3d.dcel import DcelData


def find_key_multiplier(num_points):
    key_multiplier = 1
    while num_points // key_multiplier != 0:
        key_multiplier *= 10
    return key_multiplier


## LENGTHS AND DERIVATIVES


def compute_area_derivative_autodiff(
    mesh: "DcelData", device: str = "cpu"
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute dict that maps interface (label1, label2) to array of change of area per point."""
    # Faces_membranes = extract_faces_membranes(Mesh)
    key_mult = np.amax(mesh.f[:, 3:]) + 1
    keys = mesh.f[:, 3] + key_mult * mesh.f[:, 4]
    faces_membranes = {}
    for key in np.unique(keys):
        tup = (key % key_mult, key // key_mult)
        faces_membranes[tup] = mesh.f[:, :3][np.arange(len(keys))[keys == key]]

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    areas_derivatives = {}
    for tup in sorted(faces_membranes.keys()):
        loss_area = (Compute_Area_Faces_torch(verts, torch.tensor(faces_membranes[tup]))).sum()
        loss_area.backward()
        areas_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return areas_derivatives


def compute_volume_derivative_autodiff_dict(mesh: "DcelData", device: str = "cpu") -> dict[int, NDArray[np.float64]]:
    """Compute map cell number -> derivative of volume wrt to each point."""
    # Faces_manifolds = extract_faces_manifolds(Mesh)
    faces_manifolds = {key: [] for key in mesh.materials}
    for face in mesh.f:
        a, b, c, m1, m2 = face
        faces_manifolds[m1].append([a, b, c])
        faces_manifolds[m2].append([a, c, b])

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    volumes_derivatives = {}
    for key in mesh.materials:  # 1:] :
        faces = faces_manifolds[key]
        assert len(faces) > 0
        loss_volume = -Compute_Volume_manifold_torch(verts, torch.tensor(faces))
        loss_volume.backward()
        volumes_derivatives[key] = verts.grad.numpy().copy()
        optimizer.zero_grad()

    return volumes_derivatives


def compute_length_derivative_autodiff(
    mesh: "DcelData", device: str = "cpu"
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute map trijunction edge (V1, V2) -> change of length wrt to points."""
    edges_trijunctions = extract_edges_trijunctions(mesh)

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    length_derivatives: dict[tuple[int, int], NDArray[np.float64]] = {}
    for tup in sorted(edges_trijunctions.keys()):
        loss_length = (Compute_length_edges_trijunctions_torch(verts, torch.tensor(edges_trijunctions[tup]))).sum()
        loss_length.backward()
        length_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return length_derivatives


def Compute_length_edges_trijunctions_torch(verts, Edges_trijunctions):
    Pos = verts[Edges_trijunctions]
    Lengths = torch.norm(Pos[:, 0] - Pos[:, 1], dim=1)
    return Lengths


def compute_length_trijunctions(Mesh, prints=False):
    Length_trijunctions = {}
    Edges_trijunctions = extract_edges_trijunctions(Mesh, prints)
    for key in Edges_trijunctions.keys():
        Length_trijunctions[key] = np.sum(Compute_length_edges_trijunctions(Mesh.v, Edges_trijunctions[key]))
    return Length_trijunctions


def Compute_length_edges_trijunctions(verts, Edges_trijunctions):
    Pos = verts[Edges_trijunctions]
    Lengths = np.linalg.norm(Pos[:, 0] - Pos[:, 1], axis=1)
    return Lengths


"""
    F = Mesh.f
    E = np.vstack((F[:,[0,1]],F[:,[0,2]],F[:,[1,2]]))
    E = np.sort(E,axis=1)
    Zones = np.vstack((F[:,[3,4]],F[:,[3,4]],F[:,[3,4]]))
    key_mult = find_key_multiplier(len(Mesh.v)+1)
    K = (E[:,0]+1) + (E[:,1]+1)*key_mult
    Array,Index_first_occurence,Index_inverse,Index_counts = np.unique(K, return_index=True, return_inverse=True, return_counts=True)
    if prints :
        print("Number of trijunctional edges :",np.sum(Index_counts==3))
    Edges_trijunctions = E[Index_first_occurence[Index_counts==3]]


    Indices = np.arange(len(Index_counts))
    Map = {key:[] for key in Indices[Index_counts==3]}
    Table = np.zeros(len(Index_counts))
    Table[Index_counts==3]+=1

    for i in range(len(Index_inverse)):
        inverse = Index_inverse[i]
        if Table[inverse]==1 :
            Map[inverse].append(i)

    Trijunctional_line={}
    for key in sorted(Map.keys()) :
        x = Map[key]
        regions = np.hstack((Zones[x[0]],Zones[x[1]],Zones[x[2]]))
        u = np.unique(regions)
        if len(u)>4 :
            print("oui")
            continue
        else :
            Trijunctional_line[tuple(u)]=Trijunctional_line.get(tuple(u),[])
            Trijunctional_line[tuple(u)].append(E[x[0]])
            assert(E[x[0]][0]==E[x[1]][0]==E[x[2]][0] and E[x[0]][1]==E[x[1]][1]==E[x[2]][1])

    Output_dict = {}
    for key in sorted(Trijunctional_line.keys()):
        Output_dict[key] = np.vstack(Trijunctional_line[key])
    return(Output_dict)

"""


def extract_edges_trijunctions(Mesh, prints=False):
    F = Mesh.f
    E = np.vstack((F[:, [0, 1]], F[:, [0, 2]], F[:, [1, 2]]))
    E = np.sort(E, axis=1)
    Zones = np.vstack((F[:, [3, 4]], F[:, [3, 4]], F[:, [3, 4]]))
    key_mult = find_key_multiplier(len(Mesh.v) + 1)
    K = (E[:, 0] + 1) + (E[:, 1] + 1) * key_mult
    Array, Index_first_occurence, Index_inverse, Index_counts = np.unique(
        K,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if prints:
        print("Number of trijunctional edges :", np.sum(Index_counts == 3))
    Edges_trijunctions = E[Index_first_occurence[Index_counts == 3]]

    Indices = np.arange(len(Index_counts))
    Map = {key: [] for key in Indices[Index_counts == 3]}
    Table = np.zeros(len(Index_counts))
    Table[Index_counts == 3] += 1

    for i in range(len(Index_inverse)):
        inverse = Index_inverse[i]
        if Table[inverse] == 1:
            Map[inverse].append(i)

    Trijunctional_line = {}
    for key in sorted(Map.keys()):
        x = Map[key]
        regions = np.hstack((Zones[x[0]], Zones[x[1]], Zones[x[2]]))
        u = np.unique(regions)
        if len(u) > 4:
            print("oui")
            continue
        else:
            Trijunctional_line[tuple(u)] = Trijunctional_line.get(tuple(u), [])
            Trijunctional_line[tuple(u)].append(E[x[0]])
            assert E[x[0]][0] == E[x[1]][0] == E[x[2]][0] and E[x[0]][1] == E[x[1]][1] == E[x[2]][1]

    Output_dict = {}
    for key in sorted(Trijunctional_line.keys()):
        Output_dict[key] = np.vstack(Trijunctional_line[key])
    return Output_dict


## AREAS AND DERIVATIVES


def Compute_Area_Faces(Verts, Faces):
    Pos = Verts[Faces]
    Sides = Pos - Pos[:, [2, 0, 1]]

    Lengths_sides = np.norm(Sides, dim=2)
    Half_perimeters = np.sum(Lengths_sides, axis=1) / 2
    Diffs = np.zeros(Lengths_sides.shape)
    Diffs[:, 0] = Half_perimeters - Lengths_sides[:, 0]
    Diffs[:, 1] = Half_perimeters - Lengths_sides[:, 1]
    Diffs[:, 2] = Half_perimeters - Lengths_sides[:, 2]
    Areas = (Half_perimeters * Diffs[:, 0] * Diffs[:, 1] * Diffs[:, 2]) ** (0.5)
    return Areas


def Compute_Area_Faces_torch(Verts, Faces):
    Pos = Verts[Faces]
    Sides = Pos - Pos[:, [2, 0, 1]]

    Lengths_sides = torch.norm(Sides, dim=2)
    Half_perimeters = torch.sum(Lengths_sides, axis=1) / 2
    Diffs = torch.zeros(Lengths_sides.shape)
    Diffs[:, 0] = Half_perimeters - Lengths_sides[:, 0]
    Diffs[:, 1] = Half_perimeters - Lengths_sides[:, 1]
    Diffs[:, 2] = Half_perimeters - Lengths_sides[:, 2]
    Areas = (Half_perimeters * Diffs[:, 0] * Diffs[:, 1] * Diffs[:, 2]) ** (0.5)
    return Areas


def compute_areas_faces(Mesh):
    Pos = Mesh.v[Mesh.f[:, [0, 1, 2]]]
    Sides = Pos - Pos[:, [2, 0, 1]]
    Lengths_sides = np.linalg.norm(Sides, axis=2)
    Half_perimeters = np.sum(Lengths_sides, axis=1) / 2

    Diffs = np.array([Half_perimeters] * 3).transpose() - Lengths_sides
    Areas = (Half_perimeters * Diffs[:, 0] * Diffs[:, 1] * Diffs[:, 2]) ** (0.5)
    for i, face in enumerate(Mesh.faces):
        face.area = Areas[i]


def compute_areas_cells(Mesh):
    areas = {key: 0 for key in Mesh.materials}
    for face in Mesh.faces:
        areas[face.material_1] += face.area
        areas[face.material_2] += face.area
    return areas


def compute_areas_interfaces(mesh: "DcelData") -> dict[tuple[int, int], float]:
    """Compute area of every interface (label1, label2) in mesh."""
    interfaces = {}
    for face in mesh.faces:
        materials = (face.material_1, face.material_2)
        key = (min(materials), max(materials))
        interfaces[key] = interfaces.get(key, 0) + face.area
    return interfaces


def compute_area_derivative_dict(mesh: "DcelData") -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute dict that maps interface (label1, label2) to array of change of area per point."""
    interfaces_keys: NDArray[np.int64] = np.array(sorted(compute_areas_interfaces(mesh).keys()))
    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    area_derivatives: dict[tuple[int, int], NDArray[np.float64]] = {
        tuple(t): np.zeros((len(points), 3)) for t in interfaces_keys
    }

    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    e3 = np.cross(z - y, x - z) / (np.linalg.norm(np.cross(z - y, x - z), axis=1).reshape(-1, 1))
    cross_e3_x = np.cross(e3, z - y) / 2
    cross_e3_y = np.cross(e3, x - z) / 2
    cross_e3_z = np.cross(e3, y - x) / 2

    list_indices_faces_per_vertices_x = {key: [[] for i in range(len(points))] for key in area_derivatives}
    list_indices_faces_per_vertices_y = {key: [[] for i in range(len(points))] for key in area_derivatives}
    list_indices_faces_per_vertices_z = {key: [[] for i in range(len(points))] for key in area_derivatives}

    for i in range(len(triangles)):
        i_x, i_y, i_z = triangles[i]
        a, b = labels[i]
        list_indices_faces_per_vertices_x[(a, b)][i_x].append(i)
        list_indices_faces_per_vertices_y[(a, b)][i_y].append(i)
        list_indices_faces_per_vertices_z[(a, b)][i_z].append(i)

    for key in tqdm(area_derivatives.keys()):
        for iv in range(len(points)):
            area_derivatives[key][iv] = np.vstack(
                (
                    cross_e3_x[list_indices_faces_per_vertices_x[key][iv]],
                    cross_e3_y[list_indices_faces_per_vertices_y[key][iv]],
                    cross_e3_z[list_indices_faces_per_vertices_z[key][iv]],
                ),
            ).sum(axis=0)

    return area_derivatives


##VOLUMES AND DERIVATIVES


def Compute_Volume_manifold(Verts, Faces):
    Coords = Verts[Faces]
    cross_prods = np.cross(Coords[:, 1], Coords[:, 2], axis=1)
    dots = np.sum(cross_prods * Coords[:, 0], axis=1)
    Vol = -np.sum(dots) / 6
    return Vol


def Compute_Volume_manifold_torch(Verts, Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:, 1], Coords[:, 2], axis=1)
    dots = torch.sum(cross_prods * Coords[:, 0], axis=1)
    Vol = -torch.sum(dots) / 6
    return Vol


def Compute_Volume_manifold_sequential(Verts, Faces):
    Volume = np.zeros(len(Faces))
    for i, face in enumerate(Faces):
        index = Faces[i, [0, 1, 2]]
        Coords = Verts[index]
        inc = np.linalg.det(Coords)
        Volume[i] -= inc
    Volume /= 6
    return np.sum(Volume)


def compute_volume_cells(mesh: "DcelData") -> dict[int, float]:
    """Compute map cell number -> volume."""
    volumes: dict[int, float] = {m: 0 for m in mesh.materials}
    for i, face in enumerate(mesh.faces):
        index = mesh.f[i, [0, 1, 2]]
        coords = mesh.v[index]
        inc = np.linalg.det(coords)
        volumes[face.material_1] += inc
        volumes[face.material_2] -= inc

    for key in volumes:
        volumes[key] = volumes[key] / 6
    return volumes


def compute_volume_derivative_dict(mesh: "DcelData") -> dict[int, NDArray[np.float64]]:
    """Compute map cell number -> derivative of volume wrt to each point."""
    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    materials = mesh.materials

    volumes_derivatives: dict[int, NDArray[np.float64]] = {key: np.zeros((len(points), 3)) for key in materials}

    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cross_xy = np.cross(x, y) / 6
    cross_yz = np.cross(y, z) / 6
    cross_zx = np.cross(z, x) / 6
    faces_material = {mat: np.zeros(len(triangles)) for mat in materials}
    for n in materials:
        faces_material[n][labels[:, 0] == 1] = 1
        faces_material[n][labels[:, 1] == 1] = -1

    list_indices_faces_per_vertices_pos_x = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_pos_y = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_pos_z = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_neg_x = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_neg_y = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_neg_z = {key: [[] for i in range(len(points))] for key in materials}

    for i in range(len(triangles)):
        i_x, i_y, i_z = triangles[i]
        a, b = labels[i]
        # print(i_x,i_y,i_z)
        list_indices_faces_per_vertices_pos_x[a][i_x].append(i)
        list_indices_faces_per_vertices_pos_y[a][i_y].append(i)
        list_indices_faces_per_vertices_pos_z[a][i_z].append(i)

        list_indices_faces_per_vertices_neg_x[b][i_x].append(i)
        list_indices_faces_per_vertices_neg_y[b][i_y].append(i)
        list_indices_faces_per_vertices_neg_z[b][i_z].append(i)

    for n in tqdm(materials):
        for iv in range(len(points)):
            volumes_derivatives[n][iv] = np.vstack(
                (
                    cross_yz[list_indices_faces_per_vertices_pos_x[n][iv]],
                    cross_zx[list_indices_faces_per_vertices_pos_y[n][iv]],
                    cross_xy[list_indices_faces_per_vertices_pos_z[n][iv]],
                    -cross_yz[list_indices_faces_per_vertices_neg_x[n][iv]],
                    -cross_zx[list_indices_faces_per_vertices_neg_y[n][iv]],
                    -cross_xy[list_indices_faces_per_vertices_neg_z[n][iv]],
                ),
            ).sum(axis=0)

    return volumes_derivatives


##ANGLES


def compute_angles_tri(  # noqa: C901
    mesh: "DcelData",
    unique: bool = True,
) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int, int], float], dict[tuple[int, int, int], float]]:
    """Compute three maps trijunction (id reg 1, id reg 2, id reg 3) to mean angle, mean angle (deg), length."""
    ##We compute the angles at each trijunctions. If we fall onto a quadrijunction, we skip it

    dict_length: dict[tuple[int, int, int], float] = {}
    dict_angles: dict[tuple[int, int, int], list[float]] = {}
    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            face = edge.incident_face
            faces = [face]
            sources = [edge.origin.key - edge.destination.key]
            normals = [face.normal]
            materials = [[face.material_1, face.material_2]]

            for neighbor in edge.twin:
                face_attached = mesh.half_edges[neighbor].incident_face
                faces.append(face_attached)
                sources.append(mesh.half_edges[neighbor].origin.key - mesh.half_edges[neighbor].destination.key)
                materials.append([face_attached.material_1, face_attached.material_2])
                normals.append(face_attached.normal)

            regions_id = np.array(materials)
            if len(regions_id) != 3:
                continue
                ## If we fall onto a quadrijunction, we skip it.

            normals = np.array(normals).copy()

            if regions_id[0, 0] == regions_id[1, 0] or regions_id[0, 1] == regions_id[1, 1]:
                regions_id[1] = regions_id[1][[1, 0]]
                normals[1] *= -1

            if regions_id[0, 0] == regions_id[2, 0] or regions_id[0, 1] == regions_id[2, 1]:
                regions_id[2] = regions_id[2][[1, 0]]
                normals[2] *= -1

            pairs = [[0, 1], [1, 2], [2, 0]]

            for pair in pairs:
                i1, i2 = pair
                # if np.isnan(np.arccos(np.dot(normals[i1],normals[i2]))) :
                #    print("Isnan")
                # if np.dot(normals[i1],normals[i2])>1 or np.dot(normals[i1],normals[i2])<-1 :
                #    print("Alert",np.dot(normals[i1],normals[i2]))
                angle = np.arccos(np.clip(np.dot(normals[i1], normals[i2]), -1, 1))

                if regions_id[i1][1] == regions_id[i2][0]:
                    e, f, g = regions_id[i1][0], regions_id[i1][1], regions_id[i2][1]

                elif regions_id[i1][0] == regions_id[i2][1]:
                    e, f, g = regions_id[i2][0], regions_id[i2][1], regions_id[i1][1]

                dict_angles[(min(e, g), f, max(e, g))] = dict_angles.get((min(e, g), f, max(e, g)), [])
                dict_angles[(min(e, g), f, max(e, g))].append(angle)
                dict_length[(min(e, g), f, max(e, g))] = dict_length.get((min(e, g), f, max(e, g)), 0)
                dict_length[(min(e, g), f, max(e, g))] += edge.length
                if not unique:
                    dict_angles[(min(e, g), f, max(e, g))] = dict_angles.get((min(e, g), f, max(e, g)), [])
                    dict_angles[(min(e, g), f, max(e, g))].append(angle)
                    dict_length[(min(e, g), f, max(e, g))] = dict_length.get((min(e, g), f, max(e, g)), 0)
                    dict_length[(min(e, g), f, max(e, g))] += edge.length

    dict_mean_angles: dict[tuple[int, int, int], float] = {}
    dict_mean_angles_deg: dict[tuple[int, int, int], float] = {}
    for key in dict_angles:
        dict_mean_angles[key] = np.mean(dict_angles[key])
        dict_mean_angles_deg[key] = np.mean(dict_mean_angles[key] * 180 / np.pi)

    return (dict_mean_angles, dict_mean_angles_deg, dict_length)
