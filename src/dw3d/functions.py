from time import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import expand_labels

from dw3d.dcel import DcelData
from dw3d.geometric_utilities import build_triangulation, interpolate_image
from dw3d.graph_functions import Delaunay_Graph
from dw3d.mesh_utilities import (
    clean_mesh_from_seg,
    compute_seeds_idx_from_voxel_coords,
    plot_cells_polyscope,
    renormalize_verts,
    retrieve_border_tetra_with_index_map,
    separate_faces_dict,
    write_mesh_bin,
    write_mesh_text,
)
from dw3d.Networkx_functions import seeded_watershed_map


class GeometryReconstruction3D:
    def __init__(
        self,
        labels,
        min_dist=5,
        expansion_labels=0,
        original_image=None,
        print_info=False,
    ):
        self.original_image = original_image
        if expansion_labels > 0:
            self.labels = expand_labels(labels, expansion_labels)
        else:
            self.labels = labels

        self.seeds_coords, self.seeds_indices, self.tri, self.EDT = build_triangulation(
            self.labels,
            min_distance=min_dist,
            prints=print_info,
        )

        labels = interpolate_image(self.labels)
        edt = interpolate_image(self.EDT)
        self.Delaunay_Graph = Delaunay_Graph(self.tri, edt, labels, print_info=print_info)
        self.build_graph()

        # try Matthieu Perez: use centroid & labels to create Map_end instead of watershed ?
        # self.label_nodes()

        self.watershed_seeded(print_info=print_info)

    def label_nodes(self):
        centroids = self.Delaunay_Graph.compute_nodes_centroids() + 0.5
        labels = interpolate_image(self.labels)
        self.nodes_centroids = []
        pixels_centroids = np.round(centroids).astype(np.int64)
        xc = np.minimum(156, pixels_centroids[:, 0])
        yc = np.minimum(199, pixels_centroids[:, 1])
        zc = np.minimum(156, pixels_centroids[:, 2])
        node_to_region = self.labels[xc, yc, zc]
        # node_to_region = np.round(labels((centroids))).astype(np.int64)
        values = np.unique(node_to_region)
        self.Map_end = {}
        for value in values:
            self.Map_end[value] = np.argwhere(node_to_region == value).reshape(-1)
            self.nodes_centroids.append(centroids[self.Map_end[value]])
        print(self.Map_end)
        print(self.labels[22, 139, 54])
        print(self.labels[22, 139, 55])
        print(self.labels[22, 140, 54])
        print(self.labels[22, 140, 55])
        print(labels((22.5, 139.5, 54.25)))
        # print(np.argwhere(pixels_centroids == [23, 139, 54]))

    def build_graph(self):
        self.Nx_Graph = self.Delaunay_Graph.networkx_graph_weights_and_borders()

    def watershed_seeded(self, print_info=True):
        t1 = time()
        seeds_nodes = compute_seeds_idx_from_voxel_coords(
            self.EDT,
            self.Delaunay_Graph.compute_nodes_centroids(),
            self.seeds_coords,
        )
        zero_nodes = self.Delaunay_Graph.compute_zero_nodes()
        self.Map_end = seeded_watershed_map(self.Nx_Graph, seeds_nodes, self.seeds_indices, zero_nodes)

        t2 = time()
        if print_info:
            print("Watershed done in ", np.round(t2 - t1, 3))

    def retrieve_clusters(self):
        Clusters = retrieve_border_tetra_with_index_map(self.Delaunay_Graph, self.Map_end)
        return Clusters

    def return_dcel(self):
        V, F = self.return_mesh()
        Mesh = DcelData(V, F)
        return Mesh

    def return_mesh(self):
        return clean_mesh_from_seg(self)

    def plot_cells_polyscope(self, anisotropy_factor=1.0):
        Verts, Faces = self.return_mesh()
        Verts[:, 0] *= anisotropy_factor
        plot_cells_polyscope(Verts, Faces)

    def export_mesh(self, filename, mode="bin"):
        Verts, Faces = self.return_mesh()
        if mode == "txt":
            write_mesh_text(filename, Verts, Faces)
        elif mode == "bin":
            write_mesh_bin(filename, Verts, Faces)
        else:
            print("Please choose a valid format")

    def export_segmentation(self, filename):
        Verts, Faces = self.return_mesh()
        seeds = self.seeds_coords
        image_shape = np.array(self.labels.shape)
        Mesh_dict = {
            "Verts": Verts,
            "Faces": Faces,
            "seeds": seeds,
            "image_shape": image_shape,
        }
        np.save(filename, Mesh_dict)

    def plot_in_napari(self, add_mesh=True):
        import napari

        v = napari.view_image(self.labels, name="Labels")
        v.add_image(self.EDT, name="Distance Transform")
        if self.original_image is not None:
            v.add_image(self.original_image, name="Original Image")
        if not add_mesh:
            v.add_points(
                self.seeds_coords,
                name="Watershed seeds",
                n_dimensional=True,
                face_color=np.random.rand(len(self.seeds_coords), 3),
                size=10,
            )
        v.add_points(
            self.tri.points,
            name="triangulation_points",
            n_dimensional=False,
            face_color="red",
            size=1,
        )

        # colors = ["blue", "yellow", "green"]
        # for i in range(len(self.nodes_centroids)):
        #     print("centroids for label", i)
        #     v.add_points(
        #         self.nodes_centroids[i],
        #         name=f"centroids {i}",
        #         n_dimensional=False,
        #         face_color=colors[i],
        #         size=1,
        #         text=[
        #             str(self.nodes_centroids[i][k])
        #             for k in range(len(self.nodes_centroids[i]))
        #         ],
        #     )

        if add_mesh:
            Verts, Faces = self.return_mesh()
            Clusters = separate_faces_dict(Faces)
            maxkey = np.amax(Faces[:, 3:])
            All_verts = []
            All_faces = []
            All_labels = []
            offset = 0

            for key in sorted(list(Clusters.keys())):
                if key == 0:
                    continue
                faces = np.array(Clusters[key])

                vn, fn = renormalize_verts(Verts, faces)
                ln = np.ones(len(vn)) * key / maxkey

                All_verts.append(vn.copy())
                All_faces.append(fn.copy() + offset)
                All_labels.append(ln.copy())
                offset += len(vn)
            All_verts.append(np.array([np.mean(np.vstack(All_verts), axis=0)]))
            All_labels.append(np.array([0]))
            All_verts = np.vstack(All_verts)
            All_faces = np.vstack(All_faces)
            All_labels = np.hstack(All_labels)
            v.add_points(
                self.seeds_coords,
                name="Watershed seeds",
                n_dimensional=True,
                face_color=np.array(plt.cm.viridis(np.array(sorted(list(Clusters.keys()))) / maxkey))[:, :3],
                size=10,
            )

            v.add_surface((All_verts, All_faces, All_labels), colormap="viridis")

        return v
