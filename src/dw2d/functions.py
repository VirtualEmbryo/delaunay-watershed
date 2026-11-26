from dw2d.Networkx_functions import seeded_watershed_aggregation, seeded_watershed_map, apply_zero_and_seed_overrides
from dw2d.Plotting_tools import compute_seeds_idx_from_pixel_coords, plot_polylines_ax,retrieve_border_lines,plot_triangles_coloured_ax,plot_triangulation_ax
from dw2d.Graph_functions import Delaunay_Graph
from dw2d.Geometric_utilities import build_triangulation
from dw2d.Dcel import DCEL_Data, Clean_mesh, image_border_vertices
import warnings

from dw2d.io import save_rec
from dw2d.Mask_reconstruction import reconstruct_mask_from_dict
from dw2d.pixel_grid_interpolation import BilinearOnPixelGrid
from dw2d.junction_relocation import _relocate_on_edges
from dw2d.locus_projection import _project_on_edges
from dw2d.mesh_surgery import repair_labelled_triangulation
from dw2d.sliver_relabelling import relabel_near_tie_slivers

import numpy as np
from dw2d._matplotlib import pyplot as plt  # imported on first use: matplotlib is in the `viewing` extra
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels


class _Reconstruction2D():
    """The 2D reconstruction engine: dw2d 0.1.5's `geometry_reconstruction_2d`, with the options added for 0.8.0.

    It runs at construction. `dw2d.MeshReconstructionAlgorithm` builds one per mask and reads its mesh and reports;
    the deprecated `geometry_reconstruction_2d` is this class under its 0.1.5 name.
    """

    def __init__(self, labels,min_dist = 4 ,interpolation_points=10,expansion_labels = 0,original_image = None,
                 deterministic=True, relocate_junctions=True, project_interface_vertices=False,
                 repair_pinch_vertices=True, relabel_near_tie_slivers=None, print_info=False):
        """``print_info`` (default False): print the size of the triangulation's dual graph (printed always before
        0.8.0).

        ``deterministic`` (default True):
        the reconstruction is the same on every machine and NumPy version. Three things make it so:

        * Qhull's triangulation of the integer points is not unique where four or more of them are
          cocircular, and Qhull resolves those cells differently on different machines. Every
          such cell is re-triangulated by a fixed rule, a fan from a vertex chosen by SHA-256 of the
          cell's integer coordinates, and the triangles are listed in canonical order
          (``dw2d.deterministic_triangulation``).
        * The dual-graph edges are listed in canonical order (``seeded_watershed_aggregation``).
        * The seeded watershed visits edges of equal score in that order (a stable sort).

        ``deterministic=False`` is the legacy route, dw2d 0.1.5 as written: Qhull's triangles as
        returned, the networkx iteration order and NumPy's default sort. It reproduces dw2d 0.1.5
        bitwise wherever Qhull returns the same triangulation as the machine the legacy golden master
        was recorded on (Linux x86-64), and depends on the machine elsewhere.

        ``relocate_junctions`` (default True since 0.8.0): after the mesh is built, move every junction
        vertex that has an unambiguous counterpart onto the mask's own junction (the dual-lattice
        corner where three labels meet), connectivity frozen, under the guards of
        ``dw2d.junction_relocation.relocate_junction_vertices``. Applied to the vertices that
        ``return_mesh``, ``return_dcel`` and the exports use; the report is kept in
        ``self.junction_relocation_report`` (None until ``return_mesh`` runs, or when the option is off).
        The vertices on the image border are never moved (``self.image_border_vertices``, the endpoints of the
        triangulation's hull edges, marked before anything moves; ``return_dcel`` passes the mark on), so a junction
        where a cell-cell interface meets the image border stays on the border, where foambryo's 2D inference
        recognises and excludes it.
        Measured on synthetic equilibrium foams with known tensions (53 held-out tissues, 661 images,
        exact, cropped and noisy masks with cells of 10 to 40 px radius), with foambryo's circular-arc
        reading: the junctions move from 0.65 to 0.42 px of the true ones (medians), the junction-angle
        error falls by 0.85 degrees [0.75, 0.95] and the relative tension error by 0.007 [0.004, 0.010]
        (paired medians, 95 % intervals over tissues); per tier (cell size, crop, noise level) it was
        better or not resolved in 13 of 14 and worse in one (0.5 px of boundary noise with 40 px cells:
        +0.007 [0.000, 0.012] in tension error). ``relocate_junctions=False`` gives the plain reconstruction.

        ``project_interface_vertices`` (default False, opt-in; requires ``relocate_junctions=True``,
        because measured alone it made the junction angles worse): after the relocation, move every
        interface vertex onto its pair's mask locus
        (``dw2d.locus_projection.project_interface_vertices``); report in
        ``self.locus_projection_report``. The order is fixed: relocation first, then projection.

        ``repair_pinch_vertices`` (default True): after the seeded watershed, absorb label fragments
        and repair pinch vertices of the labelled triangulation
        (``dw2d.mesh_surgery.repair_labelled_triangulation``), and let ``return_dcel`` accept several
        boundary loops per material, so that a material in two sectors around a vertex no longer
        breaks the half-edge structure. The report is kept in ``self.surgery_report`` (None when the
        repair is off). The repair changes nothing where there is nothing to repair: on every golden
        case and on every shipped mask whose half-edge structure dw2d 0.1.5 could build, the output
        is bitwise the one without it; it changes the labelling only where dw2d 0.1.5 could not
        build the half-edge structure (``return_dcel`` raised).

        ``relabel_near_tie_slivers`` (default None: on for the plain reconstruction, off when
        ``relocate_junctions=True``, hence off by default since 0.8.0): between the seeded watershed and the zero-node and seed overrides,
        give a flat membrane triangle of a pair of differently labelled flat membrane triangles its
        partner's label when the watershed's choice was a near tie (edge-score margin below 0.1 px) and
        the change strictly removes interface reversals at its three vertices
        (``dw2d.sliver_relabelling.relabel_near_tie_slivers``). It removes the Z-shaped folds the
        watershed leaves along membranes of large cells: on synthetic equilibrium foams with cells of
        40 px radius (53 images, ``min_dist=3``), 453 folds to 0, chord junction angles unchanged on every
        image; over the 52 of 159 images (cells of 10, 20, 40 px) it changed, the error of the pressures
        inferred from the plain mesh is lower by a median 0.0026 of the RMS true pressure (95 % interval
        [0.0013, 0.0053]). It is **off by default when
        ``relocate_junctions=True``** because on the mesh after relocation and projection the pressure
        change was not distinguishable from zero (median -0.00011, interval [-0.0030, +0.0015]); nothing
        became measurably worse there either, and ``relabel_near_tie_slivers=True`` turns it on for that
        route (the projection then refused no move on those images, against 438 refusals without it).
        ``False`` turns it off everywhere. It relabels nothing on the golden cases and on the
        shipped real masks. The report is kept in ``self.near_tie_sliver_report`` (None when off).

        dw2d 0.1.5 as written is ``deterministic=False, relocate_junctions=False, repair_pinch_vertices=False,
        relabel_near_tie_slivers=False`` (apart from the orientation of the edges along the image border,
        arbitrary in 0.1.5; see ``Clean_mesh``).
        """
        if project_interface_vertices and not relocate_junctions:
            raise ValueError("project_interface_vertices=True requires relocate_junctions=True "
                             "(alone, the projection made the junction angles worse on the benchmark)")
        self.deterministic = bool(deterministic)
        # On the plain reconstruction only, unless asked: on the relocated and projected mesh its gain in
        # pressure accuracy was not distinguishable from zero.
        self.relabel_near_tie_slivers = (not relocate_junctions) if relabel_near_tie_slivers is None \
            else bool(relabel_near_tie_slivers)
        self.near_tie_sliver_report = None
        self.min_dist = min_dist
        self.project_interface_vertices = bool(project_interface_vertices)
        self.locus_projection_report = None
        self.relocate_junctions = bool(relocate_junctions)
        self.junction_relocation_report = None
        self._relocated_mesh = None
        self.labels= expand_labels(labels,expansion_labels)
        self.seeds_coords,self.seeds_indices, self.tri, self.EDT = build_triangulation(
            self.labels,min_distance=min_dist,deterministic_triangulation=self.deterministic)
        self.original_image = original_image


        # Bilinear interpolants on the pixel grid; scipy.interpolate.interp2d was removed in SciPy 1.14.
        edt = BilinearOnPixelGrid(self.EDT)
        label = BilinearOnPixelGrid(self.labels)
        
        self.Delaunay_Graph = Delaunay_Graph(self.tri, edt,label,self.EDT.shape[0],self.EDT.shape[1],npoints=interpolation_points)
        # The image border, from the triangulation's structure (its hull edges), before any vertex moves.
        self.image_border_vertices = image_border_vertices(self.Delaunay_Graph)
        self.print_info = bool(print_info)
        self.build_graph()
        self.watershed_seeded()
        self.surgery_report = None
        if repair_pinch_vertices:
            self.Map_end, self.surgery_report = repair_labelled_triangulation(
                self.Delaunay_Graph, self.Map_end, self.seeds_nodes, mask_labels=self.labels)

    def build_graph(self):
        self.Nx_Graph = self.Delaunay_Graph.networkx_graph_weights_and_borders(print_info=getattr(self, "print_info", False))


    def watershed_seeded(self, print_info=False,plot_figure=False):
        seeds_nodes = compute_seeds_idx_from_pixel_coords(self.EDT,self.Delaunay_Graph.compute_nodes_centroids(),self.seeds_coords,plot_figure = plot_figure)
        zero_nodes = self.Delaunay_Graph.compute_zero_nodes()
        # kept for the repair (the seeds must never be relabelled); no effect on the output
        self.seeds_nodes, self.zero_nodes = seeds_nodes, zero_nodes
        if not self.relabel_near_tie_slivers:
            self.Map_end = seeded_watershed_map(self.Nx_Graph,seeds_nodes,self.seeds_indices,zero_nodes,deterministic=self.deterministic)
            return
        # The relabelling acts on the watershed's own labels, before the overrides: it never touches a seed
        # or zero-node triangle, whose labels the overrides would impose anyway.
        labels = seeded_watershed_aggregation(self.Nx_Graph, seeds_nodes, self.seeds_indices, deterministic=self.deterministic)
        n_membrane_points = len(peak_local_max(-self.EDT, min_distance=self.min_dist, exclude_border=False))
        protected = np.concatenate([np.asarray(seeds_nodes, np.int64).ravel(), np.asarray(zero_nodes, np.int64).ravel()])
        labels, self.near_tie_sliver_report = relabel_near_tie_slivers(self.Delaunay_Graph, labels, n_membrane_points, protected)
        self.Map_end = apply_zero_and_seed_overrides(labels, seeds_nodes, self.seeds_indices, zero_nodes)


    def return_mesh(self):
        if not self.relocate_junctions:
            return(Clean_mesh(self))
        if self._relocated_mesh is None:
            verts, edges = Clean_mesh(self)
            new_verts, self.junction_relocation_report = _relocate_on_edges(
                verts, edges, self.labels, fixed_vertices=self.image_border_vertices)
            if self.project_interface_vertices:
                new_verts, self.locus_projection_report = _project_on_edges(
                    new_verts, edges, self.labels, fixed_vertices=self.image_border_vertices)
            self._relocated_mesh = (new_verts, edges)
        verts, edges = self._relocated_mesh
        return(verts.copy(), edges.copy())
    
    def return_dcel(self): 
        verts,edges = self.return_mesh()
        # A material may have several boundary loops only after the repair; otherwise the legacy
        # single-loop contract (and its assertion) is kept.
        return(DCEL_Data(verts,edges,allow_several_loops=self.surgery_report is not None,
                         image_border_vertices=self.image_border_vertices))
    
    def export_mesh(self,filename,mode='bin'):
        verts,edges = self.return_mesh()
        # An unknown mode printed "Please choose a valid format" and wrote nothing before 0.8.0.
        if mode not in ('txt', 'bin'):
            raise ValueError(f"mode must be 'bin' or 'txt', got {mode!r}")
        save_rec(filename, verts, edges[:, :2], edges[:, 2:], binary_mode=(mode == 'bin'))
            
    def export_segmentation(self,filename):
        Verts,Edges = self.return_mesh()
        seeds = self.seeds_coords
        image_shape = np.array(self.labels.shape)
        Mesh_dict = {"Verts":Verts,
                     "Edges":Edges,
                     "seeds":seeds,
                     "image_shape":image_shape,
                     "image_border_vertices":self.image_border_vertices}
        np.save(filename, Mesh_dict)

    def simple_plot(self): 
        fig, axs = plt.subplots(1,3,figsize = (24,8))
        axs[0].imshow(self.labels,cmap = plt.cm.nipy_spectral)
        axs[0].set_title("Labels")
        
        axs[1].imshow(self.EDT,plt.cm.magma)
        axs[1].set_title("Polygonal lines")
        Lines = retrieve_border_lines(self.Delaunay_Graph, self.Map_end)
        plot_polylines_ax(axs[1],Lines,self.Delaunay_Graph.Vertices)
        
        plot_triangles_coloured_ax(axs[2],self.Delaunay_Graph, self.Map_end)
        axs[2].set_title("Triangles")

        for ax in axs : 
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0)
        
        
    def extended_plot(self): 
        
        fig, axs = plt.subplots(1,6,figsize = (32,12))
        axs[0].imshow(self.original_image,plt.cm.gray)
        axs[0].set_title("Original Image")
        axs[1].imshow(self.labels,cmap = plt.cm.nipy_spectral)
        axs[1].set_title("Labels")
        
        axs[2].imshow(self.EDT,plt.cm.gray)
        axs[2].scatter(self.seeds_coords[:,1],self.seeds_coords[:,0],c = plt.cm.nipy_spectral(np.arange(len(self.seeds_coords))/len(self.seeds_coords)),s=180)
        axs[2].set_title("Seeds from EDT")
        
        axs[3].imshow(self.EDT,plt.cm.gray)
        axs[3].set_title("Delaunay triangulation")
        plot_triangulation_ax(axs[3],self)
        
        
        axs[4].imshow(self.EDT,plt.cm.magma)
        axs[4].set_title("Polygonal Lines")
        Lines = retrieve_border_lines(self.Delaunay_Graph, self.Map_end)
        plot_polylines_ax(axs[4],Lines,self.Delaunay_Graph.Vertices)
        
        axs[5].set_title("Triangles")
        plot_triangles_coloured_ax(axs[5],self.Delaunay_Graph, self.Map_end)
        
        for ax in axs : 
            ax.set_xticks([])
            ax.set_yticks([])
        #plt.subplots_adjust(wspace=0.0, hspace=0)


    def plot_polylines_and_seeds(self):
        fig,ax = plt.subplots(figsize = (4,4))
        ax.set_title("Saved Informations")
        ax.imshow(np.ones(self.labels.shape),plt.cm.Greys)
        Lines = retrieve_border_lines(self.Delaunay_Graph, self.Map_end)
        ax.scatter(self.seeds_coords[:,1],self.seeds_coords[:,0],c = plt.cm.nipy_spectral(np.arange(len(self.seeds_coords))/len(self.seeds_coords)),s=50)
        plot_polylines_ax(ax,Lines,self.Delaunay_Graph.Vertices)
        
        plt.axis("off")
        


class geometry_reconstruction_2d(_Reconstruction2D):
    """Deprecated since 0.8.0, removed in 0.9: use `dw2d.get_default_mesh_reconstruction_algorithm` (or
    `dw2d.MeshReconstructionAlgorithmFactory`) and `construct_mesh_from_segmentation_mask`, the names dw3d uses.

    dw2d 0.1.5's reconstruction object, unchanged: it runs at construction and keeps `return_mesh`, `return_dcel`,
    `export_mesh` (binary by default), `export_segmentation` (keys `Verts`, `Edges`, ...) and the matplotlib methods.
    Its arguments map to the new names as follows: `min_dist` -> `min_distance`; `expansion_labels` ->
    `set_label_expansion`; `interpolation_points` -> `set_score_samples`; `deterministic` -> `set_canonical_triangle_order`;
    `relocate_junctions` -> `set_junction_relocation`; `project_interface_vertices` -> `set_interface_projection`;
    `repair_pinch_vertices` -> `set_pinch_vertex_repair`; `relabel_near_tie_slivers` -> `set_near_tie_sliver_relabelling`.
    `return_mesh()` gives `(Verts, Edges)` with `Edges` `(m, 4)` = the new `segments` and `labels` side by side.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "dw2d.geometry_reconstruction_2d is deprecated and will be removed in 0.9; use "
            "dw2d.get_default_mesh_reconstruction_algorithm(min_distance=...).construct_mesh_from_segmentation_mask(mask), "
            "which returns (points, segments, labels) as dw3d does.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
