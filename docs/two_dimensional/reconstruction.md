# Reconstruction in 2D

## Building the algorithm

`dw2d.get_default_mesh_reconstruction_algorithm(min_distance=4, print_info=False)` returns the default algorithm. `dw2d.MeshReconstructionAlgorithmFactory` builds another one step by step, with dw3d's names where the step has a 2-D meaning; each `set_*` method returns the factory, and `make_algorithm()` builds the algorithm:

```py
import dw2d

algorithm = (
    dw2d.MeshReconstructionAlgorithmFactory()
    .set_peak_local_points_placement_method(min_distance=3)
    .set_junction_relocation(True)
    .set_interface_projection(True)
    .make_algorithm()
)
```

Every option is also an attribute of the algorithm (`algorithm.relocate_junctions = False`) and a keyword of `dw2d.MeshReconstructionAlgorithm(...)`. One algorithm meshes any number of masks; its reports describe the last one.

dw3d's algorithm given a 2-D mask configures dw2d from its own settings that have a 2-D meaning (`min_distance`, junction relocation, canonical order, `pad_thin_medium_margins`, `print_info`); a dw3d configuration without a 2-D meaning (boundary-layer, junction-protected or link-checked placement, another distance transform, tessellation or face score, physical image faces, mesh surgery off) raises `ValueError` naming it. dw2d places its points without a random dither, so dw3d's default (dithered) and deterministic configurations mesh a 2-D mask alike.

## The options and their defaults

Measured effects are on synthetic two-dimensional foams at mechanical equilibrium with known tensions and pressures (53 held-out tissues of 4 to 64 cells, 661 images: exact masks, crops and smooth boundary noise of 0.5 to 2 px, with cells of 10, 20 and 40 px radius), read by foambryo's circular-arc angle reading unless stated otherwise.

| option (factory setter; attribute) | default | what it does, and what it was measured to change |
|---|---|---|
| `set_peak_local_points_placement_method(min_distance)`; `min_distance` | 4 pixels | The minimum distance between the triangulation's points, in pixels (dw3d: 3 voxels). Smaller gives a finer, slower mesh. |
| `set_canonical_triangle_order`; `canonical_triangle_order` | on (new in 0.8.0) | Makes the mesh the same on every machine. Where four or more points lie on one circle (frequent, since they are integer pixels), the Delaunay triangulation is not unique and Qhull picks differently on Linux x86-64 and macOS arm64; such cells are triangulated by a fixed rule, triangles and graph edges are listed in a canonical order, and the watershed sorts stably. Against dw2d 0.1.5, 2 of the 18 golden cases change (one interface length by about 0.05 %, no angle). `False` keeps the legacy route. dw3d's analogue is `canonical_tetrahedron_order`. |
| `set_junction_relocation`; `relocate_junctions` | on (since 0.8.0) | Moves every junction vertex with an unambiguous counterpart onto the corner where three labels meet between pixels, connectivity frozen, under guards against edge crossings and contacts. The junctions move from 0.65 to 0.42 px of the true ones (medians), the junction-angle error falls by 0.85° (95 % interval 0.75 to 0.95) and the relative tension error by 0.007 (0.004 to 0.010); better or not resolved in 13 of 14 tiers, including boundary noise of 0.75 px, the level measured on real 2D masks, and worse in one (0.5 px of noise with 40 px cells: +0.007, interval 0.000 to 0.012). Read with chords instead, it lowers the angle error by about 1.8° and the tension error by about 0.025. `relocation_report` says what moved. |
| `set_interface_projection`; `interface_projection` | off | With junction relocation, then moves every interface vertex onto its label pair's interface in the mask; together the two lower the angle error by about 2.5°. Alone it made the error worse, so requesting it without relocation raises `ValueError`. `interface_projection_report` says what moved. |
| `set_pinch_vertex_repair`; `pinch_vertex_repair` | on (new in 0.8.0) | Repairs a cell left in two pieces, or a material in two sectors around one vertex, by relabelling a few triangles (no point moves), so that the half-edge structure of tissue masks can be built. Where nothing needs repairing the mesh is unchanged (every golden case; 14 of 24 shipped-mask settings); on the other 10, building the half-edge structure used to fail and now succeeds. |
| `set_near_tie_sliver_relabelling`; `near_tie_sliver_relabelling` | `None`: on without junction relocation, off with it | Gives a flat membrane triangle its flat neighbour's label where the watershed's choice between them was a near tie (within 0.1 px of edge score) and the change removes a Z-shaped fold of the interface. On the plain mesh (159 images, `min_distance` 3) it removes every such fold, leaves the chord angles unchanged and makes the pressures more accurate by a median 0.0026 of the RMS true pressure (0.0013 to 0.0053); after relocation the gain was not distinguishable from zero, hence off there unless `True`. `False` turns it off everywhere. |
| `pad_thin_medium_margins` | on (new in 0.8.0) | Pads with medium every side of the image that no cell reaches and whose margin is below `2 * min_distance + 2` pixels, before meshing; see [thin margins of medium](../user-guide/thin-medium-margins.md). `medium_margin_padding` says which sides were padded. |
| `set_label_expansion(pixels)`; `label_expansion` | 0 | Grows the labels by this many pixels before meshing (`skimage.segmentation.expand_labels`), for masks whose cells do not touch. |
| `set_score_samples(n)`; `score_samples` | 10 | The number of points sampled along each triangle edge to score it on the distance map. |

dw2d 0.1.5's mesh is the one built with `canonical_triangle_order=False, relocate_junctions=False, pinch_vertex_repair=False, near_tie_sliver_relabelling=False, pad_thin_medium_margins=False`, except that its edges were stored in the opposite orientation ([migration](migration-from-0.1.5.md)).

## The image border and cropped masks

A cell that reaches the image border is cut by it. dw2d closes it there with edges along the border, whose other material is the medium (label 0), and marks the vertices on the border:

* `algorithm.image_border_vertices`: the indices of the vertices on the image border, taken from the triangulation before any vertex moves (the endpoints of its hull edges, which run along the border because the image's four corners are triangulated with the points); the junction relocation and the interface projection never move them. The half-edge structure carries them as `DCEL_Data.image_border_vertices`, and the compressed segmentation and the `.rec` appendix keep them.
* `algorithm.cut_cells`: the cells with a vertex on the image border.

foambryo uses the mark to leave the junctions on the image border out of its 2-D inference, whose wedge in the medium is the image's cut, not a membrane. Before 0.8.0, where the mask had medium pixels on the border next to two cells, the junction relocation moved the junction there 0.7 to 2.9 px inside the image (7 of 972 border junctions on 82 test meshes); it now stays on the border.

The edges along the border are oriented like every other edge (before 0.8.0 they came in an arbitrary order, 164 of 165 reversed on one shipped mask), so each material's edges run around it in one sense.

## Lengths, angles, curvatures, areas

`algorithm.last_constructed_dcel` (or `dw2d.DCEL_Data(points, edges)`, with `edges` the `(m, 4)` array `[v1, v2, m1, m2]`) is a half-edge structure:

* `compute_lengths()`: a dictionary, interface `(a, b)` → its length in pixels (or the mesh's unit), the sum of its edges' lengths;
* `compute_angles()`: the angle of each material at each junction, in radians, keyed `(a, b, c)`: the angle inside material `b` between its interfaces with `a` and `c` (each angle is listed under `(a, b, c)` and `(c, b, a)`);
* `compute_curvatures()`: interface → its curvature in 1/pixel, unsigned: the length-weighted mean of the Menger curvature at its vertices, junctions excluded;
* `compute_area_faces()`: material → its area in square pixels; the medium, label 0, comes out negative.

`compute_lengths` and `compute_curvatures` were wrong on most interfaces before 0.8.0 (the lengths by a median 7.4 %, [migration](migration-from-0.1.5.md)): recompute anything derived from them with an earlier version.

## Files and the compressed segmentation

* `algorithm.save_to_rec_mesh(filename, binary_mode=False)` writes the mesh as a `.rec` file (text by default, as dw3d), always with a metadata appendix: `dimension = 2`, `axis_order = yx`, `source_shape`, the image-border vertices, the orientation rule (`orientation = boundary_of_first_label`) and, after `construct_mesh_in_physical_units`, the frame and the pixel size. `dw2d.io.load_rec(filename)` returns `(points, segments, labels)`; `dw2d.io.save_rec` writes them; `dw3d.io.rec_dimension(filename)` tells a 2-D file from a 3-D one.
* `algorithm.last_compressed_segmentation` is a dictionary (`points`, `segments`, `labels`, `seeds`, `image_shape`, `image_border_vertices`) from which `dw2d.reconstruct_mask_from_dict(dictionary)` rebuilds the mask; `dw2d.save_compressed_segmentation(filename, dictionary)` and `dw2d.reconstruct_mask_from_saved_file_dict(filename)` go through a file.

## Warnings

* `UnresolvedCellWarning`: a label of the mask absent from the mesh, named (also a mask whose mesh has no edge at all, which is returned empty); `algorithm.unresolved_cells` lists them.
* `EnclosedMediumWarning`: a piece of label 0 that touches no side of the image; it is meshed as the outer medium (dw2d has no enclosed-medium region yet); `algorithm.enclosed_medium_pockets` lists them.
* `ThinMediumMarginWarning`: a cell left with edges on a side of the image its pixels do not reach.

They are dw2d's own classes, distinct from dw3d's classes of the same names.
