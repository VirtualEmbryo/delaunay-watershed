# Enclosed pockets of medium (lumens)

A mask's label 0 is the medium. A piece of label 0 that touches no face of the image is an **enclosed pocket**: a lumen, a cavity, or a hole in the segmentation. Before 0.7.1 such a pocket had no watershed seed (label 0 had a single one, at its deepest voxel, normally outside the tissue), so the first cell its tetrahedra reached took all of it. On a 697-cell embryo with a 389 918-voxel lumen, one lining cell came out at 2.40 times its voxel volume, and no lining cell kept a membrane against the lumen.

Since 0.7.1 a pocket becomes **its own region, flagged as a medium**, when both hold:

- its **inscribed radius** (the largest distance, in voxels of the meshed grid, from one of its voxels to the nearest voxel outside it) is at least the point placement's `min_distance + 1`: 4 voxels at the default 3. A smaller pocket holds no tetrahedron of the tesselation, so it cannot be a region of the mesh;
- **two or more cells line it**. A pocket inside a single cell is a hole in that cell's segmentation; as a region of its own its surface would have no triple line, and no method can infer the tension of such a surface.

The new labels are `max + 1, max + 2, ...`, in the order of each pocket's lowest raster index. Pieces are 6-connected (voxels sharing a face). Every other pocket stays with the cells, as before, and `EnclosedMediumWarning` names it with the cell that took it. Every piece of label 0 that touches the image box is exterior and gets its own seed.

```python
from dw3d import get_default_mesh_reconstruction_algorithm

algorithm = get_default_mesh_reconstruction_algorithm()
points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
algorithm.medium_labels                     # (0, 774): the exterior and one enclosed medium
report = algorithm.enclosed_medium_report   # each medium's voxels, inscribed radius, lining cells
report.pockets_left_to_cells                # the pockets left to the cells, and which cell took each
algorithm.save_to_rec_mesh("mesh.rec")      # the appendix records "media = 0 774"
```

A medium is a region of the mesh like a cell: its interfaces with the cells are separate interfaces, so foambryo infers their tensions apart from those against the exterior, and the medium's own pressure. foambryo 0.7.1 reads the `.rec` file's `media` and reports and draws them as media, not cells (`foambryo infer --media` names them for a mesh made elsewhere). `cell_volume_report` leaves them out.

To reproduce 0.7.0's meshes, where every pocket goes to a cell, set `algorithm.label_enclosed_media = False`.

## Regions too small for the mesh

A region's watershed seed is the tetrahedron whose centroid is nearest to its deepest voxel. When no tetrahedron centroid lies in the region's voxels (a cell a few voxels across, or thinner than `min_distance`), that tetrahedron belongs to a neighbour; before 0.7.1 the region grew inside the neighbour as a phantom. Since 0.7.1 its seed is dropped: the region is **absent from the mesh**, `UnresolvedCellWarning` names it with its voxel count and the `min_distance`, `enclosed_medium_report.unresolved` lists it, and the `.rec` file records it (`unresolved_regions`). A smaller `min_distance` may mesh it. On the benchmark, two cells are unresolved at every `min_distance` from 1 to 7: cell 2 of case 031 (62 voxels) and cell 7 of case 038 (26 voxels).

A mask with no pocket at or above the bound, one piece of exterior and every seed already inside its own region is reconstructed exactly as before: 45 of the 47 benchmark masks are.
