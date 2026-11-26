# Cropped tissues: crop faces and cut cells

When a cell of the mask reaches a face of the image box, the reconstruction closes it with triangles lying on that face. They are not contact with the medium: the tissue goes on beyond the image. The algorithm marks them as **crop faces**, and the cells they touch as **cut cells**, without changing the mesh:

```py
points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
algorithm.crop_faces          # boolean per triangle
algorithm.cut_cells           # labels of the cells the crop cuts
algorithm.crop_face_report.as_dict()   # counts per image face
```

foambryo keeps crop faces out of its equations and does not report the pressures of cut cells. A triangle the reconstruction puts on an image face that the mask's cell does not reach (the mask left a voxel or two of medium there) is not a crop face: it stays contact with the medium. If the tissue really ends on a face of the image (a coverslip), declare that face physical before reconstructing: its triangles then stay contact with the medium.

```py
algorithm.physical_image_faces = ("axis0_low",)   # index 0 along the mask's first axis
```

The image faces are named `"axis0_low"`, `"axis0_high"`, ... `"axis2_high"` (`dw3d.IMAGE_FACES`). `save_to_rec_mesh` writes the crop faces as the last block of triangles and records them in the appendix below; `dw3d.io.load_rec_with_crop_faces` reads them back.

Example: the notebook *Anisotropic voxels and crop faces* (`Examples/example_6_anisotropic_voxels_and_crop_faces.ipynb`).

## A cell a few voxels from the image box

A face of the image box that no cell reaches is not a crop: the medium goes on beyond it. Before 0.8.0 a cell a few voxels from such a face could take the medium between them (at `min_distance` 3 a 12-voxel cube 4 voxels from every face meshed to 2.98 times its voxel count). Since 0.8.0 such thin margins are padded with medium before meshing (masks of cubic voxels; a mask resampled from longer voxels is not padded unless asked), a face a cell reaches is never padded, and `ThinMediumMarginWarning` names a cell left on a face it does not reach; `algorithm.medium_margin_padding` says which faces were padded, and `algorithm.pad_thin_medium_margins = False` gives the previous meshes. On a padded face a cell's surface may pass the image box, by at most the padding. The rule, its measurements and its known issues, in 3D and in 2D: [thin margins of medium](thin-medium-margins.md).

## A cell the watershed loses (known issue)

The seeded watershed can lose a cell, most often a cell cut by the image box. A cell's seed is the tetrahedron whose centroid is nearest to the cell's deepest voxel. Against the image box this can be a thin tetrahedron whose faces all score below a face through which the rest of the cell, grouped without its seed, reaches a seeded neighbour; the neighbour then takes the whole group, the cell keeps only its seed, and mesh surgery relabels that tetrahedron. The cell is absent from the mesh, the neighbour comes out larger than its voxels, and `cut_cells` names the neighbour, which now holds the crop faces that were the lost cell's; `crop_face_report.cells_reaching_a_crop_without_a_crop_face` names the lost cell.

Since 0.8.0 such a cell is named: `UnresolvedCellWarning` lists it with the cause `"removed by mesh surgery"` (see [Enclosed media](enclosed-media.md), *Regions too small for the mesh*), and foambryo gives it no pressure. A smaller `min_distance` may mesh it. Measured on a test mask of four cells filling a ball that a 36³ image box cuts (cells 1, 3 and 4 have voxels on the box), at the default `min_distance` 3 on cubic voxels and without the padding of thin margins: cell 1 (1 648 voxels, 262 on the cut face) is absent from the mesh and cell 2 comes out at 1.37 times its voxels; at `min_distance` 1 and 2, and with z twice as long as x and y, every cell is meshed. On the same ball at 36 positions along z and `min_distance` 2 to 5, 16 cells are absent in 144 reconstructions; on 433 plane crops of the benchmark masks, `cut_cells` misses a cell on the box in 5, each lost or nearly lost by the watershed. With the padding of thin margins, the default, the ball's four faces that no cell reaches are padded and cell 1 is left only as a fragment of 4 triangles (0.010 of its voxel volume inside the box of voxel centres); the watershed's defect is unchanged, and `pad_thin_medium_margins = False` reproduces the loss above.

Compare a cut cell's mesh volume with its voxels inside the box of voxel centres, not with its voxel count: the mesh stops at the centres of the outermost voxels, so a cell cut by the box can reach at most about that volume (0.92 of the voxel count for cell 1 of the test mask above).
