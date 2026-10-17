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
