# 2D: compressing a tissue

Compress the repository's tissue mask `data/two_dimensional/Masks/mask_Cells_2.tif` (50 cells, their labels expanded by 6 pixels so that they touch) into the points, segments and seeds of its mesh (`last_compressed_segmentation`), save it, rebuild the mask from it with `reconstruct_mask_from_saved_file_dict`, and compare the rebuilt mask and the sizes with the original. Needs the `viewing` extra (matplotlib).

Data: the repository's `data/two_dimensional/`.

The notebook: [`Examples/two_dimensional/Geometry_and_mask_reconstruction_2.ipynb`](https://github.com/VirtualEmbryo/delaunay-watershed/blob/v0.8.0/Examples/two_dimensional/Geometry_and_mask_reconstruction_2.ipynb).
