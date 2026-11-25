# Mask compression and reconstruction

Compress a segmentation of nuclei into the points and triangles of its mesh, save it, and rebuild the mask from it. The notebook downloads StarDist's 3-D demonstration data and needs `csbdeep` and napari.

Data: a download (StarDist demo data).

Its outputs were made on the code of 0.8.0 in a window, since its napari cell needs one (napari does not open without a display). It compresses the third training mask (128 × 128 × 128 voxels after doubling along z, 98 nuclei) into 115 560 bytes of points, triangles and seeds: about 145 times less than the rebuilt mask held as 64-bit integers, as the notebook counts it, and 36 times less than the original 16-bit mask; 94 % of the rebuilt voxels have the original's label.

The notebook: [`Examples/example_2_mask_compression_reconstruction.ipynb`](https://github.com/VirtualEmbryo/delaunay-watershed/blob/v0.8.0/Examples/example_2_mask_compression_reconstruction.ipynb).
