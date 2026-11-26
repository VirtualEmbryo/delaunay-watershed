# 2D: geometrical analysis

Mesh the repository's mask `data/two_dimensional/Masks/mask_Cells_3.tif` (three cells) with dw2d's default algorithm, draw the mesh over the image, and read from its half-edge structure (`last_constructed_dcel`) the interface lengths, the junction angles, the curvatures and the cell areas; draw the local curvature along the interfaces; save the mesh as a `.rec` file and read it back with `dw2d.io.load_rec`. Needs the `viewing` extra (matplotlib).

Data: the repository's `data/two_dimensional/`.

The notebook: [`Examples/two_dimensional/Geometrical_analysis.ipynb`](https://github.com/VirtualEmbryo/delaunay-watershed/blob/v0.8.0/Examples/two_dimensional/Geometrical_analysis.ipynb).
