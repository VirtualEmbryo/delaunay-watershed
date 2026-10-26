# Getting started

## Installation

Delaunay-Watershed needs Python 3.11 or later:

```shell
pip install delaunay-watershed-3d               # the library
pip install "delaunay-watershed-3d[viewing]"    # and the Polyscope viewer
```

## The input and the output

The input is a 3-D array of non-negative integer labels: one label per cell (or nucleus), 0 for the medium. Axis 0 is z, axes 1 and 2 are y and x. The output is three arrays:

* `points`, `(n_points, 3)`: the vertices, in the mask's `(z, y, x)` order, in voxels, or in the unit of the spacing with `construct_mesh_in_physical_units`;
* `triangles`, `(n_triangles, 3)`: indices into `points`;
* `labels`, `(n_triangles, 2)`: the two materials on each side of each triangle; the triangle's normal (right-hand rule) points from the first label into the second.

## A first mesh

```py
import tifffile
from dw3d import get_default_mesh_reconstruction_algorithm

mask = tifffile.imread("segmentation.tif")
algorithm = get_default_mesh_reconstruction_algorithm()          # min_distance=3, junction relocation on
points, triangles, labels = algorithm.construct_mesh_in_physical_units(mask, spacing=(1.0, 0.5, 0.5), spacing_unit="um")
algorithm.save_to_rec_mesh("mesh.rec")
print(algorithm.cut_cells, algorithm.cell_volume_report)
```

Without a voxel size, `construct_mesh_from_segmentation_mask(mask)` meshes the voxel grid and returns the mesh in voxel units. `min_distance` is the minimum distance between the tessellation's points, in (cubic) voxels: smaller gives a finer, slower mesh.

To look at it (needs a display): `from dw3d.viewing import plot_cells_polyscope; plot_cells_polyscope(algorithm)`. The window shows the mask's column, row and slice as Polyscope's x, y and z (the points' columns 2, 1, 0), so the tissue is drawn as it is; a program that reads a saved mesh's columns as (x, y, z), such as ParaView, shows its mirror image (see `dw3d.io`).
