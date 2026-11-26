# Getting started

## Installation

Delaunay-Watershed needs Python 3.12 or later. Install it from PyPI, in a virtual environment:

```shell
pip install delaunay-watershed                  # dw3d and dw2d
pip install "delaunay-watershed[viewing]"       # and the viewers: Polyscope (dw3d's viewer, 2-D and 3-D), matplotlib (dw2d's figures)
```

`delaunay-watershed` holds no code: it installs the two distributions that carry the packages, at its own version, and forwards their extras (`viewing`, `benchmarks`):

* `delaunay-watershed-3d`, the package `dw3d` (3-D masks). It also requires `delaunay-watershed-2d` at the same version, so `pip install delaunay-watershed-3d` installs both packages too.
* `delaunay-watershed-2d`, the package `dw2d` (2-D masks). `pip install delaunay-watershed-2d` installs `dw2d` alone.

The imports are `from dw3d import ...` and `from dw2d import ...`, as before. Upgrading from `delaunay-watershed-3d` 0.7.1 or `delaunay-watershed-2d` 0.1.5 needs nothing special: `pip install --upgrade` of any of the three replaces the old release in place. The dependencies are NumPy 2.0, SciPy 1.13, scikit-image 0.23.1, networkx 2.8, edt 2.4.1, meshio 5.3.5 and iorec 1.4.1 or later; the `viewing` extra adds Polyscope 2.5.0 and matplotlib 3.8.4 or later.

## The input and the output

The input is an array of non-negative integer labels: one label per cell (or nucleus), 0 for the medium. **Its dimension decides the mesh's**: a 3-D stack gives a triangle mesh of surfaces, a 2-D image a mesh of polygonal lines. A 3-D array with an axis of length 1 is refused (pass `mask[0]` for a single plane); a mask whose dtype is not an integer type warns.

In 3D, axis 0 is z, axes 1 and 2 are y and x. The output is three arrays:

* `points`, `(n_points, 3)`: the vertices, in the mask's `(z, y, x)` order, in voxels, or in the unit of the spacing with `construct_mesh_in_physical_units`;
* `triangles`, `(n_triangles, 3)`: indices into `points`;
* `labels`, `(n_triangles, 2)`: the two materials on each side of each triangle, smaller first; the triangle's right-hand normal points out of the first label and into the second.

In 2D the output is `points` `(n, 2)` in `(row, column)`, `segments` `(m, 2)` and `labels` `(m, 2)`, the first label on the left of each edge in `(row, column)`: the same rule, each element the positively oriented boundary of its first label ([orientation](user-guide/orientation-in-two-and-three-dimensions.md)).

## A first mesh

```py
import tifffile
from dw3d import get_default_mesh_reconstruction_algorithm

mask = tifffile.imread("segmentation.tif")
algorithm = get_default_mesh_reconstruction_algorithm()          # min_distance=3 in 3D, junction relocation on
points, triangles, labels = algorithm.construct_mesh_in_physical_units(mask, spacing=(1.0, 0.5, 0.5), spacing_unit="um")
algorithm.save_to_rec_mesh("mesh.rec")
print(algorithm.cut_cells, algorithm.cell_volume_report)
```

Without a voxel size, `construct_mesh_from_segmentation_mask(mask)` meshes the voxel grid and returns the mesh in voxel units. `min_distance` is the minimum distance between the tessellation's points, in (cubic) voxels, 3 by default in 3D and 4 pixels in 2D: smaller gives a finer, slower mesh. The same object meshes a 2-D mask with the same calls (`construct_mesh_in_physical_units` then takes the square pixel size `(h, h)`); the [2D section](two_dimensional/overview.md) describes dw2d's options.

Warnings to read: `UnresolvedCellWarning` names every label of the mask absent from the mesh, with its cause ([enclosed media](user-guide/enclosed-media.md)); `ThinMediumMarginWarning` names a cell left on a face of the image box it does not reach ([thin margins of medium](user-guide/thin-medium-margins.md)); `CellVolumeDisagreementWarning` lists the cells whose mesh volume is outside 0.5–1.5 times their voxel count.

To look at it (needs a display and the `viewing` extra): `from dw3d.viewing import plot_cells_polyscope; plot_cells_polyscope(algorithm)`. The window shows the mask's column, row and slice as Polyscope's x, y and z (the points' columns 2, 1, 0), so the tissue is drawn as it is; a program that reads a saved mesh's columns as (x, y, z), such as ParaView, shows its mirror image (see `dw3d.io`). The same call draws a 2-D mesh (after a 2-D mask, or dw2d's three arrays) in a 2-D window: each cell's boundary as a curve, the image's column and row as x and y with row 0 at the top, as in napari and matplotlib; `plot_cells_polyscope(algorithm, image=mask)` draws the mask under it.
