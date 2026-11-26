# delaunay-watershed-2d

`delaunay-watershed-2d` carries the package `dw2d`, which reconstructs the geometry of a 2-D cell assembly from an instance segmentation: a multimaterial polygonal mesh (a half-edge structure) from which lengths, curvatures, angles and areas are read.

Since 0.8.0 it is developed and released with `dw3d` (the 3-D counterpart, distribution `delaunay-watershed-3d`) in one repository, under one version. Installing `delaunay-watershed` installs both:

```shell
pip install delaunay-watershed                  # dw3d and dw2d
pip install delaunay-watershed-2d               # dw2d alone (delaunay-watershed-3d brings it too)
pip install "delaunay-watershed-2d[viewing]"    # with matplotlib, for dw2d's figures
```

Python 3.12 or later. The import is unchanged: `from dw2d import ...`. Since 0.8.0 dw2d offers dw3d's names (`dw2d.get_default_mesh_reconstruction_algorithm().construct_mesh_from_segmentation_mask(mask)`), its defaults change results (a machine-independent mesh, junctions moved onto the mask's own junctions, pinch vertices repaired, image-border vertices marked, thin margins of medium padded), its edges follow dw3d's orientation rule, and `compute_lengths` and `compute_curvatures` are fixed; the names of 0.1.5 still work through 0.8.x, with a deprecation warning. The documentation's page on migrating from dw2d 0.1.5 lists every change.

The code, the documentation (including dw2d's own page, `docs/two_dimensional/README.md`) and the changelog are in the repository [VirtualEmbryo/delaunay-watershed](https://github.com/VirtualEmbryo/delaunay-watershed). Licence: GNU General Public License, version 3 or (at your option) any later version (SPDX `GPL-3.0-or-later`); releases 0.1.5 and earlier keep the licence they were published under.

If you use dw2d, please cite: Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H., Charras G. *Spatiotemporal mapping of the contractile and adhesive forces sculpting early C. elegans embryos.* Developmental Cell **61**(6), 1273–1288.e4 (2026), [doi:10.1016/j.devcel.2026.04.013](https://doi.org/10.1016/j.devcel.2026.04.013).
