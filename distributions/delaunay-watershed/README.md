# delaunay-watershed

`delaunay-watershed` installs the two packages of the Delaunay-Watershed method, which reconstructs the geometry of a cell assembly from an instance segmentation as a multimaterial mesh:

* `dw3d`, for 3-D masks (distribution `delaunay-watershed-3d`);
* `dw2d`, for 2-D masks (distribution `delaunay-watershed-2d`).

This distribution holds no code: it requires both at its own version. The three are released together from one repository, under one version and one changelog.

```shell
pip install delaunay-watershed                  # dw3d and dw2d
pip install "delaunay-watershed[viewing]"       # with the viewers: Polyscope (2-D and 3-D meshes) and matplotlib (dw2d's figures)
```

Python 3.12 or later. The imports are `from dw3d import ...` and `from dw2d import ...`, and the same calls mesh a 2-D or a 3-D mask: dw3d's reconstruction algorithm hands a 2-D mask to dw2d. `delaunay-watershed-3d` also requires `delaunay-watershed-2d` at the same version, so it installs both packages too; `delaunay-watershed-2d` installs `dw2d` alone. Upgrading from `delaunay-watershed-3d` 0.7.1 or `delaunay-watershed-2d` 0.1.5 replaces the old release in place.

The code, the documentation (including a page on migrating from dw2d 0.1.5) and the changelog are in the repository [VirtualEmbryo/delaunay-watershed](https://github.com/VirtualEmbryo/delaunay-watershed). Licence: GNU General Public License, version 3 or (at your option) any later version (SPDX `GPL-3.0-or-later`).

If you use Delaunay-Watershed, please cite both articles: Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7); and Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H., Charras G. *Spatiotemporal mapping of the contractile and adhesive forces sculpting early C. elegans embryos.* Developmental Cell **61**(6), 1273–1288.e4 (2026), [doi:10.1016/j.devcel.2026.04.013](https://doi.org/10.1016/j.devcel.2026.04.013).
