# Delaunay-Watershed (3D and 2D)

**Delaunay-Watershed** reconstructs, from an instance segmentation of cells or nuclei, a **multimaterial mesh** of every interface between two cells, or between a cell and the medium. It has two packages, installed together by `pip install delaunay-watershed`:

* **`dw3d`** (distribution `delaunay-watershed-3d`), for 3-D masks: one triangle mesh of every interface;
* **`dw2d`** (distribution `delaunay-watershed-2d`), for 2-D masks: one polygonal line of every interface, and a half-edge structure from which lengths, angles, curvatures and areas are read.

The same calls mesh a 2-D or a 3-D mask: the input decides the dimension. It is the reconstruction behind [foambryo](https://github.com/VirtualEmbryo/foambryo), which infers surface tensions and cell pressures from these meshes, in 3D and in 2D.

* [Getting started](getting-started.md): installation, the input, the output, a first mesh.
* User guide: [reconstruction variants](user-guide/reconstruction-variants.md), [junction relocation](user-guide/junction-relocation.md), [cropped tissues](user-guide/cropped-tissues.md), [thin margins of medium](user-guide/thin-medium-margins.md), [anisotropic voxels](user-guide/anisotropic-voxels.md), [the `.rec` metadata appendix](user-guide/rec-metadata-appendix.md), [enclosed pockets of medium](user-guide/enclosed-media.md), [orientation in 2D and 3D](user-guide/orientation-in-two-and-three-dimensions.md).
* Two dimensions: [overview](two_dimensional/overview.md), [reconstruction in 2D](two_dimensional/reconstruction.md), [migrating from dw2d 0.1.5](two_dimensional/migration-from-0.1.5.md).
* [Examples](examples/index.md): the notebooks, in 3D and 2D.
* [API reference](api/index.md), generated from the docstrings.
* [Changelog](changelog.md) and [release notes](release-notes.md).

If you use it, please cite both articles: Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7); and Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H., Charras G. *Spatiotemporal mapping of the contractile and adhesive forces sculpting early C. elegans embryos.* Developmental Cell **61**(6), 1273–1288.e4 (2026), [doi:10.1016/j.devcel.2026.04.013](https://doi.org/10.1016/j.devcel.2026.04.013).

Delaunay-Watershed is free software, under the GNU General Public License, version 3 or (at your option) any later version (`GPL-3.0-or-later`); dw2d is under this licence from 0.8.0, and its releases 0.1.5 and earlier keep the licence they were published under.
