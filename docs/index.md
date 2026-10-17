# Delaunay-Watershed 3D

**Delaunay-Watershed-3D** (package `dw3d`) reconstructs, from a 3-D instance segmentation of cells or nuclei, a **multimaterial surface mesh**: one triangle mesh of every interface between two cells, or between a cell and the medium. It is the reconstruction behind [foambryo](https://github.com/VirtualEmbryo/foambryo), which infers surface tensions and cell pressures from these meshes.

* [Getting started](getting-started.md): installation, the input, the output, a first mesh.
* User guide: [reconstruction variants](user-guide/reconstruction-variants.md), [junction relocation](user-guide/junction-relocation.md), [cropped tissues](user-guide/cropped-tissues.md), [anisotropic voxels](user-guide/anisotropic-voxels.md), [the `.rec` metadata appendix](user-guide/rec-metadata-appendix.md).
* [Examples](examples/index.md): the notebooks.
* [API reference](api/index.md), generated from the docstrings.
* [Changelog](changelog.md) and [release notes](release-notes.md).

If you use it, please cite: Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7).

Delaunay-Watershed is free software, under the GNU General Public License, version 3 or (at your option) any later version (`GPL-3.0-or-later`).
