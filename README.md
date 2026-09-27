# Delaunay-Watershed 3D

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![DOI](https://zenodo.org/badge/634561229.svg)](https://zenodo.org/badge/latestdoi/634561229)

<img src="https://raw.githubusercontent.com/sacha-ichbiah/delaunay_watershed_3d/main/Figures_readme/Figure_logo_white_arrow.png" alt="drawing" width="300"/>


**Delaunay-Watershed-3D** is an algorithm designed to reconstruct *in 3D* a sparse surface mesh representation of the geometry of multicellular structures or nuclei from instance segmentations. It accomplishes this by building multimaterial meshes from segmentation masks. These multimaterial meshes are perfectly suited for **storage, geometrical analysis, sharing** and **visualization of data**. We provide as well visualization tools based on [polyscope](https://polyscope.run) and [napari](https://napari.org).

Delaunay-Watershed was created by Sacha Ichbiah during his PhD in [Turlier Lab](https://www.turlierlab.com), and is improved and maintained by Matthieu Perez and Hervé Turlier. For support, please open an issue.
If you use this library in your work please cite the [paper](https://doi.org/10.1101/2023.04.12.536641). 

If you are interested in 2D images and meshes, please look at the [foambryo-2D](https://github.com/VirtualEmbryo/foambryo2D) package instead.

Introductory notebooks are provided for two examples (cells or nuclei in multicellular aggregates).
The algorithm takes as input 3D segmentation masks and returns multimaterial triangle meshes in 3D.

This method is used as a backend for [foambryo](https://github.com/VirtualEmbryo/foambryo), our 3D tension and pressure inference Python library.



### Installation

We recommend to install delaunay-watershed from the PyPI repository directly, in a virtual environment.

```shell
pip install delaunay-watershed-3d
```

If you want to use our visualization tools, install the package with the `viewing` option:
```shell
pip install "delaunay-watershed-3d[viewing]"
```

For developers, you may also install delaunay-watershed by cloning the source code and installing from the local directory

```shell
git clone https://github.com/VirtualEmbryo/delaunay-watershed.git
pip install pathtopackage/delaunay-watershed
```

### Quick start example 

Load an instance segmentation, construct its multimaterial mesh, save it to a file for later, and visualize it:

```py
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3D.viewing import plot_cells_polyscope

# Get a mesh reconstruction algorithm
mesh_reconstruction_algorithm = get_default_mesh_reconstruction_algorithm()

# Load the segmentation image
import skimage.io as io
segmentation_mask = io.imread("data/Images/1.tif")

# Reconstruct a multimaterial mesh from the mask using the mesh reconstruction algorithm
mesh_reconstruction_algorithm.construct_mesh_from_segmentation_mask(segmentation_mask)

# Save the last constructed mesh
mesh_reconstruction_algorithm.save_to_vtk_mesh("mesh_from_segmentation.vtk", binary_mode=True)
# Plot the last constructed mesh
plot_cells_polyscope(mesh_reconstruction_algorithm)
```

Geometry can be analyzed later, in [foambryo](https://pypi.org/project/foambryo/) for example.

### Reconstruction variants

`get_default_mesh_reconstruction_algorithm` gives you the **default** reconstruction: the
published algorithm (equivalent to dw3d 0.3.6). It is the more accurate one for downstream
tension inference today, so it is what most users should start with.

A **revised meshing pipeline** is also available, opt-in, as
`get_link_checked_mesh_reconstruction_algorithm`. It is deterministic, protects triple
junctions explicitly, and produces cleaner topology and lower interface-area bias than the
default — but as measured on `foambryo`'s benchmarking dataset, meshes it produces currently
give *worse* end-to-end tension-inference accuracy than the default, on every inference method
tested (see `BENCHMARKS.md`). Reach for it when mesh geometry and topology matter more to you
than today's downstream inference accuracy — for example, when the mesh itself, rather than
inferred tensions, is the deliverable:

```py
from dw3d import get_link_checked_mesh_reconstruction_algorithm

mesh_reconstruction_algorithm = get_link_checked_mesh_reconstruction_algorithm()
mesh_reconstruction_algorithm.construct_mesh_from_segmentation_mask(segmentation_mask)
```

### Straightening the junction lines (on by default since 0.5.0)

A reconstructed mesh's trijunction lines are ragged at voxel scale: the extraction places junction
vertices wherever the tessellation happens to bracket them. Since 0.5.0 every reconstruction moves
them onto the trijunction curves the mask itself gives, as a post-process on the finished mesh:

```py
from dw3d import get_default_mesh_reconstruction_algorithm

algorithm = get_default_mesh_reconstruction_algorithm()
points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(segmentation_mask)
print(algorithm.relocation_provenance)          # did it run, with which guard, did it warn
print(algorithm.relocation_report.as_dict())    # what it moved
```

**To reproduce the 0.4 meshes, switch it off**: `algorithm.relocate_junctions = False` before
reconstructing, or `MeshReconstructionAlgorithmFactory().set_junction_relocation(False)`. At a
point-placement `min_distance` of 5 or more the reconstruction emits
`dw3d.CoarseSpacingRelocationWarning` once: the effect on surface-tangent contact angles there lies
within the check's own noise and larger spacings are unmeasured; facet-based angles improve.

It can also be applied to a mesh you already have:

```py
from dw3d import relocate_junction_vertices
from dw3d.io import load_rec

points, triangles, labels = load_rec("mesh.rec")
points, report = relocate_junction_vertices(segmentation_mask, points, triangles, labels)
```

**It moves vertices and nothing else.** Triangles and material labels come back untouched, so a
treated mesh can be compared vertex-for-vertex with its own untreated original.

**What it buys**, measured over 40 benchmark cases at four reconstruction spacings and four point
samplers: line position error, local tangent error and discrete curvature all improve at every
spacing on every sampler, and no existing mesh-validity number changes on any case. If you are
using a seed-free sampler (`get_deterministic_mesh_reconstruction_algorithm`) it also improves
contact angles by 2.3–3.5 degrees and downstream tension error by 20–40 %, and brings the junction
lines to the default sampler's own untreated accuracy or better. On the default's seeded sampler
the lines improve and the contact angles do not move significantly. See `BENCHMARKS.md` (measured
before relocation became the default).

**What it costs**: about +0.75 s on the benchmark meshes, whose reconstruction takes about 5 s
(+15 %); 0.1–1.5 s per timepoint on real embryo volumes, whose reconstruction takes about 0.6 s
(median 0.34 s and 0.48 s on two time series). Within one real series it grows faster than linearly
with mesh size (fitted exponent 2.68 over a 2.2-fold range — not a tissue-scale law). A faster
neighbour search is planned for 0.5.1.

**It cannot make the mesh pass through itself.** Every accepted move is checked against the whole
mesh, not only the triangles around the vertex it moves, so the number of self-intersecting
triangle pairs can never rise. That check is available on its own as
`dw3d.count_self_intersections(points, triangles)`.

### `.rec` / `.arec` metadata appendix

`save_rec`/`load_rec` (`dw3d.io`) read and write `.rec`/`.arec` multimaterial mesh files
(points, triangles, and per-triangle material labels; text or binary). Optionally, a
`.rec`/`.arec` file can carry a small metadata appendix: pass `metadata={...}` to
`save_rec`, and read it back with `load_rec_appendix(filename)`.

The appendix is a trailing block of ASCII `# key = value` lines, written after the
geometry data — never before it, because every existing reader (this one included) reads
a count and then exactly that many values, so a trailing block is invisible to them,
while a leading one breaks parsing outright. Fields:

```
# rec-appendix 1
# written_by = dw3d <version>
# coordinate_frame = voxel_index
# axis_order = zyx
# spacing = 1.0 0.325 0.325
# spacing_unit = um
# source_shape = 194 200 199
# crop_offset = 0 0 0
```

- `coordinate_frame` is `voxel_index`, `physical`, or `unknown` — this is the field the
  appendix exists for, since spacing alone can't tell a reader whether coordinates are
  raw voxel indices awaiting a scale or physical units that already have it applied.
- `axis_order`, `spacing`/`spacing_unit`, `source_shape` and `crop_offset` record the
  rest of the sampling geometry, `crop_offset` even when zero (it is what will later let
  a reader distinguish a physical exterior boundary from a crop boundary).
- **A file with no appendix must be treated as `coordinate_frame == "unknown"`, never as
  isotropic spacing** — `load_rec_appendix` returns `None` in that case, and every file
  saved before this feature existed has no appendix.

For more examples and documentation, see the notebooks:
- [Mesh reconstruction and visualization](./Examples/example_1_mesh_reconstruction_visualisation.ipynb),
- [Mask compression and reconstruction](./Examples/example_2_mask_compression_reconstruction.ipynb).

There is also an advanced notebook if you want to tinkle with the algoritm: [Advanced use](./Examples/example_3_advanced_use.ipynb).


---
### Biological examples

#### Geometrical reconstruction of cell interfaces in the *P. Mammilata* embryo
See the [notebook on mesh reconstruction and visualization](./Examples/example_1_mesh_reconstruction_visualisation.ipynb).

![](https://raw.githubusercontent.com/sacha-ichbiah/delaunay_watershed_3d/main/Figures_readme/DW_3d.png "Mesh reconstruction.")

Segmentation masks from [Guignard et al.](https://www.science.org/doi/10.1126/science.aar5663)


#### Geometrical reconstruction of cell nuclei

See the [notebook on mask compression and reconstruction](./Examples/example_2_mask_compression_reconstruction.ipynb).

![](https://raw.githubusercontent.com/sacha-ichbiah/delaunay_watershed_3d/main/Figures_readme/DW_3d_nuclei.png "Mask reconstruction.")

Segmentation masks from [Stardist](https://github.com/stardist/stardist)


---

### Repository layout

The project's development record (working notes, exploratory scripts and their results) is
kept in the development repository for provenance and is not part of this distribution:
`pip install` does not ship it, and nothing under `dw3d` imports from it.

### Credits, contact, citations
If you use this tool, please cite the associated paper.
Do not hesitate to contact Matthieu Perez and Hervé Turlier for practical questions and applications. 
We hope that **Delaunay-Watershed** could help biologists and physicists to shed light on the mechanical aspects of early development.

```
@article {Ichbiah2023.04.12.536641,
	author = {Sacha Ichbiah and Fabrice Delbary and Alex McDougall and R{\'e}mi Dumollard and Herv{\'e} Turlier},
	title = {Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy},
	elocation-id = {2023.04.12.536641},
	year = {2023},
	doi = {10.1101/2023.04.12.536641},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The morphogenesis of tissues and embryos results from a tight interplay between gene expression, biochemical signaling and mechanics. Although sequencing methods allow the generation of cell-resolved spatio-temporal maps of gene expression in developing tissues, creating similar maps of cell mechanics in 3D has remained a real challenge. Exploiting the foam-like geometry of cells in embryos, we propose a robust end-to-end computational method to infer spatiotemporal atlases of cellular forces from fluorescence microscopy images of cell membranes. Our method generates precise 3D meshes of cell geometry and successively predicts relative cell surface tensions and pressures in the tissue. We validate it with 3D active foam simulations, study its noise sensitivity, and prove its biological relevance in mouse, ascidian and C. elegans embryos. 3D inference allows us to recover mechanical features identified previously, but also predicts new ones, unveiling potential new insights on the spatiotemporal regulation of cell mechanics in early embryos. Our code is freely available and paves the way for unraveling the unknown mechanochemical feedbacks that control embryo and tissue morphogenesis.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/04/13/2023.04.12.536641},
	eprint = {https://www.biorxiv.org/content/early/2023/04/13/2023.04.12.536641.full.pdf},
	journal = {bioRxiv}
}
```

### License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
