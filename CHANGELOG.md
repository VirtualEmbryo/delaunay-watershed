# Changelog

User-facing changes only, organised by what a user experiences.

## 0.7.0 — 2026-10-17

0.6.0 was prepared (4 October 2026) but not published on PyPI: its changes are part of 0.7.0, listed at the end of this section under "Prepared for 0.6.0". 0.7.0 is the first release on PyPI since 0.5.1.

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.

### Documentation

- **The README is shorter** and points to a user guide in `docs/` (reconstruction variants, junction relocation, cropped tissues, anisotropic voxels, the `.rec` metadata appendix), from which a documentation website is built with MkDocs (`mkdocs.yml`), with an API reference generated from the docstrings. A new notebook, `example_6_anisotropic_voxels_and_crop_faces.ipynb`, runs on a mask it generates. The notebooks no longer contain install cells (they say what to install), and the first notebook says that `data/Images/1.tif` records no voxel size, so the sizes it uses are illustrative. The docstrings of the public functions were checked against the code. No mesh changes.
- **Two return annotations corrected**: `both_construct_mesh_and_compressed_segmentation` returns four items (the mesh's three arrays and the compressed-segmentation dictionary), and `reconstruct_mask_from_dict` and `reconstruct_mask_from_saved_file_dict` return 64-bit integer labels, not `uint8`. What they return is unchanged.

### Fixed (changes results)

- **`set_pixel_size` now scales the right axes; results change when the xy and z sizes differ.** dw3d's mesh points are in the mask's index order, `(z, y, x)` for a microscope stack, so column 0 is z. `dw3d.mesh_utilities.set_pixel_size(points, xy_pixel_size, z_pixel_size)`, the method `algorithm.set_pixel_size(xy_pixel_size, z_pixel_size)` and `dw3d.set_points_pixel_size` now scale column 0 by the z size and columns 1 and 2 by the xy size. Before, they scaled columns 0 and 1 by the xy size and column 2 by the z size: z was multiplied by the xy size and x by the z size. Results change only when the two sizes differ; **re-run any analysis that called one of them with different sizes.** (A mask stored as `(x, y, z)` was scaled correctly before and is not now: transpose it to `(z, y, x)` before meshing.) For a mask whose voxels are longer along z, prefer `algorithm.construct_mesh_in_physical_units(mask, spacing)`, which resamples the mask to cubic voxels before meshing: measured on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size and down-sampled along z, against the same tissues meshed at full resolution, it costs +0.83, +0.71 and +1.17 degrees of contact-angle error at z/xy ratios of 5/3, 2 and 3, against +1.38, +1.97 and +4.10 for meshing the native grid and scaling the mesh.

### Faster

- **The distance transform needs less memory and the tesselation graph is built faster; the meshes are bit for bit the same.** The distance transform builds its masks in `uint8` and its field in `float32` (still returned as `float64`), and the tesselation graph pairs the faces of the tetrahedra and builds its `networkx` graph without Python loops. Measured on one machine, on a 1 866-cell tissue: the distance transform's peak memory goes from 21.0 to 14.9 GiB (38.1 to 28.1 GiB when the mask is first resampled to cubic voxels), the face table and pairing from 7.8 to 1.8 s, the graph from 5.5 to 3.0 s; on a 2 000-cell synthetic foam the whole reconstruction goes from 200 to 186 s. The meshes, crop faces and cut cells of 481 masks (benchmark, resolution sweep, simulated tissues at real cell size, cut and anisotropic masks) and of two large tissues are bitwise identical before and after. Contributed by Wenhong Li and Adam L. Taylor.

### Fixed

- **`dw3d.io.load_rec_appendix` reads `resampled_spacing` as a number.** It returned the text (`'0.3'`); it now returns a float (`0.3`), as it already did for `spacing`. `resampling` stays text.
- **`dw3d.viewing.plot_cells_polyscope(..., anisotropy_factor=...)` no longer rescales the reconstruction's mesh.** With a factor other than 1 it multiplied column 0 of the algorithm's own points in place, so every call compounded the factor and a later `save_to_rec_mesh` or `save_to_vtk_mesh` wrote the scaled points. It now shows a scaled copy and leaves the mesh as it was. If you saved a mesh after viewing it with a factor, reconstruct and save it again. To mesh a mask whose voxels are longer along z in physical units, use `algorithm.construct_mesh_in_physical_units(mask, spacing)`.
- **A reconstruction no longer prints a SciPy `DeprecationWarning`.** The seed lookup imported the k-d tree from `scipy.spatial.ckdtree`, a namespace SciPy will remove in 2.0; it now uses `scipy.spatial.cKDTree`, the same class, so the meshes are unchanged.
- **`dw3d.io.load_meshio_mesh` refuses a file without labels with a `ValueError`.** A file with no `label1` or `label2` triangle cell array raised meshio's `KeyError`; it now raises the `ValueError` the function already raised for empty label arrays, naming the file and the missing array. `load_meshio_mesh_with_crop_faces` likewise. If you caught `KeyError` there, catch `ValueError`.
- **A binary `.rec` file always reads back.** `save_rec(..., binary_mode=True)` wrote the points in their own dtype, while `load_rec` reads float64, so float32 points did not read back; they are now always written as float64. For float64 points, which is what dw3d produces, the files are byte for byte what they were. The text layout is unchanged. Both layouts are now described in `save_rec`'s docstring and in the user guide's page on the `.rec` format.
- **The compressed segmentation after `construct_mesh_in_physical_units` says which frame each entry is in.** `algorithm.last_compressed_segmentation` returns the mesh's points, which that method has scaled to physical units, with the seeds and the shape of the resampled cubic grid that was meshed. Passed as they are to `dw3d.reconstruct_mask_from_dict`, the mesh is rasterised at the wrong scale; divide the points by the resampled spacing (the shortest spacing) first, and the resampled mask is rebuilt. The docstrings now say so; nothing returned has changed.

### New

- **New: masks with voxels longer along z.** `algorithm.construct_mesh_in_physical_units(mask, spacing=(0.5, 0.3, 0.3))` resamples the mask to cubic voxels of the shortest spacing by shape-based interpolation, meshes it, and returns the mesh in the unit of `spacing`; `save_to_rec_mesh` then records `coordinate_frame = physical`, the mask's own `spacing`, `source_shape`, `resampling` and `resampled_spacing` in the file's metadata appendix. The resampling alone is `dw3d.resample_to_isotropic(mask, spacing, method="shape")`. Meshing the native grid and scaling the mesh, as before, samples the surfaces coarsely along z: measured on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size and down-sampled along z (one seed), as the median over tissues of each mesh's median line-mean contact-angle error minus that of the same tissue meshed at full resolution, it costs +1.38, +1.97 and +4.10 degrees at z/xy ratios of 5/3, 2 and 3, against +0.83, +0.71 and +1.17 when the mask is resampled first. Nothing was measured above a ratio of 3: there the method warns (`dw3d.UnmeasuredAnisotropyWarning`) and still meshes. The cost is that of a mask acquired at the finer spacing (time and memory grow with the ratio). Nothing changes for `construct_mesh_from_segmentation_mask`, its meshes or its files, and a mask with cubic voxels goes through `resample_to_isotropic` untouched.
- **New: the reconstruction records where the image cut the tissue.** When a cell of the mask reaches a face of the image box, the triangles that close it on that face are not contact with the medium: the tissue goes on beyond the image. The algorithm now marks them as **crop faces** (`algorithm.crop_faces`, a boolean per triangle) and the cells they touch as **cut cells** (`algorithm.cut_cells`); `algorithm.crop_face_report` gives the counts per image face. The mesh itself is unchanged, bit for bit: same points, same triangles in the same order, same labels. foambryo reads these marks and keeps crop faces out of its equations.
- **New: an image face can be declared physical**, for a tissue that really ends there, such as on a coverslip: `algorithm.physical_image_faces = ("axis0_low",)` before reconstructing, or `MeshReconstructionAlgorithmFactory().set_physical_image_faces("axis0_low")`. The triangles on a physical face stay contact with the medium, as before. By default every image face is a crop.
- **`.rec` and `.vtk` files keep the crop.** `save_to_rec_mesh` (and `save_rec(..., crop_faces=...)`) writes the crop faces as the last block of triangles and records the block in the file's metadata appendix as `crop_faces = <first row> <count>`, with `physical_image_faces`; `dw3d.io.load_rec_with_crop_faces` reads them back. `save_to_vtk_mesh` adds a `crop_face` cell array, read by `load_meshio_mesh_with_crop_faces`. Only meshes of masks whose cells reach the image box are affected: for every other mesh the files are byte for byte what they were. A program that does not know the new key still reads a valid mesh, in which the crop faces look like medium contact.
- Measured on the 47 benchmark masks cut by planes at 10 %, 25 % and 40 % of their extent: read as contact with the medium, crop faces put 13 % to 40 % error on the tensions foambryo infers for the interfaces that touch the crop, against 9 % to 12 % when they are kept out.
- A reconstruction can also close a cell on an image face that the mask's cell does not reach, when the mask leaves only a voxel or two of medium between the cell and the image border. Those triangles are **not** crop faces: the cell was seen to end inside the image, so they stay contact with the medium, as before. A mask none of whose cells reaches the image box therefore has no crop face and no cut cell. `algorithm.crop_face_report` counts them, and lists which cells reach each image face.

### Removed

- **The three reconstruction getters deprecated since 0.5.0 are removed**, with their top-level aliases. Each was an exact synonym of a keyword call on `get_junction_protected_algorithm` (top level: `get_junction_protected_mesh_reconstruction_algorithm`), which gives the same mesh, bit for bit; call that instead:
  - `get_offset_included_algorithm(...)` → `get_junction_protected_algorithm(...)`;
  - `get_offset_excluded_algorithm(...)` → `get_junction_protected_algorithm(..., exclude_offsets_from_surface=True)`;
  - `get_cubic_score_algorithm(...)` → `get_junction_protected_algorithm(..., spline_order=3)`.

### Dependencies

- **Higher minimum versions, the oldest with which the test suite passes** (Python 3.11): numpy 2.0.0, scipy 1.13.0, scikit-image 0.23.1, edt 2.4.1 and meshio 5.3.5 (trimesh 4.0.0 and networkx 2.8.0 unchanged); for the `viewing` extra polyscope 2.2.0 and matplotlib 3.8.4. The previous minimums had never been tested: with them (numpy 1.24) the reconstruction gives slightly different meshes and the rebuilt mask 32-bit labels, and edt 2.4.0, meshio 5.3.4 and older scipy and scikit-image releases do not work with numpy 2.

### Prepared for 0.6.0 (4 October 2026, not published)

- **Faster, and lighter in memory, on large tissues; the meshes are bit-for-bit the same.** Three steps of the reconstruction were rewritten without changing any result:
  - the seeds are found in two passes over the volume instead of one pass per label;
  - junction relocation's self-intersection guard finds its candidate vertex pairs with one batched neighbour search per range of reach, and checks only the triangles those pairs involve;
  - the trijunction curves are detected in slabs of the volume instead of on the whole volume at once.

  Measured on one machine on a 1 866-cell tissue meshed on its native grid, together with foambryo's sparse tension solve: from mask to tensions, 9.5 min instead of 47.5 min with 0.5.1, and a peak memory of 30.9 GiB instead of 54.7 GiB (measured again for 0.7.0, nothing else running). Seed extraction went from 28 min to 6 s, the relocation guard from 6.6 min to 45 s, and curve detection from 82 s to 49 s and from 37.6 to 9.7 GiB. The meshes of the 47 benchmark masks, 492 embryo timepoints, 10 cut masks and the 4 example images are bitwise identical before and after, as are those of two large tissues. The memory peak is now junction relocation's whole-mesh self-intersection check.

- **New: the cells whose mesh volume disagrees with their mask are listed, with one warning.** A reconstruction can lose a small cell, or give part of a cell to a neighbour, while the mesh stays closed and manifold, so no surface check sees it. After every reconstruction the algorithm now compares each cell's mesh volume with its voxel count and lists, in `cell_volume_report`, the cells outside 0.5–1.5 times it, with a non-positive volume, or with no surface at all; `CellVolumeDisagreementWarning` is emitted once when the list is not empty. The same check is available as `dw3d.compare_cell_volumes_with_mask`. The mesh itself is unchanged, bit for bit. Measured with the default settings: 2 of 240 cells on the 47 benchmark masks, 169 of 5 038 on two 246-timepoint embryo series (almost all in the last frames of one series, where the mask suddenly holds many very small labels), 20 cells and 45 cavities on a 1 866-cell tissue. Interfaces of the listed cells are missing or misplaced, and a pressure inferred for them is not a measurement.

- **The repository no longer ships lock files.** `uv.lock` recorded version 0.4.1 and a package index that only the lab can reach, and `locked-requirements.txt` pinned version 0.3.5. The supported dependency ranges are those in `pyproject.toml`.

- The package metadata names the Turlier lab as author; Matthieu Perez remains the maintainer.

- **Fixed: triangles on the image border were wound the wrong way.** When a cell touches the edge of the image, the triangles that close it along the image border had their normal pointing out of the cell instead of into it, against the convention every other triangle follows (the normal points from the first label into the second). Every such triangle was affected, not a random share. A cell closed by the image border then reads too small a volume, often a negative one, and any tension or pressure computed from the mesh is wrong.

  **Who is affected.** Anyone whose cells touch the image border, above all tissue cropped through its cells. Measured with the default reconstruction on ten benchmark masks cut through their middle: before the fix, 38 of the 54 cells on the cut had a negative volume; after it, none. The tensions foambryo infers from those meshes changed by up to 199 %, and every cell pressure by 37–139 %. On masks whose cells stay clear of the border nothing changes: of 543 uncut masks (47 benchmark cases, 492 embryo timepoints and the 4 example images), 542 have no border triangle and come out bit-for-bit identical. The one exception has two border triangles on a single cell, whose volume goes from 0.73 to 1.01 of its voxel count.

  Only the vertex order of border triangles changes: vertex positions, the set of triangles and their labels are identical. A cut surface is still treated as contact with the medium, as before.

- **Fixed: a few triangles lying on a flat tetrahedron were wound the wrong way.** The tetrahedral mesh the surface is cut from contains flat tetrahedra (four coplanar points of the voxel grid). A triangle whose orientation was read from such a tetrahedron got its winding from a rounding error, and about half of them came out reversed; each reversed triangle makes its two cells read a volume slightly different from the volume they actually enclose. These triangles are now oriented from the tetrahedron on their other side. Measured with the default settings: 10 triangles in 8 cells on the 47 benchmark masks (volume error 0.4 % median, 0.7 % at most), 20 triangles in 19 cells on two 246-timepoint embryo series (2 % median, 14 % at most), 9 triangles in 10 cells on a 1 866-cell tissue; after the fix, every cell's volume equals that of the tetrahedra it is made of. Only the vertex order of those triangles changes: vertex positions, the set of triangles and their labels are bit-for-bit identical on every mask measured.

- **Fixed: junction relocation could turn a very small cell inside out.** Relocation's guard checked that every triangle around a moved vertex kept its orientation, but a cell made of a single tetrahedron could still be inverted by a vertex moved across its opposite face, with every triangle around it passing that check; the cell then had a negative volume and passed every surface check. The guard now also refuses a move that would leave any cell it touches with a volume of zero or less, and nothing else: a cell may still shrink. A refused move leaves its vertex where it was; `relocation_report` counts these refusals and names the cells. Measured with the default settings: the clause changes no mesh of the 47 benchmark masks, the 141 masks of the resolution sweep, 10 masks cut through their cells, a 110-cell tissue, or 484 timepoints of two embryo series (all bit-for-bit identical). It fires on two over-segmented frames of an embryo series, where it keeps one cell from inverting on one frame, and on a 1 866-cell tissue, where it keeps one cavity from inverting.

## 0.5.1

**0.5.1 republishes the final 0.5.0 (as on GitHub's `master` since 27 September).** The 0.5.0 wheel and source archive on PyPI were built from an earlier snapshot of the release: their licence metadata says CC BY-NC-SA 4.0 instead of GPL-3.0-or-later, the `viewing` extra still installs napari, and `dw3d.viewing` still has the napari viewer that 0.5.0 removed. GitHub's `v0.5.0` tag points at that same earlier snapshot, and is left as it is. Install 0.5.1; 0.5.0 is withdrawn from PyPI (yanked).

Nothing else changes: the code is the final 0.5.0's. The package metadata names the Turlier lab as author (Matthieu Perez remains the maintainer), the README's links point to the 0.5.1 tag, and the two stale lock files 0.5.0 carried (`uv.lock` and `locked-requirements.txt`) are gone.

## 0.5.0

**New licence: from 0.5.0, `delaunay-watershed-3d` is licensed under the GNU General Public License, version 3 or (at your option) any later version** (SPDX `GPL-3.0-or-later`; the text of version 3 is in `License`), replacing CC BY-NC-SA 4.0. Commercial use is allowed, under the GPL's conditions (a distributed derivative work must be released under the same licence, with its source). Earlier releases keep the licence they were published under.

**The default changed: junction relocation is on in every reconstruction configuration.** Meshes come out with the same triangles and labels as in 0.4, and different vertex positions on the trijunction lines. To get the 0.4 meshes back exactly, switch it off: pass `relocate_junctions=False` to `MeshReconstructionAlgorithm`, call `MeshReconstructionAlgorithmFactory().set_junction_relocation(False)`, or set `algorithm.relocate_junctions = False` on an algorithm a `get_*_mesh_reconstruction_algorithm` function returned. See `RELEASE_NOTES.md` for the limitations that matter on real data.

- **Junction relocation, now the default.** `dw3d.relocate_junction_vertices` is a post-process on a finished mesh that moves its trijunction vertices onto the trijunction curves the mask itself gives. It changes vertex positions only — triangles and material labels come back untouched — so a treated mesh can be compared vertex-for-vertex with its own untreated original. Read what it did from the algorithm's `relocation_report`, and whether it ran, with which guard and whether it warned from the new `relocation_provenance`.

  **What it costs, in seconds**, because the reconstruction it follows varies by an order of magnitude between volumes: about +0.75 s on the benchmark meshes, whose reconstruction takes about 5 s (+15 %); 0.1–1.5 s per timepoint on real embryo volumes, whose reconstruction takes about 0.6 s (median 0.34 s and 0.48 s on two developmental time series; about 65 s per 154-timepoint series). Within one real series it grows faster than linearly with the number of triangles — a fitted exponent of 2.68 over a 2.2-fold range of mesh sizes, which describes that series and is not a tissue-scale law. A faster neighbour search is planned for 0.5.1; it will leave every relocated mesh bit-for-bit unchanged.

- **New warning: `dw3d.CoarseSpacingRelocationWarning`** (a `UserWarning`), emitted once per reconstruction when relocation is on and the point placement's `min_distance` is 5 or more. Its text: *"At min_distance ≥ 5, the check on surface-tangent contact angles showed a small drift that lies within the check's own noise, and larger spacings have not been measured; facet-based contact angles improve. If tangent-based angles matter for your analysis, compare with `relocate_junctions=False` (or `set_junction_relocation(False)`)."* A custom point-placement function that carries no `min_distance` never triggers it. Filter it like any warning: `warnings.filterwarnings("ignore", category=dw3d.CoarseSpacingRelocationWarning)`.

- **Removed**, as their deprecation warnings announced since 0.4.0: `get_v0_3_algorithm` (use `get_dithered_algorithm`), `get_a1b_algorithm` (use `get_deterministic_algorithm`), `get_a3_a5_algorithm` (use `get_junction_protected_algorithm`), and the matching top-level `get_v0_3_…`, `get_a1b_…` and `get_a3_a5_mesh_reconstruction_algorithm`.

- **Removed, without a deprecation period: `dw3d.viewing.plot_in_napari`.** It already failed with the napari that `pip` installs today: with napari 0.9.1 (measured on 27 Sep 2026) it raised `AttributeError`, because `napari.view_image` no longer exists, and its `add_points` calls pass an `n_dimensional` argument that napari 0.9.1 no longer accepts. **napari is no longer part of the `viewing` extra**, which now installs Polyscope and matplotlib only; the Polyscope viewer `dw3d.viewing.plot_cells_polyscope` is unchanged.

- **Deprecated, removed in 0.6.0** — three exact synonyms, which now emit a `DeprecationWarning` naming the equivalent call and still return the identical mesh: `get_offset_included_algorithm()` → `get_junction_protected_algorithm()`; `get_offset_excluded_algorithm(...)` → `get_junction_protected_algorithm(..., exclude_offsets_from_surface=True)`; `get_cubic_score_algorithm()` → `get_junction_protected_algorithm(spline_order=3)`; and their top-level `get_*_mesh_reconstruction_algorithm` aliases. Six configurations remain recommended: default, dithered, deterministic, link-checked, junction-protected and boundary-layer.

- **Golden masters.** `tests/golden/` now holds the relocated meshes; the 0.4 fixtures moved, byte for byte, to `tests/golden_without_relocation/`, and every golden-master test runs against both — the second with `relocate_junctions=False`.

- **Documentation.** `BENCHMARKS.md` is self-contained: each number states its population and how it was measured. Those numbers were measured **without** junction relocation (the 0.4 default). Docstrings no longer cite files that are not distributed. The README cites the published article (Ichbiah et al., *Nature Methods* 2023, doi:10.1038/s41592-023-02084-7) instead of its preprint, and a new `CITATION.cff` says the same. The mask-compression notebook (`Examples/example_2_mask_compression_reconstruction.ipynb`) opens its napari viewer with `napari.Viewer()` and `add_image`, since `napari.view_image` no longer exists in current napari, and says that napari must be installed separately; the advanced-use notebook (`example_3_advanced_use.ipynb`) no longer has its napari section.

The rest of this section describes what 0.5.0 adds besides the default:

  What it costs and what it buys, measured over 40 benchmark cases at four reconstruction spacings and on four point samplers — 16 independent adjudications, all passing. Line position error, local tangent error and discrete curvature all improve with their confidence intervals clear of zero at every spacing on every sampler; no pre-existing mesh-validity number changes on any case. On the seed-free samplers it also improves contact angles by 2.3–3.5 degrees and gauge-matched tension error by 20–40 %; on the shipped default's seeded sampler the line improves and the contact angle does not move significantly either way. Its cost is stated in seconds above.

  It is guarded twice. The local guard refuses any move that would invert or collapse a triangle touching the vertex. The second guard is the one that matters: a move can drive one sheet of the mesh through a distant sheet while every triangle around the moved vertex stays perfectly oriented, and no local check can see that. Measured on the benchmark cohort, all twelve existing validity numbers stayed bit-identical on 40 of 40 cases while 23 such crossings existed on 5 of them. The second guard makes the crossing count non-increasing by construction, and it costs between 0.04 % and 0.21 % of the moves to do it.

- **`dw3d.find_self_intersections` and `dw3d.count_self_intersections`.** The exact triangle–triangle intersection test the relocation certifies itself with, now part of the library. Two triangles sharing a vertex are adjacent and are never tested — three triangles meet along every trijunction edge by design — so what it counts is a genuine defect, never the intended non-manifold structure. It is a count, never a score, and no threshold is proposed.

## 0.4.2

No change to reconstruction behaviour: every default is bitwise identical to 0.4.1 (verified by hash on the mesh output of the dithered and deterministic configurations).

- **The three deprecation warnings' text no longer implies `0.5.0` already exists.** `get_v0_3_algorithm`, `get_a1b_algorithm` and `get_a3_a5_algorithm` (and the matching `dw3d/__init__.py` aliases) now say "removed in a future 0.5.0 release" instead of "removed in 0.5.0"; the replacement function and the removal commitment are unchanged.
- **New `dw3d_benchmarks/mesh_quality_comparison/` package**: compares mesh quality (M1-M7: contact angles, interface areas, cell volumes, curvature, junction-line geometry, mesh validity, cost) across the four core reconstruction configurations and across dw3d versions. A `benchmarks` extra (`pip install ".[benchmarks]"`) covers the figure scripts' matplotlib requirement without pulling in `viewing`'s napari dependency.
- One-line fix: `dw3d_benchmarks/run_case.py`'s `VARIANT_GETTERS` comment stated the default is `offset_excluded_linkcheck`; it is not — the default is `dithered`.

## 0.4.1

Release hygiene only. No change to reconstruction behaviour: every default is bitwise identical to 0.4.0 on all 47 benchmark meshes.

- **`MeshReconstructionAlgorithmFactory`'s three development-named factory methods are renamed to describe their mechanism**: `get_v0_3_algorithm` → `get_dithered_algorithm`, `get_a1b_algorithm` → `get_deterministic_algorithm`, `get_a3_a5_algorithm` → `get_offset_included_algorithm`, and the matching top-level exports in `dw3d/__init__.py` (e.g. `get_v0_3_mesh_reconstruction_algorithm` → `get_dithered_mesh_reconstruction_algorithm`). The old names still work — they now emit a `DeprecationWarning` naming the replacement — and will be removed in 0.5.0.
- **`dw3d_benchmarks/run_fingerprints.py`'s `--variant` strings are renamed to match** (`v0_3` → `dithered`, `a1b` → `deterministic`, `a3_a5` → `offset_included`); the old strings are still accepted, with a printed note pointing at the replacement.
- **Development files no longer ship or return.** Development notes and scratch scripts are removed from the repository tip and gitignored so they cannot be re-added by accident. The project's development record is kept in the development repository and has never been part of the packaged distribution; confirmed by building the wheel and listing its contents.
- Internal benchmark filenames and test names that were named after a development phase (e.g. `analyze_a5.py`, `d2_defect_patches.py`, baseline JSONs prefixed `a6_`/`a7_`/`a8_`/`a8b_`) are renamed to describe what they measure. This does not change any benchmark's recorded numbers.

## 0.4.0

### New: trijunction curves read from the mask, keyed to the mesh's own junctions

`extract_junction_curves(mask, points, triangles, labels)` returns one polyline per material triple, computed from the segmentation mask and matched against the reconstructed mesh's own trijunctions. **Purely additive**: nothing in the reconstruction pipeline calls it, no default changed, no dependency was added, and the contact-angle and mesh-validity numbers are bitwise unchanged (verified on 1974 measured values across 47 cases at two point spacings).

```python
from dw3d import extract_junction_curves
from dw3d.io import load_rec

points, triangles, labels = load_rec("000_mesh.rec")  # or your own reconstruction
curves = extract_junction_curves(mask, points, triangles, labels)
curves.matched      # the usable set: one-to-one with a mesh trijunction
curves.invented     # triples the mask says exist and the mesh does not have
curves.match_rate   # 0.900 on the 47-case benchmark at min_distance=3
```

**Why you might want it.** These curves are measurably closer to ground truth than the junction lines the mesh produces on its own — median position error 0.5846 voxels against 0.8642 at `min_distance=3`, a factor of 1.48, and more as the mesh coarsens (1.90x at `min_distance=7`). Every head-to-head comparison measured favours them on 1.000 of 10,000 bootstrap draws.

**What it costs you.** They are *less complete*: 90 % of the mesh's trijunctions get exactly one curve, and the rest arrive fragmented. `matched` and `invented` are deliberately separate fields and are never merged, because they are structurally different populations — matched curves lie a median 0.68 voxels from all three of their own interfaces, invented ones 29.7, with **none** within a voxel.

**What it is not.** The curves sit about 4.8x above the mask's own identifiability floor (0.107 voxels). This is better geometry, not correct geometry.

The default estimator is `"plain"` — detection only, about 0.8 s per benchmark case against the ~1.5 s dw3d spends reconstructing it. `estimator="anchor"` solves one small linear program per vertex, costs about six times as much, and buys 11 % of position error; its argument is that it carries a smaller systematic bias against narrow junction wedges, not that 11 %. `estimator="centroid"` exists to reproduce the research result and is not a recommendation.

See `BENCHMARKS.md` (section "Junction curves") for the full numbers and the controls, and `Examples/example_4_junction_curves.ipynb` for a worked example that runs in a few seconds on data already in this repository.

### The default reconstruction is the published `dithered` algorithm again

Between one earlier internal release and this one, the default reconstruction algorithm was a revised meshing pipeline (deterministic point placement, explicit junction detection and protection, and a topology-preserving cleanup of the extracted surface). That pipeline produces measurably better mesh geometry — lower interface-area bias against ground truth, far fewer degenerate ("all-surface") tetrahedra, better junction placement — but end-to-end tension-inference accuracy through `foambryo`, on a benchmark of 42 mechanically-equilibrated cases, is **worse** with it than with the published algorithm, on every inference method tested. Since that downstream accuracy is what most users of this package ultimately care about, **the default reconstruction algorithm has reverted to the published (`dithered`) configuration.**

The revised meshing pipeline is not removed: it remains fully available, opt-in, as `get_link_checked_mesh_reconstruction_algorithm`. Reach for it when mesh geometry and topology matter more to you than today's downstream inference accuracy. See `BENCHMARKS.md` for the numbers behind this decision and the accuracy finding stated plainly, without spin: the revised pipeline is currently the worse choice for tension inference, despite being the better mesh in isolation.

This reconstructs bit-for-bit what dw3d 0.3.6 produced on the current numpy/scipy/ scikit-image stack (see `get_default_mesh_reconstruction_algorithm`'s docstring for the one caveat: the pre-0.4.0 watershed tie-break ordering, which had no single well-defined answer, is not restored, since it depended on which sorting kernel happened to run it).

### New capability, unaffected by the default change

- `set_tiled_edt_method(..., dtype=np.float64)`: an opt-in Euclidean-distance-transform variant that uses roughly a third of the peak memory of the default EDT computation at large volumes, while reproducing it bit-for-bit — unlike the existing `float32` tiled variant, which saves the same memory but changes the output when combined with the revised meshing pipeline's normal/score computations. Not the default EDT for either reconstruction variant; a deliberate choice for memory-constrained large volumes.

### `.rec` / `.arec` files can now carry a metadata appendix

`save_rec(..., metadata={...})` appends a trailing block of ASCII `# key = value` lines recording `coordinate_frame` (`voxel_index`, `physical`, or `unknown` — the field this exists for: it is what lets a reader tell whether coordinates are raw voxel indices or already-scaled physical units), `axis_order`, `spacing`, `spacing_unit`, `source_shape` and `crop_offset`. Read it back with the new `load_rec_appendix(filename)`, which returns `None` for a file with no appendix — never a default of isotropic spacing. See `README.md` for the field list and the format.

The block is written at the *end* of the file, never the start, so this is purely additive: `save_rec` without `metadata` (the default) writes byte-for-byte the same file as before this existed, and every reader already in the wild — including dw3d 0.3.6 and released `foambryo` — loads a file with an appendix exactly as it always has. `load_rec`'s return type is unchanged; the appendix is read separately, on purpose.

### Nothing else changed

Every other named reconstruction getter (`get_deterministic_algorithm`, `get_offset_included_algorithm`, `get_boundary_layer_algorithm`, `get_offset_excluded_algorithm`, `get_cubic_score_algorithm`) is unchanged and produces exactly the output it always has.
