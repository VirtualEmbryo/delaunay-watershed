# Release notes — 0.7.1, 26 October 2026

`delaunay-watershed-3d` (this repository) and `foambryo` ship together as 0.7.1. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and foambryo's `RELEASE_NOTES.md` covers the inference side. Install with `pip install --upgrade delaunay-watershed-3d foambryo`.

## What changes for you

1. **Meshes change slightly: they no longer depend on Qhull's row order.** The watershed's tie-break between faces of exactly equal score, the seed lookup and mesh surgery read the order in which Qhull returned the tetrahedra, which changes when the same points come in another order. Every reconstruction now sorts the tetrahedra first. On the 47 benchmark masks (default configuration) 9 row orders gave 9 different meshes in the median case and now give one; against 0.7.0, 45 of the 47 meshes change by a few faces (median 13 of a few thousand), and foambryo's median relative tension error (`YoungDupre`) goes from 0.0777 to 0.0763. Set `canonical_tetrahedron_order = False` to reproduce 0.7.0's meshes on the same machine.
2. **Masks with a lumen, a medium cut in two by the image, or a region too small for the tesselation are meshed differently, and were wrong before.** An enclosed pocket of medium lined by two or more cells and at least `min_distance + 1` voxels deep is now its own region, recorded as a medium (`label_enclosed_media = False` keeps 0.7.0's behaviour); every piece of exterior gets its own seed; a seed that fell in a neighbouring region moves into its own, and a region with no tetrahedron of its own is absent from the mesh and named instead of growing as a phantom inside a neighbour. On a public 697-cell embryo with a lumen, one lining cell had 2.40 times its voxel volume; now every cell is within 0.903 to 1.024 times. Of 543 default meshes (the benchmark, two real embryo series, the test images) 533 are bit for bit unchanged.
3. **Mesh surgery no longer leaves abnormal non-manifold edges** by undoing its own repairs; none of the default route's 47 benchmark meshes changes; 4 of the 282 anisotropic benchmark meshes change by a few triangles, and 31 of the 84 meshes of simulated tissues at a real cell size change.
4. **The junction-protected configurations no longer depend on how SciPy breaks exact ties**, so they place the same points on every machine; on macOS some of their meshes change by a point or a few.
5. **The Polyscope viewer drew the mirror image of the tissue**, and now draws it the right way round. Only the picture changes: the mesh, its files and every number are unchanged. Screenshots taken with 0.7.0's viewer show the mirror image.

## New

* **Reports and warnings**: `enclosed_medium_report`, `medium_labels`, `dw3d.find_enclosed_media`; `EnclosedMediumWarning` (pockets left to the cells), `UnresolvedCellWarning` (regions absent from the mesh), `AbnormalEdgeWarning` (abnormal edges left in a finished mesh). The `.rec` appendix records the media and the unresolved regions; foambryo 0.7.1 reads them.
* **Cross-machine checks**: the golden master no longer depends on Qhull, and the test suites run on Linux x86-64 CI with the newest and the lowest supported dependencies.

## What can mislead you

1. **Between machines, meshes may still differ at exactly cospherical points** (five or more lattice points on one sphere), where Qhull's choice of tetrahedralisation itself differs; on one machine a reconstruction is reproducible.
2. **A small pocket of medium stays with the cells**: below the size bound, or lined by a single cell (a hole in that cell's segmentation). `EnclosedMediumWarning` lists them.
3. The points of 0.7.0's list still hold.

## Known issues

1. **The crop report can miss a cut cell.** On a small test mask of four cells in a ball cut by the image box, the cells with voxels on the box are 1, 3 and 4; `cut_cells` reports 2, 3 and 4, as 0.7.0 did, and foambryo, which reads the same crop faces, then gives cell 1 a pressure and cell 2 none. Not yet measured on realistic cut masks: check `cut_cells` against the mask's own border labels.
2. **`reconstruct_mask_from_dict` depends on the installed trimesh**: the same mesh is subdivided into 1 901 565 vertices with trimesh 4.12.2 and 1 319 748 with 5.1.1, and the rebuilt masks differ by 509 to 819 voxels per test image. The mesh does not depend on trimesh.
3. **dw3d 0.7.0 reads the new `.rec` keys as strings**: a file with media or unresolved regions written by 0.7.1 loads in 0.7.0, but its `media` and `unresolved_regions` are text. A file with neither is unchanged.
4. **Mesh files keep the mask's column order, `(z, y, x)`.** ParaView, MeshLab and other programs that read the columns as `(x, y, z)` show the mirror image of the tissue unless they exchange columns 0 and 2; so does napari in its default 3-D orientation.

## Credits

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.

# Release notes — 0.7.0, 17 October 2026

`delaunay-watershed-3d` (this repository) and `foambryo` ship together as 0.7.0, the first release on PyPI since 0.5.1. 0.6.0 was prepared but not published; its changes are part of 0.7.0 and are included below. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and foambryo's `RELEASE_NOTES.md` covers the inference side. Install with `pip install --upgrade delaunay-watershed-3d foambryo`.

## What changes for you

1. **`set_pixel_size` scaled the wrong axes; results change when the xy and z sizes differ.** dw3d's mesh points are in the mask's index order, `(z, y, x)` for a microscope stack, so column 0 is z. `set_pixel_size` (the function in `dw3d.mesh_utilities`, the reconstruction method and `dw3d.set_points_pixel_size`) now scales column 0 by the z size and columns 1 and 2 by the xy size. Before, z was multiplied by the xy size and x by the z size. **Re-run any analysis that called it with different xy and z sizes**; with equal sizes nothing changes. A mask stored as `(x, y, z)` was scaled correctly before and is not now: transpose it to `(z, y, x)` before meshing.
2. **Meshes of cells that touch the image border change, and were wrong before.** The triangles that close a cell along the image border were wound the wrong way, so such a cell read too small a volume, often a negative one, and any tension or pressure computed from its mesh was wrong. On ten benchmark masks cut through their middle, 38 of the 54 cells on the cut had a negative volume before the fix and none after; the tensions foambryo infers from those meshes changed by up to 199 %, and every cell pressure by 37–139 %. **Reconstruct again any mesh whose cells touch the image border**, above all tissue cropped through its cells. Only the vertex order of the border triangles changes, and meshes whose cells stay clear of the border are unchanged (542 of 543 uncut masks bit-for-bit; the exception has two border triangles on one cell).
3. **A few triangles lying on flat tetrahedra change orientation** (10 triangles in 8 cells on the 47 benchmark masks), so that every cell's volume now equals that of the tetrahedra it is made of.
4. **Junction relocation can no longer turn a very small cell inside out.** Its guard now also refuses a move that would leave a cell it touches with a volume of zero or less; `relocation_report` counts these refusals and names the cells. No mesh of the 47 benchmark masks, the resolution sweep, the cut masks or 484 embryo timepoints changes.
5. **New warning, `CellVolumeDisagreementWarning`.** After every reconstruction, the cells whose mesh volume is outside 0.5–1.5 times their voxel count, non-positive, or missing are listed in `cell_volume_report`. The interfaces of a listed cell are missing or misplaced, and a pressure inferred for it is not a measurement. Expect it on masks that hold many very small labels.
6. **Faster on large tissues, with the same meshes.** Seed extraction, the relocation guard and the trijunction-curve detection were rewritten; every mesh measured is bit-for-bit identical. On a 1 866-cell tissue meshed on its native grid, from mask to tensions with foambryo's sparse tension solve: 9.5 min instead of 47.5 min with 0.5.1, and a peak memory of 30.9 GiB instead of 54.7 GiB (one machine, nothing else running). The peak is now relocation's whole-mesh self-intersection check. The same tissue resampled to cubic voxels first (`construct_mesh_in_physical_units`) costs more: its reconstruction takes about 13.5 min and 38 GiB.
7. **No lock files.** `uv.lock` and `locked-requirements.txt` are gone; the supported dependency ranges are those in `pyproject.toml`.
8. **Viewing a mesh no longer rescales it.** `plot_cells_polyscope(..., anisotropy_factor=...)` with a factor other than 1 multiplied the reconstruction's own z column, so a mesh saved after viewing was scaled: reconstruct and save it again. Also: `load_meshio_mesh` refuses a file without labels with a `ValueError` (it raised `KeyError`), a binary `.rec` file always stores its points as float64, and a reconstruction no longer prints a SciPy deprecation warning. Reconstructions are unchanged, bit for bit (`CHANGELOG.md`).
9. **Three deprecated names are removed.** `get_offset_included_mesh_reconstruction_algorithm`, `get_offset_excluded_mesh_reconstruction_algorithm` and `get_cubic_score_mesh_reconstruction_algorithm` (and the factory's methods of the same names), deprecated since 0.5.0, are gone. Call `get_junction_protected_mesh_reconstruction_algorithm()`, the same with `exclude_offsets_from_surface=True`, or the same with `spline_order=3`: the meshes are identical, bit for bit.
10. **Higher minimum versions**: numpy 2.0.0, scipy 1.13.0, scikit-image 0.23.1, edt 2.4.1, meshio 5.3.5 (polyscope 2.2.0 and matplotlib 3.8.4 for viewing), the oldest with which the test suite passes. Older numpy releases give slightly different meshes.

## New

* **Masks with voxels longer along z**: `algorithm.construct_mesh_in_physical_units(mask, spacing)` resamples the mask to cubic voxels of the shortest spacing by shape-based interpolation, meshes it, and returns the mesh in the unit of `spacing`; `save_to_rec_mesh` records the frame, the spacing and the resampling. Measured on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size and down-sampled along z, against the same tissues meshed at full resolution: +0.83, +0.71 and +1.17 degrees of contact-angle error at z/xy ratios of 5/3, 2 and 3, against +1.38, +1.97 and +4.10 for meshing the native grid and scaling the mesh.
* **Crop faces and cut cells**: `algorithm.crop_faces`, `algorithm.cut_cells`, `algorithm.crop_face_report`, `physical_image_faces`; `.rec` and `.vtk` files keep them. The mesh is unchanged, bit for bit.
* **A lighter distance transform and a faster tesselation graph; the meshes are bit for bit the same.** The distance transform builds its masks in `uint8` and its field in `float32` (still returned as `float64`), and the tesselation graph is built without Python loops. Measured on one machine, on a 1 866-cell tissue: the distance transform's peak memory goes from 21.0 to 14.9 GiB (38.1 to 28.1 GiB when the mask is first resampled to cubic voxels), the face table and pairing from 7.8 to 1.8 s, the graph from 5.5 to 3.0 s; on a 2 000-cell synthetic foam the whole reconstruction goes from 200 to 186 s. The meshes, crop faces and cut cells of 481 masks and of two large tissues are bitwise identical before and after. Contributed by Wenhong Li and Adam L. Taylor.

## What can mislead you

1. **Above a z/xy ratio of 3 nothing was measured**: `UnmeasuredAnisotropyWarning`, and the mesh is built anyway.
2. **`min_distance` counts voxels of the resampled, cubic grid** in `construct_mesh_in_physical_units`: the mesh, its time and its memory are those of a mask acquired at the short spacing, so they grow with the ratio.
3. **A crop face is not contact with the medium.** A program that does not read the crop-face key of a `.rec` file reads a valid mesh in which the crop faces look like medium contact.
4. The limitations listed for 0.5.0 below still hold, except 7, which concerns foambryo and is fixed in its 0.7.0, and the end of 8: the faster neighbour search planned for 0.5.1 is in 0.7.0 (item 6 above), and on a 1 866-cell tissue the relocation guard takes 45 s instead of 6.6 min, with the same meshes (`CHANGELOG.md`).

## Credits

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.


# Release notes — 0.5.1

**0.5.1 is the final 0.5.0 (as on GitHub's `master` since 27 September), published again.** The 0.5.0 wheel and source archive on PyPI were built from an earlier snapshot of the release: their licence metadata says CC BY-NC-SA 4.0 instead of GPL-3.0-or-later, the `viewing` extra still installs napari, and `dw3d.viewing` still has the napari viewer. GitHub's `v0.5.0` tag points at that earlier snapshot too. Install 0.5.1 (`pip install --upgrade delaunay-watershed-3d`); 0.5.0 is withdrawn from PyPI (yanked). Reconstruction is unchanged: a mesh from 0.5.1 is the mesh the final 0.5.0 gives. The 0.5.0 notes below apply to 0.5.1 as they are.

# Release notes — 0.5.0

`delaunay-watershed-3d` (this repository) and `foambryo` ship together as 0.5.0; foambryo 0.5.0 requires `delaunay-watershed-3d>=0.5.0`. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and foambryo's `RELEASE_NOTES.md` covers the inference side.

0.5.0 is the first release on PyPI since 0.3.6; `CHANGELOG.md` also lists the changes of 0.4.0–0.4.2, which were never published there.

## Licence

From 0.5.0 `delaunay-watershed-3d` is licensed under the **GNU General Public License, version 3 or (at your option) any later version** (SPDX `GPL-3.0-or-later`), replacing CC BY-NC-SA 4.0: commercial use is allowed, under the GPL's conditions. Earlier releases keep the licence they were published under.

## The one change that affects every mesh

**Junction relocation is on by default, in every reconstruction configuration.** After the mesh is built, its trijunction vertices are moved onto the trijunction curves the segmentation mask itself gives, under a guard that cannot create a self-intersection anywhere in the mesh. Triangles and material labels are unchanged; vertex positions on trijunction lines are not. Measured over 40 equilibrium benchmark cases, four spacings and four point samplers: junction-line position, tangent and curvature errors all improve with their confidence intervals clear of zero, and no mesh-validity number changes.

**To get the 0.4 meshes back exactly**:

```py
algorithm = get_default_mesh_reconstruction_algorithm()
algorithm.relocate_junctions = False
```

or `MeshReconstructionAlgorithmFactory().set_junction_relocation(False)`, or `MeshReconstructionAlgorithm(..., relocate_junctions=False)`. Whether relocation ran on your mesh is in `algorithm.relocation_provenance`.

**At `min_distance >= 5` you will see `CoarseSpacingRelocationWarning`**, once per reconstruction: at that spacing the check on surface-tangent contact angles showed a small drift within its own noise, and larger spacings are unmeasured; facet-based angles improve. If tangent-based angles matter for your analysis, compare with `relocate_junctions=False`.

## Names removed and deprecated

Removed: `get_v0_3_algorithm`, `get_a1b_algorithm`, `get_a3_a5_algorithm` (use `get_dithered_algorithm`, `get_deterministic_algorithm`, `get_junction_protected_algorithm`). Removed without a deprecation period: `dw3d.viewing.plot_in_napari`, which already failed with napari 0.9.1; napari is no longer part of the `viewing` extra (Polyscope and matplotlib only). Deprecated, removed in 0.6.0: `get_offset_included_algorithm`, `get_offset_excluded_algorithm`, `get_cubic_score_algorithm` — exact synonyms of keyword calls on `get_junction_protected_algorithm`, which their warnings name.

## Limitations

Every item is measured; each is one sentence and its number.

1. **The benchmark under-rewards tangent-based improvements.** Its ground-truth meshes are facet-equilibrated (Surface Evolver), so an improvement in the surface-*tangent* description of a contact angle is scored against a facet-based truth and can read as no change.
2. **The second-order angle reader has its own floor, and it cannot resolve junction relocation's effect.** On a perfectly smooth analytic surface it reads contact angles with a floor of **0.245°**, and at every point spacing measured (0 of 32 spacing-by-variant cells) its check could not resolve the effect of junction relocation on tangent-based angles, in either direction.
3. **On real embryos, do not use second-order line means at sub-degree scale.** The second-order reader covers a median **0.804** of incident trijunction faces on real embryo meshes (minimum **0.454**), and its line means move by **1.75° [1.41, 2.17]** when uncovered faces are excluded instead of filled by the chord reading.
4. **Real embryo cells are small in voxels, so every benchmark statement is about one spacing step optimistic for them.** Their equivalent radius at trijunctions is a median **17.6** voxels against **38.2** in the benchmark, so real data at `min_distance 3` sits where the benchmark sits at `min_distance 4`; for cells this small, consider a smaller `min_distance`.
5. **Folded trijunctions on real meshes are mostly real geometry.** Real embryo meshes carry **3.2×** the benchmark's fraction of reflex (folded) trijunctional edges, and more than half of them sit where the segmentation mask's own wedge already exceeds 180°.
6. **The force-balance residual `rho_free` alarms on every reconstructed mesh.** Its median is **0.578** on reconstructed meshes against **0.0118** on the reference meshes, so a high value on your mesh is expected and is not by itself a sign of a bad segmentation.
7. **`ForceBalanceMAP` is not scale-invariant**, contrary to its docstring: rescaling a mesh moves its relative tensions by up to **43 %**; plain `ForceBalance` is exact under rescaling.
8. **Junction relocation, now on by default, has a cost that grows faster than linearly with mesh size.** It adds about **+0.75 s** to a ~5 s benchmark reconstruction (**+15 %**) and **0.1–1.5 s** per timepoint to a ~0.6 s real-embryo reconstruction (median **0.34 s** and **0.48 s** on two developmental time series, about 65 s per 154-timepoint series); within one real series it grows with triangle count with a fitted exponent of **2.68** over a 2.2-fold range of mesh sizes, which is not a tissue-scale law. **A faster neighbour search is planned for 0.5.1**, and will leave every relocated mesh bit-for-bit unchanged.
