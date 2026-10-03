# Release notes — 0.5.1

**0.5.1 is 0.5.0 as tagged on GitHub, published again.** The 0.5.0 wheel and source archive on PyPI were
built from an earlier snapshot of the release: their licence metadata says CC BY-NC-SA 4.0 instead of
GPL-3.0-or-later, the `viewing` extra still installs napari, and `dw3d.viewing` still has the napari viewer.
Install 0.5.1 (`pip install --upgrade delaunay-watershed-3d`); 0.5.0 is withdrawn from PyPI (yanked).
Reconstruction is unchanged: a mesh from 0.5.1 is the mesh 0.5.0 gives. Only the version, the author
metadata (the Turlier lab) and the README's links (to the 0.5.1 tag) differ, and two stale lock files are
gone from the repository. The 0.5.0 notes below apply to 0.5.1 as they are.

# Release notes — 0.5.0

`delaunay-watershed-3d` (this repository) and `foambryo` ship together as 0.5.0; foambryo 0.5.0
requires `delaunay-watershed-3d>=0.5.0`. This is what changes for you and what can mislead you;
`CHANGELOG.md` has the full list, and foambryo's `RELEASE_NOTES.md` covers the inference side.

0.5.0 is the first release on PyPI since 0.3.6; `CHANGELOG.md` also lists the changes of 0.4.0–0.4.2,
which were never published there.

## Licence

From 0.5.0 `delaunay-watershed-3d` is licensed under the **GNU General Public License, version 3 or (at your
option) any later version** (SPDX `GPL-3.0-or-later`), replacing CC BY-NC-SA 4.0: commercial use is allowed, under the GPL's conditions.
Earlier releases keep the licence they were published under.

## The one change that affects every mesh

**Junction relocation is on by default, in every reconstruction configuration.** After the mesh is
built, its trijunction vertices are moved onto the trijunction curves the segmentation mask itself
gives, under a guard that cannot create a self-intersection anywhere in the mesh. Triangles and
material labels are unchanged; vertex positions on trijunction lines are not. Measured over 40
equilibrium benchmark cases, four spacings and four point samplers: junction-line position, tangent
and curvature errors all improve with their confidence intervals clear of zero, and no mesh-validity
number changes.

**To get the 0.4 meshes back exactly**:

```py
algorithm = get_default_mesh_reconstruction_algorithm()
algorithm.relocate_junctions = False
```

or `MeshReconstructionAlgorithmFactory().set_junction_relocation(False)`, or
`MeshReconstructionAlgorithm(..., relocate_junctions=False)`. Whether relocation ran on your mesh is
in `algorithm.relocation_provenance`.

**At `min_distance >= 5` you will see `CoarseSpacingRelocationWarning`**, once per reconstruction:
at that spacing the check on surface-tangent contact angles showed a small drift within its own
noise, and larger spacings are unmeasured; facet-based angles improve. If tangent-based angles
matter for your analysis, compare with `relocate_junctions=False`.

## Names removed and deprecated

Removed: `get_v0_3_algorithm`, `get_a1b_algorithm`, `get_a3_a5_algorithm` (use
`get_dithered_algorithm`, `get_deterministic_algorithm`, `get_junction_protected_algorithm`).
Removed without a deprecation period: `dw3d.viewing.plot_in_napari`, which already failed with napari
0.9.1; napari is no longer part of the `viewing` extra (Polyscope and matplotlib only).
Deprecated, removed in 0.6.0: `get_offset_included_algorithm`, `get_offset_excluded_algorithm`,
`get_cubic_score_algorithm` — exact synonyms of keyword calls on `get_junction_protected_algorithm`,
which their warnings name.

## Limitations

Every item is measured; each is one sentence and its number.

1. **The benchmark under-rewards tangent-based improvements.** Its ground-truth meshes are
   facet-equilibrated (Surface Evolver), so an improvement in the surface-*tangent* description of
   a contact angle is scored against a facet-based truth and can read as no change.
2. **The second-order angle reader has its own floor, and it cannot resolve junction relocation's
   effect.** On a perfectly smooth analytic surface it reads contact angles with a floor of
   **0.245°**, and at every point spacing measured (0 of 32 spacing-by-variant cells) its check could
   not resolve the effect of junction relocation on tangent-based angles, in either direction.
3. **On real embryos, do not use second-order line means at sub-degree scale.** The second-order
   reader covers a median **0.804** of incident trijunction faces on real embryo meshes (minimum
   **0.454**), and its line means move by **1.75° [1.41, 2.17]** when uncovered faces are excluded
   instead of filled by the chord reading.
4. **Real embryo cells are small in voxels, so every benchmark statement is about one spacing step
   optimistic for them.** Their equivalent radius at trijunctions is a median **17.6** voxels against
   **38.2** in the benchmark, so real data at `min_distance 3` sits where the benchmark sits at
   `min_distance 4`; for cells this small, consider a smaller `min_distance`.
5. **Folded trijunctions on real meshes are mostly real geometry.** Real embryo meshes carry
   **3.2×** the benchmark's fraction of reflex (folded) trijunctional edges, and more than half of
   them sit where the segmentation mask's own wedge already exceeds 180°.
6. **The force-balance residual `rho_free` alarms on every reconstructed mesh.** Its median is
   **0.578** on reconstructed meshes against **0.0118** on the reference meshes, so a high value on
   your mesh is expected and is not by itself a sign of a bad segmentation.
7. **`ForceBalanceMAP` is not scale-invariant**, contrary to its docstring: rescaling a mesh moves
   its relative tensions by up to **43 %**; plain `ForceBalance` is exact under rescaling.
8. **Junction relocation, now on by default, has a cost that grows faster than linearly with mesh
   size.** It adds about **+0.75 s** to a ~5 s benchmark reconstruction (**+15 %**) and **0.1–1.5 s**
   per timepoint to a ~0.6 s real-embryo reconstruction (median **0.34 s** and **0.48 s** on two
   developmental time series, about 65 s per 154-timepoint series); within one real series it grows
   with triangle count with a fitted exponent of **2.68** over a 2.2-fold range of mesh sizes, which
   is not a tissue-scale law. **A faster neighbour search is planned for 0.5.1**, and will leave every
   relocated mesh bit-for-bit unchanged.
