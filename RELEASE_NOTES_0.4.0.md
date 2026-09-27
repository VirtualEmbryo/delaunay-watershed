# Release notes — 0.4.0

`delaunay-watershed-3d` (this repository) and `foambryo` ship together as 0.4.0. This is not a
changelog (see `CHANGELOG.md` for that) — it is what to run and what will mislead you, for a team
testing this on real data for the first time. `foambryo`'s own notes are
`foambryo/RELEASE_NOTES.md`; read both, since two of the three new features below are
`foambryo`-side and the caveats interact across the two packages.

**dw3d 0.3.6 is what is currently published. This release supersedes it.**

## What is new

Three opt-in features. Nothing here changes unless you ask for it — see "What is unchanged"
below.

| feature | invocation | gain |
|---|---|---|
| second-order contact angles (`foambryo`) | `infer_tensions(..., angle_reading="quadric")` | tension contrast 0.738 → 0.954 at `min_distance=3`; per-line angle error 7.35° → 4.83° |
| junction-curve extraction (this repo) | `extract_junction_curves(mask, points, triangles, labels)` | tortuosity and curvature at 1.001× / 1.013× of the identifiability floor; line-position error 0.585 vs the mesh's own 0.864 |
| `.rec` metadata appendix (this repo) | `save_rec(..., metadata={...})` | records spacing, units, coordinate frame, crop offset |

Plus one behavioural change, not opt-in: **zero abnormal non-manifold edges at every
`min_distance` in {2, 3, 4, 5, 7}**, on all 47 benchmark cases, both the 42-case equilibrium and
5-case non-equilibrium populations — where 0.3.6 leaves three at the shipped default
(`min_distance=3`). Re-verified fresh on this release's exact tip (not just cited from history):
zero abnormal edges, all cases watertight, zero degenerate triangles, zero unclosed cells, at
every one of the 5 rungs, 235 reconstructions. Historically, before the surgery-selector-sign fix, the same gate on the same 47
cases showed the pre-fix state was **not** already zero — e.g. `min_distance=3`, 42 equilibrium
cases: 3 abnormal edges before, 0 after; `min_distance=2`: 7 before, 0 after — so this is a fix
that is measurably active, not a check that was already vacuous.

## What is unchanged

Every default is bitwise identical to 0.3.6's behaviour on all 47 benchmark meshes. An existing
script produces the same numbers it always has. This is what lets a team adopt this release
without re-validating past results — the surgery-rule fix above changes reconstruction output
only where an abnormal edge would otherwise have been produced (zero cases moved without the
surgery step firing on them), and the two new features are purely additive: nothing in the
reconstruction pipeline calls `extract_junction_curves` or the metadata appendix.

## What will mislead you

In this order, because the first one is the one that will actually bite.

1. **Anisotropy. dw3d ignores voxel spacing entirely.** A 2:1 z-anisotropic stack yields a mesh
   that is 42/42 watertight with zero abnormal edges — **every validity check passes** — and
   gives tension contrast **0.610 against 0.729** isotropic, with junction lines at 1.22 voxels
   instead of 0.86 and median angle error 10.5° instead of 7.8°. At 3:1: contrast **0.484**. At
   2:1, *all* of the end-to-end damage is the pipeline's spacing-blindness rather than lost
   information — resampling the anisotropic stack to cubic voxels before reconstruction recovers
   87% of the contact-angle flattening-slope damage. At 3:1 it does not fully recover (63%), so the effect is not simply
   "resample and you're done" at every factor. **No diagnostic fires.** Anyone meshing anisotropic
   data must know its spacing and treat the numbers as unvalidated until checked against an
   isotropic or resampled control.
2. **The quadric angle reading (`foambryo`) is calibrated at this repository's benchmark lever**,
   `min_distance` 3–4. See `foambryo/RELEASE_NOTES.md` for the full caveat — briefly, its
   over-application is lever-dependent, and on the 47 ground-truth reference meshes it is worse
   than the default on 47 of 47 cases. Report both readings side by side for the first cases you
   run it on.
3. **The curves need the mask, not just the mesh.** `extract_junction_curves` matches the mesh's
   own trijunctions against the mask at a 0.900 match rate; matched and invented curves are
   returned as **separate fields and are never merged**, because they are structurally different
   populations — matched curves lie a median 0.68 voxels from all three of their own interfaces,
   invented ones 29.7, none within a voxel. Only the matched set is usable. The curves sit 4.8×
   above the mask's own identifiability floor: this is better geometry than the mesh's own
   trijunctions, not correct geometry in an absolute sense.
4. **`YoungDupreLocal` cannot be combined with `angle_reading="quadric"`** (`foambryo`) — now an
   explicit `ValueError`, because it gave +0.97 relative RMS tension error at `min_distance=4`.
   See `foambryo/RELEASE_NOTES.md`.
5. **Costs.** Curve extraction is 0.77 s/case against a ~1.5 s reconstruction with the default
   `estimator="plain"`, or 4.47 s with `estimator="anchor"` (worth 9–11% less line-position error; read the
   docstring before reaching for it by default). The quadric angle reading's cost is on the
   `foambryo` side — see its notes.

## Reproduction

- Junction curves: `curves = extract_junction_curves(mask, *load_rec("case_mesh.rec"))`, then use
  `curves.matched` only.
- Metadata appendix: `save_rec(path, points, triangles, labels, metadata={"spacing": (...),
  "spacing_unit": "micrometer", "coordinate_frame": "mask", "crop_offset": (...)})`, then
  `load_rec_appendix(path)` to read it back; `load_rec(path)` is unaffected and returns the same
  arrays whether or not a file carries an appendix.
- The zero-abnormal-edge result: reconstruct any of the 47 `benchmarking-dataset` cases with
  `dw3d.get_dithered_mesh_reconstruction_algorithm(min_distance=md, seed=42)` at any
  `md` in `{2, 3, 4, 5, 7}` and count with
  `dw3d_benchmarks.metrics.abnormal_non_manifold_edge_count`.

## Push (not yet done)

```bash
git -C delaunay-watershed-3d push origin master
git -C delaunay-watershed-3d push origin v0.4.0
```
