# Benchmarks: default (`dithered`) vs the "revised meshing" variant (`link_checked`)

> **Measured before 0.5.0's default change.** Every number in this file was measured with junction
> relocation off, which was the default up to 0.4.2. Since 0.5.0 every configuration relocates
> junction vertices by default; `relocate_junctions=False` reproduces the meshes measured here
> exactly. What relocation changes, and what it costs in seconds, is in `CHANGELOG.md` (0.5.0) and
> `RELEASE_NOTES.md`.

Short, honest comparison. Numbers below are measured, not estimated. Each section states the
population, the statistic and the configuration its numbers were measured on, so that nothing
here depends on material outside this repository.

**Common ground for every section, unless the section says otherwise.**

- *The benchmarking dataset* is `foambryo`'s `benchmarking-dataset`: 47 simulated cell
  aggregates, each with a segmentation mask, the ground-truth simulation mesh and the
  ground-truth surface tensions. 42 of the 47 are at mechanical equilibrium; the other five
  (`005`, `015`, `018`, `021`, `043`) are not, and are kept only as a negative control. The two
  groups are never pooled; the split is defined once, in `foambryo`'s
  `foambryo_benchmarks/case_groups.py`.
- *The 51-case benchmark* is those 47 masks plus the four in-repository images
  `data/Images/1–4.tif`. Quantities that need a ground truth exist on the 47 only.
- Reconstructions use `min_distance=3`.
- Length and area comparisons against the ground truth first register the ground-truth mesh
  into the mask's voxel frame, with a similarity transform fitted between label-keyed
  ground-truth and mask centroids (`similarity_to_mask_frame` in `dw3d_benchmarks/metrics.py`).
  The fit never sees the reconstruction being scored, so it is common to every variant; its
  median residual is 1.20 voxels over a median of 19 correspondences, on objects about 120
  voxels across. Angles need no registration.

## The one number that decides the default

End-to-end tension-inference accuracy (gauge-matched median relative error, 42 equilibrium
cases from `foambryo`'s benchmarking dataset, paired per case) is **worse** with the revised
meshing pipeline than with `dithered`, on every inference method measured, despite the revised
pipeline's better mesh quality in isolation (see below):

| tension-inference method | `dithered` (default) | `link_checked` (revised, opt-in) | revised wins on |
|---|---:|---:|---:|
| `YoungDupre` | 0.0882 | 0.1049 | 13 / 42 cases |
| `YoungDupreLocal` | 0.0949 | 0.1245 | 13 / 42 cases |
| `ForceBalanceKKT` | 0.1299 | 0.3436 | 4 / 42 cases |
| `BayesianMAP` | 0.1799 | 0.1962 | 20 / 42 cases |

How this was measured: each of the 42 equilibrium masks was reconstructed with both variants at
`min_distance=3`, and `foambryo`'s tension inference was run on every mesh with every method.
Per case, the error is the median over interfaces of the relative error
`|γ_inferred − γ_true| / γ_true`, after the inferred tensions are rescaled by
`mean(γ_true) / mean(γ_inferred)` over the interfaces both share (gauge matching, so that the
arbitrary overall scale of inferred tensions is not counted as error). Each cell of the table
is the median of that per-case error over the 42 cases; "revised wins on" counts the cases on
which `link_checked`'s error is the lower of the two for that method.

This is why `dithered` ships as the default: `dw3d`'s output feeds `foambryo`'s tension inference,
and today that downstream step is the one that matters most for most users.

## Mesh quality (measured independently of tension inference)

The revised pipeline is unambiguously better here — this is *why* it exists and *why* it is
kept, opt-in, rather than deleted.

| metric (51-case benchmark, `min_distance=3`) | `dithered` | `link_checked` |
|---|---:|---:|
| interface-area error vs ground truth, signed median (47 cases) | **−0.60 %** | +1.07 % (both small; `dithered` is closer to zero) |
| all-surface tetrahedra, median | 56.5 % | **0.8 %** |
| slivers (`quality < 0.01`), median | 13.5 % | **5.7 %** |
| abnormal non-manifold edges (total) | 3 | 6 |
| valence-≥4 edges (total) | 61 | 114 |
| junction-angle error vs the ground-truth Neumann angles (median over cases) | not measured before junction protection† | **10.09°** |
| junction length vs ground truth | not measured before junction protection† | **1.008×** |
| triple lines missed (total, 47 cases) | not measured before junction protection† | **19** |

† `dithered`'s committed benchmark records predate the ground-truth comparison and
junction-protection metrics (added when junction protection itself was). They were not
backfilled because regenerating `dithered`'s records would overwrite the wall times that the
whole speed budget is measured against. The interface-area row is unaffected: it is computed
directly from the meshes, not read from those records.

How these rows were measured: all variants in one run, on one machine, on the 51-case
benchmark at `min_distance=3`. Counts are totals over the cases; percentages are medians over
the cases of each case's own percentage.

- *Interface-area error* (`dw3d_benchmarks/interface_area_error.py`): for every interface,
  keyed by its label pair, the signed relative error of the reconstructed area against the
  registered ground-truth area; the row is the median over the 47 ground-truth cases of each
  case's median.
- *All-surface tetrahedra*: tetrahedra of the tesselation whose four vertices are all interface
  points, with no interior point among them.
- *Junction-angle error*: measured against the Neumann angles implied by the ground-truth
  tensions, `cos θ(ab, bc) = (γ_ac² − γ_ab² − γ_bc²) / (2 γ_ab γ_bc)`; the row is the median over
  the 47 cases of each case's median error. As a control, the ground-truth simulation mesh
  itself sits 0.34° from these Neumann angles on the same statistic.
- *Junction length*: total reconstructed trijunction length relative to the registered ground
  truth's.
- *Triple lines missed*: ground-truth triple lines with no reconstructed counterpart, summed over
  the 47 cases.

The revised pipeline explicitly detects and protects triple junctions before triangulating,
which `dithered` does not — that is the entire reason it produces cleaner topology and better
junction placement.

## Speed and memory

| | `dithered` | `link_checked` |
|---|---:|---:|
| wall time, 51-case total | 81.08 s | 92.94 s (**1.146×** `dithered`) |
| wall time, per case | — | 1.038×–1.223× `dithered` (49/51 cases inside 1.20×) |
| tesselation points, 51-case total | 277 922 | 191 415 (fewer, despite the extra junction/boundary-layer point families) |
| mesh points, 51-case total | 71 265 | 61 239 |

Wall times come from `dw3d_benchmarks/paired_timing.py`: both variants run back to back in one
process on every case, two repeats, keeping the minimum per case and variant, on an otherwise
idle machine (Apple M3 Max, 16 cores, 137 GB RAM, macOS 14.8.4), single-threaded. That
machine's wall-time ratio between repeated identical runs has a median of 0.966 and a range of
0.775–1.047. Point counts are totals over the 51 cases from the same run as the mesh-quality
table.

Both use the same dense-`float64` Euclidean distance transform, so peak memory is dominated by
the EDT stage regardless of variant. The EDT's own scaling to large volumes, independent of
which reconstruction variant reads it, was measured separately with
`dw3d_benchmarks/edt_scaling.py`, on synthetic Lloyd-relaxed foams
(`dw3d_benchmarks/synthetic_foam.py`) whose cell count grows with the volume so that the cell
size stays fixed, each variant and volume in a fresh process. At 1024³ voxels (1728 cells) the
resident memory of the EDT stage is 32 047 MB for the dense classical EDT (29.8 bytes per
voxel), against 11 993 MB for `compute_edt_float32` and 9 388 MB for `compute_edt_tiled`, whose
output is identical to the dense EDT's.

No per-variant peak-RSS comparison was measured, since the EDT stage — not the tesselation size
difference above — is the memory bottleneck at the scales this package targets.

## Force-balance residual (`rho_free`): every reconstruction is ~50x the reference mesh

**Population: the 42 equilibrium cases of the benchmarking dataset** (`005`, `015`, `018`, `021`,
`043` excluded — see `foambryo`'s `foambryo_benchmarks/case_groups.py`). Measured at
`min_distance=3`, dither seed 42, with `foambryo`'s `compute_inference_diagnostics` on every
mesh; no case failed. The table gives the median over the 42 cases. The reference row is the
ground-truth simulation mesh itself, with no reconstruction.

`rho_free` is the force-balance residual left when the tensions are free to take their
best-fitting values — that is, the part of the residual that **no** tension assignment can
remove, because it is a property of the mesh's geometry rather than of the tensions on it.

Two of the arms below are experimental and are not shipped in this package.
`move_to_mask_line` moves `dithered`'s own junction points onto the junction locus derived from
the mask at sub-voxel precision, adding and removing no point. `move_null_control` moves the same
points by the same distances in a direction that carries no information, as a control.

| arm | median `rho_free` | vs the reference mesh | contact-angle slope (length-weighted) |
|---|---:|---:|---:|
| reference simulation mesh (control) | `0.0116` | 1x | — |
| `dithered` (shipped default) | `0.5832` | **50.4x** | `-0.2343` |
| `move_to_mask_line` | `0.5766` | **49.8x** | `-0.2166` |
| `move_null_control` | `0.6260` | **54.1x** | `-0.2694` |

The contact-angle slope measures how far a mesh flattens contact angles toward 120°. For every
trijunction line and each of its three materials, it takes the contact angle `foambryo` reads on
the mesh (the per-line mean of `compute_angles_tri`) and the Neumann angle implied by the
ground-truth tensions, and regresses (measured − true) on (true − 120°) by weighted least
squares, each row weighted by its junction line's length, pooled over the 42 cases. 0 is
unbiased; −1 would mean every angle is read as 120°.

Two things follow, and the second is the useful one:

- **Every dw3d-reconstructed mesh carries ~50x the reference mesh's irreducible force-balance
  residual**, regardless of which arm produced it. This is a floor on what any force-balance
  method can achieve on these meshes, and it is not a tension-estimation error.
- **`rho_free` does not move when the contact-angle slope moves.** Across the three arms above,
  the slope moves by `0.053` (a 24 % spread) while `rho_free` stays within `0.577`–`0.626` (an 8 %
  spread) and does not even order the arms the same way. **`rho_free` is therefore not a proxy
  for contact-angle quality and must not be used as one** — an arm can improve contact angles
  measurably without moving `rho_free` at all.

This is why the angle-reading and operator-inverting method families have to be evaluated
separately: they are sensitive to different defects of the same mesh. That distinction is
documented in the `TensionComputationMethod` docstring
(`foambryo/src/foambryo/tension_inference.py`, "Which methods read angles and which invert an
operator, and why the distinction now has measured consequences") and is not restated here;
what this section adds is the quantification above, and the next subsection adds how the
angle-reading family's bias depends on mesh spacing.

### The angle-reading family's bias is spacing-dependent, and not a constant

**Population: the same 42 equilibrium cases, `dithered`, dither seed 42**, reconstructed at each
`min_distance` below; every case was reconstructed and measured at every spacing, with no
failure.

The shipped default `TensionComputationMethod.YoungDupre` is an angle-reading method, and it
weights every trijunction line by its length. In that (length-weighted) convention the
contact-angle flattening slope of `dithered` is:

| `min_distance` | median transverse lever (vox) | contact-angle slope | matched coarseness floor | excess | abnormal non-manifold edges |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.96 | `-0.5488` | `-0.0340` | `-0.5148` | **63** |
| 2 | 6.30 | `-0.2834` | `-0.0872` | `-0.1961` | 7 |
| 3 (shipped) | 8.69 | `-0.2343` | `-0.1391` | `-0.0951` | 3 |
| 5 | 13.93 | `-0.2635` | `-0.2657` | `+0.0022` | 0 |

The columns, in plain words:

- *Median transverse lever*: for each triangle incident on a junction edge, the distance from
  the edge's midpoint to the triangle's opposite vertex — the length over which a contact angle
  is read. Median per case, then median over the 42 cases.
- *Contact-angle slope*: as defined in the previous section.
- *Matched coarseness floor*: the same slope, in the same length-weighted convention, measured
  on the ground-truth mesh after it has been coarsened by edge collapse to the arm's own
  transverse lever — what a perfect mesh reads at that coarseness.
- *Excess*: slope minus floor, the part of the flattening that coarseness does not explain.
- *Abnormal non-manifold edges*: total over the 42 cases.

- **The slope is stable over `min_distance` 2–5 (spread `0.049`) and is not stable below that.**
  Do not extrapolate any angle-bias calibration below `min_distance=2`.
- **`min_distance=1` is not a usable setting.** It carries 63 abnormal non-manifold edges
  against 3 at the shipped default, and median tension contrast collapses from `0.738` to
  `0.513`. Tension contrast is, per case, the least-squares slope of the gauge-matched
  `YoungDupre` tensions against the ground-truth tensions (1 is perfect; below 1, tension
  differences are compressed), with the median taken over the cases where it is defined (36 of
  the 42 at the shipped spacing). Meshes stay watertight (42/42), so the damage is
  junction-local.
- At `min_distance=5` the flattening is entirely explained by mesh coarseness (excess
  `+0.0022`); at `min_distance=3` coarseness explains 59 %; at `min_distance=1`, 6 %.

## Junction curves: `extract_junction_curves`, a more accurate reading of the same lines

`extract_junction_curves(mask, points, triangles, labels)` reads the trijunction curves off the
segmentation mask and keys them to the reconstructed mesh's own trijunctions. It is a **new,
additive, opt-in** read: nothing in the reconstruction pipeline calls it and no default changed.

All numbers below: 47-case synthetic benchmark, 42 equilibrium and 5 non-equilibrium cases
**never pooled** (equilibrium quoted), `dithered` meshes, reference registered into the mask frame by
`similarity_to_mask_frame`, case-level bootstrap with 10,000 draws and seed `20260901`. The
position error is the median distance in voxels from a curve's vertices to the reference
polyline; lower is better.
Each arm is read at **its own** median junction-edge spacing, so this compares accuracy and not
reading resolution. Each metric's *matched-spacing floor* is the value the reference line itself
scores when subsampled at that same spacing. The numbers were measured through the shipped
`extract_junction_curves` API, whose junction detector was checked bit-identical to the research
prototype's on all 47 masks (918 polylines) and reproduces the prototype's position error to the
last digit.

### Accuracy against the reference

Median position error, in voxels:

| `min_distance` | `extract_junction_curves` (default) | the mesh's own junction lines | advantage |
|---|---:|---:|---:|
| 3 | **0.5846** | 0.8642 | **1.48x** |
| 4 | **0.6106** | 0.9277 | **1.52x** |

The research prototype, read the same way on the same 42 cases, gave the same comparison across
five spacings: 1.41x at `min_distance=2`, 1.48x at 3, 1.50x at 4, 1.64x at 5 and 1.90x at 7 — the
advantage grows as the mesh coarsens.

Against each metric's own matched-spacing floor (`min_distance=3`), the curves read tortuosity
at **1.001**, curvature at **1.013** and mean tangent error at **1.96x**, where the mesh's own
lines read 1.034, 1.172 and 4.53x. All 24 head-to-head comparisons measured here — 2 spacings x
2 estimators x 6 metrics (tortuosity, mean and 90th-percentile tangent error, curvature, median
and 90th-percentile position error), paired per case — favour the curves on **1.000** of 10,000
draws.

### Completeness, which is the trade

More accurate and **less complete**. At `min_distance=3`, 622 of the mesh's 691 trijunctions have
exactly one extracted counterpart (**match rate 0.900**; 0.908 at `min_distance=4`), with
**zero missed** — every trijunction the mesh has, the mask has a curve for. The remaining 10 % is
fragmentation: 39 split, 24 ambiguous, 6 merged, plus 26 invented triples the mesh does not have
at all. Total arclength is 0.9962 of the reference's.

**This is why `matched` and `invented` are separate fields and are never merged.** Measured
distance from a curve to **all three** of its own interfaces at once (the largest of the three
distances, so that a point near only one interface does not count), per curve, `min_distance=3`:

| | n curves | median | p90 | p99 | within 1 voxel |
|---|---:|---:|---:|---:|---:|
| `matched` | 622 | **0.681** | 1.773 | 4.502 | 0.781 |
| `invented` | 27 | 29.71 | 45.82 | 50.39 | **0.000** |

The two populations are structurally different, not two ends of one distribution. `curves.matched`
is the set you want; `curves.invented` is there so completeness is visible rather than silent.

### Cost, and why the default runs no linear program

| estimator | s/case, `min_distance=3` | position error | what it is |
|---|---:|---:|---|
| `"plain"` (default) | **0.77** | 0.5846 | detection only, no linear program |
| `"anchor"` | 4.47 | 0.5201 | one Chebyshev-centre LP per returned vertex |
| `"centroid"` | ~30x `"anchor"`'s LPs | — | reproduces the research prototype's estimator only |

Seconds per case are medians over the 42 equilibrium cases, measured on a machine shared with
other jobs, so read them as indicative rather than exact.

dw3d reconstructs one of these cases in about 1.5 s, so `"anchor"` costs roughly three times the
whole reconstruction it annotates for an 11 % improvement in position error. That is not a
default. Its real argument is a different one. Near a narrow wedge, the mask itself places the
junction slightly into the narrowest sector. Measured as the junction's displacement along the
bisector of the narrowest sector, regressed on that sector's Neumann angle minus 120°, the part
of this bias that no mask-based method can remove — where the mask's own best-fitting flat
junction sits relative to the reference — is −0.01123 voxels/degree on the 42 equilibrium cases.
On the research prototype, the polytope's deepest interior point, which `"anchor"` returns,
carries **only** that irreducible term (slope −0.01165 voxels/degree), while the centroid also
carries the feasible set's internal asymmetry, at −0.01435. Reach for `"anchor"` when a
systematic bias against narrow wedges matters, not to shave a tenth of a voxel.

### The honest residual

The mask's own identifiability floor — the best any mask-based method can do — is **0.107
voxels**: the median, over 5088 trijunction windows of radius 3 voxels from the 42 equilibrium
cases, of the smallest worst-case position error achievable within the set of flat
three-half-plane junction configurations that reproduce the mask exactly (a lower bound for that
model class). These curves sit at 0.5846, a factor of
**5.5** above it (the `"anchor"` estimator's 0.5201 is a factor of 4.9); the mesh's own lines sit at
8.1x. So this is **better geometry, not correct geometry**, and that factor is open. Two explanations for it were excluded on the research
prototype (the choice of point inside the feasible set, worth 0.06 voxels; and the extraction's
lattice coarseness, which smoothing already recovers). Three remain: the flat three-half-plane
model, the reference registration's own ~1.14-voxel residual (mean per-correspondence residual,
measured on one case), and the construction of the floor itself (computed at a single window
radius).

### Controls

* **No ground truth anywhere in the read.** Re-running the whole API over a directory holding
  only the masks and the three mesh arrays, with every `.rec` structurally unreachable, gives a
  **hash-identical** result (SHA-256 `909afa82…2db09a` over every coordinate of all 47 cases, for
  the `"anchor"` estimator; the default's output is likewise hash-identical).
* **Null.** Permuting the mask's label values with a fixed seed (`20260901`) leaves the geometry
  untouched and scrambles only the material attribution. Position error goes 0.5846 → **36.76**
  voxels and mean tangent error 4.57° → **49.90°**, both on 1.000 of draws; the match rate
  against the mesh collapses from 0.900 to **0.512**, with 290 of the mesh's trijunctions missed
  where the arm misses none.
* **Contact angles and multimaterial validity are untouched**, verified rather than assumed:
  1974 scalars (21 per case: the contact-angle and multimaterial-validity measures) re-measured
  on all 47 cases at both `min_distance` 3 and 4 against the committed values recorded when the
  junction-surgery selector-sign fix was accepted, zero differences at `1e-9`.

## Bottom line

- Use the **default** (`get_default_mesh_reconstruction_algorithm`) when the mesh feeds
  tension/pressure inference — the common case.
- Use **`get_link_checked_mesh_reconstruction_algorithm`** when mesh geometry and topology
  (junction placement, interface smoothness, low interface-area bias) matter more than
  today's downstream inference accuracy, or when comparing against future inference methods
  that may not share `ForceBalanceKKT`'s and `YoungDupre`'s sensitivity to mesh quality.
- All other named variants (`get_deterministic_algorithm`, `get_offset_included_algorithm`, ...) are intermediate
  configurations kept for reproducibility of the development history; they are not
  recommended for new work.

## Package renamed `benchmarks` -> `dw3d_benchmarks`

The top-level `benchmarks/` package was renamed to `dw3d_benchmarks/` because
both this repository and `foambryo` shipped a package named `benchmarks`, and a script that
put both repositories on `sys.path` could silently resolve the name to either one.

**Migration hazard (found in `foambryo`'s checkout, not this one):** pulling this
rename into an existing checkout can leave a stray `benchmarks/__pycache__/` directory on
disk — `git mv` moves tracked files, not gitignored build artifacts. With
`benchmarks/__init__.py` gone but the directory still present, Python imports it as an
empty implicit namespace package (PEP 420): `import benchmarks` still succeeds, with no
`__file__` and no real submodules. `tests/test_no_ambiguous_benchmarks_package.py` now
tolerates this case (only a *real*, `__file__`-bearing `benchmarks` package fails it), but
the stray directory should still be deleted on any checkout that hits it — `rm -rf
benchmarks/` once `git status` confirms it holds nothing tracked.
