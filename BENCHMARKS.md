# Benchmarks: default (`dithered`) vs the "revised meshing" variant (`link_checked`)

> **Measured on 0.7.1, with junction relocation on**, the default in every configuration since 0.5.0, and `foambryo` 0.7.0's inference (unchanged in 0.7.1). 0.7.1 puts the tetrahedra in a canonical order before reading them and its mesh surgery no longer leaves abnormal edges, so its meshes differ slightly from 0.7.0's: every section that reads meshes was measured again on 0.7.1's, with the same scripts, and a *0.7.0* value stands beside each number that moved. Where 0.5.1, the last published version before 0.7.0, gave a different number from 0.7.0, a *0.5.1* value stands beside it too; those were not measured again. 0.5.1 republished 0.5.0's code unchanged, so its numbers are 0.5.0's. Earlier versions of this file were measured with relocation off, on 0.4 meshes. **Exceptions**, said where they are printed: the speed table and the synthetic large volume (measured in early October 2026 on the code prepared for 0.6.0), the 1 866-cell tissue's run and memory (0.7.0), the EDT scaling at 1024³ (before 0.5.0), the cost table of the junction-curve section (before 0.5.0, on a machine shared with other jobs), and the junction-curve numbers measured with relocation off (0.7.0).

Short, honest comparison. Numbers below are measured, not estimated. Each section states the population, the statistic and the configuration its numbers were measured on, so that nothing here depends on material outside this repository.

**Common ground for every section, unless the section says otherwise.**

- *The benchmarking dataset* is `foambryo`'s `benchmarking-dataset`: 47 simulated cell aggregates, each with a segmentation mask, the ground-truth simulation mesh and the ground-truth surface tensions. 42 of the 47 are at mechanical equilibrium; the other five (`005`, `015`, `018`, `021`, `043`) are not, and are kept only as a negative control. The two groups are never pooled; the split is defined once, in `foambryo`'s `foambryo_benchmarks/case_groups.py`.
- *The 51-case benchmark* is those 47 masks plus the four in-repository images `data/Images/1–4.tif`. Quantities that need a ground truth exist on the 47 only.
- Reconstructions use `min_distance=3` and the shipped defaults, junction relocation included.
- Length and area comparisons against the ground truth first register the ground-truth mesh into the mask's voxel frame, with a similarity transform fitted between label-keyed ground-truth and mask centroids (`similarity_to_mask_frame` in `dw3d_benchmarks/metrics.py`). The fit never sees the reconstruction being scored, so it is common to every variant; its median residual is 1.20 voxels over a median of 19 correspondences, on objects about 120 voxels across. Angles need no registration.

## The one number that decides the default

End-to-end tension-inference accuracy (gauge-matched median relative error, 42 equilibrium cases from `foambryo`'s benchmarking dataset, paired per case) is **worse** with the revised meshing pipeline than with `dithered`, on every inference method measured, despite the revised pipeline's cleaner tesselation (see below):

| tension-inference method | `dithered` (default) | `link_checked` (revised, opt-in) | revised wins on | 0.7.0 (`dithered` / `link_checked` / wins) | 0.5.1 (`dithered` / `link_checked` / wins) |
|---|---:|---:|---:|---:|---:|
| `YoungDupre` | 0.0763 | 0.1003 | 11 / 42 cases | 0.0777 / 0.0986 / 12 | 0.0777 / 0.0975 (41 cases) / 12 |
| `YoungDupreLocal` | 0.0911 | 0.1266 | 11 / 42 cases | 0.1039 / 0.1220 / 12 | 0.1039 / 0.1198 (41 cases) / 12 |
| `ForceBalanceKKT` | 0.1138 | 0.3262 | 1 / 42 cases | 0.1150 / 0.3327 / 4 | 0.1298 / 0.3357 / 3 |
| `BayesianMAP` (`ForceBalanceMAP`) | 0.1186 | 0.3149 | 2 / 42 cases | 0.1211 / 0.3265 / 3 | 0.1841 / 0.2010 / 17 |

Between 0.5.1 and 0.7.0 the two force-balance rows changed for two reasons, neither of which moves a vertex. 0.7.0 winds correctly the few triangles that lie on a flat tetrahedron (`CHANGELOG.md`): 10 triangles in 7 of the 47 `dithered` meshes and 55 in 33 of the `link_checked` meshes turn over, while every point, label and triangle stays where it was (counted on 0.7.0's meshes). The force-balance operator reads cell volumes, so on those cases `ForceBalanceKKT` changed; the angle-reading rows kept their medians and, case by case, moved on one rewound mesh only (case 032: `YoungDupre`'s error by −0.0015). And `foambryo` now makes `ForceBalanceMAP` independent of the mesh's length unit (its `CHANGELOG.md`), which changes that method's estimate on every mesh.

How this was measured: each of the 42 equilibrium masks was reconstructed with both variants at `min_distance=3`, and `foambryo`'s tension inference was run on every mesh with every method. Per case, the error is the median over interfaces of the relative error `|γ_inferred − γ_true| / γ_true`, after the inferred tensions are rescaled by `mean(γ_true) / mean(γ_inferred)` over the interfaces both share (gauge matching, so that the arbitrary overall scale of inferred tensions is not counted as error). Each cell of the table is the median of that per-case error over the 42 cases. Before 0.7.1, case 038's `link_checked` mesh had one interface without a triple line, whose tension was undetermined, and that case was scored on its other interfaces (the 0.5.1 column left it out of the two Young-Dupré rows): it was the membrane of a 26-voxel cell that no tetrahedron lies inside, which 0.7.1 leaves out of the mesh and reports (`UnresolvedCellWarning`, `CHANGELOG.md`). That change moves the two force-balance rows: on cases 031 and 038, whose masks hold such a cell, `ForceBalanceKKT`'s error on the `dithered` mesh falls from 0.208 to 0.122 and from 0.801 to 0.109, against the same code without it. "Revised wins on" counts the cases on which `link_checked`'s error is the lower of the two for that method.

This is why `dithered` ships as the default: `dw3d`'s output feeds `foambryo`'s tension inference, and today that downstream step is the one that matters most for most users.

## Mesh quality (measured independently of tension inference)

The revised pipeline builds a much cleaner tesselation and has the smaller interface-area bias, and that is *why* it exists and *why* it is kept, opt-in, rather than deleted. It is not better on every row: `dithered` reads junction angles more accurately and has fewer valence-≥4 edges.

| metric (51-case benchmark, `min_distance=3`) | `dithered` | `link_checked` | 0.7.0, where different |
|---|---:|---:|---:|
| interface-area error vs ground truth, signed median (47 cases) | −1.01 % | **+0.71 %** (both small; `link_checked` is closer to zero) | − / +0.52 % |
| all-surface tetrahedra, median | 56.5 % | **0.8 %** | |
| slivers (`quality < 0.01`), median | 13.5 % | **5.7 %** | |
| abnormal non-manifold edges (total) | 0 | 0 | − / 1 |
| valence-≥4 edges (total) | **59** | 112 | 60 / 113 |
| junction-angle error vs the ground-truth Neumann angles (median over cases) | **6.61°** | 8.64° | 6.51° / 8.87° |
| junction length vs ground truth | 1.006× | 1.003× | − / 1.002× |
| triple lines missed (total, 47 cases) | 21 | 19 | 20 / − |

How these rows were measured: both variants with the same code and defaults, on the 51-case benchmark at `min_distance=3`. Counts are totals over the cases; percentages are medians over the cases of each case's own percentage.

- *Interface-area error* (`dw3d_benchmarks/interface_area_error.py`): for every interface, keyed by its label pair, the signed relative error of the reconstructed area against the registered ground-truth area; the row is the median over the 47 ground-truth cases of each case's median.
- *All-surface tetrahedra*: tetrahedra of the tesselation whose four vertices are all interface points, with no interior point among them.
- *Junction-angle error*: measured against the Neumann angles implied by the ground-truth tensions, `cos θ(ab, bc) = (γ_ac² − γ_ab² − γ_bc²) / (2 γ_ab γ_bc)`; the row is the median over the 47 cases of each case's median error. As a control, the ground-truth simulation mesh itself sits 0.34° from these Neumann angles on the same statistic.
- *Junction length*: total reconstructed trijunction length relative to the registered ground truth's.
- *Triple lines missed*: ground-truth triple lines with no reconstructed counterpart, summed over the 47 cases.

The revised pipeline explicitly detects and protects triple junctions before triangulating, which `dithered` does not; that is why its tesselation has far fewer all-surface tetrahedra and slivers. Junction relocation, which runs after triangulation in both variants, brings both junction lengths to within 1 % of the ground truth (with relocation off, `dithered` reads 1.045× and `link_checked` 1.012×; 1.011× on 0.7.0); `dithered` reads junction angles more accurately with relocation off as well (7.83° against 9.97°; 10.09° on 0.7.0). Since 0.7.1 the two cells of cases 031 and 038 that no tetrahedron lies inside are left out of the mesh, so their triple lines count as missed: one more in each variant.

## Speed and memory

| | `dithered` | `link_checked` | 0.5.1 (`dithered` / `link_checked`), or 0.7.0 for the mesh points |
|---|---:|---:|---:|
| wall time, 51-case total | 101.0 s | 110.6 s (**1.095×** `dithered`) | 103.7 s / 113.4 s (1.094×) |
| wall time, per case | — | 0.93×–1.17× `dithered` (51/51 cases inside 1.20×) | 0.91×–1.19× |
| tesselation points, 51-case total | 277 922 | 191 428 (fewer, despite the extra junction/boundary-layer point families) | 277 922 / 191 415 |
| mesh points, 51-case total | 71 272 | 61 295 | 71 271 / 61 271 (0.7.0 and 0.5.1) |

Wall times come from `dw3d_benchmarks/paired_timing.py` with the shipped defaults (junction relocation on): both variants run back to back in one process on every case, keeping the minimum per case and variant over two repeats, single-threaded, on one machine (Apple M3 Max, 16 cores, 137 GB RAM, macOS 14.8.4) with a 1-minute load average between 1.2 and 3.9 throughout. The two versions were run alternately (0.5.1, the development version, twice, 0.5.1), one repeat per run: paired per case, the development version took 0.974 of 0.5.1's time on `dithered` (per case 0.92–1.02) and 0.975 on `link_checked`. These times were measured in early October 2026 on the code prepared for 0.6.0 (never published; its changes are part of 0.7.0), before 0.7.0's lighter distance transform (below), which changes no mesh, and were not measured again for 0.7.0. Point counts are totals over the 51 cases from the same run as the mesh-quality table, on 0.7.1's meshes; the tesselation points are placed before the tetrahedralisation and do not depend on its order. Before 0.5.0, with relocation off and on a different day, the same totals were 81.08 s and 92.94 s; most of the difference is junction relocation, whose cost `RELEASE_NOTES.md` (0.5.0) describes.

**A large volume.** 0.7.0 rewrites several steps for large tissues without changing any vertex (`CHANGELOG.md`). Measured in early October 2026 on the code prepared for 0.6.0, before 0.7.0's lighter distance transform, and not measured again since, on a synthetic stand-in — a Kelvin foam of 15 cells voxelised at 512³ (1.3×10⁸ voxels), large in voxels rather than in cells — reconstruction with the defaults takes 56.6 s against 61.9 s on 0.5.1, and the whole run from mask to pressures peaks at **8.4 GiB of resident memory against 18.1 GiB**; at 256³, 6.2 s against 6.8 s and 2.6 against 2.9 GiB (medians of two runs per version, alternated, same machine and load as above). The two versions' meshes of these volumes differ only by the winding of one or two triangles (the flat-tetrahedron fix above). On tissues of many cells the gain in time is much larger, because the seed search used to make one pass over the volume per label.

On a 1 866-cell tissue (its mask meshed on its native grid, then `foambryo`'s tensions), the whole run takes **9.5 min against 47.5 min** on 0.5.1 (reconstruction 9.2 min against 43.5 min), and peaks at **30.9 GiB against 54.7 GiB**: two runs of 0.7.0 and one of 0.5.1, alternated, one process at a time on one machine with a 1-minute load of 3 to 7. The time went mostly from choosing the watershed's seeds (27.9 min to 5.6 s) and from the guard of junction relocation (6.6 min to 46 s). The speed-ups change no mesh; the two versions' meshes of this tissue have the same 93 097 points and labels and differ in 2 vertex positions and 166 triangle windings, from fixes of 0.6.0 (relocation refuses a move that would invert a cell; the triangles closing cells on the image border and those on flat tetrahedra are wound correctly).

Both variants use the same Euclidean distance transform, so peak memory is dominated by the EDT stage regardless of variant. Since 0.7.0 it builds its boundary masks in `uint8` and assembles its field in `float32` (`CHANGELOG.md`, 0.7.0), with the same output: on a 1 866-cell tissue the stage's peak fell from 21.0 to 14.9 GiB on the native grid and from 38.1 to 28.1 GiB on cubic voxels (three runs per version, alternated; the function measured as before is 0.5.1's). The whole reconstruction's peak there is set by junction relocation, not by this stage. The EDT's own scaling to large volumes, independent of which reconstruction variant reads it, was measured separately with `dw3d_benchmarks/edt_scaling.py`, on synthetic Lloyd-relaxed foams (`dw3d_benchmarks/synthetic_foam.py`) whose cell count grows with the volume so that the cell size stays fixed, each variant and volume in a fresh process. At 1024³ voxels (1728 cells) the resident memory of the EDT stage is 32 047 MB for the dense classical EDT (29.8 bytes per voxel), against 11 993 MB for `compute_edt_float32` and 9 388 MB for `compute_edt_tiled`, whose output is identical to the dense EDT's. That measurement predates 0.5.0 and the lighter transform above, and was not repeated since.

No per-variant peak-RSS comparison was measured, since the EDT stage — not the tesselation size difference above — is the memory bottleneck at the scales this package targets.

## Force-balance residual (`rho_free`): every reconstruction is ~50x the reference mesh

**Population: the 42 equilibrium cases of the benchmarking dataset** (`005`, `015`, `018`, `021`, `043` excluded — see `foambryo`'s `foambryo_benchmarks/case_groups.py`). Measured at `min_distance=3`, dither seed 42, with `foambryo`'s `compute_inference_diagnostics` on every mesh; no case failed. The table gives the median over the 42 cases. The reference row is the ground-truth simulation mesh itself, with no reconstruction.

`rho_free` is the force-balance residual left when the tensions are free to take their best-fitting values — that is, the part of the residual that **no** tension assignment can remove, because it is a property of the mesh's geometry rather than of the tensions on it.

Two of the arms below are experimental and are not shipped in this package. `move_to_mask_line` moves `dithered`'s own junction points onto the junction locus derived from the mask at sub-voxel precision, adding and removing no point. `move_null_control` moves the same points by the same distances in a direction that carries no information, as a control. Both act on the mesh after junction relocation.

| arm | median `rho_free` | vs the reference mesh | contact-angle slope (length-weighted) | 0.5.1, where different (`rho_free`, ratio) |
|---|---:|---:|---:|---:|
| reference simulation mesh (control) | `0.0116` | 1x | — | same |
| `dithered` (shipped default) | `0.5414` | **46.8x** | `-0.2118` | 0.7.0: `0.5511`, 47.6x, `-0.2135`; 0.5.1: `0.5556`, 48.0x |
| `move_to_mask_line` | `0.5501` | **47.6x** | `-0.2262` | 0.7.0: `0.5501`, 47.6x, `-0.2250`; 0.5.1: `0.5519`, 47.7x |
| `move_null_control` | `0.5659` | **48.9x** | `-0.2518` | 0.7.0: `0.5739`, 49.6x, `-0.2556`; 0.5.1: `0.5798`, 50.1x |

The contact-angle slopes were the same in 0.5.1 and 0.7.0; `rho_free` differed because it reads cell volumes, and 0.7.0 winds a few triangles differently (see the first section). 0.7.1's meshes move both, by up to 0.010 in `rho_free` and 0.004 in slope.

The contact-angle slope measures how far a mesh flattens contact angles toward 120°. For every trijunction line and each of its three materials, it takes the contact angle `foambryo` reads on the mesh (the per-line mean of `compute_angles_tri`) and the Neumann angle implied by the ground-truth tensions, and regresses (measured − true) on (true − 120°) by weighted least squares, each row weighted by its junction line's length, pooled over the 42 cases. 0 is unbiased; −1 would mean every angle is read as 120°.

Two things follow, and the second is the useful one:

- **Every dw3d-reconstructed mesh carries ~50x the reference mesh's irreducible force-balance residual**, regardless of which arm produced it. This is a floor on what any force-balance method can achieve on these meshes, and it is not a tension-estimation error.
- **`rho_free` hardly moves when the contact-angle slope moves.** Across the three arms above, the slope moves by `0.040` (a 19 % spread, as max/min − 1) while `rho_free` stays within `0.541`–`0.566` (a 5 % spread, same convention). On 0.7.1's meshes the two order the arms the same way (`dithered` has both the lowest `rho_free` and the shallowest slope); on 0.7.0's they did not (`move_to_mask_line` had the lowest `rho_free`, `0.5501` against `0.5511`), so the order is within the spread of the meshes. **`rho_free` is therefore not a proxy for contact-angle quality and must not be used as one** — an arm can improve contact angles measurably while `rho_free` moves by a few percent.

This is why the angle-reading and operator-inverting method families have to be evaluated separately: they are sensitive to different defects of the same mesh. That distinction is documented in the `TensionComputationMethod` docstring (`foambryo/src/foambryo/tension_inference.py`, "Which methods read angles and which invert an operator, and why the distinction now has measured consequences") and is not restated here; what this section adds is the quantification above, and the next subsection adds how the angle-reading family's bias depends on mesh spacing.

### The angle-reading family's bias is spacing-dependent, and not a constant

**Population: the same 42 equilibrium cases, `dithered`, dither seed 42**, reconstructed at each `min_distance` below; every case was reconstructed and measured at every spacing, with no failure.

The shipped default `TensionComputationMethod.YoungDupre` is an angle-reading method, and it weights every trijunction line by its length. In that (length-weighted) convention the contact-angle flattening slope of `dithered` is:

| `min_distance` | median transverse lever (vox) | contact-angle slope | matched coarseness floor | excess | abnormal non-manifold edges | 0.7.0 and 0.5.1, where different (slope / excess) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.93 | `-0.4972` | `-0.0324` | `-0.4648` | 0 | 0.7.0: `-0.4924` / `-0.4600`, 28 edges; 0.5.1: `-0.4935` / `-0.4611` |
| 2 | 6.28 | `-0.2402` | `-0.0848` | `-0.1553` | 0 | 0.7.0: `-0.2427` / `-0.1578`; 0.5.1: excess `-0.1579` |
| 3 (shipped) | 8.71 | `-0.2118` | `-0.1388` | `-0.0730` | 0 | 0.7.0: `-0.2135` / `-0.0747`; 0.5.1: excess `-0.0748` |
| 5 | 13.96 | `-0.2455` | `-0.2653` | `+0.0198` | 0 | 0.7.0: `-0.2481` / `+0.0171` |

The columns, in plain words:

- *Median transverse lever*: for each triangle incident on a junction edge, the distance from the edge's midpoint to the triangle's opposite vertex — the length over which a contact angle is read. Median per case, then median over the 42 cases.
- *Contact-angle slope*: as defined in the previous section.
- *Matched coarseness floor*: the same slope, in the same length-weighted convention, measured on the ground-truth mesh after it has been coarsened by edge collapse to the arm's own transverse lever — what a perfect mesh reads at that coarseness.
- *Excess*: slope minus floor, the part of the flattening that coarseness does not explain.
- *Abnormal non-manifold edges*: total over the 42 cases.

- **The slope is stable over `min_distance` 2–5 (spread `0.034`) and is not stable below that.** Do not extrapolate any angle-bias calibration below `min_distance=2`.
- **`min_distance=1` is not a usable setting.** Its slope is more than twice the shipped default's (`-0.497` against `-0.212`) and median tension contrast falls from `0.787` to `0.595`. On 0.7.0 it also carried 28 abnormal non-manifold edges (contrast `0.791` to `0.612`); 0.7.1's mesh surgery leaves none, so the damage is now in the angles alone. Tension contrast is, per case, the least-squares slope of the gauge-matched `YoungDupre` tensions against the ground-truth tensions (1 is perfect; below 1, tension differences are compressed), with the median taken over the cases where it is defined (36 of the 42 at the shipped spacing). Meshes stay watertight (42/42), so the damage is junction-local.
- At `min_distance=3` coarseness explains 66 % of the flattening; at `min_distance=1`, 7 %. At `min_distance=5` it explains all of it, and the excess turns slightly positive (`+0.0198`): the relocated mesh reads angles a little *less* flattened than a perfect mesh of the same coarseness, the drift that `CoarseSpacingRelocationWarning` warns about at coarse spacings.

## Junction curves: `extract_junction_curves`, a more accurate reading of the same lines

`extract_junction_curves(mask, points, triangles, labels)` reads the trijunction curves off the segmentation mask and keys them to the reconstructed mesh's own trijunctions. It is an **additive, opt-in** read: nothing in the reconstruction pipeline calls it. Junction relocation, the default since 0.5.0, uses the same curves as its targets, so on a relocated mesh the mesh's own lines already sit close to them, and most of the advantage below is now built into the default mesh.

All numbers below: 47-case synthetic benchmark, 42 equilibrium and 5 non-equilibrium cases **never pooled** (equilibrium quoted), `dithered` meshes (relocation on), reference registered into the mask frame by `similarity_to_mask_frame`, case-level bootstrap with 10,000 draws and seed `20260901`. The position error is the median distance in voxels from a curve's vertices to the reference polyline; lower is better. Each arm is read at **its own** median junction-edge spacing, so this compares accuracy and not reading resolution. Each metric's *matched-spacing floor* is the value the reference line itself scores when subsampled at that same spacing. The numbers were measured through the shipped `extract_junction_curves` API, whose junction detector was checked bit-identical to the research prototype's on all 47 masks (918 polylines) and reproduces the prototype's position error to the last digit.

### Accuracy against the reference

Median position error, in voxels:

| `min_distance` | `extract_junction_curves` (default) | the mesh's own junction lines | advantage |
|---|---:|---:|---:|
| 3 | **0.5796** | 0.6735 | **1.16x** |
| 4 | **0.6146** | 0.7449 | **1.21x** |

Without relocation (`relocate_junctions=False`, on 0.7.0, which then reproduced the 0.4 meshes; not measured again) the mesh's own lines sit at 0.8642 and 0.9277, and the curves' advantage is 1.48x and 1.52x. The first implementation (not distributed), read the same way on the same 42 cases, gives the same comparison across five spacings: 1.07x at `min_distance=2` (1.08x on 0.7.0), 1.16x at 3, 1.20x at 4, 1.38x at 5 and 1.93x at 7 — the advantage grows as the mesh coarsens.

Against each metric's own matched-spacing floor (`min_distance=3`), the curves read tortuosity at **1.001**, curvature at **1.018** and mean tangent error at **1.99x**, where the mesh's own lines read 1.002, 1.024 and 3.35x (0.7.0: curvature 1.019 and 1.027, tangent error 3.33x). Of the 24 head-to-head comparisons measured here — 2 spacings x 2 estimators x 6 metrics (tortuosity, mean and 90th-percentile tangent error, curvature, median and 90th-percentile position error), paired per case over the 41 equilibrium cases that enter the pairing — the 16 on tangent and position error favour the curves on at least 9,999 of 10,000 draws (15 of them on every draw); on tortuosity and curvature the two are not distinguished (the curves win on 29–92 % of draws; 26–92 % on 0.7.1's meshes before its enclosed-media fix). These 24 counts were read from the same bootstrap as the numbers above, on 0.7.1's final meshes.

### Completeness, which is the trade

More accurate and **less complete**. At `min_distance=3`, 621 of the mesh's 690 trijunctions have exactly one extracted counterpart (**match rate 0.900**; 0.908 at `min_distance=4`), with **zero missed** — every trijunction the mesh has, the mask has a curve for. The remaining 10 % is fragmentation: 39 split, 24 ambiguous, 6 merged, plus 27 invented triples the mesh does not have at all (26 before 0.7.1's fix of cells no tetrahedron lies inside: case 038's 26-voxel cell no longer has a mesh trijunction, while the mask keeps its curve). Total arclength is 0.9965 of the reference's.

**This is why `matched` and `invented` are separate fields and are never merged.** Measured distance from a curve to **all three** of its own interfaces at once (the largest of the three distances, so that a point near only one interface does not count), per curve, `min_distance=3`:

| | n curves | median | p90 | p99 | within 1 voxel |
|---|---:|---:|---:|---:|---:|
| `matched` | 621 | **0.523** | 1.470 | 4.426 | 0.834 |
| `invented` | 27 | 29.71 | 45.82 | 50.39 | **0.000** |

(0.7.0: matched 622 curves, median 0.522, p90 1.477, p99 4.418, 0.830 within 1 voxel; `invented` unchanged.) The two populations are structurally different, not two ends of one distribution. `curves.matched` is the set you want; `curves.invented` is there so completeness is visible rather than silent.

### Cost, and why the default runs no linear program

| estimator | s/case, `min_distance=3` | position error | what it is |
|---|---:|---:|---|
| `"plain"` (default) | **0.77** | 0.5796 | detection only, no linear program |
| `"anchor"` | 4.47 | 0.5208 | one Chebyshev-centre LP per returned vertex |
| `"centroid"` | ~30x `"anchor"`'s LPs | — | reproduces the first implementation's estimator only |

Seconds per case are medians over the 42 equilibrium cases, measured before 0.5.0 on a machine shared with other jobs and not repeated since, so read them as upper bounds rather than exact; 0.7.0 detects the curves in slabs of the volume, which changes their memory and not their output.

Before 0.5.0, dw3d reconstructed one of these cases in about 1.5 s, so `"anchor"` costs roughly three times the whole reconstruction it annotates, for a 10 % improvement in position error. That is not a default. Its real argument is a different one. Near a narrow wedge, the mask itself places the junction slightly into the narrowest sector. Measured as the junction's displacement along the bisector of the narrowest sector, regressed on that sector's Neumann angle minus 120°, the part of this bias that no mask-based method can remove — where the mask's own best-fitting flat junction sits relative to the reference — is −0.01123 voxels/degree on the 42 equilibrium cases. On the first implementation, the polytope's deepest interior point, which `"anchor"` returns, carries **only** that irreducible term (slope −0.01165 voxels/degree), while the centroid also carries the feasible set's internal asymmetry, at −0.01435. Reach for `"anchor"` when a systematic bias against narrow wedges matters, not to shave a tenth of a voxel.

### The honest residual

The mask's own identifiability floor — the best any mask-based method can do — is **0.107 voxels**: the median, over 5088 trijunction windows of radius 3 voxels from the 42 equilibrium cases, of the smallest worst-case position error achievable within the set of flat three-half-plane junction configurations that reproduce the mask exactly (a lower bound for that model class). These curves sit at 0.5796, a factor of **5.4** above it (the `"anchor"` estimator's 0.5208 is a factor of 4.9); the relocated mesh's own lines sit at 6.3x. So this is **better geometry, not correct geometry**, and that factor is open. Two explanations for it were excluded on the first implementation (the choice of point inside the feasible set, worth 0.06 voxels; and the extraction's lattice coarseness, which smoothing already recovers). Three remain: the flat three-half-plane model, the reference registration's own ~1.14-voxel residual (mean per-correspondence residual, measured on one case), and the construction of the floor itself (computed at a single window radius).

### Controls

* **No ground truth anywhere in the read.** Re-running the whole API over a directory holding only the masks and the three mesh arrays, with every `.rec` structurally unreachable, gives a **hash-identical** result (SHA-256 `042cec82…e320d5` over every coordinate of all 47 cases, for the `"anchor"` estimator, on 0.7.1's meshes; `f6145dfb…5b7a5f` on 0.7.0's; the default's output is likewise hash-identical).
* **Null.** Permuting the mask's label values with a fixed seed (`20260901`) leaves the geometry untouched and scrambles only the material attribution. Position error goes 0.5796 → **36.48** voxels and mean tangent error 4.62° → **49.93°**, both on 1.000 of draws; the match rate against the mesh collapses from 0.900 to **0.513**, with 289 of the mesh's trijunctions missed where the arm misses none.
* **Contact angles and multimaterial validity are untouched by the read**, verified rather than assumed: 1974 scalars (21 per case: the contact-angle and multimaterial-validity measures) re-measured on all 47 cases at both `min_distance` 3 and 4 after the API has run, against the same scalars measured on the same meshes without it, zero differences at `1e-9`, and no mesh array modified.

## Bottom line

- Use the **default** (`get_default_mesh_reconstruction_algorithm`) when the mesh feeds tension/pressure inference — the common case.
- Use **`get_link_checked_mesh_reconstruction_algorithm`** when the quality of the tesselation itself (fewer all-surface tetrahedra and slivers) or a low interface-area bias matters more than today's downstream inference accuracy, or when comparing against future inference methods that may not share `ForceBalanceKKT`'s and `YoungDupre`'s sensitivity to mesh quality.
- All other named variants (`get_deterministic_mesh_reconstruction_algorithm`, `get_junction_protected_mesh_reconstruction_algorithm`, ...) are intermediate configurations kept for reproducibility of the development history; they are not recommended for new work.

## Package renamed `benchmarks` -> `dw3d_benchmarks`

The top-level `benchmarks/` package was renamed to `dw3d_benchmarks/` because both this repository and `foambryo` shipped a package named `benchmarks`, and a script that put both repositories on `sys.path` could silently resolve the name to either one.

**Migration hazard (found in `foambryo`'s checkout, not this one):** pulling this rename into an existing checkout can leave a stray `benchmarks/__pycache__/` directory on disk — `git mv` moves tracked files, not gitignored build artifacts. With `benchmarks/__init__.py` gone but the directory still present, Python imports it as an empty implicit namespace package (PEP 420): `import benchmarks` still succeeds, with no `__file__` and no real submodules. `tests/test_no_ambiguous_benchmarks_package.py` now tolerates this case (only a *real*, `__file__`-bearing `benchmarks` package fails it), but the stray directory should still be deleted on any checkout that hits it — `rm -rf benchmarks/` once `git status` confirms it holds nothing tracked.
