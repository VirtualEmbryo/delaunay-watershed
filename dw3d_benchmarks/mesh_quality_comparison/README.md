# Mesh-quality comparison

One script that compares one or more dw3d reconstruction configurations ("mesh sets")
against `benchmarking-dataset`'s ground truth, on the same seven metric families every time,
and writes one flat results JSON plus four figures. It compares **mesh sets**, not dw3d
versions — pointing it at meshes built by two different dw3d versions is the natural next
step, but is out of scope here (see `REPRODUCE.md`'s "What this is not").

Run `python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets --help` for the
three subcommands (`profile`, `validate`, `batch`); `REPRODUCE.md` gives the exact command
the reference run used, its expected wall-clock, and what each output file contains.

**Requirements.** The `benchmarks` extra (`pip install -e ".[benchmarks]"`) covers plotting
only (matplotlib). This package's mesh-quality metrics also import `foambryo` directly,
which must be installed separately, **editable from a local checkout, not from PyPI**: the
published `foambryo` (0.3.5) is older than the crash fix these numbers depend on and pulls
in `torch`/CUDA as a core dependency. See `REPRODUCE.md`'s "Requirements" for the exact
install command.

## Cohort

45 of the dataset's 47 cases. `031` and `038` are excluded by case id: both masks had their
largest cell emptied into the exterior in the single labelling pass that produced all 47
masks (2022-03-11); the ground-truth meshes (2022-03-08) predate that pass and are intact.
This script never reads, repairs or deletes those two masks.

Of the 45, 40 are at mechanical equilibrium and carry every accuracy claim; the 5 that are
not (`005`, `015`, `018`, `021`, `043`) are reported separately, never pooled with the 40 —
enforced structurally by `case_metadata.require_single_group`. Case `004` is kept in the
equilibrium group with an integrity flag (2 malformed triple-line edges of 539); every
aggregate is reported with and without it.

## What is measured

- **M1 Contact angles** — production `foambryo.geometry.compute_angles_tri(unique=False)`,
  against the ground-truth mesh's own angles and against the true Neumann angles from the
  ground-truth tensions. Signed error, flattening slope (unweighted and length-weighted),
  matched-spacing floor.
- **M2 Interface areas** — signed relative error per interface, plus a registration-free
  area-fraction error.
- **M3 Cell volumes** — `foambryo.validation.mesh_io.signed_cell_volumes`; exterior reported
  separately from cells.
- **M4 Interface mean curvature** — `foambryo.curvature.compute_curvature_interfaces`, one
  function applied identically to the reconstruction and the ground-truth mesh. Absolute
  error below a stated crossover, relative above it.
- **M5 Junction line geometry** — tortuosity, tangent error, curvature ratio, line position
  error, quadruple-point tangent agreement. **Provisional**: no independent anchor exists for
  this metric; treat its numbers accordingly.
- **M6 Validity** — watertightness, abnormal non-manifold edges, degenerate/duplicate
  triangles, quadjunction edges (an expected discretisation artefact, not corruption), sliver
  fraction, interface identity against ground truth.
- **M7 Budget and cost** — points, triangles, triangles per cell, wall-clock, peak RSS.

Every median is a case-level bootstrap CI (10000 draws, seed `20260901`). Undetermined
values use `case_metadata.median_of_determined` — never a bare median.

## The three comparisons this package can make

| script | varies | holds fixed |
|---|---|---|
| `compare_mesh_sets.py` | the reconstruction configuration | dw3d version, seed |
| `compare_versions.py` | the dw3d version | configuration, seed |
| `seed_sensitivity.py` | the dither seed | dw3d version, configuration |

`compare_mesh_sets.py` accepts either a named configuration (built here) or `--mesh-dir
NAME=PATH`, a directory of prebuilt `<case>_mesh.rec` files. The second form is what makes a
cross-version comparison sound: `compare_versions.py --export-to` writes each version's meshes,
and then **one** metric code measures them all, so a difference is attributable to the mesh and
not to the measuring code. (Measuring each version's meshes with its own metric code would
confound the two -- and for a pre-0.4.0 arm it is impossible anyway, since
`dw3d_benchmarks.metrics` imports `points_on_edt.boundary_layer_families`, absent in 0.3.6.)

`seed_sensitivity.py` exists because the shipped default (`dithered`) places points with a
**randomly dithered** rule fixed by a seed. "Reproducible at a fixed seed" and "better than the
alternatives" are different claims, and the second one measured at a single seed can be a lucky
draw. This script reports how far the cohort median moves across seeds, so a configuration gap
can be read against the seed's own influence rather than assumed to exceed it.

## Version provenance

Every figure carries a footer stating the exact `dw3d` and `foambryo` commit used, and how
that commit relates to what's actually pushed (`N commits behind origin/master` / `N commits
ahead of origin/main, unreleased` / equal). This is a **code-version** label, not a
mesh-set label: all four configurations (`dithered`, `deterministic`, `offset_included`,
`link_checked`) in one run share the same single `dw3d` commit — they differ only in which
factory-preset reconstruction algorithm each name calls, never in code version. Comparing the
*same* configuration built at two different `dw3d` commits is the separate, later comparison
`REPRODUCE.md`'s "What this is not" describes; this script does not do that on its own.

## Validation gate

Four numbers this project already knows, from four independent sources, checked before any
figure exists. Anchors 2, 3 and 4 are hard gates: a failure stops the run before the batch.
Anchor 1 is a diagnostic that never stops anything. See `validation_gate.py` and the results
report for the values obtained.

## Design notes

- **No ground truth in the estimator.** Every mesh set is reconstructed from the mask alone;
  ground truth is read only in this script's analysis functions, never passed to a dw3d
  reconstruction call.
- **No path outside this repository.** `foambryo` is a declared dependency (`setup.cfg`'s
  `benchmarks` extra), imported as a package — not a workspace-relative path into a sibling
  checkout. The dataset cohort constants and `median_of_determined` in `case_metadata.py` are
  restated from `foambryo_benchmarks` rather than imported, for the same reason; a test checks
  the two copies agree when a sibling `foambryo` checkout happens to be present.
- **Shared palette.** `figures/_style.py` copies (not imports) the validated categorical
  palette from `foambryo_benchmarks/figures/_style.py`, with the same fixed hue order. A test
  asserts the two colour lists are identical.
