# Changelog

User-facing changes only, organised by what a user experiences.

## 0.5.0

**The default changed: junction relocation is on in every reconstruction configuration.** Meshes
come out with the same triangles and labels as in 0.4, and different vertex positions on the
trijunction lines. To get the 0.4 meshes back exactly, switch it off: pass
`relocate_junctions=False` to `MeshReconstructionAlgorithm`, call
`MeshReconstructionAlgorithmFactory().set_junction_relocation(False)`, or set
`algorithm.relocate_junctions = False` on an algorithm a `get_*_mesh_reconstruction_algorithm`
function returned. See `RELEASE_NOTES.md` for the limitations that matter on real data.

- **Junction relocation, now the default.** `dw3d.relocate_junction_vertices` is a post-process on a
  finished mesh that moves its trijunction vertices onto the trijunction curves the mask itself
  gives. It changes vertex positions only — triangles and material labels come back untouched — so
  a treated mesh can be compared vertex-for-vertex with its own untreated original. Read what it
  did from the algorithm's `relocation_report`, and whether it ran, with which guard and whether it
  warned from the new `relocation_provenance`.

  **What it costs, in seconds**, because the reconstruction it follows varies by an order of
  magnitude between volumes: about +0.75 s on the benchmark meshes, whose reconstruction takes about
  5 s (+15 %); 0.1–1.5 s per timepoint on real embryo volumes, whose reconstruction takes about
  0.6 s (median 0.34 s and 0.48 s on two developmental time series; about 65 s per 154-timepoint
  series). Within one real series it grows faster than linearly with the number of triangles — a
  fitted exponent of 2.68 over a 2.2-fold range of mesh sizes, which describes that series and is
  not a tissue-scale law. A faster neighbour search is planned for 0.5.1; it will leave every
  relocated mesh bit-for-bit unchanged.

- **New warning: `dw3d.CoarseSpacingRelocationWarning`** (a `UserWarning`), emitted once per
  reconstruction when relocation is on and the point placement's `min_distance` is 5 or more. Its
  text: *"At min_distance ≥ 5, the check on surface-tangent contact angles showed a small drift that
  lies within the check's own noise, and larger spacings have not been measured; facet-based contact
  angles improve. If tangent-based angles matter for your analysis, compare with
  `relocate_junctions=False` (or `set_junction_relocation(False)`)."* A custom point-placement
  function that carries no `min_distance` never triggers it. Filter it like any warning:
  `warnings.filterwarnings("ignore", category=dw3d.CoarseSpacingRelocationWarning)`.

- **Removed**, as their deprecation warnings announced since 0.4.0: `get_v0_3_algorithm` (use
  `get_dithered_algorithm`), `get_a1b_algorithm` (use `get_deterministic_algorithm`),
  `get_a3_a5_algorithm` (use `get_junction_protected_algorithm`), and the matching top-level
  `get_v0_3_…`, `get_a1b_…` and `get_a3_a5_mesh_reconstruction_algorithm`.

- **Deprecated, removed in 0.6.0** — three exact synonyms, which now emit a `DeprecationWarning`
  naming the equivalent call and still return the identical mesh:
  `get_offset_included_algorithm()` → `get_junction_protected_algorithm()`;
  `get_offset_excluded_algorithm(...)` → `get_junction_protected_algorithm(..., exclude_offsets_from_surface=True)`;
  `get_cubic_score_algorithm()` → `get_junction_protected_algorithm(spline_order=3)`; and their
  top-level `get_*_mesh_reconstruction_algorithm` aliases. Six configurations remain recommended:
  default, dithered, deterministic, link-checked, junction-protected and boundary-layer.

- **Golden masters.** `tests/golden/` now holds the relocated meshes; the 0.4 fixtures moved,
  byte for byte, to `tests/golden_without_relocation/`, and every golden-master test runs against
  both — the second with `relocate_junctions=False`.

- **Documentation.** `BENCHMARKS.md` is self-contained: each number states its population and how
  it was measured. Those numbers were measured **without** junction relocation (the 0.4 default).
  Docstrings no longer cite files that are not distributed.

The rest of this section describes what 0.5.0 adds besides the default:

  What it costs and what it buys, measured over 40 benchmark cases at four reconstruction spacings
  and on four point samplers — 16 independent adjudications, all passing. Line position error, local
  tangent error and discrete curvature all improve with their confidence intervals clear of zero at
  every spacing on every sampler; no pre-existing mesh-validity number changes on any case. On the
  seed-free samplers it also improves contact angles by 2.3–3.5 degrees and gauge-matched tension
  error by 20–40 %; on the shipped default's seeded sampler the line improves and the contact angle
  does not move significantly either way. Its cost is stated in seconds above.

  It is guarded twice. The local guard refuses any move that would invert or collapse a triangle
  touching the vertex. The second guard is the one that matters: a move can drive one sheet of the
  mesh through a distant sheet while every triangle around the moved vertex stays perfectly
  oriented, and no local check can see that. Measured on the benchmark cohort, all twelve existing
  validity numbers stayed bit-identical on 40 of 40 cases while 23 such crossings existed on 5 of
  them. The second guard makes the crossing count non-increasing by construction, and it costs
  between 0.04 % and 0.21 % of the moves to do it.

- **`dw3d.find_self_intersections` and `dw3d.count_self_intersections`.** The exact
  triangle–triangle intersection test the relocation certifies itself with, now part of the
  library. Two triangles sharing a vertex are adjacent and are never tested — three triangles meet
  along every trijunction edge by design — so what it counts is a genuine defect, never the intended
  non-manifold structure. It is a count, never a score, and no threshold is proposed.

## 0.4.2

No change to reconstruction behaviour: every default is bitwise identical to 0.4.1 (verified
by hash on the mesh output of the dithered and deterministic configurations).

- **The three deprecation warnings' text no longer implies `0.5.0` already exists.**
  `get_v0_3_algorithm`, `get_a1b_algorithm` and `get_a3_a5_algorithm` (and the matching
  `dw3d/__init__.py` aliases) now say "removed in a future 0.5.0 release" instead of
  "removed in 0.5.0"; the replacement function and the removal commitment are unchanged.
- **New `dw3d_benchmarks/mesh_quality_comparison/` package**: compares mesh quality (M1-M7:
  contact angles, interface areas, cell volumes, curvature, junction-line geometry, mesh
  validity, cost) across the four core reconstruction configurations and across dw3d
  versions. A `benchmarks` extra (`pip install ".[benchmarks]"`) covers the figure scripts'
  matplotlib requirement without pulling in `viewing`'s napari dependency.
- One-line fix: `dw3d_benchmarks/run_case.py`'s `VARIANT_GETTERS` comment stated the default
  is `offset_excluded_linkcheck`; it is not — the default is `dithered`.

## 0.4.1

Release hygiene only. No change to reconstruction behaviour: every default is bitwise identical
to 0.4.0 on all 47 benchmark meshes.

- **`MeshReconstructionAlgorithmFactory`'s three development-named factory methods are renamed
  to describe their mechanism**: `get_v0_3_algorithm` → `get_dithered_algorithm`,
  `get_a1b_algorithm` → `get_deterministic_algorithm`, `get_a3_a5_algorithm` →
  `get_offset_included_algorithm`, and the matching top-level exports in `dw3d/__init__.py`
  (e.g. `get_v0_3_mesh_reconstruction_algorithm` → `get_dithered_mesh_reconstruction_algorithm`).
  The old names still work — they now emit a `DeprecationWarning` naming the replacement — and
  will be removed in 0.5.0.
- **`dw3d_benchmarks/run_fingerprints.py`'s `--variant` strings are renamed to match**
  (`v0_3` → `dithered`, `a1b` → `deterministic`, `a3_a5` → `offset_included`); the old strings
  are still accepted, with a printed note pointing at the replacement.
- **Development files no longer ship or return.** Development notes and scratch scripts are
  removed from the repository tip and gitignored so they cannot be re-added by accident. The
  project's development record is kept in the development repository and has
  never been part of the packaged distribution; confirmed by building the wheel and listing its
  contents.
- Internal benchmark filenames and test names that were named after a development phase (e.g.
  `analyze_a5.py`, `d2_defect_patches.py`, baseline JSONs prefixed `a6_`/`a7_`/`a8_`/`a8b_`) are
  renamed to describe what they measure. This does not change any benchmark's recorded numbers.

## 0.4.0

### New: trijunction curves read from the mask, keyed to the mesh's own junctions

`extract_junction_curves(mask, points, triangles, labels)` returns one polyline per material
triple, computed from the segmentation mask and matched against the reconstructed mesh's own
trijunctions. **Purely additive**: nothing in the reconstruction pipeline calls it, no default
changed, no dependency was added, and the contact-angle and mesh-validity numbers are bitwise
unchanged (verified on 1974 measured values across 47 cases at two point spacings).

```python
from dw3d import extract_junction_curves
from dw3d.io import load_rec

points, triangles, labels = load_rec("000_mesh.rec")  # or your own reconstruction
curves = extract_junction_curves(mask, points, triangles, labels)
curves.matched      # the usable set: one-to-one with a mesh trijunction
curves.invented     # triples the mask says exist and the mesh does not have
curves.match_rate   # 0.900 on the 47-case benchmark at min_distance=3
```

**Why you might want it.** These curves are measurably closer to ground truth than the junction
lines the mesh produces on its own — median position error 0.5846 voxels against 0.8642 at
`min_distance=3`, a factor of 1.48, and more as the mesh coarsens (1.90x at `min_distance=7`).
Every head-to-head comparison measured favours them on 1.000 of 10,000 bootstrap draws.

**What it costs you.** They are *less complete*: 90 % of the mesh's trijunctions get exactly one
curve, and the rest arrive fragmented. `matched` and `invented` are deliberately separate fields
and are never merged, because they are structurally different populations — matched curves lie a
median 0.68 voxels from all three of their own interfaces, invented ones 29.7, with **none**
within a voxel.

**What it is not.** The curves sit about 4.8x above the mask's own identifiability floor
(0.107 voxels). This is better geometry, not correct geometry.

The default estimator is `"plain"` — detection only, about 0.8 s per benchmark case against the
~1.5 s dw3d spends reconstructing it. `estimator="anchor"` solves one small linear program per
vertex, costs about six times as much, and buys 11 % of position error; its argument is that it
carries a smaller systematic bias against narrow junction wedges, not that 11 %.
`estimator="centroid"` exists to reproduce the research result and is not a recommendation.

See `BENCHMARKS.md` (section "Junction curves") for the full numbers and the controls, and
`Examples/example_4_junction_curves.ipynb` for a worked example that runs in a few seconds on
data already in this repository.

### The default reconstruction is the published `dithered` algorithm again

Between one earlier internal release and this one, the default reconstruction algorithm was
a revised meshing pipeline (deterministic point placement, explicit junction detection and
protection, and a topology-preserving cleanup of the extracted surface). That pipeline
produces measurably better mesh geometry — lower interface-area bias against ground truth,
far fewer degenerate ("all-surface") tetrahedra, better junction placement — but end-to-end
tension-inference accuracy through `foambryo`, on a benchmark of 42 mechanically-equilibrated
cases, is **worse** with it than with the published algorithm, on every inference method
tested. Since that downstream accuracy is what most users of this package ultimately care
about, **the default reconstruction algorithm has reverted to the published (`dithered`)
configuration.**

The revised meshing pipeline is not removed: it remains fully available, opt-in, as
`get_link_checked_mesh_reconstruction_algorithm`. Reach for it when mesh geometry and
topology matter more to you than today's downstream inference accuracy. See `BENCHMARKS.md`
for the numbers behind this decision and the accuracy finding stated plainly, without
spin: the revised pipeline is currently the worse choice for tension inference, despite
being the better mesh in isolation.

This reconstructs bit-for-bit what dw3d 0.3.6 produced on the current numpy/scipy/
scikit-image stack (see `get_default_mesh_reconstruction_algorithm`'s docstring for the one
caveat: the pre-0.4.0 watershed tie-break ordering, which had no single well-defined answer,
is not restored, since it depended on which sorting kernel happened to run it).

### New capability, unaffected by the default change

- `set_tiled_edt_method(..., dtype=np.float64)`: an opt-in Euclidean-distance-transform
  variant that uses roughly a third of the peak memory of the default EDT computation at
  large volumes, while reproducing it bit-for-bit — unlike the existing `float32` tiled
  variant, which saves the same memory but changes the output when combined with the
  revised meshing pipeline's normal/score computations. Not the default EDT for either
  reconstruction variant; a deliberate choice for memory-constrained large volumes.

### `.rec` / `.arec` files can now carry a metadata appendix

`save_rec(..., metadata={...})` appends a trailing block of ASCII `# key = value` lines
recording `coordinate_frame` (`voxel_index`, `physical`, or `unknown` — the field this
exists for: it is what lets a reader tell whether coordinates are raw voxel indices or
already-scaled physical units), `axis_order`, `spacing`, `spacing_unit`, `source_shape`
and `crop_offset`. Read it back with the new `load_rec_appendix(filename)`, which returns
`None` for a file with no appendix — never a default of isotropic spacing. See
`README.md` for the field list and the format.

The block is written at the *end* of the file, never the start, so this is purely
additive: `save_rec` without `metadata` (the default) writes byte-for-byte the same file
as before this existed, and every reader already in the wild — including dw3d 0.3.6 and
released `foambryo` — loads a file with an appendix exactly as it always has.
`load_rec`'s return type is unchanged; the appendix is read separately, on purpose.

### Nothing else changed

Every other named reconstruction getter (`get_deterministic_algorithm`,
`get_offset_included_algorithm`, `get_boundary_layer_algorithm`,
`get_offset_excluded_algorithm`, `get_cubic_score_algorithm`) is unchanged and produces
exactly the output it always has.
