# Reproducing this run

## Requirements

The `benchmarks` extra (`pip install -e ".[benchmarks]"`) covers plotting only
(matplotlib). This package's mesh-quality metrics also import `foambryo` directly, which is
**not** part of that extra and must be installed separately, **editable from a local
checkout, not from PyPI**: the published `foambryo` (0.3.5) is older than the crash fix this
package's numbers depend on and pulls in `torch`/CUDA as a core dependency -- a multi-GB
footprint no plotting extra should impose. Install `foambryo>=0.4.2` editable:

```bash
uv pip install -p .venv/bin/python -e /path/to/foambryo-checkout-at-integration-0.4.2
```

## Environment

```bash
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python -e ".[benchmarks]"
# foambryo pinned to the local checkout that carries the crash fix (see Requirements above):
uv pip install -p .venv/bin/python -e /path/to/foambryo-checkout-at-integration-0.4.2
```

Assert the install before trusting any number:

```bash
PYTHONPATH="src:." .venv/bin/python -c "import foambryo, dw3d; print(foambryo.__file__); print(dw3d.__file__)"
```

## Commands

```bash
DATASET=/path/to/benchmarking-dataset

# 1. Profile one case, see the projected total, before launching anything larger.
PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets profile \
    --dataset-dir "$DATASET" --mesh-sets dithered --cases 000 --min-distances 3 --workers 6

# 2. Validate the four anchors (hard gates 2/3/4, diagnostic 1). Stops on a hard-gate failure.
PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets validate \
    --dataset-dir "$DATASET" --workers 6 \
    --out dw3d_benchmarks/mesh_quality_comparison/results/validation_gate.json

# 3. Core batch: 4 configurations x 45 cases at min_distance=3. Re-running is resumable
#    (skips whatever is already in results/raw/); pass --force to redo everything.
PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets batch \
    --dataset-dir "$DATASET" \
    --mesh-sets dithered deterministic offset_included link_checked \
    --min-distances 3 --workers 6 \
    --out dw3d_benchmarks/mesh_quality_comparison/results/mesh_quality_core.json

# 4. Extension: dithered and link_checked across min_distance in {2, 3, 5, 7}.
PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets batch \
    --dataset-dir "$DATASET" \
    --mesh-sets dithered link_checked \
    --min-distances 2 3 5 7 --workers 6 \
    --out dw3d_benchmarks/mesh_quality_comparison/results/mesh_quality_all.json

# 5. Figures, from the results JSON only (never a re-run).
PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.figures.make_figures \
    --results dw3d_benchmarks/mesh_quality_comparison/results/mesh_quality_all.json \
    --out-dir dw3d_benchmarks/mesh_quality_comparison/figures
```

## Version-vs-version (the other axis)

`compare_mesh_sets.py` varies the *configuration* inside one checkout. To vary the *code
version* with the configuration held fixed, give each version its own worktree and venv, then
fingerprint them:

```bash
for V in v0.4.0 v0.4.1; do
    git worktree add --detach ~/dw3d-ver-${V//./-} "$V"
    (cd ~/dw3d-ver-${V//./-} && uv venv --python 3.11 .venv \
        && uv pip install -p .venv/bin/python -e .)
done

PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.compare_versions \
    --arm v0.4.0=$HOME/dw3d-ver-v0-4-0 \
    --arm v0.4.1=$HOME/dw3d-ver-v0-4-1 \
    --arm master=$PWD \
    --dataset-dir "$DATASET" \
    --out dw3d_benchmarks/mesh_quality_comparison/results/version_comparison.json
```

It exits 0 when every `(configuration, case)` pair hashes identically in every arm — i.e. the
versions are behaviourally identical and no metric can differ — and 2 when some pair differs,
in which case run `compare_mesh_sets.py` inside each arm and diff the two scorecards for the
configurations named in `differing_pairs`. Configurations are addressed by their legacy
factory names, which are stable across versions; see the module docstring for why
`get_default_...` is fingerprinted separately instead.

## Expected wall-clock

Measured on this machine (Apple Silicon, 6 worker processes): about 4 s per
`(mesh_set, case, min_distance)` job. Anchor 2 (225 reconstructions, abnormal-edge count
only) took 73 s. The core batch (180 jobs) took about 2 minutes; the extension (360 jobs)
about 4 minutes. A different machine's core count changes the wall-clock, not the result:
every number in the results JSON is a property of the mesh, not of how many workers built it.

## What each output file contains

- `results/validation_gate.json` — the four anchors, reproduced or not, with the values
  obtained.
- `results/raw/<mesh_set>__<case>__md<N>.json` — one complete M1-M7 record per job. This is
  the resumability unit: delete one file to force that job to re-run, or pass `--force` to
  redo everything.
- `results/mesh_quality_core.json` / `results/mesh_quality_all.json` — the flat aggregate:
  `{"manifest": {...provenance, cohort, configuration list...}, "records": [...]}`. Figures
  are generated from this file only.
- `figures/*.png` (talk) and `figures/*.pdf` (paper) — the four figures; filenames are
  self-explanatory (`scorecard`, `paired_per_case`, `flattening`, `geometry_fidelity`).

## What this is not

This compares **mesh sets** — four reconstruction configurations, or a directory of
already-built meshes — never dw3d **versions**. Comparing meshes built by two different dw3d
releases is "point this script at the two directories"; it needs no version-comparison
machinery, and is not attempted here.
