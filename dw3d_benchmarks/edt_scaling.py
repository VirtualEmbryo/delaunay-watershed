"""The EDT stage's cost, measured -- wall time, peak RSS, and the scaling curve.

Why every measurement here runs in a fresh subprocess
-----------------------------------------------------
`resource.getrusage(...).ru_maxrss` is a **high-water mark that never decreases** for the
life of a process. Timing two EDT variants in one process therefore reports the larger of
the two for both, which would make the "peak RSS reduced by >= 3x" criterion unmeasurable --
and would report a *pass* as a fail or vice versa depending on the order. `benchmarks/profiling.py`'s
`cost_profile` has this shape (it reads `_peak_rss_mb()` after a reconstruction in a process
that has already done other work), which is fine for its purpose but not for this one's. So each
`(variant, case)` pair here gets its own `python -c` subprocess and reports that process's
own peak, and the driver records the harness's own baseline RSS so it can be subtracted.

Run-to-run noise
----------------
Re-running the 51-case benchmark on code that changes no output measured this machine's
wall-time ratio noise (new / committed records) as **median 0.966, range 0.775-1.047**. This
module's "wall time not worse than baseline" criterion is therefore read against that band and
not against a single run:
`--repeats` defaults to 5 and the driver reports the median ratio and the full range, so a
1.04x reading is correctly called *inside the noise* rather than a regression.

Usage:
    uv run python benchmarks/edt_scaling.py --images                 # the 4 in-repo images
    uv run python benchmarks/edt_scaling.py --synthetic 256 512      # the sparse/tiled ladder
    uv run python benchmarks/edt_scaling.py --plateau-scaling        # the packing loop vs area
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Variants measured. `classical` is the original baseline every ratio is taken against.
# `tiled_float64` is the tiled path emitting float64 rather than float32: the only
# variant here whose field composes with the current default reconstruction, since that reads the
# EDT's dtype and not only its values (the float32 tiled field moves the watershed scores on
# 8 of 8 in-repo configurations; promoted to float64 it is bit-identical). It is NOT in the default
# tuple, so the historical three-variant ladder is unchanged unless it is asked for by name.
EDT_VARIANTS = ("classical", "float32", "tiled")

_CHILD = r"""
import json, resource, sys, time
import numpy as np
sys.path.insert(0, {repo!r})

_SCALE = (1 / 1024**2) if sys.platform == "darwin" else (1 / 1024)
def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _SCALE

baseline_rss = rss()
mask = {load}
after_mask_rss = rss()

from dw3d.edt import compute_edt_classical, compute_edt_float32, compute_edt_tiled
from dw3d.edt_band import narrow_band_seeding
from dw3d.points_on_edt import plateau_packing_extrema

variant = {variant!r}
repeats = {repeats}
times = []
first_run_rss = None
for _ in range(repeats):
    t0 = time.perf_counter()
    if variant == "classical":
        edt_image = compute_edt_classical(mask)
    elif variant == "float32":
        edt_image = compute_edt_float32(mask)
    elif variant == "tiled":
        edt_image = compute_edt_tiled(mask, tile_size={tile_size}, halo={halo})
    elif variant == "tiled_float64":
        edt_image = compute_edt_tiled(mask, tile_size={tile_size}, halo={halo}, dtype=np.float64)
    elif variant == "narrow_band_seeding":
        edt_image = None
        result = narrow_band_seeding(mask, {min_distance}, tile_size={tile_size}, halo={halo})
    else:
        raise SystemExit("unknown variant " + variant)
    times.append(time.perf_counter() - t0)
    if first_run_rss is None:
        # `ru_maxrss` is a high-water mark that never falls, and freed arrays are not
        # returned to the OS, so reading it after `repeats` iterations reports the
        # allocator's accumulated footprint rather than one run's peak. Only the reading
        # taken after the *first* iteration is the single-run peak this phase claims.
        first_run_rss = rss()

row = {{
    "variant": variant,
    "n_voxels": int(mask.size),
    "shape": list(mask.shape),
    "wall_time_s_median": float(np.median(times)),
    "wall_time_s_all": [float(t) for t in times],
    "peak_rss_mb": first_run_rss,
    "peak_rss_mb_all_repeats": rss(),
    "baseline_rss_mb": baseline_rss,
    "mask_rss_mb": after_mask_rss,
    "edt_stage_rss_mb": first_run_rss - after_mask_rss,
}}

if edt_image is not None:
    row["dtype"] = str(edt_image.dtype)
    row["edt_nbytes"] = int(edt_image.nbytes)
    # The point set is what the tiled EDT's correctness criterion (identical seeding output) is
    # about; time it too, because the junction-protection work's timings make the packing the
    # next candidate bottleneck.
    for maximise, name in ((True, "maxima"), (False, "minima")):
        t0 = time.perf_counter()
        points = plateau_packing_extrema(edt_image, {min_distance}, maximise=maximise)
        row["plateau_" + name + "_s"] = time.perf_counter() - t0
        row["n_" + name] = int(len(points))
    row["peak_rss_mb_after_seeding"] = rss()
else:
    row["wall_time_s_median"] = float(np.median(times))
    row["n_maxima"] = int(len(result["maxima"]))
    row["n_minima"] = int(len(result["minima"]))
    row["report_minima"] = {{k: v for k, v in result["report_minima"].items() if not isinstance(v, np.ndarray)}}

print("__ROW__" + json.dumps(row))
"""

_LOAD_IMAGE = "__import__('skimage.io', fromlist=['imread']).imread({path!r})"
_LOAD_SYNTHETIC = (
    "__import__('benchmarks.synthetic_foam', fromlist=['g']).generate_foam_mask_low_memory({side}, n_cells={cells})"
)
_LOAD_NPY = "__import__('numpy').load({path!r})"


def _run_child(load: str, variant: str, repeats: int, min_distance: int, tile_size: int, halo: int) -> dict:
    """Run one `(variant, case)` measurement in a fresh interpreter and return its row."""
    source = _CHILD.format(
        repo=str(REPO_ROOT),
        load=load,
        variant=variant,
        repeats=repeats,
        min_distance=min_distance,
        tile_size=tile_size,
        halo=halo,
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, source built from this module's own template
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        return {"variant": variant, "failed": True, "stderr": completed.stderr[-2000:]}
    for line in completed.stdout.splitlines():
        if line.startswith("__ROW__"):
            return json.loads(line[len("__ROW__") :])
    return {"variant": variant, "failed": True, "stderr": "no row emitted; stdout=" + completed.stdout[-2000:]}


def _report(rows: list[dict], label: str) -> None:
    """Print a per-case table of ratios against `classical`, with the noise band in mind."""
    by_variant = {row["variant"]: row for row in rows if not row.get("failed")}
    base = by_variant.get("classical")
    print(f"\n=== {label} ===")
    for row in rows:
        if row.get("failed"):
            print(f"  {row['variant']:<22} FAILED: {row['stderr'].strip().splitlines()[-1][:120]}")
            continue
        stage = row["edt_stage_rss_mb"]
        line = (
            f"  {row['variant']:<22} t={row['wall_time_s_median']:7.3f}s  "
            f"edt_stage_rss={stage:8.1f}MB  peak={row['peak_rss_mb']:8.1f}MB"
        )
        if base and row is not base and base["edt_stage_rss_mb"] > 0:
            line += (
                f"  |  t_ratio={row['wall_time_s_median'] / base['wall_time_s_median']:.3f}"
                f"  rss_gain={base['edt_stage_rss_mb'] / max(stage, 1e-9):.2f}x"
            )
        if "plateau_minima_s" in row:
            line += f"  |  packing={row['plateau_maxima_s'] + row['plateau_minima_s']:.3f}s"
        print(line)


def measure_images(repeats: int, min_distance: int, tile_size: int, halo: int) -> list[dict]:
    """The four in-repo images, every EDT variant, plus the narrow-band seeding path."""
    out = []
    for name in ("1.tif", "2.tif", "3.tif", "4.tif"):
        path = REPO_ROOT / "data" / "Images" / name
        if not path.exists():
            print(f"skipping {name}: not found (data/ is gitignored)")
            continue
        load = _LOAD_IMAGE.format(path=str(path))
        rows = [
            _run_child(load, variant, repeats, min_distance, tile_size, halo)
            for variant in (*EDT_VARIANTS, "narrow_band_seeding")
        ]
        for row in rows:
            row["case"] = name
        _report(rows, f"{name} (min_distance={min_distance})")
        out.extend(rows)
    return out


def measure_prebuilt(
    path: Path,
    repeats: int,
    min_distance: int,
    tile_size: int,
    halo: int,
    variants: tuple[str, ...] = (*EDT_VARIANTS, "narrow_band_seeding"),
) -> list[dict]:
    """Measure the variants on a mask already on disk as `.npy`.

    Needed from 1024^3 up. `measure_synthetic` regenerates the mask inside every child, which
    is right at 256^3-512^3 (it keeps each measurement hermetic and costs a second or two) and
    wrong at 1024^3, where the generator's `cKDTree` query over 1.07e9 voxels dominates the
    thing being measured and would be paid once per variant. Generating once and loading keeps
    the comparison fair -- every child pays the same `np.load`, and `after_mask_rss` subtracts
    it out of the EDT-stage figure exactly as it subtracts the generator's cost elsewhere.
    """
    load = _LOAD_NPY.format(path=str(path))
    rows = [_run_child(load, variant, repeats, min_distance, tile_size, halo) for variant in variants]
    for row in rows:
        row["case"] = path.stem
        row["mask_source"] = str(path)
    _report(rows, f"{path.stem} (min_distance={min_distance}, from {path})")
    return rows


def measure_synthetic(sides: list[int], repeats: int, min_distance: int, tile_size: int, halo: int) -> list[dict]:
    """The synthetic scaling ladder. `classical` is attempted at every size and allowed to fail loudly."""
    from dw3d_benchmarks.synthetic_foam import default_cell_count

    out = []
    for side in sides:
        cells = default_cell_count(side)
        load = _LOAD_SYNTHETIC.format(side=side, cells=cells)
        rows = [
            _run_child(load, variant, repeats, min_distance, tile_size, halo)
            for variant in (*EDT_VARIANTS, "narrow_band_seeding")
        ]
        for row in rows:
            row["case"] = f"synthetic_{side}"
            row["n_cells"] = cells
        _report(rows, f"synthetic {side}^3, {cells} cells (min_distance={min_distance})")
        out.extend(rows)
    return out


def measure_plateau_scaling(sides: list[int], min_distance: int) -> list[dict]:
    """The deterministic placer's plateau-packing loop against interface **area**, not volume.

    The reason this is measured as part of the EDT scaling work: the packing's accepted-point
    loop is Python, and it visits one lattice cell per occupied cell of the plateau set -- so it
    scales with the *area* of the interfaces, not the volume of the image. Junction protection's
    shell coarsening cut the point budget 23 % and its own stratum detection is 0.06 s against
    0.43 s for the EDT, so once a cheaper EDT stops being the dominant cost the packing is the
    next candidate. The
    exponent fitted here says whether that is true.
    """
    source_template = r"""
import json, sys, time
sys.path.insert(0, {repo!r})
import numpy as np
from dw3d_benchmarks.synthetic_foam import generate_foam_mask_low_memory, default_cell_count
from dw3d.edt import compute_edt_float32, region_codes, CODE_BOUNDARY
from dw3d.points_on_edt import plateau_packing_extrema

side = {side}
cells = default_cell_count(side)
mask = generate_foam_mask_low_memory(side, n_cells=cells)
code = region_codes(mask)
area_voxels = int((code == CODE_BOUNDARY).sum())
edt_image = compute_edt_float32(mask)
row = {{"side": side, "n_voxels": int(mask.size), "n_cells": cells, "boundary_voxels": area_voxels}}
for maximise, name in ((True, "maxima"), (False, "minima")):
    t0 = time.perf_counter()
    points = plateau_packing_extrema(edt_image, {min_distance}, maximise=maximise)
    row["plateau_" + name + "_s"] = time.perf_counter() - t0
    row["n_" + name] = int(len(points))
t0 = time.perf_counter()
compute_edt_float32(mask)
row["edt_s"] = time.perf_counter() - t0
print("__ROW__" + json.dumps(row))
"""
    rows = []
    for side in sides:
        source = source_template.format(repo=str(REPO_ROOT), side=side, min_distance=min_distance)
        completed = subprocess.run(  # noqa: S603 - fixed argv, source from this module
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        row = None
        for line in completed.stdout.splitlines():
            if line.startswith("__ROW__"):
                row = json.loads(line[len("__ROW__") :])
        if row is None:
            print(f"  side {side}: FAILED {completed.stderr[-300:]}")
            continue
        rows.append(row)
        packing = row["plateau_maxima_s"] + row["plateau_minima_s"]
        print(
            f"  side {row['side']:5d}  voxels {row['n_voxels']:>12,}  boundary {row['boundary_voxels']:>11,}  "
            f"edt {row['edt_s']:7.3f}s  packing {packing:7.3f}s  ratio {packing / row['edt_s']:6.2f}x",
        )

    if len(rows) >= 3:
        packing = np.array([r["plateau_maxima_s"] + r["plateau_minima_s"] for r in rows])
        area = np.array([r["boundary_voxels"] for r in rows], dtype=float)
        volume = np.array([r["n_voxels"] for r in rows], dtype=float)
        for name, predictor in (("boundary voxels (area)", area), ("total voxels (volume)", volume)):
            slope, intercept = np.polyfit(np.log(predictor), np.log(packing), 1)
            residual = np.log(packing) - (slope * np.log(predictor) + intercept)
            print(f"  packing time ~ {name}**{slope:.3f}   (log-residual s.d. {residual.std():.4f})")
    return rows


def measure_sparse_footprint(sides: list[int], cell_counts: list[int] | None = None) -> list[dict]:
    """`SparseEdt`'s storage as a fraction of the dense `float32` array, against side and cell size.

    The question this answers, and it is not the obvious one. `SparseEdt` stores the EDT's zero
    set, so its footprint is `O(interface area)` while the dense array is `O(volume)` -- which
    invites the conclusion that the fraction falls as the image grows. **It does not**, if the
    cells keep their size: area then grows *with* volume, and the fraction is set by the cells'
    inradius, not by `side`. That distinction decides whether the sparse container is a
    scaling win at the 2048^3 target or merely a constant-factor one, because that target
    is "~2048^3 voxels, a few thousand cells" -- which is an inradius of roughly 70 voxels,
    i.e. about the same as `data/Images/3.tif`'s.

    Two sweeps, therefore: `sides` at fixed inradius (`default_cell_count` scales the count
    with the volume), and `cell_counts` at fixed side, which varies the inradius directly.
    """
    from dw3d_benchmarks.synthetic_foam import default_cell_count, generate_foam_mask_low_memory
    from dw3d.edt import CODE_BOUNDARY, region_codes, sparse_edt

    rows = []

    def one(side: int, cells: int, sweep: str) -> dict:
        mask = generate_foam_mask_low_memory(side, n_cells=cells)
        container = sparse_edt(mask)
        dense_nbytes = mask.size * 4  # the dense float32 array this replaces
        boundary = int((region_codes(mask) == CODE_BOUNDARY).sum())
        row = {
            "sweep": sweep,
            "side": side,
            "n_cells": cells,
            "n_voxels": int(mask.size),
            "inradius_voxels": 0.5 * side / cells ** (1 / 3),
            "boundary_voxels": boundary,
            "n_zero_voxels": int(container.n_zero_voxels),
            "bytes_stored": int(container.bytes_stored),
            "dense_float32_bytes": int(dense_nbytes),
            "fraction_of_dense": container.bytes_stored / dense_nbytes,
        }
        print(
            f"  {sweep:<14} side {side:5d}  cells {cells:6d}  inradius {row['inradius_voxels']:6.1f}  "
            f"zero {row['n_zero_voxels']:>12,}  sparse/dense {row['fraction_of_dense']:6.3f}",
        )
        return row

    print("\n=== SparseEdt footprint vs side, at FIXED cell inradius ===")
    rows += [one(side, default_cell_count(side), "fixed-inradius") for side in sides]
    print("\n=== SparseEdt footprint vs cell count, at FIXED side ===")
    for cells in cell_counts or [8, 27, 64, 216]:
        rows.append(one(sides[-1], cells, "fixed-side"))
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--synthetic", type=int, nargs="*", default=None)
    parser.add_argument("--mask-npy", type=Path, default=None, help="measure a pre-generated .npy mask")
    parser.add_argument("--variants", type=str, nargs="*", default=None)
    parser.add_argument("--plateau-scaling", action="store_true")
    parser.add_argument("--sparse-footprint", action="store_true")
    parser.add_argument("--sparse-sides", type=int, nargs="*", default=[128, 192, 256, 320])
    parser.add_argument("--plateau-sides", type=int, nargs="*", default=[128, 192, 256, 384])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--halo", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    if args.images:
        rows += measure_images(args.repeats, args.min_distance, args.tile_size, args.halo)
    if args.synthetic is not None:
        rows += measure_synthetic(args.synthetic or [256, 512], args.repeats, args.min_distance, args.tile_size,
                                  args.halo)
    if args.mask_npy is not None:
        variants = tuple(args.variants) if args.variants else (*EDT_VARIANTS, "narrow_band_seeding")
        rows += measure_prebuilt(args.mask_npy, args.repeats, args.min_distance, args.tile_size, args.halo,
                                 variants)
    if args.plateau_scaling:
        print("\n=== deterministic plateau packing vs interface area ===")
        rows += measure_plateau_scaling(args.plateau_sides, args.min_distance)

    if args.sparse_footprint:
        rows += measure_sparse_footprint(args.sparse_sides)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
