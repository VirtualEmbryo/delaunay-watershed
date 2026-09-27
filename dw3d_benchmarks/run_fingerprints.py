#!/usr/bin/env python
"""Take bit-exactness fingerprints over the 4 in-repo images x min_distance in {3, 5}.

This is the harness determinism is judged with. `tests/golden/` only asserts `rtol=1e-10`
on summary statistics and would not see a pure labelling change; see
`benchmarks/fingerprint.py` for why that matters here.

Usage::

    python -m benchmarks.run_fingerprints --out /tmp/fp_after.json
    python -m benchmarks.run_fingerprints --variant dithered --out /tmp/fp_dithered.json
    python -m benchmarks.run_fingerprints --compare /tmp/fp_before.json /tmp/fp_after.json

`--variant dithered` selects the historical dithered point placement
(`MeshReconstructionAlgorithmFactory.get_dithered_algorithm`); `--variant deterministic`
selects an earlier default, no dither and no boundary layer (`get_deterministic_algorithm`);
`--variant boundary_layer` selects the boundary-layer scheme
(`get_boundary_layer_algorithm`); `--variant offset_included` / `--variant
junction_protected` select the boundary-layer + junction-protection scheme that was a
former default (`get_offset_included_algorithm`); `--variant offset_excluded_linkcheck`
selects the link-checked collapse, which **is** the default, so `fingerprints.json` and
`fingerprints_offset_excluded_linkcheck.json` must carry the same digests. On a tree
that predates the requested variant that factory method does not exist, and the script
says so and falls back to the default algorithm. That fallback is what lets the identical
script run in a worktree of a parent commit.

The old variant names (`v0_3`, `a1b`, `a3_a5`) are still accepted, mapped to the renamed
ones above with a printed warning, so a saved command keeps working.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import skimage.io as io

from dw3d_benchmarks.fingerprint import FINGERPRINT_FIELDS, environment_record, fingerprint_reconstruction
from dw3d import (
    MeshReconstructionAlgorithmFactory,
    get_default_mesh_reconstruction_algorithm,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_REPO_IMAGES = [REPO_ROOT / "data" / "Images" / f"{i}.tif" for i in range(1, 5)]
MIN_DISTANCES = (3, 5)


# Variant -> (factory method, keyword arguments). The three variants that used to name the
# synonyms deprecated in 0.5.0 are spelled as the keyword calls those synonyms stand for.
_VARIANT_GETTERS = {
    "dithered": ("get_dithered_algorithm", {}),
    "deterministic": ("get_deterministic_algorithm", {}),
    "cubic_score": ("get_junction_protected_algorithm", {"spline_order": 3}),
    "boundary_layer": ("get_boundary_layer_algorithm", {}),
    "offset_included": ("get_junction_protected_algorithm", {}),
    "junction_protected": ("get_junction_protected_algorithm", {}),
    "offset_excluded": ("get_junction_protected_algorithm", {"exclude_offsets_from_surface": True}),
    "offset_excluded_linkcheck": ("get_link_checked_algorithm", {}),
}

# Old variant names, still accepted (with a warning) so a saved command keeps working.
_LEGACY_VARIANT_ALIASES = {
    "v0_3": "dithered",
    "a1b": "deterministic",
    "a3_a5": "offset_included",
}


def _resolve_variant(variant: str) -> str:
    """Map a legacy variant name to its replacement, warning once if it is used."""
    canonical = _LEGACY_VARIANT_ALIASES.get(variant)
    if canonical is not None:
        print(f"  note: --variant {variant!r} is deprecated; use --variant {canonical!r} instead.")
        return canonical
    return variant


def _algorithm_factory(variant: str, min_distance: int):  # noqa: ANN202 - returns a closure
    """Zero-argument factory for the requested algorithm variant."""
    variant = _resolve_variant(variant)
    entry = _VARIANT_GETTERS.get(variant)
    if entry is not None:
        method, keywords = entry
        getter = getattr(MeshReconstructionAlgorithmFactory, method, None)
        if getter is None:
            print(f"  note: {method}() absent on this tree; using the default algorithm instead.")
            return lambda: get_default_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False)
        return lambda: getter(min_distance=min_distance, print_info=False, **keywords)
    return lambda: get_default_mesh_reconstruction_algorithm(min_distance=min_distance, print_info=False)


def run_all(variant: str, repeats: int = 1) -> dict:
    """Fingerprint every (image, min_distance) configuration `repeats` times."""
    variant = _resolve_variant(variant)
    record = {"variant": variant, "repeats": repeats, "environment": environment_record(), "cases": {}}
    for image_path in IN_REPO_IMAGES:
        if not image_path.exists():
            print(f"  SKIP {image_path} (not found)")
            continue
        mask = io.imread(image_path)
        for min_distance in MIN_DISTANCES:
            key = f"{image_path.stem}_md{min_distance}"
            factory = _algorithm_factory(variant, min_distance)
            runs = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fingerprint = fingerprint_reconstruction(mask, factory)
                fingerprint["wall_time_s"] = time.perf_counter() - t0
                runs.append(fingerprint)
            record["cases"][key] = runs[0] if repeats == 1 else {"runs": runs}
            joints = {run["joint"] for run in runs}
            flag = "" if len(joints) == 1 else "  <-- NOT REPRODUCIBLE ACROSS REPEATS"
            print(f"  {key}: {runs[0]['joint'][:16]} ({runs[0]['wall_time_s']:.2f}s){flag}")
    return record


def _case_runs(case: dict) -> list[dict]:
    return case.get("runs", [case])


def compare(path_a: Path, path_b: Path) -> int:
    """Print a field-by-field comparison of two fingerprint files. Return the mismatch count."""
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    print(f"A: {path_a}  variant={a['variant']}  {a['environment']}")
    print(f"B: {path_b}  variant={b['variant']}  {b['environment']}")

    keys = sorted(set(a["cases"]) | set(b["cases"]))
    n_mismatch = 0
    for key in keys:
        if key not in a["cases"] or key not in b["cases"]:
            print(f"  {key}: present in only one file")
            n_mismatch += 1
            continue
        run_a, run_b = _case_runs(a["cases"][key])[0], _case_runs(b["cases"][key])[0]
        differing = [name for name in FINGERPRINT_FIELDS if run_a[name] != run_b[name]]
        if not differing:
            print(f"  {key}: IDENTICAL")
        else:
            n_mismatch += 1
            sizes = (
                f"points {run_a['n_points']}->{run_b['n_points']}, "
                f"triangles {run_a['n_triangles']}->{run_b['n_triangles']}, "
                f"tets {run_a['n_tetrahedra']}->{run_b['n_tetrahedra']}"
            )
            print(f"  {key}: DIFFERS in {', '.join(differing)}  ({sizes})")
    print(f"{len(keys) - n_mismatch}/{len(keys)} configurations identical")
    return n_mismatch


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "default",
            "dithered",
            "deterministic",
            "cubic_score",
            "boundary_layer",
            "offset_included",
            "junction_protected",
            "offset_excluded",
            "offset_excluded_linkcheck",
            # Deprecated, still accepted (see _LEGACY_VARIANT_ALIASES):
            "v0_3",
            "a1b",
            "a3_a5",
        ),
        default="default",
    )
    parser.add_argument("--repeats", type=int, default=1, help="fingerprint each configuration N times")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON record here")
    parser.add_argument("--compare", type=Path, nargs=2, default=None, metavar=("A", "B"))
    args = parser.parse_args()

    if args.compare is not None:
        compare(*args.compare)
        return

    record = run_all(args.variant, repeats=args.repeats)
    if args.out is not None:
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
