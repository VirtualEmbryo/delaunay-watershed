#!/usr/bin/env python
r"""Compare dw3d reconstruction configurations ("mesh sets") against ground truth.

One script, added to the repository, re-runnable for every new dw3d version: point it at one
or more mesh sets (a named reconstruction configuration this dw3d exposes, or a directory of
already-built meshes) and it measures M1-M7 against `benchmarking-dataset`'s ground truth,
writing one flat results JSON and four figures. See `README.md` for what is measured and
`REPRODUCE.md` for the exact command the reference run used.

**Validate before you batch.** `--validate-only` runs the four anchors and exits; the batch
subcommand runs them itself before touching a single case, and stops on a hard-gate failure
(anchors 2, 3, 4) rather than producing a results file next to a bad measurement. Anchor 1 is
a diagnostic and never stops the run.

**Resumable, incremental, parallel.** Each `(mesh_set, case, min_distance)` triple is one
worker job; its result is written to its own file the moment it completes, and a re-invocation
skips whatever is already on disk unless `--force`. A crash costs the one job in flight, not
the run.

Usage::

    # 1. Profile one case, see the projected total.
    python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets profile \
        --dataset-dir benchmarking-dataset --mesh-sets dithered --cases 000

    # 2. Check the validation anchors (hard gates 2/3/4, diagnostic 1).
    python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets validate \
        --dataset-dir benchmarking-dataset

    # 3. Run the batch (resumable; re-run the same command to continue after a crash).
    python -m dw3d_benchmarks.mesh_quality_comparison.compare_mesh_sets batch \
        --dataset-dir benchmarking-dataset --mesh-sets dithered deterministic offset_included link_checked \
        --min-distances 3 --workers 5 --out results/mesh_quality_core.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import skimage.io as io
from foambryo.dcel import DcelData
from foambryo.validation.mesh_io import mean_edge_length

from dw3d.io import load_rec
from dw3d_benchmarks import metrics as dwm
from dw3d_benchmarks.angle_error_budget import _triple_line_vertices_and_lengths, synthetic_wedge_sensitivity

from . import bootstrap, case_metadata, case_metrics, mesh_sets, validation_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_MIN_DISTANCE = 3
CORE_CONFIGURATIONS = ("dithered", "deterministic", "offset_included", "link_checked")
EXTENSION_CONFIGURATIONS = ("dithered", "link_checked")
EXTENSION_MIN_DISTANCES = (2, 3, 5, 7)


def _git_commit(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(  # noqa: S603 - fixed argv, git assumed on PATH
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(  # noqa: S603 - fixed argv, git assumed on PATH
            ["git", "-C", str(repo_dir), *args],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_version_info(repo_dir: Path, remote_refs: tuple[str, ...] = ("origin/master", "origin/main")) -> dict:
    """This checkout's commit plus its relationship to whichever remote branch exists.

    Best-effort throughout: a missing repo, no network for `git fetch`, or a detached/foreign
    remote layout all degrade to partial information rather than raising, since this is
    reported metadata, not something the run depends on. `git fetch` is attempted once so
    "commits behind the pushed branch" reflects the remote's current state, not a stale local
    tracking ref.
    """
    head = _git(repo_dir, "rev-parse", "HEAD")
    if head is None:
        return {"available": False}
    info: dict = {
        "available": True,
        "commit": head,
        "short_commit": _git(repo_dir, "rev-parse", "--short", "HEAD"),
        "branch": _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD"),
        "date": _git(repo_dir, "log", "-1", "--format=%ad", "--date=short"),
        "subject": _git(repo_dir, "log", "-1", "--format=%s"),
    }
    _git(repo_dir, "fetch", "origin", "--quiet")
    for remote_ref in remote_refs:
        remote_commit = _git(repo_dir, "rev-parse", remote_ref)
        if remote_commit is None:
            continue
        merge_base = _git(repo_dir, "merge-base", "HEAD", remote_ref)
        ahead = _git(repo_dir, "rev-list", "--count", f"{remote_ref}..HEAD")
        behind = _git(repo_dir, "rev-list", "--count", f"HEAD..{remote_ref}")
        info["last_pushed"] = {
            "ref": remote_ref,
            "commit": remote_commit,
            "short_commit": _git(repo_dir, "rev-parse", "--short", remote_ref),
            "date": _git(repo_dir, "log", "-1", "--format=%ad", "--date=short", remote_ref),
            "subject": _git(repo_dir, "log", "-1", "--format=%s", remote_ref),
            "is_ancestor_of_head": merge_base == remote_commit,
            "commits_ahead_of_remote": int(ahead) if ahead is not None else None,
            "commits_behind_remote": int(behind) if behind is not None else None,
        }
        break
    return info


def _dataset_file_list_hash(dataset_dir: Path, case_ids: list[str]) -> str:
    """SHA-256 over the sorted list of `(filename, size)` for every file this run reads."""
    entries = []
    for case_id in case_ids:
        for suffix in ("_labels_filled.tif", "_mesh.rec", "_dict_tensions.npy"):
            path = dataset_dir / f"{case_id}{suffix}"
            if path.exists():
                entries.append(f"{path.name}:{path.stat().st_size}")
    digest = hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()
    return digest


def provenance(dataset_dir: Path, case_ids: list[str]) -> dict:
    """Provenance block: dw3d/foambryo commits, each vs. its last-pushed remote branch.

    `dw3d_version`/`foambryo_version` carry `commit`, `date`, `subject`, `branch`, and
    (best-effort) `last_pushed`: the remote branch's own commit and whether this checkout is
    ahead of it, behind it, or diverged -- so a report or figure can say plainly "this used
    commit X, which is/[n] commits ahead of/behind what's actually pushed" rather than a bare
    hash. Every configuration this run built (dithered/deterministic/offset_included/
    link_checked) shares this *same* single dw3d commit -- the four differ only in which
    factory-preset algorithm they call, not in code version -- so one version block covers all
    of them; see `README.md`'s "What this is not" for comparing across dw3d versions instead.
    """
    import foambryo

    import dw3d

    foambryo_root = Path(foambryo.__file__).resolve().parents[1]
    return {
        "dw3d_commit": _git_commit(REPO_ROOT),
        "dw3d_file": str(Path(dw3d.__file__).resolve()),
        "dw3d_version": _git_version_info(REPO_ROOT),
        "foambryo_commit": _git_commit(foambryo_root),
        "foambryo_file": str(Path(foambryo.__file__).resolve()),
        "foambryo_version": _git_version_info(foambryo_root),
        "dataset_file_list_hash": _dataset_file_list_hash(dataset_dir, case_ids),
        "bootstrap_seed": bootstrap.SEED,
        "bootstrap_n_draws": bootstrap.N_DRAWS,
    }


# ---------------------------------------------------------------------------
# One (mesh_set, case, min_distance) job
# ---------------------------------------------------------------------------


def compute_record(
    dataset_dir: str,
    configuration: str,
    case_id: str,
    min_distance: int,
    mesh_dir: str | None = None,
) -> dict:
    """Compute every M1-M7 block for this (mesh_set, case, min_distance).

    Two input modes, per the two kinds of mesh set:

    * `mesh_dir is None` -- build the mesh here, from the named configuration. The normal path.
    * `mesh_dir` given -- load `<mesh_dir>/<case>_mesh.rec` instead of reconstructing. This is
      what makes a **cross-version** comparison sound: meshes built by any dw3d version are
      measured by *this* version's metric code, so a difference is attributable to the mesh
      rather than to the measuring code. Wall-clock and peak RSS are then not this process's
      to report and are recorded as `None` with a note, never as a fabricated number.
    """
    dataset_path = Path(dataset_dir)
    mask = io.imread(dataset_path / f"{case_id}_labels_filled.tif")

    if mesh_dir is None:
        algo = mesh_sets.get_algorithm(configuration, min_distance)
        (points, triangles, labels), wall_time_s, peak_rss = case_metrics.timed_reconstruction(algo, mask)
        budget = case_metrics.compute_m7_budget(points, triangles, labels, wall_time_s, peak_rss)
    else:
        rec_path = Path(mesh_dir) / f"{case_id}_mesh.rec"
        if not rec_path.exists():
            message = f"{rec_path} not found; export it first (compare_versions.py --export-to)"
            raise FileNotFoundError(message)
        points, triangles, labels = load_rec(rec_path)
        budget = case_metrics.compute_m7_budget(points, triangles, labels, None, None)
        budget["_note"] = "mesh loaded from disk; wall-clock and peak RSS belong to the run that built it"

    record: dict = {
        "mesh_set": configuration,
        "case": case_id,
        "min_distance": min_distance,
        "group": case_metadata.group_of(case_id),
        "integrity_flagged": case_metadata.carries_integrity_flag(case_id),
        "mesh_source": str(mesh_dir) if mesh_dir else "built here",
        "m7_budget": budget,
    }

    gt = case_metrics.load_ground_truth(dataset_path, case_id)
    if gt is None:
        record["ground_truth"] = None
        return record

    record["m6_validity"] = case_metrics.compute_m6_validity(points, triangles, labels, gt)

    registration = dwm.similarity_to_mask_frame(mask, gt["points"], gt["triangles"], gt["labels"])
    scale = registration.get("scale")
    record["m2_interface_areas"] = case_metrics.compute_m2_interface_areas(points, triangles, labels, mask, gt)
    record["m3_cell_volumes"] = case_metrics.compute_m3_cell_volumes(points, triangles, labels, mask, gt)

    mesh = DcelData(points, triangles, labels)
    gt_mesh = DcelData(gt["points"], gt["triangles"], gt["labels"])

    _, ell = _triple_line_vertices_and_lengths(points, triangles, labels)
    median_ell = float(np.median(ell)) if len(ell) else None
    floor_deg = None
    if median_ell:
        wedge = synthetic_wedge_sensitivity(edge_length=median_ell)
        floor_deg = next(p["mean_deg"] for p in wedge["modes"]["all"]["curve"] if p["epsilon"] == 0.5)
    neumann = case_metrics.neumann_wedge_angles_deg(gt["tensions"]) if gt["tensions"] is not None else None
    record["m1_contact_angles"] = case_metrics.compute_m1_contact_angles(mesh, gt_mesh, neumann, floor_deg)
    # A count of how often production's triple-key pooling merges disconnected lines. Attached
    # to the M1 block because it qualifies M1's own population; it changes no M1 number.
    record["m1_contact_angles"]["_component_coverage"] = case_metrics.compute_m1_component_coverage(
        points, triangles, labels,
    )

    if scale:
        # `mean_edge_length` reads the ground-truth mesh in its own (non-voxel) frame; the
        # crossover must be expressed in the same 1/voxel units as the curvature comparison
        # (`compute_m4_curvature` divides ground-truth curvature by `scale` to get there), so
        # the edge length is converted the same way lengths convert: multiplied by `scale`.
        crossover = 1.0 / (2.0 * scale * mean_edge_length(gt["points"], gt["triangles"]))
        record["m4_curvature"] = case_metrics.compute_m4_curvature(mesh, gt_mesh, mask, gt, crossover)
    else:
        record["m4_curvature"] = {"error": "registration_failed"}
    record["m5_junction_geometry"] = case_metrics.compute_m5_junction_geometry(
        points, triangles, labels, gt, registration,
    )
    # The same geometry with connected-line identity, reported **alongside** the triple-keyed
    # production reproduction above and never instead of it. The triple-keyed path scores one
    # chain per material triple; this one scores every matched connected line and reports what
    # arclength fraction the other path therefore covers.
    record["m5_component_geometry"] = case_metrics.compute_m5_component_geometry(
        points, triangles, labels, gt, registration,
    )
    return record


def _job_path(out_dir: Path, configuration: str, case_id: str, min_distance: int) -> Path:
    return out_dir / f"{configuration}__{case_id}__md{min_distance}.json"


def _run_one_job(
    dataset_dir: str,
    out_dir: str,
    configuration: str,
    case_id: str,
    min_distance: int,
    mesh_dir: str | None = None,
) -> str:
    """Worker entry point: compute one record and write it, atomically, to its own file."""
    record = compute_record(dataset_dir, configuration, case_id, min_distance, mesh_dir)
    path = _job_path(Path(out_dir), configuration, case_id, min_distance)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    tmp.rename(path)
    return str(path)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile one case; print the projected total for the requested batch shape."""
    configuration = args.mesh_sets[0]
    case_id = args.cases[0]
    t0 = time.perf_counter()
    compute_record(args.dataset_dir, configuration, case_id, args.min_distances[0])
    per_job_s = time.perf_counter() - t0

    n_jobs = len(args.mesh_sets) * len(args.cases_full or [case_id]) * len(args.min_distances)
    projected_wall_s = per_job_s * n_jobs / max(args.workers, 1)
    print(f"[profile] one job ({configuration}, {case_id}, md={args.min_distances[0]}): {per_job_s:.2f} s")
    print(f"[profile] projected: {n_jobs} jobs / {args.workers} workers -> ~{projected_wall_s / 60:.1f} min wall-clock")


def cmd_validate(args: argparse.Namespace) -> None:
    """Run the four validation anchors and print/write the result."""
    case_ids = case_metadata.cohort_case_ids(args.dataset_dir)
    result = validation_gate.run_all_anchors(args.dataset_dir, case_ids, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    if not result["all_hard_gates_passed"]:
        message = "one or more hard-gate validation anchors failed; see the printed report."
        raise SystemExit(message)


def _resolve_mesh_dirs(args: argparse.Namespace) -> dict[str, str]:
    """Parse `--mesh-dir NAME=PATH` specs, adding each `NAME` to `args.mesh_sets` in place.

    A mesh set may be a *directory* of already-built meshes rather than a configuration to
    build here -- that is how meshes from another dw3d version get measured by this one.
    """
    mesh_dirs: dict[str, str] = {}
    for spec in args.mesh_dir or []:
        if "=" not in spec:
            message = f"--mesh-dir expects NAME=PATH, got {spec!r}"
            raise SystemExit(message)
        name, path = spec.split("=", 1)
        mesh_dirs[name] = str(Path(path).expanduser())
        if name not in args.mesh_sets:
            args.mesh_sets = [*args.mesh_sets, name]
    return mesh_dirs


def _run_batch_jobs(
    jobs: list[tuple[str, str, int]],
    args: argparse.Namespace,
    out_dir: Path,
    mesh_dirs: dict[str, str],
) -> tuple[int, int]:
    """Run `jobs` in a process pool, printing progress every 10 completions.

    Returns:
        tuple[int, int]: `(n_done, n_failed)`.
    """
    t0 = time.perf_counter()
    n_done, n_failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one_job,
                args.dataset_dir,
                str(out_dir),
                configuration,
                case_id,
                md,
                mesh_dirs.get(configuration),
            ): (configuration, case_id, md)
            for configuration, case_id, md in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                n_done += 1
            except Exception as exc:
                n_failed += 1
                print(f"[batch] FAILED {job}: {exc!r}")
            if (n_done + n_failed) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[batch] {n_done + n_failed}/{len(jobs)} done "
                    f"({n_failed} failed), {elapsed / 60:.1f} min elapsed",
                )
    print(f"[batch] finished: {n_done} ok, {n_failed} failed, {(time.perf_counter() - t0) / 60:.1f} min")
    return n_done, n_failed


def cmd_batch(args: argparse.Namespace) -> None:
    """Validate (unless skipped), then run the batch: resumable, incremental, parallel."""
    case_ids = case_metadata.cohort_case_ids(args.dataset_dir) if not args.cases else list(args.cases)
    out_dir = Path(args.out).parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dirs = _resolve_mesh_dirs(args)

    if not args.skip_validate:
        gate = validation_gate.run_all_anchors(args.dataset_dir, case_ids, workers=args.workers)
        gate_path = Path(args.out).parent / "validation_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True, default=float) + "\n")
        print(json.dumps(gate, indent=2, sort_keys=True, default=float))
        if not gate["all_hard_gates_passed"]:
            message = "hard-gate validation anchor failed; stopping before the batch. See validation_gate.json."
            raise SystemExit(message)

    jobs = [
        (configuration, case_id, md)
        for configuration in args.mesh_sets
        for case_id in case_ids
        for md in args.min_distances
    ]
    if not args.force:
        jobs = [j for j in jobs if not _job_path(out_dir, *j).exists()]
    print(f"[batch] {len(jobs)} jobs to run (of {len(args.mesh_sets) * len(case_ids) * len(args.min_distances)} total)")

    _n_done, n_failed = _run_batch_jobs(jobs, args, out_dir, mesh_dirs)

    all_records = [json.loads(path.read_text()) for path in sorted(out_dir.glob("*.json"))]
    # The manifest describes what is actually in `all_records` -- which, on a resumed or
    # incrementally-extended run, is everything ever written to `out_dir`, not just this
    # invocation's own `--mesh-sets`/`--min-distances` -- so a later, narrower run (e.g. the
    # extension sweep) never silently narrows what earlier runs (e.g. the core batch) put there.
    distinct_mesh_sets = {r["mesh_set"] for r in all_records}
    ordered_mesh_sets = [c for c in CORE_CONFIGURATIONS if c in distinct_mesh_sets] + sorted(
        distinct_mesh_sets - set(CORE_CONFIGURATIONS),
    )
    manifest = {
        "provenance": provenance(Path(args.dataset_dir), case_ids),
        "mesh_sets": ordered_mesh_sets,
        "min_distances": sorted({r["min_distance"] for r in all_records}),
        "cases": sorted({r["case"] for r in all_records}),
        "n_records": len(all_records),
        "n_failed": n_failed,
        "this_invocation": {
            "mesh_sets": list(args.mesh_sets),
            "min_distances": list(args.min_distances),
            "cases": case_ids,
        },
    }
    payload = {"manifest": manifest, "records": all_records}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(f"[batch] wrote {args.out} ({len(all_records)} records)")


def build_parser() -> argparse.ArgumentParser:
    """Build the `profile`/`validate`/`batch` subcommand parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset-dir", required=True)
        p.add_argument("--mesh-sets", nargs="+", default=list(CORE_CONFIGURATIONS))
        p.add_argument("--cases", nargs="*", default=None)
        p.add_argument("--min-distances", nargs="+", type=int, default=[CORE_MIN_DISTANCE])
        p.add_argument("--workers", type=int, default=5)

    p_profile = sub.add_parser("profile")
    _common(p_profile)
    p_profile.add_argument("--cases-full", nargs="*", default=None, help="full cohort size, for the projection only")
    p_profile.set_defaults(func=cmd_profile)

    p_validate = sub.add_parser("validate")
    _common(p_validate)
    p_validate.add_argument("--out", default=None)
    p_validate.set_defaults(func=cmd_validate)

    p_batch = sub.add_parser("batch")
    _common(p_batch)
    p_batch.add_argument("--mesh-dir", action="append", default=None, metavar="NAME=PATH",
                         help="treat mesh set NAME as a directory of prebuilt <case>_mesh.rec files "
                              "instead of a configuration to build; repeatable")
    p_batch.add_argument("--out", required=True)
    p_batch.add_argument("--force", action="store_true")
    p_batch.add_argument("--skip-validate", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    return parser


def main() -> None:
    """CLI entry point: parse args and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    if args.cases is not None and len(args.cases) == 0:
        args.cases = None
    args.func(args)


if __name__ == "__main__":
    main()
