#!/usr/bin/env python
r"""Compare dw3d **versions** by mesh fingerprint: are two checkouts behaviourally identical?

`compare_mesh_sets.py` compares *configurations* inside one checkout. This compares *versions*
of the code, holding the configuration fixed -- the other axis, and the one a release needs
before it can claim "no behavioural change".

**Why a fingerprint first, and not the full metric suite.** If two versions produce
bitwise-identical meshes, every M1-M7 metric is identical by construction, and running the
full suite on both would burn hours to print the same table twice. So this reports identity
first, on cheap SHA-256 mesh fingerprints, and tells you to run `compare_mesh_sets.py` per arm
only for the (configuration, case) pairs that actually differ.

**Each arm runs in its own interpreter.** An arm is a dw3d checkout with its own `.venv`; this
script shells into `<arm>/.venv/bin/python`, so two versions with incompatible dependencies
never have to coexist in one process. Nothing is imported from an arm into this process.

**Configurations are addressed by a list of factory names, tried in order**: the name each
configuration has from 0.5.0 on, then the legacy name it had from v0.4.0 to v0.4.2 (`get_v0_3_...`,
`get_a1b_...`, `get_a3_a5_...`, removed in 0.5.0). Addressing only one of the two would silently
fail on the arms that lack it, and using `get_default_...` would conflate a code change with a *policy* change -- which
default a version ships is a decision, not behaviour. `get_default_...` is therefore fingerprinted
separately and reported as "resolves to <configuration>", so a default that moved between
versions shows up as exactly that rather than as a phantom behavioural difference.

Usage::

    PYTHONPATH="src:." .venv/bin/python -m dw3d_benchmarks.mesh_quality_comparison.compare_versions \
        --arm v0.4.0=/path/to/dw3d-ver-v0-4-0 \
        --arm v0.4.1=/path/to/dw3d-ver-v0-4-1 \
        --arm master=/path/to/dw3d-mesh-quality-comparison \
        --dataset-dir /path/to/benchmarking-dataset \
        --out results/version_comparison.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import case_metadata

#: Configuration name -> factory names to try in order: the 0.5.0 name, then the v0.4.x legacy
#: name. From 0.5.0 on, every configuration relocates junction vertices by default, so a 0.4 arm
#: and a 0.5 arm differ on every relocated case -- which is a real behavioural difference.
STABLE_GETTERS: dict[str, list[str]] = {
    "dithered": ["get_dithered_mesh_reconstruction_algorithm", "get_v0_3_mesh_reconstruction_algorithm"],
    "deterministic": ["get_deterministic_mesh_reconstruction_algorithm", "get_a1b_mesh_reconstruction_algorithm"],
    "offset_included": [
        "get_junction_protected_mesh_reconstruction_algorithm",
        "get_a3_a5_mesh_reconstruction_algorithm",
    ],
    "link_checked": ["get_link_checked_mesh_reconstruction_algorithm"],
}

#: Arms older than v0.4.0 predate the named variants and need an explicit mapping, supplied
#: with `--getter ARM:CONFIGURATION=GETTER_NAME`. This is deliberately **not** a silent
#: fallback to `get_default_...`: mapping a configuration onto an arm's *default* asserts that
#: the two are the same algorithm in that version, which is a claim about that release and must
#: be written down by the caller, not guessed by this script. For dw3d 0.3.6 the claim is
#: sound -- that release shipped exactly one algorithm, the dithered one, as its default --
#: and the mapping to use is:
#:     --getter 0.3.6:dithered=get_default_mesh_reconstruction_algorithm
#: with `--only-configurations dithered`, since 0.3.6 has no counterpart for the other three.

_WORKER = r"""
import hashlib, json, sys
import numpy as np
import skimage.io as io
import dw3d

dataset, min_distance = sys.argv[1], int(sys.argv[2])
getters = json.loads(sys.argv[3])
cases = sys.argv[4:]

def mesh_hash(points, triangles, labels):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(points, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(triangles, dtype=np.uint64).tobytes())
    h.update(np.ascontiguousarray(labels, dtype=np.uint64).tobytes())
    return h.hexdigest()

out = {"dw3d_file": dw3d.__file__, "fingerprints": {}, "defaults": {}}
for case in cases:
    mask = io.imread(f"{dataset}/{case}_labels_filled.tif")
    for name, getter_names in getters.items():
        if isinstance(getter_names, str):
            getter_names = [getter_names]
        getter = next((getattr(dw3d, n) for n in getter_names if hasattr(dw3d, n)), None)
        if getter is None:
            out["fingerprints"][f"{name}|{case}"] = None
            continue
        p, t, l = getter(min_distance=min_distance, print_info=False).construct_mesh_from_segmentation_mask(mask)
        out["fingerprints"][f"{name}|{case}"] = mesh_hash(p, t, l)
    default = getattr(dw3d, "get_default_mesh_reconstruction_algorithm")
    p, t, l = default(min_distance=min_distance, print_info=False).construct_mesh_from_segmentation_mask(mask)
    out["defaults"][case] = mesh_hash(p, t, l)
print("@@JSON@@" + json.dumps(out))
"""


_EXPORT_WORKER = r"""
import json, sys
import skimage.io as io
import dw3d

dataset, min_distance, getter_names, out_dir = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4]
cases = sys.argv[5:]

getter_names = [getter_names] if isinstance(getter_names, str) else getter_names
getter = next((getattr(dw3d, n) for n in getter_names if hasattr(dw3d, n)), None)
getter_name = " or ".join(getter_names)
if getter is None:
    print("@@JSON@@" + json.dumps({"error": f"{getter_name} absent in this version"}))
    raise SystemExit(1)

written = {}
for case in cases:
    mask = io.imread(f"{dataset}/{case}_labels_filled.tif")
    p, t, l = getter(min_distance=min_distance, print_info=False).construct_mesh_from_segmentation_mask(mask)
    path = f"{out_dir}/{case}_mesh.rec"
    # binary_mode=True: raw float64, so the exported mesh is bit-exact. Text mode would round
    # the coordinates and silently make a version comparison measure the serialiser.
    dw3d.save_rec(path, p, t, l, binary_mode=True)
    written[case] = {"path": path, "n_points": int(len(p)), "n_triangles": int(len(t))}
print("@@JSON@@" + json.dumps({"dw3d_file": dw3d.__file__, "written": written}))
"""


def export_arm_meshes(
    arm_dir: Path,
    dataset_dir: Path,
    cases: list[str],
    out_dir: Path,
    getter_name: str | list[str],
    min_distance: int,
) -> dict:
    """Reconstruct `cases` inside `arm_dir`'s interpreter and write each mesh as a binary `.rec`.

    This is what makes a cross-version *metric* comparison sound: meshes are produced by each
    version, then measured by **one** version's metric code (`compare_mesh_sets.py --mesh-dir`).
    Running each arm's own metric code instead would confound "the mesh changed" with "the
    measuring code changed" -- and for a pre-0.4.0 arm it is not even possible, since
    `dw3d_benchmarks.metrics` imports `points_on_edt.boundary_layer_families`, which 0.3.6
    does not have.
    """
    python = arm_dir / ".venv" / "bin" / "python"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The worker runs with `cwd=arm_dir`, so every path handed to it must be absolute or it
    # resolves against the *arm's* directory instead of the caller's.
    out_dir = out_dir.resolve()
    dataset_dir = Path(dataset_dir).resolve()
    result = subprocess.run(  # noqa: S603 - fixed argv, python interpreter is an absolute path
        [str(python), "-c", _EXPORT_WORKER, str(dataset_dir), str(min_distance), json.dumps(getter_name), str(out_dir),
         *cases],
        cwd=arm_dir,
        env={"PYTHONPATH": str(arm_dir / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    payload = next((ln for ln in result.stdout.splitlines() if ln.startswith("@@JSON@@")), None)
    if result.returncode != 0 or payload is None:
        message = f"export from {arm_dir} failed:\n{result.stderr[-3000:]}\n{result.stdout[-1500:]}"
        raise SystemExit(message)
    return json.loads(payload.removeprefix("@@JSON@@"))


def arm_version_block(arm_dir: Path) -> dict:
    """An arm's version: git metadata for a checkout, installed distribution version for a wheel.

    Shared by the compare and export paths so a wheel arm (e.g. 0.3.6 from PyPI, which has no
    git checkout) is labelled the same way in both outputs instead of coming out as all-`None`
    in one of them. The `importlib.metadata` query runs with `cwd=arm_dir` and a clean env --
    see the note in `fingerprint_arm` for the mislabelling that happens otherwise.
    """
    block = _git_describe(arm_dir)
    if block.get("commit") is not None:
        return block
    python = arm_dir / ".venv" / "bin" / "python"
    installed = subprocess.run(  # noqa: S603 - fixed argv, python interpreter is an absolute path
        [str(python), "-c", "from importlib.metadata import version; print(version('delaunay-watershed-3d'))"],
        cwd=arm_dir,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "installed_version": installed.stdout.strip() or None,
        "describe": installed.stdout.strip() or None,
        "source": "wheel (no git checkout)",
    }


def _git_describe(arm_dir: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(  # noqa: S603 - fixed argv, git assumed on PATH
                ["git", "-C", str(arm_dir), *args],  # noqa: S607
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "short_commit": run("rev-parse", "--short", "HEAD"),
        "describe": run("describe", "--tags", "--always"),
        "date": run("log", "-1", "--format=%ad", "--date=short"),
        "subject": run("log", "-1", "--format=%s"),
    }


def fingerprint_arm(
    arm_dir: Path,
    dataset_dir: Path,
    cases: list[str],
    min_distance: int,
    getters: dict[str, str] | None = None,
) -> dict:
    """Run one arm's interpreter and collect its mesh fingerprints.

    `getters` maps configuration -> factory name for *this* arm; defaults to
    `STABLE_GETTERS`. An arm installed from a wheel has no `src/` directory and no git
    metadata; both are handled (the import comes from site-packages, and the version block
    falls back to the installed distribution version).
    """
    getters = getters or STABLE_GETTERS
    python = arm_dir / ".venv" / "bin" / "python"
    if not python.exists():
        message = (
            f"{python} not found. Each arm needs its own venv:\n"
            f"  cd {arm_dir} && uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e ."
        )
        raise SystemExit(message)
    result = subprocess.run(  # noqa: S603 - fixed argv, python interpreter is an absolute path
        [
            str(python),
            "-c",
            _WORKER,
            str(dataset_dir),
            str(min_distance),
            json.dumps(getters),
            *cases,
        ],
        cwd=arm_dir,
        env={"PYTHONPATH": str(arm_dir / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = f"arm {arm_dir} failed:\n{result.stderr[-3000:]}"
        raise SystemExit(message)
    payload = next((ln for ln in result.stdout.splitlines() if ln.startswith("@@JSON@@")), None)
    if payload is None:
        message = f"arm {arm_dir} produced no result payload:\n{result.stdout[-2000:]}"
        raise SystemExit(message)
    data = json.loads(payload.removeprefix("@@JSON@@"))
    data["version"] = _git_describe(arm_dir)
    data["arm_dir"] = str(arm_dir)
    data["getters_used"] = getters
    if data["version"].get("commit") is None:
        # A wheel arm: no git checkout. Report the installed distribution version instead, so
        # the arm is still identified by something verifiable rather than by its name alone.
        #
        # This MUST run with the same cwd/env isolation as the worker. Run from another
        # checkout's directory, `python -c` puts that cwd on `sys.path` and
        # `importlib.metadata` then resolves the *wrong* project's dist-info -- measured: the
        # 0.3.6 arm reported "0.4.1" that way. The fingerprints were never affected (the
        # worker is already isolated), but the label would have been wrong on every figure.
        installed = subprocess.run(  # noqa: S603 - fixed argv, python interpreter is an absolute path
            [str(python), "-c", "from importlib.metadata import version; print(version('delaunay-watershed-3d'))"],
            cwd=arm_dir,
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=False,
        )
        data["version"] = {
            "installed_version": installed.stdout.strip() or None,
            "source": "wheel (no git checkout)",
        }
    # Surface where dw3d was actually imported from, so a mislabelled arm is self-evident
    # in the report rather than needing to be taken on trust.
    data["version"]["dw3d_file"] = data.get("dw3d_file")
    return data


def compare(arms: dict[str, dict], cases: list[str], configurations: list[str] | None = None) -> dict:
    """Reduce per-arm fingerprints to a per-(configuration, case) identical/differs verdict."""
    arm_names = list(arms)
    configurations = configurations or list(STABLE_GETTERS)
    differing: list[dict] = []
    n_compared = 0
    for configuration in configurations:
        for case in cases:
            key = f"{configuration}|{case}"
            hashes = {name: arms[name]["fingerprints"].get(key) for name in arm_names}
            if any(h is None for h in hashes.values()):
                differing.append(
                    {"configuration": configuration, "case": case, "reason": "absent_in_some_arm", "hashes": hashes},
                )
                continue
            n_compared += 1
            if len(set(hashes.values())) > 1:
                differing.append(
                    {"configuration": configuration, "case": case, "reason": "hash_mismatch", "hashes": hashes},
                )

    default_resolution = {}
    for name in arm_names:
        per_arm = {}
        for case in cases:
            default_hash = arms[name]["defaults"][case]
            matches = [c for c in configurations if arms[name]["fingerprints"].get(f"{c}|{case}") == default_hash]
            per_arm[case] = matches or ["<none of the compared configurations>"]
        # collapse when every case agrees, which is the normal case
        distinct = {tuple(v) for v in per_arm.values()}
        default_resolution[name] = list(distinct.pop()) if len(distinct) == 1 else per_arm
    default_moved = len({json.dumps(v, sort_keys=True) for v in default_resolution.values()}) > 1

    return {
        "arms": {
            name: arms[name]["version"]
            | {"arm_dir": arms[name]["arm_dir"], "getters_used": arms[name].get("getters_used")}
            for name in arm_names
        },
        "configurations_compared": configurations,
        "n_configurations": len(configurations),
        "n_cases": len(cases),
        "n_pairs_compared": n_compared,
        "identical_across_all_arms": len(differing) == 0,
        "n_differing_pairs": len(differing),
        "differing_pairs": differing,
        "default_resolves_to": default_resolution,
        "default_policy_moved_between_arms": default_moved,
    }


def _run_export_to(
    export_to: Path,
    arm_specs: list[str],
    configurations: list[str],
    min_distance: int,
    cases: list[str],
    dataset_dir: Path,
    overrides: dict[str, dict[str, str]],
) -> None:
    """Handle `--export-to`: write each arm's meshes to disk instead of comparing, then exit.

    Split out of `main` (same behaviour, unchanged) to keep that function's branching within
    the project's complexity budget.
    """
    if len(configurations) != 1:
        message = "--export-to needs exactly one --only-configurations value"
        raise SystemExit(message)
    configuration = configurations[0]
    manifest: dict = {"configuration": configuration, "min_distance": min_distance, "arms": {}}
    for spec in arm_specs:
        name, path = spec.split("=", 1)
        arm_dir = Path(path).expanduser()
        getter_name = overrides.get(name, {}).get(configuration, STABLE_GETTERS[configuration])
        out_dir = export_to / name
        print(f"[versions] exporting {name} ({configuration} via {getter_name}) -> {out_dir}")
        exported = export_arm_meshes(arm_dir, dataset_dir, cases, out_dir, getter_name, min_distance)
        manifest["arms"][name] = {
            "arm_dir": str(arm_dir),
            "getter": getter_name,
            "mesh_dir": str(out_dir),
            "dw3d_file": exported.get("dw3d_file"),
            "version": arm_version_block(arm_dir),
            "n_cases_written": len(exported.get("written", {})),
        }
    manifest_path = export_to / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[versions] wrote {manifest_path}")
    sys.exit(0)


def main() -> None:
    """CLI entry point: fingerprint each arm and compare, or export meshes with `--export-to`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                        help="a dw3d checkout with its own .venv; repeatable")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--cases", nargs="*", default=None, help="default: the 45-case cohort")
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--only-configurations", nargs="*", default=None,
                        help=f"subset of {sorted(STABLE_GETTERS)}; default: all four")
    parser.add_argument("--getter", action="append", default=[], metavar="ARM:CONFIG=GETTER",
                        help="override the factory name for one arm's configuration (pre-0.4.0 arms); repeatable")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--export-to", type=Path, default=None,
                        help="instead of comparing, write each arm's meshes as binary .rec under "
                             "<dir>/<arm>/ so one metric suite can measure them all "
                             "(compare_mesh_sets.py --mesh-dir). Requires a single configuration.")
    args = parser.parse_args()

    cases = args.cases or case_metadata.cohort_case_ids(args.dataset_dir)
    configurations = args.only_configurations or list(STABLE_GETTERS)
    unknown = set(configurations) - set(STABLE_GETTERS)
    if unknown:
        message = f"unknown configuration(s) {sorted(unknown)}; known: {sorted(STABLE_GETTERS)}"
        raise SystemExit(message)

    overrides: dict[str, dict[str, str]] = {}
    for spec in args.getter:
        if ":" not in spec or "=" not in spec:
            message = f"--getter expects ARM:CONFIG=GETTER, got {spec!r}"
            raise SystemExit(message)
        arm_name, rest = spec.split(":", 1)
        config_name, getter_name = rest.split("=", 1)
        overrides.setdefault(arm_name, {})[config_name] = getter_name

    if args.export_to is not None:
        _run_export_to(args.export_to, args.arm, configurations, args.min_distance, cases, args.dataset_dir, overrides)

    arms: dict[str, dict] = {}
    for spec in args.arm:
        if "=" not in spec:
            message = f"--arm expects NAME=PATH, got {spec!r}"
            raise SystemExit(message)
        name, path = spec.split("=", 1)
        getters = {c: STABLE_GETTERS[c] for c in configurations} | overrides.get(name, {})
        print(f"[versions] fingerprinting arm {name} ({len(cases)} cases x {len(getters)} configurations)...")
        arms[name] = fingerprint_arm(Path(path).expanduser(), args.dataset_dir, cases, args.min_distance, getters)

    report = compare(arms, cases, configurations)
    print(json.dumps({k: v for k, v in report.items() if k != "differing_pairs"}, indent=2, sort_keys=True))
    if report["identical_across_all_arms"]:
        print(
            f"\n[versions] IDENTICAL: all {report['n_pairs_compared']} (configuration, case) pairs hash the same "
            f"in every arm. No behavioural change between these versions; the M1-M7 metrics cannot differ.",
        )
    else:
        print(f"\n[versions] {report['n_differing_pairs']} pair(s) DIFFER -- run compare_mesh_sets.py in each arm "
              "and diff the resulting scorecards for those configurations.")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[versions] wrote {args.out}")
    sys.exit(0 if report["identical_across_all_arms"] else 2)


if __name__ == "__main__":
    main()
