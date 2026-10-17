"""The abnormal-edge patch tool must not write into the workspace root by default.

Its ``--out`` default used to be ``<workspace root>/abnormal_edge_diagnostics/default``, so a run
without ``--out`` added a new top-level entry to the workspace. The default is now the Git-ignored
``dw3d_benchmarks/outputs/`` inside this repository.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from dw3d_benchmarks import abnormal_edge_patches


def test_default_output_is_inside_the_repository() -> None:
    repo = abnormal_edge_patches.REPO_ROOT
    out = abnormal_edge_patches.DEFAULT_OUTPUT_DIR
    assert out.is_relative_to(repo)
    assert out == repo / "dw3d_benchmarks" / "outputs" / "abnormal_edge_diagnostics" / "default"


def test_default_output_is_ignored_by_git() -> None:
    repo = abnormal_edge_patches.REPO_ROOT
    probe = abnormal_edge_patches.DEFAULT_OUTPUT_DIR.relative_to(repo) / "d2_2_summary.json"
    git = shutil.which("git")
    assert git is not None
    checkout = subprocess.run(  # noqa: S603 — fixed arguments + trusted paths, a test subprocess
        [git, "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
    )
    if checkout.returncode != 0:
        pytest.skip("not a Git checkout (an unpacked source distribution): nothing is ignored or tracked")
    ignored = subprocess.run(  # noqa: S603 — fixed arguments + trusted paths, a test subprocess
        [git, "-C", str(repo), "check-ignore", "-q", str(probe)],
        check=False,
    )
    assert ignored.returncode == 0, f"{probe} is not ignored by Git"
