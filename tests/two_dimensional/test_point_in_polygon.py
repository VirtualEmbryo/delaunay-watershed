"""dw2d reconstructs without matplotlib, and its own point-in-polygon test agrees with matplotlib's.

Since 0.8.0 matplotlib is in the `viewing` extra. Until then the half-edge structure oriented the boundary loops
of a material with several loops (a medium with enclosed cells, a cell with a hole) with
``matplotlib.path.Path.contains_point``; ``dw2d.Dcel.point_strictly_inside_polygon`` replaced it, an even-odd
crossing test whose side decisions are exact. Pinned here:

* on every pair (probe, polygon) that the orientation of the loops evaluates, over every golden case and every
  shipped mask (labels as they are and grown by 6 px, junctions relocated and not), it gives matplotlib's answer;
  the number of pairs is asserted non-zero, so the comparison cannot pass empty;
* on random non-convex polygons and random points off their boundary, the same;
* in a Python where matplotlib cannot be imported, ``import dw2d`` and a reconstruction with several loops per
  material work, and a plotting call raises an ``ImportError`` that names the `viewing` extra.
"""
import contextlib
import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from .conftest import MASKS, golden_cases

ROUTES = {"relocated": {"relocate_junctions": True}, "plain": {"relocate_junctions": False}}


def _cases():
    import skimage.io as skio

    cases = [(name, lab, md, exp) for name, lab, md, exp in golden_cases()]
    for path in sorted(MASKS.glob("mask_Cells_*.tif")):
        lab = skio.imread(path)
        lab = lab[..., 0] if lab.ndim == 3 else lab
        cases.extend((f"{path.stem}__expansion{expansion}__md3", lab, 3, expansion) for expansion in (0, 6))
    return cases


CASES = _cases()


@pytest.fixture(scope="module")
def pairs_evaluated_by_the_loop_orientation():
    """Every (probe, polygon) pair ``_orient_several_loops`` asks about, over every case and route."""
    from dw2d import Dcel, geometry_reconstruction_2d

    seen = []
    original = Dcel.point_strictly_inside_polygon

    def recording(point, polygon):
        answer = original(point, polygon)
        seen.append((np.array(point, dtype=np.float64), np.array(polygon, dtype=np.float64), answer))
        return answer

    Dcel.point_strictly_inside_polygon = recording
    try:
        for _name, labels, min_dist, expansion in CASES:
            for options in ROUTES.values():
                with contextlib.redirect_stdout(io.StringIO()):
                    geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion,
                                               **options).return_dcel()
    finally:
        Dcel.point_strictly_inside_polygon = original
    return seen


def test_agrees_with_matplotlib_on_every_loop_of_the_golden_cases_and_shipped_masks(
        pairs_evaluated_by_the_loop_orientation):
    path = pytest.importorskip("matplotlib.path")
    pairs = pairs_evaluated_by_the_loop_orientation
    assert len(pairs) > 0, "no material with several loops: the comparison would be empty"
    assert any(answer for _, _, answer in pairs), "no probe inside another loop: the comparison would be one-sided"
    disagreements = [
        i for i, (p, poly, answer) in enumerate(pairs) if bool(path.Path(poly).contains_point(p)) != answer
    ]
    assert disagreements == []


def _random_polygon(rng, n):
    """A star-shaped, generally non-convex polygon of ``n`` vertices around (0, 0)."""
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    radii = rng.uniform(0.3, 1.0, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def test_agrees_with_matplotlib_on_random_polygons():
    path = pytest.importorskip("matplotlib.path")
    from dw2d.Dcel import point_strictly_inside_polygon

    rng = np.random.default_rng(20261109)
    n_inside = n_outside = 0
    for _ in range(300):
        polygon = _random_polygon(rng, int(rng.integers(3, 40)))
        for point in rng.uniform(-1.1, 1.1, (20, 2)):
            ours = point_strictly_inside_polygon(point, polygon)
            assert ours == bool(path.Path(polygon).contains_point(point))
            n_inside += ours
            n_outside += not ours
    assert n_inside > 100
    assert n_outside > 100


def test_exact_cases():
    """A ray through a vertex counts it once; a closing vertex repeated changes nothing."""
    from dw2d.Dcel import point_strictly_inside_polygon

    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    diamond = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    assert point_strictly_inside_polygon((1.0, 1.0), square)
    assert not point_strictly_inside_polygon((3.0, 1.0), square)
    assert point_strictly_inside_polygon((0.5, 1.0), diamond)  # the ray passes through the vertex (2, 1)
    assert not point_strictly_inside_polygon((-0.5, 1.0), np.vstack([diamond, diamond[:1]]))
    assert point_strictly_inside_polygon((1.0, 1.0), np.vstack([square, square[:1]]))


def test_reconstruction_and_import_work_without_matplotlib():
    """In a subprocess where ``import matplotlib`` fails: dw2d imports, reconstructs, and names the extra to plot."""
    mask = MASKS / "mask_Cells_8.tif"  # 68 cells enclosed by the medium: the medium has several loops
    src = Path(__file__).resolve().parents[2] / "src"
    script = textwrap.dedent(f"""
        import sys, contextlib, io
        sys.modules["matplotlib"] = None  # every import of matplotlib or of a submodule now raises ImportError
        sys.path.insert(0, {str(src)!r})
        import skimage.io
        import dw2d
        import dw2d.Dcel
        lab = skimage.io.imread({str(mask)!r})
        lab = lab[..., 0] if lab.ndim == 3 else lab
        with contextlib.redirect_stdout(io.StringIO()):
            dw = dw2d.geometry_reconstruction_2d(lab, min_dist=3, expansion_labels=0)
            mesh = dw.return_dcel()
        assert len(mesh.compute_lengths()) > 0
        assert "matplotlib" not in {{k for k, v in sys.modules.items() if v is not None}}
        try:
            dw.plot_polylines_and_seeds()
        except ImportError as error:
            assert "delaunay-watershed-2d[viewing]" in str(error), str(error)
            print("plotting refused with the extra named")
        else:
            raise AssertionError("plotting without matplotlib did not raise")
    """)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr[-3000:]
    assert "plotting refused with the extra named" in result.stdout
