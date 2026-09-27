"""Shared style, palette and I/O helpers for the mesh-quality-comparison figure set.

The palette values and conventions (fixed categorical hue order, text/grid colours, the
save-and-verify discipline) are copied from `foambryo_benchmarks/figures/_style.py` so the
two projects' figures read as one system. Copied rather than imported: `dw3d` must stay
independently runnable from its own repository, with no path reaching into a sibling
checkout of `foambryo`. `tests/test_mesh_quality_palette_matches_foambryo.py` asserts the
two colour lists stay identical (skipped when no sibling `foambryo` checkout is present, so
this independence is never accidentally required at runtime).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --- Paths -------------------------------------------------------------------------
# This file lives at <dw3d_repo>/dw3d_benchmarks/mesh_quality_comparison/figures/_style.py
FIGURES_SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_OUT = FIGURES_SCRIPT_DIR

# --- Palette (dataviz skill reference palette, fixed categorical order) ------------
# Bit-identical to foambryo_benchmarks/figures/_style.py; see this module's docstring.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

DIVERGING = (BLUE, "#f0efec", RED)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8a86"
GRID = "#e3e2dd"
SURFACE = "#fcfcfb"

SINGLE_COL_IN = 3.5  # inches, single-column journal width


def apply_style() -> None:
    """Apply the shared matplotlib rcParams (fonts, grid, spines, colours)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        },
    )


def strip_spines(ax, keep=("left", "bottom")) -> None:  # noqa: ANN001
    """Hide all spines except `keep`, and the matching tick marks."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(top=False, right=False)


def mesh_set_color(mesh_set: str, mesh_sets_in_order: list[str]) -> str:
    """One fixed colour per mesh set, assigned once by its position in `mesh_sets_in_order`.

    Every figure must call this with the *same* `mesh_sets_in_order` (the results JSON's own
    `mesh_sets` list) so a given configuration gets the same colour in every panel.
    """
    return CATEGORICAL[mesh_sets_in_order.index(mesh_set) % len(CATEGORICAL)]


def save_fig(fig, name: str) -> tuple[Path, Path]:  # noqa: ANN001
    """Save `fig` as both PDF (vector, for a paper) and PNG (for a talk), then verify both.

    Returns `(pdf_path, png_path)`. Raises if either file is empty, unreadable, or a flat
    (blank) image, so a script never claims success for a figure it did not actually
    produce.
    """
    pdf_path = FIGURES_OUT / f"{name}.pdf"
    png_path = FIGURES_OUT / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    for path in (pdf_path, png_path):
        if path.stat().st_size == 0:
            message = f"{path} is zero bytes"
            raise RuntimeError(message)

    with Path(pdf_path).open("rb") as fh:
        header = fh.read(5)
        if header != b"%PDF-":
            message = f"{pdf_path} does not start with a PDF header"
            raise RuntimeError(message)

    from PIL import Image

    with Image.open(png_path) as im:
        im.verify()
    with Image.open(png_path) as im:
        extrema = im.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            message = f"{png_path} is a flat/blank image"
            raise RuntimeError(message)

    print(
        f"[figures] wrote {pdf_path.name} ({pdf_path.stat().st_size} B) and "
        f"{png_path.name} ({png_path.stat().st_size} B), both verified re-openable",
    )
    return pdf_path, png_path
