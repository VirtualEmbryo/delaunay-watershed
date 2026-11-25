"""Draw the two figures of the 2-D pages of the documentation.

* `two_dimensional_mesh_over_its_mask.png`: dw2d's default mesh of `mask_Cells_3.tif` over the mask.
* `route_of_0_1_5_against_the_0_8_0_defaults.png`: `mask_Cells_2.tif`, its labels expanded by 6 pixels, meshed by the route of dw2d 0.1.5 and with the
  0.8.0 defaults, side by side.

Run from the repository's root, with delaunay-watershed installed with its `viewing` extra:

    python docs/two_dimensional/figures/make_the_two_dimensional_figures.py

It prints, for the second figure, how many vertices the two routes share.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.collections import LineCollection

import dw2d

HERE = Path(__file__).resolve().parent
MASKS = HERE.parents[2] / "data" / "two_dimensional" / "Masks"


def junction_vertices(segments: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Vertices where three or more materials meet."""
    materials: dict[int, set[int]] = {}
    for (v0, v1), (a, b) in zip(segments, labels, strict=True):
        for v in (v0, v1):
            materials.setdefault(int(v), set()).update((int(a), int(b)))
    return np.array(sorted(v for v, m in materials.items() if len(m) >= 3), dtype=np.int64)


def draw(ax, mask, points, segments, labels, border, colour, title):
    """Draw a 2-D mesh over its mask, in the image's frame: column to the right, row down."""
    ax.imshow(mask > 0, cmap="Greys", vmin=0, vmax=4, interpolation="nearest")
    xy = points[:, ::-1]  # (row, column) -> (x, y) = (column, row) for imshow
    ax.add_collection(LineCollection(xy[segments], colors=colour, linewidths=0.8))
    vertices = np.unique(segments)
    ax.plot(xy[vertices, 0], xy[vertices, 1], ".", color=colour, markersize=1.5)
    junctions = junction_vertices(segments, labels)
    ax.plot(xy[junctions, 0], xy[junctions, 1], "o", color="tab:red", markersize=2.5, label="junction vertices")
    used = np.intersect1d(border, vertices)
    if len(used):
        ax.plot(xy[used, 0], xy[used, 1], "s", color="tab:blue", markersize=2.5, label="image-border vertices")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    ax.set_xlim(-0.5, mask.shape[1] - 0.5)
    ax.set_ylim(mask.shape[0] - 0.5, -0.5)


def mesh_over_its_mask() -> None:
    mask = tifffile.imread(MASKS / "mask_Cells_3.tif")
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=150)
    draw(ax, mask, points, segments, labels, algorithm.image_border_vertices, "black",
         f"mask_Cells_3, min_distance {algorithm.min_distance} px, 0.8.0 defaults")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "two_dimensional_mesh_over_its_mask.png")
    plt.close(fig)


def route_of_0_1_5_against_the_defaults() -> None:
    mask = tifffile.imread(MASKS / "mask_Cells_2.tif")
    legacy = (
        dw2d.MeshReconstructionAlgorithmFactory()
        .set_canonical_triangle_order(False)
        .set_junction_relocation(False)
        .set_pinch_vertex_repair(False)
        .set_near_tie_sliver_relabelling(False)
        .set_label_expansion(6)
        .make_algorithm()
    )
    legacy.pad_thin_medium_margins = False
    default = dw2d.MeshReconstructionAlgorithmFactory().set_label_expansion(6).make_algorithm()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=150)
    meshes = []
    for ax, algorithm, colour, title in (
        (axes[0], legacy, "dimgrey", "the route of dw2d 0.1.5"),
        (axes[1], default, "black", "the 0.8.0 defaults"),
    ):
        points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(mask)
        meshes.append((points, segments, labels))
        draw(ax, mask, points, segments, labels, algorithm.image_border_vertices, colour,
             f"mask_Cells_2 expanded by 6 px, min_distance {algorithm.min_distance} px:\n{title}")
    axes[1].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "route_of_0_1_5_against_the_0_8_0_defaults.png")
    plt.close(fig)
    (p0, s0, l0), (p1, s1, l1) = meshes
    same_vertices = len({tuple(p) for p in p0[np.unique(s0)]} & {tuple(p) for p in p1[np.unique(s1)]})
    print(default.relocation_report)
    print(
        f"0.1.5 route: {len(np.unique(s0))} vertices, {len(s0)} edges; "
        f"0.8.0 defaults: {len(np.unique(s1))} vertices, {len(s1)} edges; vertices at the same position: {same_vertices}; "
        f"junction vertices: {len(junction_vertices(s0, l0))} and {len(junction_vertices(s1, l1))}"
    )


if __name__ == "__main__":
    mesh_over_its_mask()
    route_of_0_1_5_against_the_defaults()
