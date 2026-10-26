"""Store the tetrahedralisation of every opt-in configuration's points on the golden images.

Writes `tests/golden/opt_in_configuration_tetrahedralisations.npz`: per point set of
`pinned_tetrahedralisations.OPT_IN_POINT_SETS`, golden image and `min_distance` (3, 5), the points
(uint16), the weights (float64, for a weighted configuration) and the tetrahedra the configuration's
own tesselation function gives, in canonical order (uint16). Checks first that the four
junction-protected variants place the same weighted points, so that they can share one entry.

Run from the repository root, once, on the machine whose Qhull made the golden fixtures (macOS arm64):

    python -m tests.make_opt_in_configuration_tetrahedralisations

The fixture tests then reconstruct from these tetrahedra, so a regenerated file must leave every
fixture test passing; regenerating it is a deliberate decision, as for the fixtures themselves.
"""

import numpy as np

from dw3d.tesselation import canonical_tetrahedron_order
from tests.conftest import IMAGE_NAMES
from tests.pinned_tetrahedralisations import (
    OPT_IN_POINT_SETS,
    OPT_IN_TETRAHEDRALISATIONS,
    TESSELATED_POINT_SETS,
    _key,
    live_tetrahedralisation,
    placed_points_and_weights,
)

MIN_DISTANCES = (3, 5)


def main() -> None:
    arrays = {}
    for image_name in IMAGE_NAMES:
        for min_distance in MIN_DISTANCES:
            for point_set in OPT_IN_POINT_SETS:
                points, weights = placed_points_and_weights(image_name, min_distance, point_set)
                for configuration, shared in TESSELATED_POINT_SETS.items():
                    if shared != point_set or configuration == point_set:
                        continue
                    other_points, other_weights = placed_points_and_weights(image_name, min_distance, configuration)
                    assert np.array_equal(points, other_points), f"{configuration} places other points than {point_set}"
                    assert (weights is None) == (other_weights is None)
                    assert weights is None or np.array_equal(weights, other_weights), f"{configuration}: other weights"
                tetrahedra = canonical_tetrahedron_order(live_tetrahedralisation(image_name, min_distance, point_set))
                assert np.array_equal(points, np.rint(points))
                assert points.min() >= 0
                assert points.max() < 2**16
                assert len(points) < 2**16
                key = f"{point_set}_{_key(image_name, min_distance)}"
                arrays[f"{key}_points"] = points.astype(np.uint16)
                arrays[f"{key}_tetrahedra"] = tetrahedra.astype(np.uint16)
                if weights is not None:
                    arrays[f"{key}_weights"] = weights
                print(f"{key}: {len(points)} points, {len(tetrahedra)} tetrahedra, weighted {weights is not None}")
    np.savez_compressed(OPT_IN_TETRAHEDRALISATIONS, **arrays)
    print(f"wrote {OPT_IN_TETRAHEDRALISATIONS} ({OPT_IN_TETRAHEDRALISATIONS.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
