"""Regression: the AABB-shaped abnormal edge stays resolved at the shipped rung.

Measurement found three `md=3` survivors sharing one mechanism -- the AABB shape,
`{(A,B): 2, (A,C): 2}` -- and specified a widened `_find_candidate_for_label_switching`.
This fix implements it (`src/dw3d/mesh_surgery.py`). This pins the headline claim end to
end, through a real reconstruction rather than a label-cycle fixture:
`benchmarking-dataset` cases 011, 013 and 036, at `min_distance=3`, each have zero
abnormal non-manifold edges.

Skips (does not fail) if `benchmarking-dataset` cannot be found beside the repo -- same
contract as the rest of the suite (see `tests/conftest.py`).
"""

import pytest
import skimage.io as io

from dw3d import get_dithered_mesh_reconstruction_algorithm
from dw3d.mesh_surgery import _find_abnormal_non_manifold_edges
from tests.conftest import dataset_masks

# The three md=3 survivors this fix targets, reproduced against the real reconstruction.
NAMED_CASES = ("011", "013", "036")

_ALL_MASKS = {path.name.split("_")[0]: path for path in dataset_masks()}
_CASE_PATHS = [_ALL_MASKS[case] for case in NAMED_CASES if case in _ALL_MASKS]
_CASE_IDS = [path.name.split("_")[0] for path in _CASE_PATHS]


@pytest.fixture(params=_CASE_PATHS, ids=_CASE_IDS)
def named_case_mask(request):
    return io.imread(request.param)


@pytest.mark.skipif(
    len(_CASE_PATHS) < len(NAMED_CASES),
    reason="benchmarking-dataset not found beside the repo, or missing one of 011/013/036",
)
def test_aabb_survivor_resolves_to_zero_abnormal_edges_at_md3(named_case_mask):
    algo = get_dithered_mesh_reconstruction_algorithm(min_distance=3, print_info=False)
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(named_case_mask)
    assert len(_find_abnormal_non_manifold_edges(points, triangles, labels)) == 0
