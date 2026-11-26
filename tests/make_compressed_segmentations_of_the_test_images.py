"""Write `tests/golden/compressed_segmentations_of_the_test_images.npz`, the input of the mask-rebuild tests.

For each test image (`data/Images/1.tif` to `4.tif`): the compressed segmentation of its default mesh at
`min_distance` 3, reconstructed on the stored tetrahedralisation (`tests/pinned_tetrahedralisations.py`), so
that the file does not depend on Qhull. Keys `<image>_points`, `<image>_triangles`, `<image>_seeds` and
`<image>_image_shape`, with `<image>` the image's stem. Run from the repository root:

    PYTHONPATH="$PWD/src:$PWD" .venv/bin/python tests/make_compressed_segmentations_of_the_test_images.py
"""

import numpy as np

from dw3d import get_default_mesh_reconstruction_algorithm
from tests.conftest import GOLDEN_DIR, IMAGE_NAMES, load_image_or_skip
from tests.pinned_tetrahedralisations import pinned_tesselation_function

MIN_DISTANCE = 3
OUTPUT = GOLDEN_DIR / "compressed_segmentations_of_the_test_images.npz"


def compressed_segmentation(image_name: str) -> dict:
    algorithm = get_default_mesh_reconstruction_algorithm(min_distance=MIN_DISTANCE, print_info=False)
    algorithm.tesselation_creation_function = pinned_tesselation_function(image_name, MIN_DISTANCE)
    algorithm.construct_mesh_from_segmentation_mask(load_image_or_skip(image_name))
    return algorithm.last_compressed_segmentation


if __name__ == "__main__":
    arrays = {}
    for name in IMAGE_NAMES:
        stem = name.rsplit(".", 1)[0]
        compressed = compressed_segmentation(name)
        arrays[f"{stem}_points"] = np.asarray(compressed["points"], dtype=np.float64)
        arrays[f"{stem}_triangles"] = np.asarray(compressed["triangles"], dtype=np.int32)
        arrays[f"{stem}_seeds"] = np.asarray(compressed["seeds"], dtype=np.int32)
        arrays[f"{stem}_image_shape"] = np.asarray(compressed["image_shape"], dtype=np.int32)
    np.savez_compressed(OUTPUT, **arrays)
    print(OUTPUT, {key: value.shape for key, value in arrays.items()})
