"""The return annotations of the compressed-segmentation helpers say what they return.

`both_construct_mesh_and_compressed_segmentation` returns four items, the mesh's three arrays and the
compressed-segmentation dictionary, where its annotation listed three; `reconstruct_mask_from_dict` (and
`reconstruct_mask_from_saved_file_dict`) return the watershed's 64-bit integer labels, where their
annotations said uint8. Only the annotations changed: the values returned are those of before.
"""

import typing

import numpy as np

from dw3d import get_default_mesh_reconstruction_algorithm, reconstruct_mask_from_dict
from dw3d.mask_reconstruction import reconstruct_mask_from_saved_file_dict
from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm


def two_cells() -> np.ndarray:
    z, y, x = np.mgrid[:30, :40, :30].astype(float)
    mask = np.zeros(z.shape, dtype=np.uint16)
    mask[(z - 15) ** 2 + (y - 13) ** 2 + (x - 15) ** 2 < 9**2] = 1
    mask[(z - 15) ** 2 + (y - 27) ** 2 + (x - 15) ** 2 < 8**2] = 2
    return mask


def dtype_argument(annotation: object) -> object:
    """The scalar type of an `NDArray[...]` annotation, whichever way numpy nests its arguments."""
    for argument in typing.get_args(annotation):
        if isinstance(argument, type) and issubclass(argument, np.generic):
            return argument
        found = dtype_argument(argument)
        if found is not None:
            return found
    return None


def test_both_construct_returns_the_four_items_its_annotation_lists() -> None:
    hints = typing.get_type_hints(MeshReconstructionAlgorithm.both_construct_mesh_and_compressed_segmentation)
    annotated = typing.get_args(hints["return"])
    result = get_default_mesh_reconstruction_algorithm().both_construct_mesh_and_compressed_segmentation(two_cells())
    assert len(result) == len(annotated) == 4
    assert typing.get_origin(annotated[3]) is dict
    assert isinstance(result[3], dict)
    assert set(result[3]) == {"points", "triangles", "seeds", "image_shape"}


def test_the_rebuilt_mask_is_int64_as_annotated() -> None:
    algorithm = get_default_mesh_reconstruction_algorithm()
    algorithm.construct_mesh_from_segmentation_mask(two_cells())
    rebuilt = reconstruct_mask_from_dict(algorithm.last_compressed_segmentation)
    assert rebuilt.dtype == np.int64
    for function in (reconstruct_mask_from_dict, reconstruct_mask_from_saved_file_dict):
        assert dtype_argument(typing.get_type_hints(function)["return"]) is np.int64
