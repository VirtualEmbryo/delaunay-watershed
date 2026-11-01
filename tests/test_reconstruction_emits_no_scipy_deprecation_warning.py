"""A reconstruction imports SciPy's k-d tree from its public namespace and warns about no deprecated SciPy name.

Up to 0.7.0 the seed lookup imported `scipy.spatial.ckdtree`, a namespace SciPy deprecated (to be removed in SciPy
2.0.0), so every reconstruction printed a `DeprecationWarning`. The public `scipy.spatial.cKDTree` is the very same
class (checked below), so the meshes cannot change; the golden-master tests pin them bitwise. The seed lookup now
goes through `dw3d.nearest_neighbours`, which breaks exact ties itself; its k-d tree is the same public class.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.spatial

from dw3d import get_default_mesh_reconstruction_algorithm, nearest_neighbours


def two_cells(n: int = 32) -> np.ndarray:
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[(z - 16) ** 2 + (y - 11) ** 2 + (x - 16) ** 2 < 7**2] = 1
    mask[(z - 16) ** 2 + (y - 21) ** 2 + (x - 16) ** 2 < 6**2] = 2
    return mask


def test_the_seed_lookup_uses_the_public_kd_tree() -> None:
    assert nearest_neighbours.cKDTree is scipy.spatial.cKDTree


def test_a_reconstruction_emits_no_scipy_deprecation_warning() -> None:
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algorithm.construct_mesh_from_segmentation_mask(two_cells())
    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not [m for m in deprecations if "scipy" in m.lower()], deprecations
