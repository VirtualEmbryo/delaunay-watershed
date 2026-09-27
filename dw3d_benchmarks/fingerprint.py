"""Bit-exactness fingerprints of a whole reconstruction (the determinism fix).

Why this exists, and why the golden masters are not enough
----------------------------------------------------------
`tests/golden/` asserts `rtol=1e-10` on *summary statistics* — sorted interface areas,
sorted cell volumes, sorted junction lengths, plus mesh point/triangle counts. That is a
good regression net for geometry, but it is blind to the failure mode the determinism fix
is about:
a watershed **labelling** change that permutes which tetrahedron belongs to which cell
without moving any of those aggregates by more than `1e-10`, or a change confined to the
tesselation-boundary bookkeeping (`lone_faces`, `nodes_on_the_border`) that never reaches
the output mesh at all.

The defect-cleanup work established that 26-33 % of consecutive watershed face-score gaps are
**exactly zero** across the 8 in-repo configurations, so a third of the edge ordering in
`_seeded_watershed_aggregation` is decided by tie-breaking rather than by the score field.
Determinism claims therefore have to be judged on the raw internal state, not on the
summaries. That work did it with a one-off script; this module
is the same measurement made reproducible and checked in.

What is fingerprinted
---------------------
Seven arrays, hashed individually and then jointly, chosen to cover every stage whose
output a tie-break could perturb:

===========================  ============================================================
`points`                     output mesh geometry
`triangles`                  output mesh topology
`labels`                     output mesh materials (the two labels either side)
`scores`                     the raw watershed score of every tesselation face
`map_node_id_to_label`       the watershed labelling of every tetrahedron
`lone_faces`                 faces on the tesselation boundary (removed by the border cleanup)
`nodes_on_the_border`        tetrahedra flagged by those faces (removed by the border cleanup)
===========================  ============================================================

Float arrays are hashed from their raw IEEE-754 bytes in C order, so the hash is exact,
not tolerance-based: two runs agree here only if they agree bit for bit. Byte order is
forced to little-endian so a fingerprint is comparable across architectures.

Reading a mismatch
------------------
Compare the per-array digests, not just the joint one. `scores` differing means the
perturbation is upstream of the watershed (points, tesselation or interpolation);
`scores` identical with `map_node_id_to_label` differing means the perturbation is the
edge ordering itself, i.e. tie-breaking.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm

# The fingerprinted arrays, in the order they are combined into the joint digest.
FINGERPRINT_FIELDS = (
    "points",
    "triangles",
    "labels",
    "scores",
    "map_node_id_to_label",
    "lone_faces",
    "nodes_on_the_border",
)


def _digest(array: NDArray) -> str:
    """SHA-256 of an array's raw bytes, plus its shape and canonical dtype.

    Shape and dtype are hashed alongside the buffer so that two arrays with the same
    bytes but a different shape (or a different integer width) do not collide. The array
    is forced to little-endian and C order first, so the digest does not depend on the
    machine's byte order or on whether the caller handed us a view.
    """
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder == ">":
        array = array.astype(array.dtype.newbyteorder("<"))
    hasher = hashlib.sha256()
    hasher.update(f"{array.shape}|{array.dtype.str}|".encode())
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def fingerprint_algorithm(algo: MeshReconstructionAlgorithm) -> dict[str, Any]:
    """Fingerprint an algorithm that has already run `construct_mesh_from_segmentation_mask`.

    Args:
        algo: a `MeshReconstructionAlgorithm` whose reconstruction has completed.

    Returns:
        dict with one hex digest per field of `FINGERPRINT_FIELDS`, a `joint` digest over
        all of them in order, and the plain sizes that make a mismatch readable at a
        glance (`n_points`, `n_triangles`, `n_tetrahedra`, `n_lone_faces`).
    """
    graph = algo._tesselation_graph  # measurement reaches into internals on purpose
    points, triangles, labels = algo.last_constructed_mesh

    arrays = {
        "points": np.asarray(points),
        "triangles": np.asarray(triangles),
        "labels": np.asarray(labels),
        "scores": np.asarray(graph.scores),
        "map_node_id_to_label": np.asarray(algo._map_node_id_to_label),
        "lone_faces": np.asarray(graph.lone_faces),
        "nodes_on_the_border": np.asarray(graph.nodes_on_the_border),
    }

    digests = {name: _digest(arrays[name]) for name in FINGERPRINT_FIELDS}
    joint = hashlib.sha256("|".join(digests[name] for name in FINGERPRINT_FIELDS).encode()).hexdigest()

    return {
        **digests,
        "joint": joint,
        "n_points": len(points),
        "n_triangles": len(triangles),
        "n_tetrahedra": int(graph.n_simplices),
        "n_lone_faces": len(graph.lone_faces),
    }


def fingerprint_reconstruction(
    mask: NDArray[np.uint],
    algorithm_factory: Callable[[], MeshReconstructionAlgorithm],
) -> dict[str, Any]:
    """Run one reconstruction and fingerprint it.

    Args:
        mask: the segmentation mask.
        algorithm_factory: zero-argument callable returning a fresh
            `MeshReconstructionAlgorithm`. A callable rather than an instance so that
            each fingerprint is taken on a pristine algorithm object.
    """
    algo = algorithm_factory()
    algo.construct_mesh_from_segmentation_mask(mask)
    return fingerprint_algorithm(algo)


def environment_record() -> dict[str, str]:
    """Identify the dependency stack a fingerprint was taken in.

    Fingerprints are only interesting when compared *across* these, so every record
    carries one. Kept to the libraries that can plausibly move a bit: the interpreter,
    numpy (RNG, sorting, linear algebra), scipy (Qhull), scikit-image (`peak_local_max`)
    and `edt` (the distance transform itself).
    """
    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for module_name, key in (("numpy", "numpy"), ("scipy", "scipy"), ("skimage", "skimage"), ("edt", "edt")):
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - every one of these is a hard dependency
            versions[key] = "absent"
    return versions


def compare_fingerprints(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return the names of the fingerprinted fields that differ between two records."""
    return [name for name in FINGERPRINT_FIELDS if a.get(name) != b.get(name)]
