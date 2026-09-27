"""Resolve a **mesh set** name to a reconstruction algorithm, or to a directory of meshes.

A mesh set is either a named reconstruction configuration this `dw3d` exposes (built and
cached per case), or a directory of already-reconstructed meshes (one `.rec`/`.npz` per
case, matched by case id). This module only concerns the first kind; a directory mesh set is
read directly by `compare_mesh_sets.py` with no algorithm involved.

Names here are release-hygiene names for the four configurations this comparison covers, not
`dw3d_benchmarks.run_case.VARIANT_GETTERS`'s internal keys -- kept as a separate, small,
explicit map so a user reading this file never has to know that `link_checked` is
`VARIANT_GETTERS["offset_excluded_linkcheck"]` internally, or that `dw3d`'s *current* default
is `dithered` (`get_default_mesh_reconstruction_algorithm` is `get_dithered_algorithm` at its
own defaults, restored after the link-checked configuration was found to *regress*
end-to-end tension-inference accuracy despite better mesh geometry in isolation -- see
`dw3d.reconstruction_algorithm_factory.MeshReconstructionAlgorithmFactory.get_default_algorithm`'s
docstring). `dw3d_benchmarks/run_case.py`'s own comment claiming the default is
`offset_excluded_linkcheck` is stale, left over from when that was briefly true; this module
does not repeat it.
"""

from __future__ import annotations

from dw3d_benchmarks.run_case import VARIANT_GETTERS

#: Release-hygiene configuration name -> `VARIANT_GETTERS` key.
NAMED_CONFIGURATIONS: dict[str, str] = {
    "dithered": "dithered",  # dw3d's current shipped default (get_default_mesh_reconstruction_algorithm)
    "deterministic": "deterministic",  # the default before the boundary-layer/junction-protection work
    "offset_included": "offset_included",  # boundary layer + junction protection; +15.7% area regression (anchor 1)
    "link_checked": "offset_excluded_linkcheck",  # junction-protected, link-condition-checked offset exclusion
}


def get_algorithm(configuration: str, min_distance: int, **kwargs):  # noqa: ANN003, ANN201
    """Return the reconstruction algorithm for a named configuration at `min_distance`."""
    if configuration not in NAMED_CONFIGURATIONS:
        message = f"unknown mesh-set configuration {configuration!r}; known: {sorted(NAMED_CONFIGURATIONS)}"
        raise KeyError(message)
    getter = VARIANT_GETTERS[NAMED_CONFIGURATIONS[configuration]]
    return getter(min_distance=min_distance, print_info=False, **kwargs)
