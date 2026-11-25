"""Build dw2d's mesh reconstruction algorithm, step by step or at its defaults, with dw3d's names.

dw3d's factory chooses among several EDTs, point placements, tesselations and face scores; dw2d has one of each, so
its factory keeps the setters that have a 2D meaning (`set_peak_local_points_placement_method`,
`set_junction_relocation`, `set_canonical_triangle_order`) and adds the 2D-only options as setters
(`set_interface_projection`, `set_pinch_vertex_repair`, `set_near_tie_sliver_relabelling`, `set_label_expansion`,
`set_score_samples`).
"""

from __future__ import annotations

from typing import Self

from dw2d.reconstruction_algorithm import DEFAULT_MIN_DISTANCE, MeshReconstructionAlgorithm


class MeshReconstructionAlgorithmFactory:
    """Factory of 2-D mesh reconstruction algorithms. Each `set_*` method returns the factory, so calls chain;
    `make_algorithm` builds the algorithm. A new factory builds the default algorithm.
    """

    def __init__(self, print_info: bool = False) -> None:
        """Start from the default configuration (`get_default_algorithm`).

        Args:
            print_info (bool, optional): print the size of the triangulation's dual graph. Defaults to False.
        """
        self.print_info = bool(print_info)
        self._options: dict = {
            "min_distance": DEFAULT_MIN_DISTANCE,
            "relocate_junctions": True,
            "canonical_triangle_order": True,
            "interface_projection": False,
            "pinch_vertex_repair": True,
            "near_tie_sliver_relabelling": None,
            "label_expansion": 0,
            "score_samples": 10,
        }

    def set_peak_local_points_placement_method(self, min_distance: int = DEFAULT_MIN_DISTANCE) -> Self:
        """Place the triangulation's points at the maxima of the distance map, at least `min_distance` pixels apart.

        Args:
            min_distance (int, optional): in pixels. Defaults to 4 (dw3d's default is 3 voxels).
        """
        self._options["min_distance"] = int(min_distance)
        return self

    def set_junction_relocation(self, enabled: bool = True) -> Self:
        """Move the junction vertices onto the mask's own junctions after meshing (on by default since 0.8.0;
        `dw2d.junction_relocation` says what it was measured to buy).
        """
        self._options["relocate_junctions"] = bool(enabled)
        return self

    def set_canonical_triangle_order(self, enabled: bool = True) -> Self:
        """Make the triangulation and the watershed's tie order independent of the machine (on by default).

        Qhull's triangulation of the integer points is not unique where four or more are cocircular, and Qhull
        resolves those cells differently on different machines; with this option such cells are re-triangulated by a
        fixed rule and triangles, dual-graph edges and equal scores are taken in canonical order. `False` is dw2d
        0.1.5's route (named `deterministic=False` before 0.8.0). dw3d's analogue is `canonical_tetrahedron_order`.
        """
        self._options["canonical_triangle_order"] = bool(enabled)
        return self

    def set_interface_projection(self, enabled: bool = True) -> Self:
        """After the junction relocation, move interface vertices onto the mask's interface locus (opt-in; requires
        the relocation, because alone it made junction angles worse: `dw2d.locus_projection`).
        """
        self._options["interface_projection"] = bool(enabled)
        return self

    def set_pinch_vertex_repair(self, enabled: bool = True) -> Self:
        """Absorb label fragments and repair pinch vertices after the watershed (on by default; `dw2d.mesh_surgery`)."""
        self._options["pinch_vertex_repair"] = bool(enabled)
        return self

    def set_near_tie_sliver_relabelling(self, enabled: bool | None = None) -> Self:
        """Relabel near-tie slivers after the watershed (`dw2d.sliver_relabelling`): `None` (default) turns it on
        without junction relocation and off with it, `True` or `False` force it.
        """
        self._options["near_tie_sliver_relabelling"] = enabled
        return self

    def set_label_expansion(self, pixels: int = 0) -> Self:
        """Grow every label by `pixels` before meshing (`skimage.segmentation.expand_labels`), for masks whose cells
        are separated by background and never meet. Defaults to 0 (dw2d 0.1.5's `expansion_labels`).
        """
        self._options["label_expansion"] = int(pixels)
        return self

    def set_score_samples(self, n: int = 10) -> Self:
        """Sample `n` points along each triangle edge to score it (dw2d 0.1.5's `interpolation_points`)."""
        self._options["score_samples"] = int(n)
        return self

    def make_algorithm(self) -> MeshReconstructionAlgorithm:
        """Build a new algorithm with the options set so far.

        Raises:
            ValueError: for the interface projection without the junction relocation.
        """
        if self._options["interface_projection"] and not self._options["relocate_junctions"]:
            error = "the interface projection requires the junction relocation (alone it made junction angles worse)"
            raise ValueError(error)
        return MeshReconstructionAlgorithm(print_info=self.print_info, **self._options)

    @staticmethod
    def get_default_algorithm(min_distance: int = DEFAULT_MIN_DISTANCE, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """The default 2-D algorithm: points at least `min_distance` pixels apart, canonical triangle order, pinch-vertex
        repair and junction relocation on, interface projection off.

        Args:
            min_distance (int, optional): in pixels. Defaults to 4.
            print_info (bool, optional): print the size of the triangulation's dual graph. Defaults to False.
        """
        factory = MeshReconstructionAlgorithmFactory(print_info=print_info)
        return factory.set_peak_local_points_placement_method(min_distance).make_algorithm()


get_default_mesh_reconstruction_algorithm = MeshReconstructionAlgorithmFactory.get_default_algorithm
"""dw2d's default mesh reconstruction algorithm, named as in dw3d.

The same function as `MeshReconstructionAlgorithmFactory.get_default_algorithm`, which gives its arguments
(`min_distance`, `print_info`) and what it returns.
"""
