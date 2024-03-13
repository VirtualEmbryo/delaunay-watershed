"""Module for factory class MeshReconstructionAlgorithmFactory.

This class allows to build custom algorithms for mesh reconstruction from segmentation masks.

Matthieu Perez 2024
"""

from functools import partial
from typing import Self

from dw3d.edt import compute_edt_classical
from dw3d.points_on_edt import peak_local_points
from dw3d.reconstruction_algorithm import (
    EdtCreationFunction,
    MeshReconstructionAlgorithm,
    PointPlacingFunction,
    ScoreComputationFunction,
    TesselationCreationFunction,
)
from dw3d.score_computation import compute_scores_by_max_value, compute_scores_by_mean_value
from dw3d.tesselation import simple_delaunay_tesselation


class MeshReconstructionAlgorithmFactory:
    """Factory to build custom algorithms for mesh reconstruction from segmentation masks."""

    def __init__(self, print_info: bool = False, perform_mesh_postprocess_surgery: bool = True) -> None:
        """Initialize the factory."""
        self.print_info = print_info
        self._edt_creation_function: EdtCreationFunction = _edt_creation_function_classical(print_info=print_info)
        self._point_placing_function: PointPlacingFunction = _point_placing_function_peak_local(print_info=print_info)
        self._tesselation_creation_function: TesselationCreationFunction = simple_delaunay_tesselation
        self._score_computation_function: ScoreComputationFunction = compute_scores_by_mean_value
        self.perform_mesh_postprocess_surgery = perform_mesh_postprocess_surgery

    def set_classical_edt_method(self) -> Self:
        """Use the classical method to create the Euclidean Distance Transform from a segmentation mask."""
        self._edt_creation_function = _edt_creation_function_classical(self.print_info)
        return self

    def set_peak_local_points_placement_method(self, min_distance: int = 5) -> Self:
        """Place points on the EDT image using local extrema of the EDT (and corners).

        Args:
            min_distance (int, optional): Minimum distance between extrema. Defaults to 5.
        """
        self._point_placing_function = _point_placing_function_peak_local(min_distance, self.print_info)
        return self

    def set_delaunay_tesselation_method(self) -> Self:
        """Use Delaunay algorithm to create the tesselation from a list of points."""
        self._tesselation_creation_function = simple_delaunay_tesselation
        return self

    def set_score_computation_by_mean_value(self) -> Self:
        """Use the mean value of the EDT image on triangle faces of the tesselation to compute scores for Watershed."""
        self._score_computation_function = compute_scores_by_mean_value
        return self

    def set_score_computation_by_max_value(self) -> Self:
        """Use the max value of the EDT image on triangle faces of the tesselation to compute scores for Watershed."""
        self._score_computation_function = compute_scores_by_max_value
        return self

    def make_algorithm(self) -> MeshReconstructionAlgorithm:
        """Build a new mesh reconstruction algorithm using the previously set methods.

        Returns:
            MeshReconstructionAlgorithm: Mesh Reconstruction Algorithm ready to be executed.
        """
        return MeshReconstructionAlgorithm(
            print_info=self.print_info,
            edt_creation_function=self._edt_creation_function,
            point_placing_function=self._point_placing_function,
            tesselation_creation_function=self._tesselation_creation_function,
            score_computation_function=self._score_computation_function,
            perform_mesh_postprocess_surgery=self.perform_mesh_postprocess_surgery,
        )

    @staticmethod
    def get_default_algorithm(min_distance: int = 5, print_info: bool = False) -> MeshReconstructionAlgorithm:
        """Return the default mesh reconstruction algorithm with sensible default values.

        Args:
            min_distance (int, optional): Minimum distance (in pixels) between extrema when placing
                tesselation points. Defaults to 5.
            print_info (bool, optional): Print algorithm details while executing.
        """
        return MeshReconstructionAlgorithm(
            print_info=print_info,
            edt_creation_function=_edt_creation_function_classical(print_info=print_info),
            point_placing_function=_point_placing_function_peak_local(min_distance=min_distance, print_info=print_info),
            tesselation_creation_function=simple_delaunay_tesselation,
            score_computation_function=compute_scores_by_mean_value,
            perform_mesh_postprocess_surgery=True,
        )


def _edt_creation_function_classical(
    print_info: bool = False,
) -> EdtCreationFunction:
    """Get a function that compute an EDT from a segmentation mask.

    Args:
        print_info (bool, optional): Print detals about the algorithm. Defaults to False.

    Returns:
        EdtCreationFunction:
            - the actual function which takes a segmentation mask and return an Euclidean Distance Transform image.
    """
    return partial(compute_edt_classical, print_info=print_info)


def _point_placing_function_peak_local(
    min_distance: int = 5,
    print_info: bool = False,
) -> PointPlacingFunction:
    """Get a function that peaks local min and max points (+ corner points) from an EDT image.

    Args:
        min_distance (int, optional): Minimum distance between extrema. Defaults to 5.
        print_info (bool, optional): Print detals about the algorithm. Defaults to False.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.uint]]:
            - the actual function which takes only an EDT image and return an array of 3D pixel coordinates
              of local min & max of the EDT (+ corners)
    """
    return partial(peak_local_points, min_distance=min_distance, print_info=print_info)
