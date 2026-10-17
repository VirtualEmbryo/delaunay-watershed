# Masks with anisotropic voxels (longer along z)

Most microscope stacks have a z step larger than their pixel size. Meshing such a mask on its own grid and scaling the mesh afterwards samples the surfaces coarsely along z, and every contact angle read from the mesh suffers. Resample the mask to cubic voxels first, by shape-based interpolation, and mesh in physical units:

```py
algorithm = get_default_mesh_reconstruction_algorithm()
points, triangles, labels = algorithm.construct_mesh_in_physical_units(mask, spacing=(0.5, 0.3, 0.3), spacing_unit="um")
algorithm.save_to_rec_mesh("mesh.rec")   # records the frame and the source spacing in the appendix
```

The mesh comes out in the unit of `spacing` (do not call `set_pixel_size` on it); `min_distance` counts voxels of the resampled, cubic grid, so the mesh has as many points as for a mask acquired at the finer spacing, and costs as much: memory and time grow with the z/xy ratio. Measured on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size and down-sampled along z (one seed), as the median over tissues of each mesh's median line-mean contact-angle error minus that of the same tissue meshed at full resolution: meshing the native grid costs +1.38, +1.97 and +4.10 degrees at ratios 5/3, 2 and 3; resampling first, +0.83, +0.71 and +1.17. Nothing was measured above 3 at that cell size: above it the method warns (`dw3d.UnmeasuredAnisotropyWarning`) and still meshes. Keep foambryo's default (chord) reading: its quadric reading made every route worse at real cell size. The resampling alone is `dw3d.resample_to_isotropic(mask, spacing)`. Repeating slices instead (`method="nearest"`), coarsening x and y to the z step, or telling only the distance transform about the spacing were all measured worse.

Example: the notebook *Anisotropic voxels and crop faces* (`Examples/example_6_anisotropic_voxels_and_crop_faces.ipynb`).

**Scaling a mesh afterwards.** `set_pixel_size(xy_pixel_size, z_pixel_size)` (the algorithm's method, the function in `dw3d.mesh_utilities`, and `dw3d.set_points_pixel_size`) scales column 0 of the points (the mask's axis 0, z) by `z_pixel_size` and columns 1 and 2 by `xy_pixel_size`. Before 0.7.0 it scaled columns 0 and 1 by the xy size and column 2 by the z size: re-run any analysis that called it with different sizes. A mask stored as `(x, y, z)` must be transposed to `(z, y, x)` first.
