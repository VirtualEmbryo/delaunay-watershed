# The `.rec` / `.arec` metadata appendix

`save_rec`/`load_rec` (`dw3d.io`) read and write `.rec`/`.arec` multimaterial mesh files (points, triangles, and per-triangle material labels; text or binary). The text layout is the point count, one `x y z` line per point, the triangle count, and one `v0 v1 v2 label1 label2` line per triangle. Since 0.8.0 the files are read and written by [iorec](https://pypi.org/project/iorec/) (`iorec>=1.4.1`), with the same names and signatures in `dw3d.io`, and the same arrays and typed appendix values as before. The binary layout (`binary_mode=True`), little-endian on every machine, is the point count (uint64), the points (3 float64 each), the triangle count (uint64), then per triangle its 3 vertex indices (int64) and its 2 labels (int32). The points are always written as float64, whatever their dtype in memory, so a file reads back the same. Optionally, a `.rec`/`.arec` file can carry a small metadata appendix: pass `metadata={...}` to `save_rec`, and read it back with `load_rec_appendix(filename)`.

The appendix is a trailing block of ASCII `# key = value` lines, written after the geometry data — never before it, because every existing reader (this one included) reads a count and then exactly that many values, so a trailing block is invisible to them, while a leading one breaks parsing outright. Fields:

```
# rec-appendix 1
# written_by = dw3d <version>
# writer_library = iorec <version>
# dimension = 3
# coordinate_frame = voxel_index
# axis_order = zyx
# spacing = 1.0 0.325 0.325
# spacing_unit = um
# source_shape = 194 200 199
# crop_offset = 0 0 0
```

- `coordinate_frame` is `voxel_index`, `physical`, or `unknown` — this is the field the appendix exists for, since spacing alone can't tell a reader whether coordinates are raw voxel indices awaiting a scale or physical units that already have it applied.
- `axis_order`, `spacing`/`spacing_unit`, `source_shape` and `crop_offset` record the rest of the sampling geometry, `crop_offset` even when zero (where the mask sat in a larger image; the crop faces themselves are recorded by `crop_faces`, below).
- **A file with no appendix must be treated as `coordinate_frame == "unknown"`, never as isotropic spacing** — `load_rec_appendix` returns `None` in that case, and every file saved before this feature existed has no appendix.
- `crop_faces = <first row> <count>` says that the last `<count>` triangle rows, from `<first row>` on, are crop faces (see [Cropped tissues](cropped-tissues.md)), and `physical_image_faces` which image faces were declared physical (`none` by default). Written by `save_to_rec_mesh` whenever a cell of the mask reaches the image box where the mesh lies on it; a file without the key records nothing about a crop.
- `media = 0 L ...` lists the labels that are media, not cells: the exterior 0, then the enclosed pockets of medium the reconstruction made their own regions (see [Enclosed pockets of medium](enclosed-media.md)), with `enclosed_media_voxels` and `enclosed_media_inscribed_radius` aligned with the enclosed labels. Written only when the mesh has an enclosed medium; a file without the key has the exterior 0 as its only medium.
- `unresolved_regions = L ...` and `unresolved_regions_voxels` list the labels of the mask absent from the mesh, whatever the cause (see [Enclosed pockets of medium](enclosed-media.md#regions-too-small-for-the-mesh)), with their voxel counts; before 0.8.0 only those with no tetrahedron of the tesselation in their voxels. Written only when there is one.
- `construct_mesh_in_physical_units` writes `coordinate_frame = physical`, the mask's own `spacing` (the mesh is already scaled by it), `source_shape`, `resampling` and `resampled_spacing` (see [Masks with anisotropic voxels](anisotropic-voxels.md)).
- `dimension` is `3` for dw3d's meshes and `2` for dw2d's. `dw3d.io.rec_dimension(filename)` reads it, or, for a file without it, tells the dimension from the file's layout (the coordinates per point row of a text file; which binary layout accounts exactly for the geometry's bytes), and refuses a file it cannot classify; `dw3d.io.load_rec` refuses a 2-D file and names `dw2d.io.load_rec`. dw3d writes `dimension` whenever it writes an appendix; a file without one is written byte for byte as before. dw2d's `save_to_rec_mesh` (and dw3d's, given a 2-D mask) always writes an appendix, with `dimension = 2`, `axis_order = yx`, `source_shape`, the image-border vertices and `orientation = boundary_of_first_label`, the rule its edges follow ([orientation](orientation-in-two-and-three-dimensions.md)); `dw2d.io.save_rec` writes one only when given `metadata`, and the deprecated `export_mesh` writes none.
- `writer_library` names the iorec version that wrote the file; `written_by` names dw3d's (or dw2d's) version.

**The appendix written is version 1 (`# key = value` lines), on purpose.** iorec also reads and writes a version 2 (`# key: <JSON>` lines), and delaunay-watershed 0.8.0 reads it, typing its values by their JSON. It does not write it: delaunay-watershed-3d 0.7.1 and foambryo 0.7.1 read a version-2 block without any error but lose every key of it (crop faces, media, unresolved regions, spacing), while they read every key of version 1. `dw3d.io.APPENDIX_VERSION_WRITTEN` is 1.
