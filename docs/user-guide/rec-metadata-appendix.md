# The `.rec` / `.arec` metadata appendix

`save_rec`/`load_rec` (`dw3d.io`) read and write `.rec`/`.arec` multimaterial mesh files (points, triangles, and per-triangle material labels; text or binary). The text layout is the point count, one `x y z` line per point, the triangle count, and one `v0 v1 v2 label1 label2` line per triangle. The binary layout (`binary_mode=True`), in the machine's byte order (little-endian on x86 and ARM), is the point count (uint64), the points (3 float64 each), the triangle count (uint64), then per triangle its 3 vertex indices (int64) and its 2 labels (int32). The points are always written as float64, whatever their dtype in memory, so a file reads back the same. Optionally, a `.rec`/`.arec` file can carry a small metadata appendix: pass `metadata={...}` to `save_rec`, and read it back with `load_rec_appendix(filename)`.

The appendix is a trailing block of ASCII `# key = value` lines, written after the geometry data — never before it, because every existing reader (this one included) reads a count and then exactly that many values, so a trailing block is invisible to them, while a leading one breaks parsing outright. Fields:

```
# rec-appendix 1
# written_by = dw3d <version>
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
- `construct_mesh_in_physical_units` writes `coordinate_frame = physical`, the mask's own `spacing` (the mesh is already scaled by it), `source_shape`, `resampling` and `resampled_spacing` (see [Masks with anisotropic voxels](anisotropic-voxels.md)).
