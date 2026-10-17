# Reconstruction variants

`get_default_mesh_reconstruction_algorithm` gives you the **default** reconstruction: the published algorithm (equivalent to dw3d 0.3.6). It is the more accurate one for downstream tension inference today, so it is what most users should start with.

A **revised meshing pipeline** is also available, opt-in, as `get_link_checked_mesh_reconstruction_algorithm`. It is deterministic, protects triple junctions explicitly, and produces cleaner topology and lower interface-area bias than the default — but as measured on `foambryo`'s benchmarking dataset, meshes it produces currently give *worse* end-to-end tension-inference accuracy than the default, on every inference method tested (see [`BENCHMARKS.md`](https://github.com/VirtualEmbryo/delaunay-watershed/blob/v0.7.0/BENCHMARKS.md)). Reach for it when mesh geometry and topology matter more to you than today's downstream inference accuracy — for example, when the mesh itself, rather than inferred tensions, is the deliverable:

```py
from dw3d import get_link_checked_mesh_reconstruction_algorithm

mesh_reconstruction_algorithm = get_link_checked_mesh_reconstruction_algorithm()
mesh_reconstruction_algorithm.construct_mesh_from_segmentation_mask(segmentation_mask)
```
