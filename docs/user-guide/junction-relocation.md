# Straightening the junction lines (on by default since 0.5.0)

A reconstructed mesh's trijunction lines are ragged at voxel scale: the extraction places junction vertices wherever the tessellation happens to bracket them. Since 0.5.0 every reconstruction moves them onto the trijunction curves the mask itself gives, as a post-process on the finished mesh:

```py
from dw3d import get_default_mesh_reconstruction_algorithm

algorithm = get_default_mesh_reconstruction_algorithm()
points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(segmentation_mask)
print(algorithm.relocation_provenance)          # did it run, with which guard, did it warn
print(algorithm.relocation_report.as_dict())    # what it moved
```

**To reproduce the 0.4 meshes, switch it off**: `algorithm.relocate_junctions = False` before reconstructing, or `MeshReconstructionAlgorithmFactory().set_junction_relocation(False)`. At a point-placement `min_distance` of 5 or more the reconstruction emits `dw3d.CoarseSpacingRelocationWarning` once: the effect on surface-tangent contact angles there lies within the check's own noise and larger spacings are unmeasured; facet-based angles improve.

It can also be applied to a mesh you already have:

```py
from dw3d import relocate_junction_vertices
from dw3d.io import load_rec

points, triangles, labels = load_rec("mesh.rec")
points, report = relocate_junction_vertices(segmentation_mask, points, triangles, labels)
```

**It moves vertices and nothing else.** Triangles and material labels come back untouched, so a treated mesh can be compared vertex-for-vertex with its own untreated original.

**What it buys**, measured over 40 benchmark cases at four reconstruction spacings and four point samplers: line position error, local tangent error and discrete curvature all improve at every spacing on every sampler, and no existing mesh-validity number changes on any case. If you are using a seed-free sampler (`get_deterministic_mesh_reconstruction_algorithm`) it also improves contact angles by 2.3–3.5 degrees and downstream tension error by 20–40 %, and brings the junction lines to the default sampler's own untreated accuracy or better. On the default's seeded sampler the lines improve and the contact angles do not move significantly. See [`BENCHMARKS.md`](https://github.com/VirtualEmbryo/delaunay-watershed/blob/v0.7.0/BENCHMARKS.md) (measured before relocation became the default).

**What it costs**: about +0.75 s on the benchmark meshes, whose reconstruction takes about 5 s (+15 %); 0.1–1.5 s per timepoint on real embryo volumes, whose reconstruction takes about 0.6 s (median 0.34 s and 0.48 s on two time series). Within one real series it grows faster than linearly with mesh size (fitted exponent 2.68 over a 2.2-fold range — not a tissue-scale law). Since 0.6.0 the guard finds its candidate vertex pairs with a batched neighbour search, which leaves every relocated mesh bit-for-bit unchanged: on a 1 866-cell tissue the guard takes 45 s instead of 6.6 min.

**It cannot make the mesh pass through itself.** Every accepted move is checked against the whole mesh, not only the triangles around the vertex it moves, so the number of self-intersecting triangle pairs can never rise. That check is available on its own as `dw3d.count_self_intersections(points, triangles)`.

Example: the notebook *Default junction relocation* (`Examples/example_5_default_junction_relocation.ipynb`).
