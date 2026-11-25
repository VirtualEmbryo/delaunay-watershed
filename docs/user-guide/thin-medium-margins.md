# Thin margins of medium

A face of the image box (a side of the image, in 2D) that no cell reaches is not a crop: the medium goes on beyond it. Before 0.8.0, a cell a few voxels from such a face could take the medium between them. Since 0.8.0 thin margins are padded with medium before meshing, in both packages, with one rule.

## The defect

The distance field the points are placed on is 0 on the image box's outer voxels, below the membranes (0.73 on a flat membrane), and a point is placed only where no lower value lies within `min_distance`. So the membrane facing a nearby face of the box lost its points to the box (at a margin of 2 or 3 voxels it had none at all, at `min_distance` 3), the tetrahedra joining the cell to the box spanned the medium, and the watershed gave them to the cell. Measured before 0.8.0 at `min_distance` 3: a 12-voxel cube 4 voxels from every face meshed to 2.98 times its voxel count, and the two boxes of a 24-voxel test mask to 2.46 and 2.44 times; the range of margins affected grows with `min_distance` (up to margins of 5–7 voxels at `min_distance` 3, 7–8 at 4, 8–10 at 5). In 2D, at `min_distance` 4, a 20-pixel square 2, 3 or 4 pixels from every side meshed to 1.25, 1.37 and 1.28 times its pixels, and a 12-pixel square 3 or 4 pixels from one side collapsed to a triangle of 0.23 times its pixels at `min_distance` 5.

## The padding

Every face that no cell reaches and whose margin (its smallest gap to a cell) is below `2 * min_distance + 2` voxels or pixels is padded with label 0 up to that bound; the mesh is built on the padded mask and returned in the mask's frame.

* `pad_thin_medium_margins` (attribute or constructor argument of the algorithm, in dw3d and dw2d): `True` by default since 0.8.0; `False` gives the meshes of earlier versions.
* `algorithm.medium_margin_padding`: which faces were padded, and by how many voxels or pixels.
* A face that a cell reaches is a crop: it is never padded, and no vertex passes it.
* `ThinMediumMarginWarning`: a cell left with triangles (edges, in 2D) on a face its voxels do not reach, named with the face and its margin.
* On a padded face the cell is meshed as if it had been imaged with a wider margin, so its surface may reach a few voxels into the medium there, and past the image box when the margin is thinner than the padding: **by at most the face's padding**. Measured: up to 7 voxels on a 16-voxel cube 1 voxel from every face, 1 to 4 voxels on foambryo's test masks, 0 to 2 voxels on the lab's data.

**In 3D** the padding is decided from a first mesh: the mask is padded when that mesh has a vertex on a face no cell reaches, and is then meshed again. The mask's own voxels keep the random dither of the point placement they have unpadded, so the mesh changes only within the padded faces' reach. Measured on cubes of side 12 to 24 voxels at margins of 2 to 5 voxels, the volume ratio is now 0.75–1.09 (the 12-voxel cube 4 voxels from every face: 1.09; the test mask's two boxes 1.01 and 0.81). Of the lab's 684 test masks (the 47 benchmark masks, the 141 masks of the resolution sweep, 492 frames of two real embryo series and four test images), 18 change, and the others are meshed bit for bit as before, among them every benchmark mask; [the benchmarks](https://github.com/VirtualEmbryo/delaunay-watershed/blob/v0.8.0/BENCHMARKS.md) say what the change does to the 13 sweep masks it affects.

**In 2D** dw2d places its points without a dither, so it decides from the mask before meshing, with one run. On squares, squares near one side, a doublet and disks at `min_distance` 3, 4 and 5, every cell at a margin of 1 to 10 pixels now meshes to 0.84–0.99 times its pixels (0.23–1.73 before). None of the repository's 2-D masks is padded (the smallest margin of a side no cell reaches is 12 pixels), so their meshes are unchanged. dw3d's algorithm passes its `pad_thin_medium_margins` to dw2d for a 2-D mask.

## Masks resampled from longer voxels are not padded by default

`construct_mesh_in_physical_units` resamples a mask whose voxels are longer along one axis to cubic voxels. It does **not** pad such a mask unless asked for that call (`construct_mesh_in_physical_units(mask, spacing, pad_thin_medium_margins=True)`); a mask of cubic voxels is padded on both routes. Reason, measured on a test mask of four cells filling a ball 4 voxels from the image box, with z twice as long as x and y, over 100 seeds of the point placement's dither: a cell kept less than half its voxel volume on 20 of 100 seeds with the padding (95 % interval 0.13–0.29) against 3 of 100 without it (0.006–0.085; McNemar exact p = 1.5e-5); with cubic voxels, 13 against 10 (p = 0.51). The cell is lost in the watershed, not at its seed nor in mesh surgery; why this happens more often on resampled masks was not found. Unpadded, a resampled mask keeps the 0.7 meshes near thin margins, with their defect (on the same ball with x 2.5 times as long, one cell comes out at 7.47 times its voxel volume unpadded, 0.78 padded), and `ThinMediumMarginWarning` names each cell that lies on a face it does not reach.

## Known issues

* **Resampled masks**: above; the padding is off there by default because it made the watershed lose cells more often (20 of 100 seeds against 3 of 100).
* **In 3D, a cell that collapses without reaching the box is not padded.** The decision is taken from the first mesh's vertices, so a small cell that collapses near a face without a vertex on it is missed: on cubes of side 6 to 12 voxels at 1 to 5 voxels from one face or from every face, at `min_distance` 3 to 5 and 10 dither seeds (1 200 reconstructions), 25 cubes collapsed to 0.00–0.49 of their voxels with no vertex on a face of the box, so the mask was not padded; padded, the same masks give 0.58–0.80. dw2d decides from the mask and has no such blind spot; the two packages therefore decide differently in this release.
* **The surface can pass the image box** on a padded face, by at most the padding (above).
* **An 8-voxel cube** no longer exceeds 1.5 times its voxels, though it can fall to 0.10, as it does on 2 of 20 dither seeds 8 voxels from every face without any padding.

API: [`dw3d.thin_medium_margins`](../api/thin-medium-margins.md), [`dw2d.thin_medium_margins`](../api/dw2d-thin-medium-margins.md).
