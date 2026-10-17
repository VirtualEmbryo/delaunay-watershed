# Examples

The examples are Jupyter notebooks in the repository's [`Examples/`](https://github.com/VirtualEmbryo/delaunay-watershed/tree/v0.7.0/Examples) folder. Install the package (with `[viewing]` for the viewer cells) in a terminal, then open a notebook from inside `Examples/`, where its relative paths resolve.

| notebook | what it shows | data |
|---|---|---|
| [Mesh reconstruction and visualisation](mesh-reconstruction-and-visualisation.md) | Reconstruct the multimaterial mesh of a cell aggregate (`data/Images/1.tif`, in the repository), scale it, centre it, save it as `.rec` and `.vtk`, and show it in the Polyscope viewer. | the repository's `data/Images/` |
| [Mask compression and reconstruction](mask-compression.md) | Compress a segmentation of nuclei into the points and triangles of its mesh, save it, and rebuild the mask from it. | a download (StarDist demo data) |
| [Advanced use: custom reconstruction algorithms](advanced-use.md) | Build reconstruction algorithms from their steps with `MeshReconstructionAlgorithmFactory`. | the repository's `data/Images/` |
| [Junction curves read from the mask](junction-curves.md) | Extract the trijunction curves the mask itself gives (`extract_junction_curves`), match them to the mesh's junction lines and measure the distance between the two. | the repository's `data/Images/` |
| [Default junction relocation](default-junction-relocation.md) | What junction relocation, on by default since 0.5.0, moves: the same mask reconstructed with and without it, and its report. | the repository's `data/Images/` |
| [Anisotropic voxels and crop faces](anisotropic-voxels-and-crop-faces.md) | A mask with voxels twice as long along z, resampled to cubic voxels and meshed in physical units (`construct_mesh_in_physical_units`), what meshing the native grid does instead, and a mask cut by the image box: crop faces, cut cells, the `.rec` file that keeps them, and a face declared physical. | nothing: the mask is generated in the notebook |
