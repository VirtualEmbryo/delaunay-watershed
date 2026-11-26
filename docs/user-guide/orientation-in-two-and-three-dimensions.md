# Orientation of the meshes in 2D and 3D

Both packages store, with every mesh element, the two regions it separates, smaller label first, and orient the element by one rule. foambryo reads that rule to know which region is on which side of an interface, which is what the sign of a pressure difference depends on. This page states the rule once, for triangles and for edges, with the test that decides it.

## The rule

Notation: a mask's index frame is the order of its axes, $(i, j, k) = (z, y, x)$ for a 3-D stack (columns 0, 1, 2 of `points`) and $(\text{row}, \text{column})$ for a 2-D image (columns 0, 1). Both frames are taken right-handed in that order. `labels[n] = (a, b)` with $a < b$.

**Each mesh element is the positively oriented boundary of its first label $a$, in the index frame.**

* **Triangles (dw3d)**, vertices $p_0, p_1, p_2$ in $(i, j, k)$: the right-hand normal $$ \mathbf{n} = (p_1 - p_0) \times (p_2 - p_0) $$ points out of $a$ and into $b$.
* **Edges (dw2d)**, vertices $p_0 \to p_1$ in $(\text{row}, \text{column})$, tangent $\mathbf{t} = p_1 - p_0 = (\Delta r, \Delta c)$: material $a$ lies on the left of $\mathbf{t}$, so the normal $$ \mathbf{n} = (\Delta c, -\Delta r), $$ the tangent turned clockwise, points out of $a$ and into $b$.

"Positively oriented boundary" is Stokes' convention, outward normal first: a triangle is positively oriented as boundary of a region $R$ when $\det(\mathbf{n}_\text{out}, p_1 - p_0, p_2 - p_0) > 0$, i.e. when its right-hand normal points out of $R$; an edge when $\det(\mathbf{n}_\text{out}, \mathbf{t}) > 0$, i.e. when $R$ lies on the left of $\mathbf{t}$. Equivalently, each material's elements, taken as stored where it is $a$ and reversed where it is $b$, run counter-clockwise around it: seen from outside in 3D, in the $(\text{row}, \text{column})$ plane in 2D.

## The test that decides it: signed volume and signed area

Take one cell (label 1) alone in the medium (label 0): every element carries $(a, b) = (0, 1)$ and is, by the rule, the positive boundary of the medium. By the divergence theorem, the signed volume and the signed area of the stored elements,

$$ V = \frac{1}{6} \sum_n \det(p_0, p_1, p_2), \qquad A = \frac{1}{2} \sum_n \left( p_{0,\text{row}}\, p_{1,\text{column}} - p_{0,\text{column}}\, p_{1,\text{row}} \right), $$

are minus the enclosed measure: $V < 0$ and $A < 0$ (they would be positive if the elements were the cell's boundary, i.e. under the opposite rule). Measured on a ball of radius 9 voxels (dw3d's default algorithm, a 28-voxel cube) and a disk of radius 20 pixels (dw2d's, a 60-pixel square): $V = -2407$ cubic voxels (3112 voxels in the ball; every right-hand normal points to the centre, into the cell) and $A = -1244$ square pixels (1264 pixels in the disk; the normal $(\Delta c, -\Delta r)$ points into the cell on every edge). `tests/test_orientation_of_two_and_three_dimensional_meshes.py` pins both signs, on every route that returns a 2-D mesh (the algorithm, the half-edge structure, the compressed segmentation, the `.rec` file, dw3d's object given a 2-D mask).

The same test, cell by cell, tells the rule of any 2-D mesh: the signed area of a cell's boundary, each edge counted as stored where the cell is $a$ and reversed where it is $b$, is positive for every cell under the rule. foambryo measures it this way on every 2-D mesh it reads.

## Before 0.8.0

dw2d stored every edge the other way round until 0.8.0: $a$ lay on the right of $p_0 \to p_1$ in $(\text{row}, \text{column})$ (on its left in $(x, y) = (\text{column}, \text{row})$, the frame of dw2d's former docstrings), so each edge was the boundary of its second label, the opposite of dw3d's rule in index order, and the disk above had $A = +1244$. Since 0.8.0 the two vertices of each edge are swapped; no vertex, edge, label or edge order changes, and dw2d's half-edge quantities (angles, lengths, curvatures, cell areas) are bitwise those of before. dw3d's triangles do not change.

A 2-D `.rec` file written by dw2d's algorithm since 0.8.0 records the rule in its appendix, `orientation = boundary_of_first_label` (`dw2d.io.ORIENTATION_RECORDED`). Files written before carry no such key. foambryo does not rely on the key: it measures each 2-D mesh's orientation from its geometry, reverses a mesh in the former orientation (so the tensions and pressures are the same from either), and refuses a mesh whose cells disagree, or whose geometry contradicts the rule its file records.
