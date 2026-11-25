"""Record the golden outputs of dw2d's default (deterministic) route on the 18 golden cases.

Usage, from the repository root: python -B tests/two_dimensional/golden/record_the_deterministic_route_outputs.py

Writes ``tests/two_dimensional/golden/dw2d_deterministic_route_outputs.npz`` (per case: mesh vertices and edges,
watershed seeds, edge scores, triangles, the label of each triangle after the watershed) and
``.json`` (versions; per case the DCEL angles, lengths, curvatures and cell areas at full
precision). The default route does not depend on the machine (its fixed rule and canonical orders make it so), and
``tests/two_dimensional/test_deterministic_route.py`` checks on every machine that the same outputs come from that
machine's Qhull and from the x86-64 triangulation stored with the legacy golden master.
"""
import contextlib
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS))

from two_dimensional.conftest import GOLDEN, golden_cases  # noqa: E402


def key(k):
    return "|".join(str(int(x)) for x in (k if isinstance(k, tuple) else (k,)))


def label_of_each_triangle(dw):
    out = np.full(len(dw.tri.simplices), np.iinfo(np.int64).min, dtype=np.int64)
    for label, triangles in dw.Map_end.items():
        out[np.asarray(triangles, dtype=np.int64)] = int(label)
    return out


def outputs(labels, min_dist, expansion):
    from dw2d import geometry_reconstruction_2d

    with contextlib.redirect_stdout(io.StringIO()):
        dw = geometry_reconstruction_2d(labels, min_dist=min_dist, expansion_labels=expansion)
        verts, edges = dw.return_mesh()
        mesh = dw.return_dcel()
        quantities = {q: {key(k): float(v) for k, v in f().items()}
                      for q, f in (("angles", mesh.compute_angles), ("lengths", mesh.compute_lengths),
                                   ("curvatures", mesh.compute_curvatures), ("cell_areas", mesh.compute_area_faces))}
    arrays = dict(verts=np.asarray(verts), edges=np.asarray(edges), seeds_coords=np.asarray(dw.seeds_coords),
                  seeds_indices=np.asarray(dw.seeds_indices), edge_scores=np.asarray(dw.Delaunay_Graph.Scores),
                  triangles=np.asarray(dw.tri.simplices), triangle_labels=label_of_each_triangle(dw))
    return arrays, quantities


def main():
    import networkx
    import scipy
    import skimage
    arrays, record = {}, dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                            skimage=skimage.__version__, networkx=networkx.__version__,
                                            machine=f"{platform.system()} {platform.machine()}"), cases={})
    for name, labels, min_dist, expansion in golden_cases():
        a, q = outputs(labels, min_dist, expansion)
        arrays.update({f"{name}__{k}": v for k, v in a.items()})
        record["cases"][name] = dict(min_dist=min_dist, expansion_labels=expansion, **q)
    np.savez_compressed(GOLDEN / "dw2d_deterministic_route_outputs.npz", **arrays)
    json.dump(record, open(GOLDEN / "dw2d_deterministic_route_outputs.json", "w"), indent=1, sort_keys=True)
    print("written", len(record["cases"]), "cases")


if __name__ == "__main__":
    main()
