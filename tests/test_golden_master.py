"""Golden-master regression tests for dw3d's mesh reconstruction pipelines.

Four fixture sets, four different jobs:

`tests/golden/` — the **current default** algorithm, which is `dithered` again (the
boundary layer + junction protection + link-condition-checked offset exclusion had been
the default for a time; it is reverted because end-to-end tension-inference accuracy is
worse with it than with `dithered` on every method measured — see
`get_default_algorithm`'s docstring and `BENCHMARKS.md` at the repository root). Asserted to
`rtol=1e-10`. Since point placement was made deterministic this pipeline contains no
random number and no undetermined ordering at all, so the tolerance is not really a
tolerance: `benchmarks/run_fingerprints.py` shows the whole internal state is reproduced
bit for bit. These fixtures were re-baselined four times: once for the determinism fix
(from `v0_3/`), once when the default moved to the junction-protected configuration (from
the deterministic fixtures), once when it moved to the link-checked configuration (from `junction_protected/`)
and once when it reverted (back to `dithered`, freshly computed on the current stack — see
`test_dithered_algorithm_reproduces_its_fixture` for why this is not byte-identical to
`tests/golden/dithered/`'s own, historical fixture). Every metric that moved at the first
three was quantified when it moved; the reversion
moves nothing new — it is a re-pointing of `get_default_algorithm`, not a re-tuning of any
algorithm.

`tests/golden/link_checked/` — the link-checked configuration, **a former default**, kept
as the opt-in "revised meshing" variant (see
`get_link_checked_mesh_reconstruction_algorithm`). It is generated independently from
`get_link_checked_mesh_reconstruction_algorithm` rather than moved, so it stays a live,
checkable record of that configuration regardless of what the default points at.

`tests/golden/junction_protected/` — boundary layer + junction protection, a former
default (`e598909`), and what `get_offset_included_mesh_reconstruction_algorithm` returns.
It is kept, and its own test kept green, because it is the configuration carrying the
+15.7 % interface-area regression the link-checked flip removes: a regression that can be
re-measured is worth more than one that is only described.
`test_the_historical_fixture_sets_are_actually_different` now covers it, so it cannot
silently become a duplicate of the default set.

`tests/golden/deterministic/` — the deterministic algorithm, an earlier default. Kept
because every acceptance criterion from the boundary-layer work onwards is stated
against it: the boundary layer's "all-surface tets 42.0 % -> < 15 %", junction
protection's "valence->=4 <= 133", and the wall-time budget are all deterministic-baseline
numbers. Checked against `get_deterministic_mesh_reconstruction_algorithm` at
`rtol=1e-10` — it is as deterministic as the default and its fixtures are the *same files*
that were `tests/golden/` before the flip, moved with `git mv` rather than regenerated.

`tests/golden/dithered/` — the original fixtures, kept verbatim as the historical record of
dw3d 0.3.6, and checked against `get_dithered_mesh_reconstruction_algorithm`, which
restores the historical **randomly dithered point placement**. It does *not* restore the
historical watershed edge ordering, and cannot: that ordering was `np.argsort` with no
tie-break over a score field where 24-33 % of consecutive gaps are exactly zero, so it had
no single well-defined answer — it was whatever numpy's sorting kernel happened to do. The
dithered test therefore pins the mesh *size* exactly and the geometry to `rtol=1e-4`, the
scale at which the historical ordering was itself undefined (measured: the ordering change
alone moves interface areas by at most 1.1e-5 and cell volumes by at most 9.4e-5,
relative, over the 8 configurations; junction lengths are unaffected). Widening the
tolerance further would make the test vacuous; the exact mesh-size assertions are what
keep it sharp.

If either fixture set ever needs regenerating, that must be a deliberate, reviewed
decision (a real algorithm change), never an automatic "fix the test" action.
"""

import json

import numpy as np
import pytest

from dw3d_benchmarks import metrics as m
from functools import partial

from dw3d import (
    get_boundary_layer_mesh_reconstruction_algorithm,
    get_default_mesh_reconstruction_algorithm,
    get_deterministic_mesh_reconstruction_algorithm,
    get_dithered_mesh_reconstruction_algorithm,
    get_junction_protected_mesh_reconstruction_algorithm,
    get_link_checked_mesh_reconstruction_algorithm,
    get_offset_included_mesh_reconstruction_algorithm,
)
from tests.conftest import (
    BOUNDARY_LAYER_GOLDEN_DIR,
    CUBIC_SCORE_GOLDEN_DIR,
    DETERMINISTIC_GOLDEN_DIR,
    DITHERED_GOLDEN_DIR,
    GOLDEN_DIR,
    IMAGE_NAMES,
    JUNCTION_PROTECTED_GOLDEN_DIR,
    LINK_CHECKED_GOLDEN_DIR,
    OFFSET_EXCLUDED_GOLDEN_DIR,
    golden_dir,
    load_image_or_skip,
)

# The keyword calls the deprecated synonyms `get_cubic_score_...` and `get_offset_excluded_...`
# stand for; tests/test_deprecated_aliases.py pins that each synonym reproduces its call exactly.
get_cubic_score = partial(get_junction_protected_mesh_reconstruction_algorithm, spline_order=3)
get_offset_excluded = partial(get_junction_protected_mesh_reconstruction_algorithm, exclude_offsets_from_surface=True)

# Every golden-master test runs twice: with the 0.5.0 default (junction relocation on) against
# `tests/golden/`, and with `relocate_junctions=False` set explicitly against the 0.4 fixtures in
# `tests/golden_without_relocation/`, which were moved there byte for byte, not regenerated.
RELOCATION = pytest.mark.parametrize("relocate_junctions", [True, False], ids=["relocated", "without_relocation"])

MIN_DISTANCES = [3, 5]

# The historical watershed ordering was undefined at this scale; see the module docstring.
DITHERED_GEOMETRY_RTOL = 1e-4


def _compute_golden_record(image_name: str, min_distance: int, algorithm_getter, relocate_junctions: bool) -> dict:
    mask = load_image_or_skip(image_name)
    algo = algorithm_getter(min_distance=min_distance, print_info=False)
    algo.relocate_junctions = relocate_junctions
    points, triangles, labels = algo.construct_mesh_from_segmentation_mask(mask)

    interface_areas = m.interface_areas(points, triangles, labels)
    cell_volumes = m.cell_volumes_from_tetrahedra(algo._tesselation_graph, algo._map_label_to_nodes_ids)
    edge_stats = m.edge_topology_stats(points, triangles, labels)

    return {
        "n_points": len(points),
        "n_triangles": len(triangles),
        "sorted_interface_areas": sorted(interface_areas.values()),
        "sorted_cell_volumes": sorted(cell_volumes.values()),
        "sorted_junction_lengths": sorted(edge_stats["triple_line_lengths"].values()),
    }


def _fixture_path(directory, image_name: str, min_distance: int):
    stem = image_name.rsplit(".", 1)[0]
    return directory / f"{stem}_min_distance_{min_distance}.json"


def _golden(directory, image_name: str, min_distance: int, relocate_junctions: bool):
    return _fixture_path(golden_dir(directory, relocate_junctions), image_name, min_distance)


def _assert_matches_fixture(record: dict, fixture_path, rtol: float) -> None:
    if not fixture_path.exists():
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"No baseline fixture yet: generated {fixture_path}. Re-run to verify reproduction.")

    expected = json.loads(fixture_path.read_text())

    assert record["n_points"] == expected["n_points"]
    assert record["n_triangles"] == expected["n_triangles"]
    for key in ("sorted_interface_areas", "sorted_cell_volumes", "sorted_junction_lengths"):
        assert len(record[key]) == len(expected[key]), f"{key}: different number of entries"
        np.testing.assert_allclose(record[key], expected[key], rtol=rtol, err_msg=key)


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_golden_master_reproduces_fixture(image_name, min_distance, relocate_junctions):
    record = _compute_golden_record(
        image_name, min_distance, get_default_mesh_reconstruction_algorithm, relocate_junctions,
    )
    _assert_matches_fixture(record, _golden(GOLDEN_DIR, image_name, min_distance, relocate_junctions), rtol=1e-10)


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_deterministic_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    """The deterministic algorithm, an earlier default (see the module docstring).

    `rtol=1e-10`, unchanged from when these were `tests/golden/`: the flip moved the files,
    it did not regenerate them, so a failure here means the deterministic path itself moved.
    """
    record = _compute_golden_record(
        image_name, min_distance, get_deterministic_mesh_reconstruction_algorithm, relocate_junctions,
    )
    _assert_matches_fixture(
        record,
        _golden(DETERMINISTIC_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_dithered_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    record = _compute_golden_record(
        image_name, min_distance, get_dithered_mesh_reconstruction_algorithm, relocate_junctions,
    )
    _assert_matches_fixture(
        record,
        _golden(DITHERED_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=DITHERED_GEOMETRY_RTOL,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_boundary_layer_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    """Boundary-layer variant (opt-in, not the default).

    Asserted at `rtol=1e-10`: the variant is fully deterministic and RNG-free, and
    `benchmarks/fingerprints_boundary_layer.json` shows it is bit-exact across runs, so —
    unlike the dithered set — there is no ordering ambiguity to absorb with a loose tolerance.
    """
    record = _compute_golden_record(
        image_name, min_distance, get_boundary_layer_mesh_reconstruction_algorithm, relocate_junctions,
    )
    _assert_matches_fixture(
        record,
        _golden(BOUNDARY_LAYER_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_junction_protected_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    """Junction-protected variant, boundary layer + junction protection -- a former default.

    `rtol=1e-10` for the same reason as the boundary-layer set: no RNG anywhere, and
    `benchmarks/baseline/fingerprints_junction_protected.json` pins bit-exactness across
    runs. The fixture is the *recommended* junction-protection configuration — the
    junction boundary layer is off and shell coarsening is 3, per
    `get_junction_protected_algorithm`'s defaults and the measurements recorded in
    its docstring. These files were not regenerated by the link-checked flip, so a
    failure here means the boundary-layer + junction-protection path itself moved.
    """
    record = _compute_golden_record(
        image_name, min_distance, get_junction_protected_mesh_reconstruction_algorithm, relocate_junctions,
    )
    _assert_matches_fixture(
        record,
        _golden(JUNCTION_PROTECTED_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_offset_included_algorithm_reproduces_the_junction_protected_fixture(
    image_name,
    min_distance,
    relocate_junctions,
):
    """`get_offset_included_algorithm` and `get_junction_protected_algorithm` are one configuration.

    Two names, one code path: `offset_included` is what the configuration *is* (and what
    the link-checked flip demoted), `junction_protected` is the mechanism it is built from
    and the key its 51-case records are stored under. If they ever diverge — say a later
    change gives one of them a different parameter — this fails and forces the divergence
    to be declared instead of leaving two names quietly meaning two things.
    """
    with pytest.warns(DeprecationWarning, match="0.6.0"):
        record = _compute_golden_record(
            image_name, min_distance, get_offset_included_mesh_reconstruction_algorithm, relocate_junctions,
        )
    _assert_matches_fixture(
        record,
        _golden(JUNCTION_PROTECTED_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_link_checked_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    """Link-checked collapse: boundary layer + junction protection with the offsets excluded.

    **This is the default**, and `tests/golden/link_checked/` is generated from
    `get_link_checked_mesh_reconstruction_algorithm` independently of `tests/golden/` so
    that the two can be compared rather than assumed equal — see
    `test_the_default_is_exactly_the_link_checked_configuration`.

    `rtol=1e-10` for the same reason as every deterministic set: no RNG anywhere, and
    `benchmarks/baseline/fingerprints_offset_excluded_linkcheck.json` pins bit-exactness
    across runs. Relative to `junction_protected/` this set should show fewer points and
    triangles, identical cell volumes (they come from the tetrahedra, which the collapse
    does not touch) and smaller interface areas — the +15.7 % that the boundary-layer
    flaps added. `test_offset_exclusion.py` and `test_link_condition_collapse.py` assert
    each of those directly rather than leaving them implicit in the numbers here.
    """
    record = _compute_golden_record(
        image_name, min_distance, get_link_checked_mesh_reconstruction_algorithm, relocate_junctions,
    )
    _assert_matches_fixture(
        record,
        _golden(LINK_CHECKED_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_cubic_score_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    """Cubic-interpolation variant: the default with a cubic B-spline EDT interpolant for the watershed scores.

    `rtol=1e-10` for the same reason as the other deterministic sets: no RNG anywhere, and
    `benchmarks/baseline/fingerprints_cubic_score.json` pins bit-exactness across runs.
    Opt-in, not the default: it changes the scores, hence the labelling, hence the mesh.
    """
    record = _compute_golden_record(image_name, min_distance, get_cubic_score, relocate_junctions)
    _assert_matches_fixture(
        record,
        _golden(CUBIC_SCORE_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_offset_excluded_algorithm_reproduces_its_fixture(image_name, min_distance, relocate_junctions):
    """Offset-excluded variant: the default, with the boundary-layer offsets kept out of the surface.

    `rtol=1e-10` for the same reason as every deterministic set: no RNG anywhere, and
    `benchmarks/baseline/fingerprints_offset_excluded.json` pins bit-exactness across runs.
    Opt-in. Note what this fixture set *should* show relative to
    `GOLDEN_DIR`'s: fewer points and triangles, the same cell volumes (the volumes come from
    the tetrahedra, which this exclusion does not touch) and smaller interface areas (the
    boundary-layer flaps carried area no interface has). `test_offset_exclusion.py` asserts
    each of those directly rather than leaving them implicit in the numbers here.
    """
    record = _compute_golden_record(image_name, min_distance, get_offset_excluded, relocate_junctions)
    _assert_matches_fixture(
        record,
        _golden(OFFSET_EXCLUDED_GOLDEN_DIR, image_name, min_distance, relocate_junctions),
        rtol=1e-10,
    )


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@pytest.mark.parametrize(
    "historical_dir",
    [DETERMINISTIC_GOLDEN_DIR, JUNCTION_PROTECTED_GOLDEN_DIR, LINK_CHECKED_GOLDEN_DIR],
    ids=["deterministic", "offset_included", "link_checked"],
)
@RELOCATION
def test_the_historical_fixture_sets_are_actually_different(
    image_name,
    min_distance,
    historical_dir,
    relocate_junctions,
):
    """Guard against a re-baselining having silently produced the current numbers again.

    Every configuration the default has ever pointed away from (the deterministic
    algorithm's, from the dithered one; the junction-protection work's, to boundary layer
    + junction protection; the link-checked flip's, to the link-condition-checked
    collapse) must differ from the current default's mesh. `dithered` itself is
    deliberately **not** included here: it is now the default, so `tests/golden/` and
    `tests/golden/dithered/` are expected to agree (to the extent
    `test_dithered_algorithm_reproduces_its_fixture`'s tolerance allows — see its
    docstring for why they are not byte-identical). If this ever passed vacuously for one
    of the three kept here — because someone regenerated a historical set with the
    *current* default — that set's test would become a duplicate of the first one and
    would stop protecting anything.

    Offset-included and `link_checked` are the interesting ones: the link-checked collapse
    changes only the *extraction*, so the tesselation, the labelling and the cell volumes
    are bit-identical between those two. Point and triangle counts are the quantities that
    must move, and they are what is compared.
    """
    current_path = _golden(GOLDEN_DIR, image_name, min_distance, relocate_junctions)
    historical_path = _golden(historical_dir, image_name, min_distance, relocate_junctions)
    if not (current_path.exists() and historical_path.exists()):
        pytest.skip("fixtures not generated yet")
    current = json.loads(current_path.read_text())
    historical = json.loads(historical_path.read_text())
    assert (current["n_points"], current["n_triangles"]) != (historical["n_points"], historical["n_triangles"])


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_the_default_is_exactly_the_dithered_configuration(image_name, min_distance, relocate_junctions):
    """The revert to the dithered default must be a re-pointing, not a re-tuning.

    `get_default_algorithm` delegates to `get_dithered_algorithm` at its own defaults, so
    the two must produce byte-identical output — computed live here, deliberately not read
    from `tests/golden/dithered/`'s fixture file, because that file is the historical dw3d
    0.3.6 record and is only asserted at `DITHERED_GEOMETRY_RTOL` elsewhere (see
    `test_dithered_algorithm_reproduces_its_fixture`'s docstring). If a later change gives
    the default a different parameter, this fails and forces the divergence to be declared
    — the same guard the junction-protection work kept between the default and
    `junction_protected/`, and the link-checked flip kept between the default and
    `link_checked/`.
    """
    default_record = _compute_golden_record(
        image_name, min_distance, get_default_mesh_reconstruction_algorithm, relocate_junctions,
    )
    dithered_record = _compute_golden_record(
        image_name, min_distance, get_dithered_mesh_reconstruction_algorithm, relocate_junctions,
    )
    assert default_record == dithered_record


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
@pytest.mark.parametrize("min_distance", MIN_DISTANCES)
@RELOCATION
def test_link_checked_kept_the_cell_volumes_and_shrank_the_interface_areas_of_offset_included(
    image_name,
    min_distance,
    relocate_junctions,
):
    """What the link-condition-checked collapse is *for*, asserted on the fixtures rather than described.

    Not about the default any more — both configurations compared here are opt-in
    variants. The collapse changes only which tesselation vertices may become surface
    vertices, so, relative to boundary layer + junction protection
    (`get_offset_included_algorithm`):

    * **cell volumes are bit-identical** — they are computed from the tetrahedra, which the
      collapse never touches;
    * **total interface area falls** — the boundary-layer offsets sat `delta` voxels off the
      surface and their triangles were fans reaching out and back, adding area no interface
      has. Over the 47 ground-truth cases that is +15.73 % -> +1.07 % signed error against
      the registered ground truth; here it is only asserted
      to be a *decrease*, because these four images have no ground truth to score against.
    """
    link_checked_path = _golden(LINK_CHECKED_GOLDEN_DIR, image_name, min_distance, relocate_junctions)
    offset_included_path = _golden(JUNCTION_PROTECTED_GOLDEN_DIR, image_name, min_distance, relocate_junctions)
    if not (link_checked_path.exists() and offset_included_path.exists()):
        pytest.skip("fixtures not generated yet")
    link_checked = json.loads(link_checked_path.read_text())
    offset_included = json.loads(offset_included_path.read_text())

    np.testing.assert_allclose(
        link_checked["sorted_cell_volumes"],
        offset_included["sorted_cell_volumes"],
        rtol=0,
        atol=0,
        err_msg="the link-condition-checked collapse must not move a cell volume",
    )
    assert link_checked["n_points"] < offset_included["n_points"]
    assert link_checked["n_triangles"] < offset_included["n_triangles"]
    assert sum(link_checked["sorted_interface_areas"]) < sum(offset_included["sorted_interface_areas"])
