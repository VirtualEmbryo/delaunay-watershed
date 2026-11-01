"""The CI's reading of the cross-machine checks (`tests/report_cross_machine_checks.py`), on JUnit reports made up here.

Hervé's rule for the CI: a Qhull difference inside cospherical sets is reported (a warning and counts
in the job summary), not a failure; any other skip of these checks, in particular of the tests on the
stored (injected) tetrahedralisation, or a module where no test ran, fails the job.
"""

from tests import pinned_tetrahedralisations
from tests import report_cross_machine_checks as report

MODULES = ["tests/test_canonical_tetrahedron_order.py", "tests/test_golden_master.py"]
COSPHERICAL = f"{report.COSPHERICAL_SKIP}: 141 tetrahedra differ, all inside cospherical sets (exact check)"


def _junit(tmp_path, cases):
    body = []
    for module, name, outcome, message in cases:
        inner = "" if outcome == "passed" else f'<{outcome} message="{message}"/>'
        body.append(f'<testcase classname="tests.{module}" name="{name}">{inner}</testcase>')
    path = tmp_path / "junit.xml"
    path.write_text(f'<testsuites><testsuite name="pytest">{"".join(body)}</testsuite></testsuites>')
    return path


def _run(tmp_path, cases, capsys):
    summary = tmp_path / "summary.md"
    status = report.main(["report", str(_junit(tmp_path, cases)), str(summary), *MODULES])
    return status, capsys.readouterr().out, summary.read_text()


BASE = [
    ("test_canonical_tetrahedron_order", "test_qhull_gives_the_pinned_tetrahedralisation[3-2.tif]", "passed", ""),
    ("test_golden_master", "test_golden_master_reproduces_fixture[True-3-2.tif]", "passed", ""),
]


def test_the_skip_message_is_the_one_the_tests_write():
    assert report.COSPHERICAL_SKIP == pinned_tetrahedralisations.COSPHERICAL_SKIP


def test_all_passed_is_clean(tmp_path, capsys):
    status, out, summary = _run(tmp_path, BASE, capsys)
    assert status == 0
    assert "::warning" not in out
    assert "test_golden_master: 1 passed" in summary


def test_a_cospherical_skip_of_a_qhull_test_warns_and_passes(tmp_path, capsys):
    cases = [
        *BASE,
        (
            "test_golden_master",
            "test_golden_master_reproduces_fixture_with_qhull[True-3-2.tif]",
            "skipped",
            COSPHERICAL,
        ),
    ]
    status, out, summary = _run(tmp_path, cases, capsys)
    assert status == 0
    assert "::warning title=Qhull differs inside cospherical sets::1 tests skipped" in out
    assert "1 skipped, Qhull differs inside cospherical sets" in summary
    assert "141 tetrahedra differ" in summary


def test_any_skip_of_a_test_on_the_stored_tetrahedralisation_fails(tmp_path, capsys):
    cases = [
        *BASE,
        ("test_golden_master", "test_golden_master_reproduces_fixture[False-5-1.tif]", "skipped", COSPHERICAL),
    ]
    status, _, summary = _run(tmp_path, cases, capsys)
    assert status == 1
    assert "Skipped for another reason" in summary


def test_another_reason_to_skip_a_qhull_test_fails(tmp_path, capsys):
    cases = [
        *BASE,
        (
            "test_canonical_tetrahedron_order",
            "test_qhull_gives_the_pinned_tetrahedralisation[5-1.tif]",
            "skipped",
            "data/Images/1.tif not found",
        ),
    ]
    status, _, _ = _run(tmp_path, cases, capsys)
    assert status == 1


def test_a_module_where_nothing_ran_fails(tmp_path, capsys):
    status, _, summary = _run(tmp_path, BASE[:1], capsys)
    assert status == 1
    assert "No test ran in: test_golden_master" in summary


def test_a_difference_outside_cospherical_sets_is_a_failure_and_is_counted(tmp_path, capsys):
    cases = [
        *BASE,
        (
            "test_canonical_tetrahedron_order",
            "test_qhull_gives_the_pinned_tetrahedralisation[5-1.tif]",
            "failure",
            "Qhull's tetrahedralisation differs from the stored one outside cospherical sets",
        ),
    ]
    status, _, summary = _run(tmp_path, cases, capsys)
    assert status == 1
    assert "1 cross-machine checks failed" in summary


def test_the_tests_allowed_a_cospherical_skip_are_exactly_those_that_call_it():
    """Every test of the two modules that calls the cospherical skip is in `QHULL_LIVE_TESTS`, and only those.

    A test added with the skip but left out of the list fails the CI on every machine where Qhull splits a
    cospherical set otherwise (it happened once: the opt-in configurations' check was not listed).
    """
    import ast
    from pathlib import Path

    calling = set()
    for module in ("test_canonical_tetrahedron_order.py", "test_golden_master.py"):
        tree = ast.parse((Path(__file__).parent / module).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if "skip_where_qhull_differs_inside_cospherical_sets" in names:
                    calling.add(node.name)
    assert calling == set(report.QHULL_LIVE_TESTS)
