"""Read the cross-machine checks out of a pytest JUnit report: warn on Qhull's cospherical choices, fail on the rest.

Run by the CI after the test suite (`.github/workflows/test-suites.yml`):

    python tests/report_cross_machine_checks.py JUNIT_XML SUMMARY_MARKDOWN MODULE.py [MODULE.py ...]

The modules are `tests/test_canonical_tetrahedron_order.py` and `tests/test_golden_master.py`. The
golden masters run on stored tetrahedralisations, so they do not depend on Qhull; four tests
compare with Qhull's own tetrahedralisation instead (`QHULL_LIVE_TESTS`), and where Qhull splits a
set of five or more cospherical lattice points otherwise than the machine the fixtures were made on,
they skip with a message that starts with `COSPHERICAL_SKIP` (after an exact check that the difference
lies only inside such sets; a difference anywhere else fails them). Rules:

* such a skip, of one of those four tests, is reported: a GitHub warning annotation on standard
  output and a table of counts in the job summary; the job does not fail;
* any other skip in these modules (the tests on the stored tetrahedralisation included), or a module
  that ran no test, fails the job (exit status 1);
* failures and errors are already failures of the pytest step; they are counted here too.
"""

import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path

#: Must equal `tests.pinned_tetrahedralisations.COSPHERICAL_SKIP` (tested).
COSPHERICAL_SKIP = "Qhull triangulates the cospherical sets differently here"
#: The tests that use Qhull's own tetrahedralisation and may skip for that reason, and only them.
QHULL_LIVE_TESTS = (
    "test_qhull_gives_the_pinned_tetrahedralisation",
    "test_golden_master_reproduces_fixture_with_qhull",
    "test_qhull_tetrahedron_order_reproduces_the_0_7_default",
    "test_qhull_gives_the_opt_in_configurations_their_stored_tetrahedralisation",
)


def classify(junit: Path, modules: list[str]) -> tuple[dict[str, Counter], list[tuple[str, str]], list[str]]:
    """Per module, the outcome counts; the cospherical skips (test id, message); the other skips, as lines."""
    counts = {Path(m).stem: Counter() for m in modules}
    cospherical, other = [], []
    # The report is the one pytest wrote in the same job, not data from outside.
    for case in ElementTree.parse(junit).iter("testcase"):  # noqa: S314
        module = case.get("classname", "").split(".")[-1]
        if module not in counts:
            continue
        element = next((c for c in case if c.tag in ("skipped", "failure", "error")), None)
        outcome = "passed" if element is None else element.tag
        name = case.get("name", "")
        if outcome == "skipped":
            message = (element.get("message") or "").removeprefix("Skipped: ")
            if name.split("[")[0] in QHULL_LIVE_TESTS and message.startswith(COSPHERICAL_SKIP):
                outcome = "skipped, Qhull differs inside cospherical sets"
                cospherical.append((f"{module}::{name}", message))
            else:
                other.append(f"{module}::{name}: {message}")
        counts[module][outcome] += 1
    return counts, cospherical, other


def main(argv: list[str]) -> int:
    junit, summary, modules = Path(argv[1]), Path(argv[2]), argv[3:]
    counts, cospherical, other = classify(junit, modules)
    lines = ["", "#### Cross-machine checks", ""]
    for module, count in counts.items():
        outcomes = ", ".join(f"{n} {o}" for o, n in sorted(count.items())) or "none ran"
        lines.append(f"cross-machine checks, {module}: {outcomes}")
    if cospherical:
        print(
            f"::warning title=Qhull differs inside cospherical sets::{len(cospherical)} tests skipped: on this "
            "machine Qhull splits sets of cospherical lattice points otherwise than the stored tetrahedralisation "
            "(another Delaunay choice, checked exactly); the golden master on the stored tetrahedralisation ran.",
        )
        warning = f"**Warning: {len(cospherical)} tests skipped because Qhull differs inside cospherical sets.**"
        lines += ["", warning, ""]
        lines += ["| test | reason |", "|---|---|"] + [f"| `{name}` | {message} |" for name, message in cospherical]
    failed = [m for m, c in counts.items() if not sum(c.values())]
    bad = sum(c["failure"] + c["error"] for c in counts.values())
    if other:
        lines += ["", "**Skipped for another reason (fails the job):**", ""] + [f"* `{line}`" for line in other]
    if failed:
        lines.append(f"\n**No test ran in: {', '.join(failed)} (fails the job).**")
    if bad:
        lines.append(f"\n**{bad} cross-machine checks failed.**")
    text = "\n".join(lines) + "\n"
    print(text)
    with summary.open("a") as handle:
        handle.write(text)
    return 1 if (other or failed or bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
