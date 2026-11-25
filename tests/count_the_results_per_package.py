"""Print, from a pytest JUnit XML report, the test results of dw3d's suite and of dw2d's suite separately.

The CI jobs run both suites in one `pytest tests` (dw2d's is the package `tests/two_dimensional`); a changed skip
count is a finding, and it must be visible per package. Usage: python tests/count_the_results_per_package.py
<junit.xml>. Prints one Markdown line per package. Standard library only.
"""

import sys
import xml.etree.ElementTree as ET
from collections import Counter


def package_of(classname: str) -> str:
    """`dw2d` for a test of tests/two_dimensional, `dw3d` for any other."""
    return "dw2d" if "two_dimensional" in classname.split(".") else "dw3d"


def outcome_of(case: ET.Element) -> str:
    """passed, failed, error, skipped or xfailed, as pytest reports them in JUnit XML."""
    for child in case:
        if child.tag == "failure":
            return "failed"
        if child.tag == "error":
            return "error"
        if child.tag == "skipped":
            return "xfailed" if (child.get("type") or "").endswith("xfail") else "skipped"
    return "passed"


def main() -> None:
    """Count the outcomes per package and print them."""
    tree = ET.parse(sys.argv[1])  # noqa: S314 - a report pytest wrote in the same job
    counts: dict[str, Counter] = {"dw3d": Counter(), "dw2d": Counter()}
    for case in tree.iter("testcase"):
        counts[package_of(case.get("classname", ""))][outcome_of(case)] += 1
    for package, counter in counts.items():
        if counter:
            parts = ", ".join(
                f"{counter[k]} {k}" for k in ("passed", "failed", "error", "skipped", "xfailed") if counter[k]
            )
            print(f"* {package}: {parts}")


if __name__ == "__main__":
    main()
