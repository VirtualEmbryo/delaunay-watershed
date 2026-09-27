"""The `benchmarking-dataset` cohort this comparison uses: exclusions, groups, undetermined values.

Three independent facts are recorded here, each from a different source, kept separate so
none of them is silently forgotten by an aggregate:

1. **`031` and `038` are excluded from this comparison entirely.** Both masks have had their
   largest cell emptied into the exterior in a single labelling pass (2022-03-11): case
   `031` label 2 holds 62 voxels where an independent voxelisation of the same embryo has
   493 532, and case `038` label 7 holds 26 voxels against 252 092. The ground-truth meshes
   predate the masks (2022-03-08) and still carry both cells intact, so the defect is in the
   two masks, not in the reference data. Regenerating the masks is a separate, parallel
   task; this comparison does not read, repair or delete them, and reports the exclusion by
   case id in every output.
2. **The equilibrium split** (`005`, `015`, `018`, `021`, `043` are not at mechanical
   equilibrium) and **case `004`'s integrity flag** (2 malformed triple-line edges of 539)
   are `foambryo_benchmarks.case_groups`'s facts, restated here rather than imported: this
   package must run from a clean `delaunay-watershed-3d` checkout with no path reaching
   into a sibling `foambryo` repository, and `case_groups` is not part of the installed
   `foambryo` package. `tests/test_mesh_quality_case_metadata_matches_foambryo.py` checks the
   two copies agree whenever a sibling `foambryo` checkout is available.
3. **`median_of_determined`** mirrors `foambryo_benchmarks.metrics.median_of_determined`
   for the same reason: a metric can be structurally undetermined for a specific case (a
   junction graph with no triple line to gauge on, a triple-line quantity on a cell with no
   surviving triple line) and a bare `median`/`nanmedian` would either propagate that as a
   silent `nan` or silently shrink `n`. This refuses unless the caller opts in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

#: Both masks had their largest cell emptied into the exterior in the 2022-03-11 labelling
#: pass; the ground-truth meshes (2022-03-08) are unaffected. Never read, never repaired here.
DAMAGED_MASK_CASE_IDS: frozenset[str] = frozenset({"031", "038"})

#: foambryo_benchmarks.case_groups.NON_EQUILIBRIUM_CASE_IDS, restated (see module docstring).
NON_EQUILIBRIUM_CASE_IDS: frozenset[str] = frozenset({"005", "015", "018", "021", "043"})

#: foambryo_benchmarks.case_groups.INTEGRITY_FLAGGED_CASE_IDS, restated.
INTEGRITY_FLAGGED_CASE_IDS: frozenset[str] = frozenset({"004"})

EQUILIBRIUM = "equilibrium"
NON_EQUILIBRIUM = "non_equilibrium"
GROUP_ORDER: tuple[str, str] = (EQUILIBRIUM, NON_EQUILIBRIUM)


def full_dataset_case_ids(dataset_dir) -> list[str]:  # noqa: ANN001
    """All case ids present in `benchmarking-dataset`, sorted, from its `*_labels_filled.tif` files."""
    from pathlib import Path

    return sorted(p.name.removesuffix("_labels_filled.tif") for p in Path(dataset_dir).glob("*_labels_filled.tif"))


def cohort_case_ids(dataset_dir) -> list[str]:  # noqa: ANN001
    """The 45 cases this comparison runs: the full dataset minus the two damaged masks."""
    return [c for c in full_dataset_case_ids(dataset_dir) if c not in DAMAGED_MASK_CASE_IDS]


def accuracy_cohort_case_ids(dataset_dir) -> list[str]:  # noqa: ANN001
    """The 40 equilibrium cases this comparison's accuracy claims are computed over.

    45 cases actually run, minus the 5 non-equilibrium cases. `004` is included, with its
    integrity flag travelling separately (see `carries_integrity_flag`).
    """
    return [c for c in cohort_case_ids(dataset_dir) if c not in NON_EQUILIBRIUM_CASE_IDS]


def non_equilibrium_cohort_case_ids(dataset_dir) -> list[str]:  # noqa: ANN001
    """The 5 non-equilibrium cases, reported separately, never pooled with the 40."""
    return [c for c in cohort_case_ids(dataset_dir) if c in NON_EQUILIBRIUM_CASE_IDS]


def group_of(case_id: str) -> str:
    """`EQUILIBRIUM` or `NON_EQUILIBRIUM` for a given case id."""
    return NON_EQUILIBRIUM if str(case_id) in NON_EQUILIBRIUM_CASE_IDS else EQUILIBRIUM


def carries_integrity_flag(case_id: str) -> bool:
    """Whether `case_id` is `004` (or any future case so flagged)."""
    return str(case_id) in INTEGRITY_FLAGGED_CASE_IDS


def split_records(records: Sequence[dict], case_key: str = "case") -> dict[str, list[dict]]:
    """Partition per-case records into `{EQUILIBRIUM: [...], NON_EQUILIBRIUM: [...]}`."""
    groups: dict[str, list[dict]] = {group: [] for group in GROUP_ORDER}
    for record in records:
        groups[group_of(record[case_key])].append(record)
    return groups


def require_single_group(records: Sequence[dict], case_key: str = "case") -> str:
    """Assert `records` are all from one equilibrium group; return that group's name.

    Mirrors `foambryo_benchmarks.case_groups.require_single_group`: the structural guard
    that stops an aggregate from silently pooling the 40 equilibrium cases with the 5
    non-equilibrium ones, which would report a data defect as a method's accuracy.

    Raises:
        ValueError: if `records` is empty or spans both groups.
    """
    if not records:
        message = "cannot aggregate an empty record list: the group is undefined"
        raise ValueError(message)
    present = {group_of(record[case_key]) for record in records}
    if len(present) > 1:
        offenders = sorted(record[case_key] for record in records if group_of(record[case_key]) == NON_EQUILIBRIUM)
        message = (
            "refusing to aggregate across the equilibrium split: these cases are not at "
            f"mechanical equilibrium and must be reported separately -> {offenders}. "
            "Use case_metadata.split_records first."
        )
        raise ValueError(message)
    return present.pop()


def median_of_determined(values: Iterable[float], *, allow_partial: bool = False) -> dict:
    """Median over the finite values in `values`, refusing to silently hide the rest.

    Mirrors `foambryo_benchmarks.metrics.median_of_determined`. A value can be legitimately
    undetermined (`nan`) -- e.g. a case whose reconstruction has no triple line to compare, or
    a curvature crossover with no interface below it. Neither `np.nanmedian` (silently
    shrinks `n`) nor a bare `np.median` (propagates one `nan` into the whole aggregate) makes
    that visible; this refuses unless the caller opts in with `allow_partial=True`, and the
    returned counts must travel with the median wherever it is reported.

    Returns:
        dict: `{"median": float | None, "n_determined": int, "n_total": int}`. `median` is
            `None` (not `nan`) when there are zero determined values.

    Raises:
        ValueError: if any value is non-finite and `allow_partial` is not `True`.
    """
    values = list(values)
    n_total = len(values)
    determined = [v for v in values if math.isfinite(v)]
    n_determined = len(determined)
    if n_determined < n_total and not allow_partial:
        message = (
            f"{n_total - n_determined}/{n_total} values are undetermined (non-finite); "
            "pass allow_partial=True to accept a median over the rest, and report "
            "n_determined/n_total alongside it."
        )
        raise ValueError(message)
    if n_determined == 0:
        return {"median": None, "n_determined": 0, "n_total": n_total}
    determined.sort()
    mid = n_determined // 2
    median = determined[mid] if n_determined % 2 else 0.5 * (determined[mid - 1] + determined[mid])
    return {"median": float(median), "n_determined": n_determined, "n_total": n_total}
