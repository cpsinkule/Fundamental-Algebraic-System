"""
Batch exact coefficient search for monomial/extra-row pairs.

This CLI consumes either:
- the summary JSON written by structure_function_search.py --output
- the dedicated task artifact written by structure_function_search.py --task-output

For each distinct (extra_row, monomial_cli) pair, it computes the exact
coefficient of that monomial in the original minor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp

from structure_function_search import build_task_artifact
from targeted_fas_minor import compute_monomial_coefficient, parse_monomial_spec


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class MonomialPairResult:
    extra_row: Tuple[int, int, int]
    monomial_cli: str
    coefficient: sp.Expr
    task: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        task_data = dict(self.task)
        task_data["extra_row"] = list(self.extra_row)
        task_data["coefficient"] = str(self.coefficient)
        task_data["coefficient_srepr"] = sp.srepr(self.coefficient)
        task_data["coefficient_is_zero"] = self.coefficient == 0
        task_data["coefficient_free_symbols"] = sorted(
            str(sym) for sym in getattr(self.coefficient, "free_symbols", set())
        )
        return task_data


@dataclass
class MonomialPairSearchSummary:
    input_path: str
    char_tuples: List[Tuple[int, ...]]
    total_tasks: int
    processed_tasks: int
    results: List[MonomialPairResult]
    elapsed_seconds: float
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "generated_at": _utc_timestamp(),
            "char_tuples": [list(t) for t in self.char_tuples],
            "total_tasks": self.total_tasks,
            "processed_tasks": self.processed_tasks,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "errors": self.errors,
            "results": [result.to_dict() for result in self.results],
        }


def _load_task_artifact(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if "tasks" in payload:
        return payload
    if "findings" in payload:
        summary_payload = payload
        summary = type("SummaryProxy", (), {
            "char_tuples": [tuple(t) for t in summary_payload["char_tuples"]],
            "target_structure_function": summary_payload["target_structure_function"],
            "diff_order": summary_payload["diff_order"],
            "findings": [],
        })()
        for finding in summary_payload["findings"]:
            summary.findings.append(
                type("FindingProxy", (), {
                    "extra_row": tuple(finding["extra_row"]),
                    "u_monomial": finding["u_monomial"],
                    "selected_vars": finding.get("selected_vars", []),
                    "monomial_cli": finding["monomial_cli"],
                    "coefficient": finding["coefficient"],
                    "classification": finding["classification"],
                })()
            )
        return build_task_artifact(summary, source_summary_path=path)
    raise ValueError("Input JSON must be either a structure-function summary or a monomial task artifact")


def run_monomial_pair_search_from_artifact(
    artifact: Dict[str, Any],
    *,
    input_label: str = "<in-memory artifact>",
    progress_callback: Optional[Callable[[int, int, Tuple[int, int, int], str], None]] = None,
) -> MonomialPairSearchSummary:
    char_tuples = [tuple(t) for t in artifact["char_tuples"]]
    tasks = artifact["tasks"]
    errors: List[Dict[str, Any]] = []
    results: List[MonomialPairResult] = []
    t0 = datetime.now(timezone.utc)

    for idx, task in enumerate(tasks, start=1):
        extra_row = tuple(task["extra_row"])
        monomial_cli = task["monomial_cli"]
        if progress_callback:
            progress_callback(idx, len(tasks), extra_row, monomial_cli)
        try:
            monomial_spec = parse_monomial_spec(monomial_cli)
            coefficient = compute_monomial_coefficient(
                char_tuples,
                extra_row,
                monomial_spec,
                match="exact",
            )
        except Exception as exc:
            errors.append(
                {
                    "extra_row": list(extra_row),
                    "monomial_cli": monomial_cli,
                    "error": str(exc),
                }
            )
            continue
        results.append(
            MonomialPairResult(
                extra_row=extra_row,
                monomial_cli=monomial_cli,
                coefficient=coefficient,
                task=task,
            )
        )

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    return MonomialPairSearchSummary(
        input_path=input_label,
        char_tuples=char_tuples,
        total_tasks=len(tasks),
        processed_tasks=len(results),
        results=results,
        elapsed_seconds=elapsed,
        errors=errors,
    )


def run_monomial_pair_search(
    input_path: str,
    *,
    progress_callback: Optional[Callable[[int, int, Tuple[int, int, int], str], None]] = None,
) -> MonomialPairSearchSummary:
    artifact = _load_task_artifact(input_path)
    return run_monomial_pair_search_from_artifact(
        artifact,
        input_label=input_path,
        progress_callback=progress_callback,
    )


def _stderr_progress(current: int, total: int, extra_row: Tuple[int, int, int], monomial_cli: str) -> None:
    print(f"[{current}/{total}] row={extra_row} monomial={monomial_cli}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute exact coefficients for monomial/extra-row search artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python monomial_pair_search.py --input results.json
  python monomial_pair_search.py --input tasks.json --output pair_results.json --quiet
""",
    )
    parser.add_argument("--input", required=True, help="Path to structure-function summary JSON or task artifact JSON.")
    parser.add_argument("--output", "-o", default=None, help="Optional JSON output path.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-task progress output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = run_monomial_pair_search(
        args.input,
        progress_callback=None if args.quiet else _stderr_progress,
    )

    print(f"Input: {summary.input_path}")
    print(f"Total tasks: {summary.total_tasks}")
    print(f"Processed tasks: {summary.processed_tasks}")
    print(f"Elapsed: {summary.elapsed_seconds:.2f}s")

    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for err in summary.errors:
            print(f"  row={err['extra_row']} monomial={err['monomial_cli']} error={err['error']}")

    if summary.results:
        print(f"\nResults ({len(summary.results)}):")
        for result in summary.results:
            print(f"  row={result.extra_row} monomial={result.monomial_cli} -> {result.coefficient}")
    else:
        print("\nNo coefficients computed.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(summary.to_dict(), fp, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
