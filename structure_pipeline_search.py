"""
One-shot automation for the structure-function to exact-monomial pipeline.

This CLI runs:
1. structure_function_search
2. task artifact generation from search findings
3. exact monomial/extra-row coefficient search

Sparse search and live finding output are enabled by default.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import sympy as sp

from monomial_pair_search import _stderr_progress as _pair_progress
from monomial_pair_search import run_monomial_pair_search_from_artifact
from structure_function_search import (
    _format_char_tuples_arg,
    _parse_tuples,
    _stderr_progress,
    build_task_artifact,
    search_simple_coefficients,
)
from structure_function_search import SimpleCoefficientResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run structure-function search and exact monomial search in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python structure_pipeline_search.py \\
    --tuples "3,1,5;2,1,3" \\
    --structure-function "c^{1,(0,1)}_{(0,(1,2)),(0,2)}"

  python structure_pipeline_search.py \\
    --tuples "3,1,5;2,1,3" \\
    --structure-function "c^{1,(0,1)}_{(0,(1,2)),(0,2)}" \\
    --prefix prop634 --output-dir runs --live-targeted-command
""",
    )
    parser.add_argument(
        "--tuples", required=True,
        help='Characteristic tuples, semicolon-separated (e.g., "3,1,5;2,1,3")',
    )
    parser.add_argument(
        "--structure-function", required=True,
        help="Target structure function symbol name.",
    )
    parser.add_argument(
        "--diff-order", type=int, default=1,
        help="Differentiation order (default: 1).",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for summary, task, and pair-results JSON outputs.",
    )
    parser.add_argument(
        "--prefix", default="pipeline",
        help="Prefix for generated output files.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output from both pipeline stages.",
    )
    parser.add_argument(
        "--live-targeted-command",
        action="store_true",
        help="For live search findings, print copy-paste targeted_fas_minor.py commands.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Disable the default sparse structure-function search.",
    )
    parser.add_argument(
        "--no-live", action="store_true",
        help="Disable the default live finding output during structure-function search.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    char_tuples = _parse_tuples(args.tuples)
    sf_symbol = sp.Symbol(args.structure_function)
    use_sparse = not args.dense
    live_enabled = not args.no_live

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.json")
    task_path = os.path.join(args.output_dir, f"{args.prefix}_tasks.json")
    pair_results_path = os.path.join(args.output_dir, f"{args.prefix}_pair_results.json")

    search_progress = None if args.quiet else _stderr_progress
    pair_progress = None if args.quiet else _pair_progress

    finding_count = [0]
    tuples_arg = _format_char_tuples_arg(char_tuples)

    def _live_finding(result: SimpleCoefficientResult) -> None:
        finding_count[0] += 1
        if args.live_targeted_command:
            row_arg = ",".join(str(value) for value in result.extra_row)
            print(
                f'python targeted_fas_minor.py --tuples "{tuples_arg}" '
                f'--row "{row_arg}" --monomial "{result.monomial_cli}" --coeff-mode exact',
                flush=True,
            )
            return
        print(
            f"  [{finding_count[0]}] Row {result.extra_row}: "
            f"{result.u_monomial} vars={result.selected_vars_literal()} "
            f"-> {result.coefficient} [{result.classification}]",
            flush=True,
        )

    search_summary = search_simple_coefficients(
        char_tuples,
        sf_symbol,
        diff_order=args.diff_order,
        use_sparse=use_sparse,
        progress_callback=search_progress,
        finding_callback=_live_finding if live_enabled else None,
    )

    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(search_summary.to_dict(), fp, indent=2)

    task_artifact = build_task_artifact(search_summary, source_summary_path=summary_path)
    with open(task_path, "w", encoding="utf-8") as fp:
        json.dump(task_artifact, fp, indent=2)

    pair_summary = run_monomial_pair_search_from_artifact(
        task_artifact,
        input_label=task_path,
        progress_callback=pair_progress,
    )
    with open(pair_results_path, "w", encoding="utf-8") as fp:
        json.dump(pair_summary.to_dict(), fp, indent=2)

    print(f"System: {search_summary.char_tuples}")
    print(f"Target: {search_summary.target_structure_function}")
    print(f"Diff order: {search_summary.diff_order}")
    print(f"Sparse search: {'yes' if use_sparse else 'no'}")
    print(f"Live search output: {'yes' if live_enabled else 'no'}")
    print(f"Extra rows searched: {search_summary.total_extra_rows_searched}")
    print(f"Total monomials examined: {search_summary.total_monomials_examined}")
    print(f"Simple coefficients found: {search_summary.total_simple_found}")
    print(f"Deduplicated tasks: {task_artifact['total_tasks']}")
    print(f"Exact coefficients computed: {pair_summary.processed_tasks}")
    print(f"Exact-step errors: {len(pair_summary.errors)}")
    print(f"Search elapsed: {search_summary.elapsed_seconds:.2f}s")
    print(f"Exact elapsed: {pair_summary.elapsed_seconds:.2f}s")
    print(f"Summary written to {summary_path}")
    print(f"Task artifact written to {task_path}")
    print(f"Pair results written to {pair_results_path}")

    if search_summary.errors:
        print(f"\nSearch errors ({len(search_summary.errors)}):")
        for err in search_summary.errors:
            print(f"  Row {err['extra_row']}: [{err['stage']}] {err['error']}")

    if pair_summary.errors:
        print(f"\nExact-step errors ({len(pair_summary.errors)}):")
        for err in pair_summary.errors:
            print(
                f"  row={err['extra_row']} monomial={err['monomial_cli']} "
                f"error={err['error']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
