"""Compute a single b-entry with deferred p(u)-style variable zeroing.

This module mirrors the semantics of ``compute_minor_with_p_vars`` for the
single ``b`` column entry in a requested row:

- build the requested row recursively with full symbolic values
- only after the row is fully constructed, zero all u-variables that are not
  in the global p(u) root-product support, plus any user-specified exceptions

The intended use case is exploring or testing b-entries under the same
post-recursion filtering regime already used for targeted minor searches.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from targeted_fas_minor import (
    DeterminantComputer,
    FASMinorCalculator,
    _parse_tuples,
    format_monomial_spec,
    parse_monomial_spec,
)


def _parse_row_spec(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 row values (graph_idx,vertex,layer), got {len(parts)}"
        )
    return tuple(int(part.strip()) for part in parts)  # type: ignore[return-value]


def _combine_p_with_additional_vars(
    char_tuples: List[Tuple[int, ...]],
    additional_vars: Optional[Sequence[Tuple]] = None,
) -> Dict[Tuple, int]:
    temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    temp_det = DeterminantComputer(temp_calc)
    p_spec = temp_det.base_A_root_product_spec()

    combined_spec: Dict[Tuple, int] = {key: 1 for key in p_spec.keys()}
    if additional_vars:
        for var_key in additional_vars:
            combined_spec[var_key] = 1
    return combined_spec


def compute_b_entry_with_p_vars(
    char_tuples: List[Tuple[int, ...]],
    row: Tuple[int, int, int],
    additional_vars: Optional[Sequence[Tuple]] = None,
) -> sp.Expr:
    """Compute one b-entry keeping only p(u) vars plus requested exceptions.

    Variables not in the global p(u) support or ``additional_vars`` are zeroed
    only after the full row has been recursively constructed.
    """
    combined_spec = _combine_p_with_additional_vars(char_tuples, additional_vars)
    calc = FASMinorCalculator.from_characteristic_tuples(
        char_tuples,
        target_monomial_spec=combined_spec,
    )
    row_expr = calc.get_row(*row)
    return row_expr[0, -1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a single b-entry with p(u)-style deferred zeroing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python b_entry_with_p_vars.py \\
    --tuples "3,1,5;2,1,3" \\
    --row "0,0,2"

  python b_entry_with_p_vars.py \\
    --tuples "3,1,5;2,1,3" \\
    --row "0,0,2" \\
    --keep "v:0,1;e:1,(1,2)"
""",
    )
    parser.add_argument(
        "--tuples",
        required=True,
        help='Characteristic tuples, e.g. "3,1,5;2,1,3"',
    )
    parser.add_argument(
        "--row",
        required=True,
        help='Row as "graph_idx,vertex,layer", e.g. "0,0,2"',
    )
    parser.add_argument(
        "--keep",
        help=(
            'Additional variables to keep beyond p(u), using the same syntax as '
            '--monomial in targeted_fas_minor.py, e.g. "v:0,1;e:1,(1,2)". '
            "Exponents are accepted but ignored."
        ),
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand the final b-entry before printing.",
    )
    parser.add_argument(
        "--show-kept-vars",
        action="store_true",
        help="Print the retained variable set before the expression.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        char_tuples = _parse_tuples(args.tuples)
    except ValueError as exc:
        parser.error(f"Error parsing --tuples: {exc}")

    try:
        row = _parse_row_spec(args.row)
    except ValueError as exc:
        parser.error(f"Error parsing --row: {exc}")

    additional_vars: List[Tuple] = []
    if args.keep:
        try:
            keep_spec = parse_monomial_spec(args.keep)
        except ValueError as exc:
            parser.error(f"Error parsing --keep: {exc}")
        additional_vars = list(keep_spec.keys())

    expr = compute_b_entry_with_p_vars(char_tuples, row, additional_vars)
    if args.expand:
        expr = sp.expand(expr)

    if args.show_kept_vars:
        combined_spec = _combine_p_with_additional_vars(char_tuples, additional_vars)
        print(f"kept_vars={format_monomial_spec(combined_spec)}")

    print(expr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
