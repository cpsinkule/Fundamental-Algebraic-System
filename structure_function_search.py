"""
Automated search for simple structure function coefficients in FAS minors.

For the fixed system with characteristic tuples (3,1,5) and (2,1,3), this
module iterates over all candidate extra rows from Gamma_0, computes each
maximal minor (keeping ALL u-variables), differentiates by a target structure
function, and reports monomials whose resulting coefficients are "simple"
(purely numeric or alpha-only).

Library usage:
    from structure_function_search import search_simple_coefficients
    summary = search_simple_coefficients(
        [(3, 1, 5), (2, 1, 3)],
        structure_function=[(0, (0, 1)), (0, 0), (0, 1)],  # compact spec
    )
    for f in summary.findings:
        print(f.extra_row, f.u_monomial, f.coefficient, f.classification)

CLI usage:
    python structure_function_search.py \\
        --tuples "3,1,5;2,1,3" \\
        --structure-function "c^{0,(0,1)}_{(0,0),(0,1)}" \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp

from sparse_u_monomials import (
    differentiate_by_structure_function,
    structure_function_symbol,
)
from targeted_fas_minor import ComponentGraph, compute_minor_with_p_vars


# ----------------------------- Type aliases ----------------------------------

StructureFunctionSpec = Union[sp.Symbol, str, Sequence[Tuple]]


# ----------------------------- Data classes ----------------------------------

@dataclass(frozen=True)
class SimpleCoefficientResult:
    """A monomial whose differentiated coefficient is simple."""

    extra_row: Tuple[int, int, int]
    u_monomial: sp.Expr
    coefficient: sp.Expr
    classification: str  # "numeric" or "alpha_only"

    def to_dict(self) -> dict:
        return {
            "extra_row": list(self.extra_row),
            "u_monomial": str(self.u_monomial),
            "coefficient": str(self.coefficient),
            "coefficient_srepr": sp.srepr(self.coefficient),
            "classification": self.classification,
        }


@dataclass
class SearchSummary:
    """Aggregate results from a full search run."""

    char_tuples: List[Tuple[int, ...]]
    target_structure_function: str
    diff_order: int
    total_extra_rows_searched: int
    total_monomials_examined: int
    total_simple_found: int
    findings: List[SimpleCoefficientResult]
    elapsed_seconds: float
    errors: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "char_tuples": [list(t) for t in self.char_tuples],
            "target_structure_function": self.target_structure_function,
            "diff_order": self.diff_order,
            "total_extra_rows_searched": self.total_extra_rows_searched,
            "total_monomials_examined": self.total_monomials_examined,
            "total_simple_found": self.total_simple_found,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "errors": self.errors,
            "findings": [f.to_dict() for f in self.findings],
        }


# ----------------------------- Helpers ---------------------------------------

def _is_structure_function_symbol(symbol: sp.Symbol) -> bool:
    return symbol.name.startswith("c^{")


def _is_alpha_symbol(symbol: sp.Symbol) -> bool:
    return symbol.name.startswith("α_{") or symbol.name.startswith("α_")


def is_simple_coefficient(expr: sp.Expr) -> Tuple[bool, str]:
    """Check if a coefficient is simple: numeric or alpha-only.

    Returns:
        (is_simple, classification) where classification is one of
        "numeric", "alpha_only", or "not_simple".
    """
    if expr.is_number:
        return True, "numeric"
    free = expr.free_symbols
    if any(_is_structure_function_symbol(s) for s in free):
        return False, "not_simple"
    if all(_is_alpha_symbol(s) for s in free):
        return True, "alpha_only"
    return False, "not_simple"


# ----------------------- Row complement enumeration --------------------------

def enumerate_row_complements(
    char_tuples: List[Tuple[int, ...]],
    component: int = 0,
) -> List[Tuple[int, int, int]]:
    """Enumerate candidate extra rows from a single component.

    For the specified component, for each layer s in {1, ..., omega+1},
    collects vertices v where depth(v) < s.

    Args:
        char_tuples: Characteristic tuples for all components.
        component: Which component to enumerate complements for (default 0).

    Returns:
        Sorted list of (graph_idx, vertex, layer) tuples.
    """
    ct = char_tuples[component]
    graph = ComponentGraph.from_characteristic_tuple(ct)
    omega_plus_1 = graph.num_roots  # num_roots = omega + 1

    complements = []
    for layer in range(1, omega_plus_1 + 1):
        for vertex in sorted(graph.vertices):
            if graph.vertex_depths[vertex] < layer:
                complements.append((component, vertex, layer))
    return complements


# ----------------------- Build all-vars list ---------------------------------

def build_all_vars(
    char_tuples: List[Tuple[int, ...]],
) -> List[Tuple]:
    """Build additional_vars list containing every u-variable.

    Passing this to compute_minor_with_p_vars effectively disables zeroing,
    keeping all vertex and edge variables alive in the minor.
    """
    all_vars: List[Tuple] = []
    for g_idx, ct in enumerate(char_tuples):
        graph = ComponentGraph.from_characteristic_tuple(ct)
        for vertex in sorted(graph.vertices):
            all_vars.append(("vertex", g_idx, vertex))
        for edge in graph.edges:
            all_vars.append(("edge", g_idx, edge))
    return all_vars


# ----------------------------- Main search -----------------------------------

def search_simple_coefficients(
    char_tuples: List[Tuple[int, ...]],
    structure_function: StructureFunctionSpec,
    *,
    diff_order: int = 1,
    extra_rows: Optional[List[Tuple[int, int, int]]] = None,
    progress_callback: Optional[Callable[[int, int, Tuple[int, int, int]], None]] = None,
) -> SearchSummary:
    """Search minors for monomials with simple differentiated coefficients.

    For each candidate extra row from Gamma_0:
    1. Compute the full minor (all u-variables kept).
    2. Differentiate by the target structure function.
    3. Expand into Poly over u-generators.
    4. Check each coefficient for simplicity.

    Args:
        char_tuples: System characteristic tuples.
        structure_function: Target structure function (symbol, string, or
            compact index spec accepted by sparse_u_monomials).
        diff_order: Differentiation order (default 1).
        extra_rows: Explicit list of extra rows. If None, uses
            enumerate_row_complements for component 0.
        progress_callback: Optional (current, total, extra_row) -> None.

    Returns:
        SearchSummary with all findings.
    """
    t0 = time.monotonic()

    if extra_rows is None:
        extra_rows = enumerate_row_complements(char_tuples, component=0)

    all_vars = build_all_vars(char_tuples)

    findings: List[SimpleCoefficientResult] = []
    total_monomials = 0
    errors: List[dict] = []

    for idx, row in enumerate(extra_rows):
        if progress_callback:
            progress_callback(idx + 1, len(extra_rows), row)

        # Step 1: compute minor with all variables kept
        try:
            minor, u_gens = compute_minor_with_p_vars(
                char_tuples,
                row,
                additional_vars=all_vars,
                return_u_gens=True,
            )
        except Exception as exc:
            errors.append({"extra_row": list(row), "stage": "minor", "error": str(exc)})
            continue

        if minor == 0:
            continue

        # Step 2: differentiate by the target structure function
        try:
            diff_expr = differentiate_by_structure_function(
                minor, structure_function, order=diff_order,
            )
        except ValueError:
            # Structure function not present in this minor
            continue

        if diff_expr == 0:
            continue

        # Step 3: expand into Poly to extract monomials and coefficients
        try:
            expanded = sp.expand(diff_expr)
            poly = sp.Poly(expanded, *u_gens, domain="EX")
        except Exception as exc:
            errors.append({"extra_row": list(row), "stage": "poly", "error": str(exc)})
            continue

        # Step 4: check each coefficient
        for monom_tuple, coeff in poly.as_dict().items():
            total_monomials += 1
            simple, classification = is_simple_coefficient(coeff)
            if simple:
                # Reconstruct monomial expression
                u_monomial = sp.Integer(1)
                for sym, exp in zip(u_gens, monom_tuple):
                    if exp:
                        u_monomial *= sym ** exp
                findings.append(SimpleCoefficientResult(
                    extra_row=row,
                    u_monomial=u_monomial,
                    coefficient=coeff,
                    classification=classification,
                ))

    elapsed = time.monotonic() - t0

    return SearchSummary(
        char_tuples=list(char_tuples),
        target_structure_function=str(structure_function),
        diff_order=diff_order,
        total_extra_rows_searched=len(extra_rows),
        total_monomials_examined=total_monomials,
        total_simple_found=len(findings),
        findings=findings,
        elapsed_seconds=elapsed,
        errors=errors,
    )


# --------------------------------- CLI ---------------------------------------

def _parse_tuples(s: str) -> List[Tuple[int, ...]]:
    tuples: List[Tuple[int, ...]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        values = tuple(int(x.strip()) for x in part.split(","))
        tuples.append(values)
    return tuples


def _stderr_progress(current: int, total: int, extra_row: Tuple[int, int, int]) -> None:
    sys.stderr.write(f"\r[{current}/{total}] Processing extra row {extra_row}...")
    sys.stderr.flush()
    if current == total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search FAS minors for simple structure function coefficients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python structure_function_search.py \\
    --tuples "3,1,5;2,1,3" \\
    --structure-function "c^{0,(0,1)}_{(0,0),(0,1)}"

  python structure_function_search.py \\
    --tuples "3,1,5;2,1,3" \\
    --structure-function "c^{0,(0,1)}_{(0,0),(0,1)}" \\
    --output results.json --quiet
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
        "--output", "-o", default=None,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    char_tuples = _parse_tuples(args.tuples)
    sf_symbol = sp.Symbol(args.structure_function)

    callback = None if args.quiet else _stderr_progress

    summary = search_simple_coefficients(
        char_tuples,
        sf_symbol,
        diff_order=args.diff_order,
        progress_callback=callback,
    )

    # Print human-readable summary
    print(f"System: {summary.char_tuples}")
    print(f"Target: {summary.target_structure_function}")
    print(f"Diff order: {summary.diff_order}")
    print(f"Extra rows searched: {summary.total_extra_rows_searched}")
    print(f"Total monomials examined: {summary.total_monomials_examined}")
    print(f"Simple coefficients found: {summary.total_simple_found}")
    print(f"Elapsed: {summary.elapsed_seconds:.2f}s")

    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for err in summary.errors:
            print(f"  Row {err['extra_row']}: [{err['stage']}] {err['error']}")

    if summary.findings:
        print(f"\nFindings ({summary.total_simple_found}):")
        for f in summary.findings:
            print(f"  Row {f.extra_row}: {f.u_monomial} -> {f.coefficient} [{f.classification}]")
    else:
        print("\nNo simple coefficients found.")

    if args.output:
        with open(args.output, "w") as fp:
            json.dump(summary.to_dict(), fp, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


__all__ = [
    "SimpleCoefficientResult",
    "SearchSummary",
    "enumerate_row_complements",
    "build_all_vars",
    "is_simple_coefficient",
    "search_simple_coefficients",
]

if __name__ == "__main__":
    raise SystemExit(main())
