"""
Standalone witness search for mixed Type 2 structure functions in FAS minors.

This module intentionally leaves ``targeted_fas_minor`` and
``sparse_u_monomials`` unchanged. It reuses their public APIs to search full
maximal minors for u-monomials whose differentiated coefficients isolate a
chosen mixed structure function.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import sympy as sp

from sparse_u_monomials import (
    ExponentVector,
    TooManyTermsError,
    differentiate_by_structure_function,
    exponent_vector_to_monomial_expr,
    expr_to_sparse_u_poly,
    iter_sorted_terms,
)
from targeted_fas_minor import DeterminantComputer, FASMinorCalculator


StructureFunctionKey = Tuple[str, Tuple, str, Tuple, str, Tuple]
StructureFunctionSpec = Union[sp.Symbol, str, StructureFunctionKey, Sequence[Tuple]]

_MIXED_TYPE2_SYMBOL_RE = re.compile(
    r"^c\^\{(?P<gk>-?\d+),\((?P<ksrc>-?\d+),(?P<ktgt>-?\d+)\)\}_"
    r"\{\((?P<gl>-?\d+),\((?P<lsrc>-?\d+),(?P<ltgt>-?\d+)\)\),"
    r"\((?P<gi>-?\d+),(?P<vi>-?\d+)\)\}$"
)


@dataclass(frozen=True)
class _ResolvedMixedType2Target:
    target_symbol: sp.Symbol
    target_variants: Tuple[sp.Symbol, ...]
    upper_edge: Tuple[int, Tuple[int, int]]
    lower_edge: Tuple[int, Tuple[int, int]]
    lower_vertex: Tuple[int, int]
    lower_component: int


@dataclass(frozen=True)
class StructureWitness:
    extra_row: Tuple[int, int, int]
    target_symbol: sp.Symbol
    target_variants: Tuple[sp.Symbol, ...]
    u_gens: Tuple[sp.Symbol, ...]
    u_exponents: ExponentVector
    u_monomial: sp.Expr
    coefficient: sp.Expr
    differentiated_expr: Optional[sp.Expr] = None
    minor: Optional[sp.Expr] = None

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "extra_row": list(self.extra_row),
            "target_symbol": str(self.target_symbol),
            "target_variants": [str(symbol) for symbol in self.target_variants],
            "u_gens": [str(symbol) for symbol in self.u_gens],
            "u_exponents": list(self.u_exponents),
            "u_monomial": str(self.u_monomial),
            "coefficient": str(self.coefficient),
            "differentiated_expr": None if self.differentiated_expr is None else str(self.differentiated_expr),
            "minor": None if self.minor is None else str(self.minor),
        }


def _infer_index_type(index: Tuple) -> str:
    if not isinstance(index, tuple) or len(index) != 2:
        raise ValueError(f"Invalid index tuple: {index}")
    _, local_id = index
    if isinstance(local_id, int):
        return "vertex"
    if isinstance(local_id, tuple) and len(local_id) == 2:
        return "edge"
    raise ValueError(f"Unable to infer index type from tuple: {index}")


def _build_type2_symbol(
    upper_edge: Tuple[int, Tuple[int, int]],
    lower_edge: Tuple[int, Tuple[int, int]],
    lower_vertex: Tuple[int, int],
) -> sp.Symbol:
    g_k, edge_k = upper_edge
    g_l, edge_l = lower_edge
    g_i, vertex_i = lower_vertex
    k_src, k_tgt = edge_k
    l_src, l_tgt = edge_l
    symbol_name = f"c^{{{g_k},({k_src},{k_tgt})}}_{{({g_l},({l_src},{l_tgt})),({g_i},{vertex_i})}}"
    return sp.Symbol(symbol_name)


def _parse_mixed_type2_symbol(symbol: Union[sp.Symbol, str]) -> _ResolvedMixedType2Target:
    symbol_name = symbol.name if isinstance(symbol, sp.Symbol) else symbol
    match = _MIXED_TYPE2_SYMBOL_RE.match(symbol_name)
    if match is None:
        raise ValueError(
            "This witness-search module currently supports only mixed Type 2 "
            "structure functions of the form c^k_{l,i} with k,l edges and i a vertex."
        )
    g_k = int(match.group("gk"))
    k_src = int(match.group("ksrc"))
    k_tgt = int(match.group("ktgt"))
    g_l = int(match.group("gl"))
    l_src = int(match.group("lsrc"))
    l_tgt = int(match.group("ltgt"))
    g_i = int(match.group("gi"))
    vertex_i = int(match.group("vi"))
    if g_l != g_i:
        raise ValueError(
            "Mixed Type 2 structure function must have lower edge and lower vertex "
            f"in the same component (got {g_l} and {g_i})."
        )
    target_symbol = sp.Symbol(symbol_name)
    return _ResolvedMixedType2Target(
        target_symbol=target_symbol,
        target_variants=(target_symbol,),
        upper_edge=(g_k, (k_src, k_tgt)),
        lower_edge=(g_l, (l_src, l_tgt)),
        lower_vertex=(g_i, vertex_i),
        lower_component=g_l,
    )


def _resolve_mixed_type2_target(
    structure_function: StructureFunctionSpec,
) -> _ResolvedMixedType2Target:
    if isinstance(structure_function, sp.Symbol):
        return _parse_mixed_type2_symbol(structure_function)
    if isinstance(structure_function, str):
        return _parse_mixed_type2_symbol(structure_function)
    if isinstance(structure_function, tuple) and len(structure_function) == 6:
        index_type_a, val_a, index_type_b, val_b, index_type_c, val_c = structure_function
        if index_type_a != "edge":
            raise ValueError("Mixed Type 2 structure function must have an edge superscript.")
        lower_entries = [(index_type_b, val_b), (index_type_c, val_c)]
        edge_entries = [value for kind, value in lower_entries if kind == "edge"]
        vertex_entries = [value for kind, value in lower_entries if kind == "vertex"]
        if len(edge_entries) != 1 or len(vertex_entries) != 1:
            raise ValueError(
                "Mixed Type 2 structure function must have exactly one edge and one vertex lower index."
            )
        lower_edge = edge_entries[0]
        lower_vertex = vertex_entries[0]
        if lower_edge[0] != lower_vertex[0]:
            raise ValueError(
                "Mixed Type 2 structure function must have lower edge and lower vertex "
                "in the same component."
            )
        target_symbol = _build_type2_symbol(val_a, lower_edge, lower_vertex)
        return _ResolvedMixedType2Target(
            target_symbol=target_symbol,
            target_variants=(target_symbol,),
            upper_edge=val_a,
            lower_edge=lower_edge,
            lower_vertex=lower_vertex,
            lower_component=lower_edge[0],
        )
    if not isinstance(structure_function, (list, tuple)) or len(structure_function) != 3:
        raise ValueError(
            "structure_function must be a SymPy symbol, an exact symbol name string, "
            "a six-tuple structure function key, or a compact [upper, lower1, lower2] list/tuple"
        )

    upper_index, lower_index_a, lower_index_b = structure_function
    if _infer_index_type(upper_index) != "edge":
        raise ValueError("Mixed Type 2 structure function must have an edge superscript.")

    lower_entries = [lower_index_a, lower_index_b]
    edge_entries = [index for index in lower_entries if _infer_index_type(index) == "edge"]
    vertex_entries = [index for index in lower_entries if _infer_index_type(index) == "vertex"]
    if len(edge_entries) != 1 or len(vertex_entries) != 1:
        raise ValueError(
            "Compact mixed Type 2 structure function must have exactly one edge and one vertex lower index."
        )
    lower_edge = edge_entries[0]
    lower_vertex = vertex_entries[0]
    if lower_edge[0] != lower_vertex[0]:
        raise ValueError(
            "Mixed Type 2 structure function must have lower edge and lower vertex in the same component."
        )

    target_symbol = _build_type2_symbol(upper_index, lower_edge, lower_vertex)
    return _ResolvedMixedType2Target(
        target_symbol=target_symbol,
        target_variants=(target_symbol,),
        upper_edge=upper_index,
        lower_edge=lower_edge,
        lower_vertex=lower_vertex,
        lower_component=lower_edge[0],
    )


def build_full_u_gens(calc: FASMinorCalculator) -> Tuple[sp.Symbol, ...]:
    """Stable ordered list of all u symbols: vertices first, then edges."""
    verts = sorted(
        [
            (key, symbol)
            for key, symbol in calc.vertex_variables.items()
            if isinstance(symbol, sp.Symbol)
        ],
        key=lambda item: (item[0][0], item[0][1]),
    )
    edges = sorted(
        [
            (key, symbol)
            for key, symbol in calc.edge_variables.items()
            if isinstance(symbol, sp.Symbol)
        ],
        key=lambda item: (item[0][0], item[0][1][0], item[0][1][1]),
    )
    return tuple(symbol for _, symbol in verts) + tuple(symbol for _, symbol in edges)


def iter_candidate_rows_for_structure_function(
    char_tuples: List[Tuple[int, ...]],
    structure_function: StructureFunctionSpec,
    *,
    candidate_rows: str = "target_component",
    layer_bound: str = "base_range",
) -> Iterable[Tuple[int, int, int]]:
    """Yield candidate extra rows for a mixed Type 2 structure-function witness search."""
    if candidate_rows != "target_component":
        raise ValueError(f"Unsupported candidate_rows mode: {candidate_rows}")
    if layer_bound != "base_range":
        raise ValueError(f"Unsupported layer_bound mode: {layer_bound}")

    target = _resolve_mixed_type2_target(structure_function)
    calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    graph = calc.graphs[target.lower_component]

    for layer in range(1, graph.num_roots + 1):
        for vertex in sorted(graph.vertices):
            if graph.get_vertex_depth(vertex) >= layer:
                yield (target.lower_component, vertex, layer)


def _is_structure_function_symbol(symbol: sp.Symbol) -> bool:
    return symbol.name.startswith("c^{")


def _is_alpha_symbol(symbol: sp.Symbol) -> bool:
    return symbol.name.startswith("α_{")


def _coefficient_matches_isolation_rule(
    coefficient: sp.Expr,
    *,
    target_variants: Tuple[sp.Symbol, ...],
    isolation: str,
) -> bool:
    if coefficient == 0:
        return False
    free_symbols = getattr(coefficient, "free_symbols", set())
    structure_symbols = {symbol for symbol in free_symbols if _is_structure_function_symbol(symbol)}
    allowed_structure_symbols = set(target_variants)

    if isolation == "linear":
        if not structure_symbols <= allowed_structure_symbols:
            return False
    elif isolation == "exact":
        if structure_symbols:
            return False
    else:
        raise ValueError(f"Unsupported isolation mode: {isolation}")

    non_structure_symbols = free_symbols - structure_symbols
    return all(_is_alpha_symbol(symbol) for symbol in non_structure_symbols)


def find_structure_function_witnesses(
    char_tuples: List[Tuple[int, ...]],
    structure_function: StructureFunctionSpec,
    *,
    candidate_rows: str = "target_component",
    layer_bound: str = "base_range",
    diff_order: int = 1,
    isolation: str = "linear",
    max_total_degree: Optional[int] = None,
    max_degree_per_var: Optional[Union[int, Sequence[Optional[int]]]] = None,
    max_terms: Optional[int] = None,
    limit: Optional[int] = None,
    return_minor: bool = False,
    return_diff_expr: bool = False,
) -> List[StructureWitness]:
    """Search full minors for u-monomial witnesses isolating a mixed Type 2 structure function."""
    if diff_order < 0:
        raise ValueError("diff_order must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    target = _resolve_mixed_type2_target(structure_function)
    calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    det = DeterminantComputer(calc)
    u_gens = build_full_u_gens(calc)
    witnesses: List[StructureWitness] = []

    for extra_row in iter_candidate_rows_for_structure_function(
        char_tuples,
        target.target_symbol,
        candidate_rows=candidate_rows,
        layer_bound=layer_bound,
    ):
        minor = det.compute_minor_fast(*extra_row)
        try:
            differentiated = differentiate_by_structure_function(
                minor,
                target.target_symbol,
                order=diff_order,
            )
        except ValueError:
            continue
        if differentiated == 0:
            continue

        sparse_poly = expr_to_sparse_u_poly(
            differentiated,
            u_gens,
            max_total_degree=max_total_degree,
            max_degree_per_var=max_degree_per_var,
            max_terms=max_terms,
        )
        for exponents, coefficient in iter_sorted_terms(sparse_poly, descending=True):
            if not _coefficient_matches_isolation_rule(
                coefficient,
                target_variants=target.target_variants,
                isolation=isolation,
            ):
                continue
            witness = StructureWitness(
                extra_row=extra_row,
                target_symbol=target.target_symbol,
                target_variants=target.target_variants,
                u_gens=u_gens,
                u_exponents=exponents,
                u_monomial=exponent_vector_to_monomial_expr(exponents, u_gens),
                coefficient=coefficient,
                differentiated_expr=differentiated if return_diff_expr else None,
                minor=minor if return_minor else None,
            )
            witnesses.append(witness)
            if limit is not None and len(witnesses) >= limit:
                return witnesses
    return witnesses


def _parse_tuples(s: str) -> List[Tuple[int, ...]]:
    tuples: List[Tuple[int, ...]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        tuple_vals = tuple(int(x.strip()) for x in part.split(","))
        if len(tuple_vals) < 2:
            raise ValueError(f"Tuple must have at least 2 elements: {part}")
        tuples.append(tuple_vals)
    if not tuples:
        raise ValueError("No valid tuples found in input")
    return tuples


def _parse_structure_function_argument(s: str) -> StructureFunctionSpec:
    text = s.strip()
    if not text:
        raise ValueError("Empty structure-function specification")
    if text[0] in "[(":
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid compact structure-function specification: {text}") from exc
    return text


def _parse_max_degree_per_var(
    s: Optional[str],
) -> Optional[Union[int, Tuple[Optional[int], ...]]]:
    if s is None:
        return None
    text = s.strip()
    if not text:
        return None
    if "," not in text:
        return int(text)
    degrees: List[Optional[int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part or part.lower() == "none":
            degrees.append(None)
        else:
            degrees.append(int(part))
    return tuple(degrees)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search FAS maximal minors for u-monomial witnesses isolating a mixed Type 2 structure function.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python structure_witness_search.py \\
      --tuples "2,1,4;2,1,3" \\
      --structure-function "[(0,(0,1)), (1,(0,2)), (1,0)]" \\
      --result-limit 3

The structure-function argument accepts either:
  - an exact symbol name, e.g. c^{0,(0,1)}_{(1,(0,2)),(1,0)}
  - a Python literal compact form, e.g. [(0,(0,1)), (1,(0,2)), (1,0)]
""",
    )
    parser.add_argument("--tuples", required=True, help='Characteristic tuples, e.g. "2,1,4;2,1,3"')
    parser.add_argument(
        "--structure-function",
        required=True,
        help="Target mixed Type 2 structure function as an exact symbol name or compact literal.",
    )
    parser.add_argument("--result-limit", type=int, default=None, help="Maximum number of witnesses to return.")
    parser.add_argument("--max-total-degree", type=int, default=None, help="Optional maximum total u-degree.")
    parser.add_argument(
        "--max-degree-per-var",
        default=None,
        help='Optional max degree bound per u generator, e.g. "2" or "2,1,None,0".',
    )
    parser.add_argument("--max-terms", type=int, default=None, help="Maximum sparse terms before aborting.")
    parser.add_argument("--show-diff", action="store_true", help="Include differentiated expressions in output.")
    parser.add_argument("--show-minor", action="store_true", help="Include full minor expressions in output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        char_tuples = _parse_tuples(args.tuples)
        structure_function = _parse_structure_function_argument(args.structure_function)
        max_degree_per_var = _parse_max_degree_per_var(args.max_degree_per_var)
        witnesses = find_structure_function_witnesses(
            char_tuples,
            structure_function,
            limit=args.result_limit,
            max_total_degree=args.max_total_degree,
            max_degree_per_var=max_degree_per_var,
            max_terms=args.max_terms,
            return_minor=args.show_minor,
            return_diff_expr=args.show_diff,
        )
    except (TooManyTermsError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    if not witnesses:
        print("No witnesses found.")
        return 0

    for idx, witness in enumerate(witnesses, start=1):
        print(
            f"[{idx}] row={witness.extra_row} "
            f"u_monomial={witness.u_monomial} "
            f"coefficient={witness.coefficient}"
        )
        print(json.dumps(witness.to_jsonable(), ensure_ascii=False))
    return 0


__all__ = [
    "StructureWitness",
    "build_full_u_gens",
    "find_structure_function_witnesses",
    "iter_candidate_rows_for_structure_function",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
