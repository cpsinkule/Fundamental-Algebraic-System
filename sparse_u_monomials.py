"""
Sparse monomial extraction for SymPy expressions over selected u-generators.

This module leaves the existing FAS determinant code untouched. It provides a
separate path that treats the selected u-variables as polynomial generators and
everything else as coefficient data. That avoids the global normalization cost
of ``sympy.Poly(expr, *u_gens)`` on large nested expressions.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp

from targeted_fas_minor import DeterminantComputer, FASMinorCalculator, compute_minor_with_p_vars


ExponentVector = Tuple[int, ...]
SparsePolynomial = Dict[ExponentVector, sp.Expr]
StructureFunctionKey = Tuple[str, Tuple, str, Tuple, str, Tuple]


class TooManyTermsError(RuntimeError):
    """Raised when sparse expansion exceeds the configured term limit."""


def _zero_vector(length: int) -> ExponentVector:
    return (0,) * length


def _normalize_max_degrees(
    u_gens: Sequence[sp.Symbol],
    max_degree_per_var: Optional[int | Sequence[Optional[int]]],
) -> Optional[Tuple[Optional[int], ...]]:
    if max_degree_per_var is None:
        return None
    if isinstance(max_degree_per_var, int):
        if max_degree_per_var < 0:
            raise ValueError("max_degree_per_var must be non-negative")
        return tuple(max_degree_per_var for _ in u_gens)
    max_degrees = tuple(max_degree_per_var)
    if len(max_degrees) != len(u_gens):
        raise ValueError(
            f"max_degree_per_var length ({len(max_degrees)}) must match "
            f"u_gens length ({len(u_gens)})"
        )
    for degree in max_degrees:
        if degree is not None and degree < 0:
            raise ValueError("max_degree_per_var entries must be non-negative or None")
    return max_degrees


def _add_term(result: SparsePolynomial, exponents: ExponentVector, coeff: sp.Expr) -> None:
    if coeff == 0:
        return
    updated = result.get(exponents, sp.Integer(0)) + coeff
    if updated == 0:
        result.pop(exponents, None)
        return
    result[exponents] = updated


def _check_term_limit(poly: SparsePolynomial, max_terms: Optional[int]) -> None:
    if max_terms is not None and len(poly) > max_terms:
        raise TooManyTermsError(
            f"Sparse expansion exceeded max_terms={max_terms}. "
            "Add stronger pruning or keep fewer u-variables."
        )


def _within_degree_bounds(
    exponents: ExponentVector,
    *,
    max_total_degree: Optional[int],
    max_degrees: Optional[Tuple[Optional[int], ...]],
) -> bool:
    if max_total_degree is not None and sum(exponents) > max_total_degree:
        return False
    if max_degrees is not None:
        for exp, bound in zip(exponents, max_degrees):
            if bound is not None and exp > bound:
                return False
    return True


def add_sparse_polynomials(
    left: SparsePolynomial,
    right: SparsePolynomial,
    *,
    max_terms: Optional[int] = None,
) -> SparsePolynomial:
    """Add two sparse polynomials."""
    result = dict(left)
    for exponents, coeff in right.items():
        _add_term(result, exponents, coeff)
    _check_term_limit(result, max_terms)
    return result


def multiply_sparse_polynomials(
    left: SparsePolynomial,
    right: SparsePolynomial,
    *,
    max_total_degree: Optional[int] = None,
    max_degree_per_var: Optional[int | Sequence[Optional[int]]] = None,
    max_terms: Optional[int] = None,
) -> SparsePolynomial:
    """Multiply two sparse polynomials with optional degree pruning."""
    if not left or not right:
        return {}
    nvars = len(next(iter(left.keys())))
    max_degrees = _normalize_max_degrees([sp.Symbol("_")] * nvars, max_degree_per_var)
    result: SparsePolynomial = {}
    for left_exps, left_coeff in left.items():
        for right_exps, right_coeff in right.items():
            exponents = tuple(a + b for a, b in zip(left_exps, right_exps))
            if not _within_degree_bounds(
                exponents,
                max_total_degree=max_total_degree,
                max_degrees=max_degrees,
            ):
                continue
            _add_term(result, exponents, left_coeff * right_coeff)
        _check_term_limit(result, max_terms)
    return result


def sparse_pow(
    poly: SparsePolynomial,
    exponent: int,
    *,
    max_total_degree: Optional[int] = None,
    max_degree_per_var: Optional[int | Sequence[Optional[int]]] = None,
    max_terms: Optional[int] = None,
) -> SparsePolynomial:
    """Raise a sparse polynomial to a non-negative integer power."""
    if exponent < 0:
        raise ValueError("Exponent must be non-negative")
    nvars = len(next(iter(poly.keys()))) if poly else 0
    result: SparsePolynomial = {_zero_vector(nvars): sp.Integer(1)}
    base = poly
    power = exponent
    while power > 0:
        if power & 1:
            result = multiply_sparse_polynomials(
                result,
                base,
                max_total_degree=max_total_degree,
                max_degree_per_var=max_degree_per_var,
                max_terms=max_terms,
            )
        power >>= 1
        if power:
            base = multiply_sparse_polynomials(
                base,
                base,
                max_total_degree=max_total_degree,
                max_degree_per_var=max_degree_per_var,
                max_terms=max_terms,
            )
    return result


def expr_to_sparse_u_poly(
    expr: sp.Expr,
    u_gens: Sequence[sp.Symbol],
    *,
    max_total_degree: Optional[int] = None,
    max_degree_per_var: Optional[int | Sequence[Optional[int]]] = None,
    max_terms: Optional[int] = None,
) -> SparsePolynomial:
    """
    Convert a SymPy expression into a sparse polynomial in the selected u-generators.

    Any symbol or expression not involving ``u_gens`` is treated as part of the
    coefficient ring.
    """
    u_gens = tuple(u_gens)
    max_degrees = _normalize_max_degrees(u_gens, max_degree_per_var)
    u_index = {sym: idx for idx, sym in enumerate(u_gens)}
    u_set = set(u_gens)
    zero_vector = _zero_vector(len(u_gens))
    memo: Dict[sp.Expr, SparsePolynomial] = {}

    def visit(node: sp.Expr) -> SparsePolynomial:
        if node in memo:
            return memo[node]
        free_symbols = getattr(node, "free_symbols", set())
        if not (free_symbols & u_set):
            result = {zero_vector: node} if node != 0 else {}
            memo[node] = result
            return result
        if node.is_Symbol:
            idx = u_index[node]
            exponents = [0] * len(u_gens)
            exponents[idx] = 1
            result = {tuple(exponents): sp.Integer(1)}
            memo[node] = result
            return result
        if node.is_Add:
            result: SparsePolynomial = {}
            for arg in node.args:
                result = add_sparse_polynomials(result, visit(arg), max_terms=max_terms)
            memo[node] = result
            return result
        if node.is_Mul:
            result = {zero_vector: sp.Integer(1)}
            for arg in node.args:
                result = multiply_sparse_polynomials(
                    result,
                    visit(arg),
                    max_total_degree=max_total_degree,
                    max_degree_per_var=max_degrees,
                    max_terms=max_terms,
                )
                if not result:
                    break
            memo[node] = result
            return result
        if node.is_Pow:
            base, exponent = node.as_base_exp()
            if not (getattr(base, "free_symbols", set()) & u_set):
                result = {zero_vector: node}
                memo[node] = result
                return result
            if exponent.is_Integer and int(exponent) >= 0:
                result = sparse_pow(
                    visit(base),
                    int(exponent),
                    max_total_degree=max_total_degree,
                    max_degree_per_var=max_degrees,
                    max_terms=max_terms,
                )
                memo[node] = result
                return result
            raise ValueError(f"Expression is not polynomial in selected generators: {node}")
        raise ValueError(f"Unsupported expression node while extracting u-monomials: {node.func}")

    return visit(expr)


def sparse_poly_to_expr(poly: SparsePolynomial, u_gens: Sequence[sp.Symbol]) -> sp.Expr:
    """Reconstruct a SymPy expression from a sparse polynomial."""
    result = sp.Integer(0)
    for exponents, coeff in poly.items():
        term = coeff
        for sym, exponent in zip(u_gens, exponents):
            if exponent:
                term *= sym ** exponent
        result += term
    return result


def exponent_vector_to_monomial_expr(exponents: ExponentVector, u_gens: Sequence[sp.Symbol]) -> sp.Expr:
    """Convert an exponent vector into the corresponding monomial expression."""
    term = sp.Integer(1)
    for sym, exponent in zip(u_gens, exponents):
        if exponent:
            term *= sym ** exponent
    return term


def filter_sparse_poly(
    poly: SparsePolynomial,
    *,
    must_divide: Optional[Sequence[int]] = None,
    max_total_degree: Optional[int] = None,
    max_degree_per_var: Optional[int | Sequence[Optional[int]]] = None,
    max_terms: Optional[int] = None,
) -> SparsePolynomial:
    """Filter sparse polynomial terms by divisibility and degree bounds."""
    if not poly:
        return {}
    nvars = len(next(iter(poly.keys())))
    max_degrees = _normalize_max_degrees([sp.Symbol("_")] * nvars, max_degree_per_var)
    divide_vector = tuple(must_divide) if must_divide is not None else None
    if divide_vector is not None and len(divide_vector) != nvars:
        raise ValueError("must_divide length must match the sparse polynomial exponent length")
    result: SparsePolynomial = {}
    for exponents, coeff in poly.items():
        if divide_vector is not None:
            if any(exp < required for exp, required in zip(exponents, divide_vector)):
                continue
        if not _within_degree_bounds(
            exponents,
            max_total_degree=max_total_degree,
            max_degrees=max_degrees,
        ):
            continue
        result[exponents] = coeff
        _check_term_limit(result, max_terms)
    return result


def iter_sorted_terms(
    poly: SparsePolynomial,
    *,
    descending: bool = True,
) -> List[Tuple[ExponentVector, sp.Expr]]:
    """Return sparse terms sorted by total degree, then lexicographically."""
    items = list(poly.items())
    items.sort(key=lambda item: (sum(item[0]), item[0]), reverse=descending)
    return items


def format_sparse_terms(
    poly: SparsePolynomial,
    u_gens: Sequence[sp.Symbol],
    *,
    descending: bool = True,
) -> List[Tuple[sp.Expr, sp.Expr]]:
    """Format sparse terms as ``(monomial_expr, coefficient)`` pairs."""
    return [
        (exponent_vector_to_monomial_expr(exponents, u_gens), coeff)
        for exponents, coeff in iter_sorted_terms(poly, descending=descending)
    ]


def _infer_index_type(index: Tuple) -> str:
    if not isinstance(index, tuple) or len(index) != 2:
        raise ValueError(f"Invalid index tuple: {index}")
    _, local_id = index
    if isinstance(local_id, int):
        return "vertex"
    if isinstance(local_id, tuple) and len(local_id) == 2:
        return "edge"
    raise ValueError(f"Unable to infer index type from tuple: {index}")


def _structure_function_symbol_from_compact_indices(
    upper_index: Tuple,
    lower_index_a: Tuple,
    lower_index_b: Tuple,
    *,
    strict: bool,
) -> Optional[sp.Symbol]:
    upper_type = _infer_index_type(upper_index)
    lower_type_a = _infer_index_type(lower_index_a)
    lower_type_b = _infer_index_type(lower_index_b)

    if upper_type == "edge" and lower_type_a == "vertex" and lower_type_b == "vertex":
        g_k, edge_k = upper_index
        g_i, vertex_i = lower_index_a
        g_j, vertex_j = lower_index_b
        k_src, k_tgt = edge_k
        symbol_name = f"c^{{{g_k},({k_src},{k_tgt})}}_{{({g_i},{vertex_i}),({g_j},{vertex_j})}}"
        return sp.Symbol(symbol_name)

    if upper_type == "edge" and lower_type_a == "edge" and lower_type_b == "vertex":
        g_k, edge_k = upper_index
        g_l, edge_l = lower_index_a
        g_i, vertex_i = lower_index_b
        k_src, k_tgt = edge_k
        l_src, l_tgt = edge_l
        symbol_name = f"c^{{{g_k},({k_src},{k_tgt})}}_{{({g_l},({l_src},{l_tgt})),({g_i},{vertex_i})}}"
        return sp.Symbol(symbol_name)

    if upper_type == "vertex" and lower_type_a == "vertex" and lower_type_b == "vertex":
        g_l, vertex_l = upper_index
        g_w, vertex_w = lower_index_a
        g_v, vertex_v = lower_index_b
        symbol_name = f"c^{{({g_l},{vertex_l})}}_{{({g_w},{vertex_w}),({g_v},{vertex_v})}}"
        return sp.Symbol(symbol_name)

    if strict:
        raise ValueError(
            "Unsupported compact structure function signature. "
            "Expected upper/lower indices matching one of the supported "
            "signatures: edge-vertex-vertex, edge-edge-vertex, or vertex-vertex-vertex."
        )
    return None


def _is_compact_structure_function_spec(structure_function: object) -> bool:
    if not isinstance(structure_function, (list, tuple)) or len(structure_function) != 3:
        return False
    return all(isinstance(item, tuple) and len(item) == 2 for item in structure_function)


def _parse_structure_function_string(
    name: str,
) -> Optional[Tuple[Tuple, Tuple, Tuple]]:
    """Parse a structure function symbol name into compact index spec.

    Recognizes the three formats produced by
    ``_structure_function_symbol_from_compact_indices``:

    Type 1 (edge-vertex-vertex):
        ``c^{g,(s,t)}_{(g,v),(g,v)}``
    Type 2 (edge-edge-vertex):
        ``c^{g,(s,t)}_{(g,(s,t)),(g,v)}``  — note nested parens for edge index
    Type 3 (vertex-vertex-vertex):
        ``c^{(g,v)}_{(g,v),(g,v)}``

    Returns compact spec ``(upper, lower_a, lower_b)`` or *None* if the
    string does not match any known format.
    """
    import re

    if not name.startswith("c^{"):
        return None

    # Split into upper and lower parts at "}_{" boundary
    m = re.match(r'^c\^{(.+)}_\{(.+)\}$', name)
    if not m:
        return None

    upper_str = m.group(1)
    lower_str = m.group(2)

    def _parse_index(s: str) -> Tuple:
        """Parse a single index like '(0,2)' or '(0,(0,1))' into a tuple."""
        s = s.strip()
        if not s.startswith('(') or not s.endswith(')'):
            raise ValueError(f"Cannot parse index: {s}")
        inner = s[1:-1]
        # Check for nested tuple: (g,(s,t))
        nested = re.match(r'^(\d+),\((\d+),(\d+)\)$', inner)
        if nested:
            g = int(nested.group(1))
            a = int(nested.group(2))
            b = int(nested.group(3))
            return (g, (a, b))
        # Simple pair: (g,v)
        simple = re.match(r'^(\d+),(\d+)$', inner)
        if simple:
            return (int(simple.group(1)), int(simple.group(2)))
        raise ValueError(f"Cannot parse index: {s}")

    def _parse_upper(s: str) -> Tuple:
        """Parse the upper index: either 'g,(s,t)' (edge) or '(g,v)' (vertex)."""
        s = s.strip()
        # Edge upper: g,(s,t)
        edge_m = re.match(r'^(\d+),\((\d+),(\d+)\)$', s)
        if edge_m:
            g = int(edge_m.group(1))
            src = int(edge_m.group(2))
            tgt = int(edge_m.group(3))
            return (g, (src, tgt))
        # Vertex upper: (g,v)
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            parts = re.match(r'^(\d+),(\d+)$', inner)
            if parts:
                return (int(parts.group(1)), int(parts.group(2)))
        raise ValueError(f"Cannot parse upper index: {s}")

    def _split_lower(s: str) -> Tuple[str, str]:
        """Split lower indices at the top-level comma.

        Handles nested parens like '(0,(0,1)),(0,2)'.
        """
        depth = 0
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                return s[:i], s[i + 1:]
        raise ValueError(f"Cannot split lower indices: {s}")

    try:
        upper = _parse_upper(upper_str)
        lower_a_str, lower_b_str = _split_lower(lower_str)
        lower_a = _parse_index(lower_a_str)
        lower_b = _parse_index(lower_b_str)
        return (upper, lower_a, lower_b)
    except ValueError:
        return None


def _resolve_structure_function_variants(
    structure_function: sp.Symbol | str | StructureFunctionKey | Sequence[Tuple],
) -> List[sp.Symbol]:
    if isinstance(structure_function, sp.Symbol):
        parsed = _parse_structure_function_string(structure_function.name)
        if parsed is not None:
            return _resolve_structure_function_variants(parsed)
        return [structure_function]
    if isinstance(structure_function, str):
        parsed = _parse_structure_function_string(structure_function)
        if parsed is not None:
            return _resolve_structure_function_variants(parsed)
        return [sp.Symbol(structure_function)]
    if _is_compact_structure_function_spec(structure_function):
        upper_index, lower_index_a, lower_index_b = structure_function
        primary = _structure_function_symbol_from_compact_indices(
            upper_index,
            lower_index_a,
            lower_index_b,
            strict=False,
        )
        swapped = _structure_function_symbol_from_compact_indices(
            upper_index,
            lower_index_b,
            lower_index_a,
            strict=False,
        )
        variants: List[sp.Symbol] = []
        if primary is not None:
            variants.append(primary)
        if swapped is not None and swapped != primary:
            variants.append(swapped)
        if not variants:
            _structure_function_symbol_from_compact_indices(
                upper_index,
                lower_index_a,
                lower_index_b,
                strict=True,
            )
        return variants
    if not isinstance(structure_function, tuple) or len(structure_function) != 6:
        raise ValueError(
            "structure_function must be a SymPy symbol, a symbol name string, "
            "a compact [upper, lower1, lower2] index list/tuple, or a six-tuple structure function key"
        )

    return [structure_function_symbol(structure_function)]


def structure_function_symbol(
    structure_function: sp.Symbol | str | StructureFunctionKey | Sequence[Tuple],
) -> sp.Symbol:
    """
    Convert a structure-function reference into the exact SymPy symbol.

    Accepted inputs:
    - an existing SymPy symbol
    - the exact symbol name string
    - the six-tuple key format used internally by ``targeted_fas_minor``
    """
    if isinstance(structure_function, sp.Symbol):
        return structure_function
    if isinstance(structure_function, str):
        return sp.Symbol(structure_function)
    if _is_compact_structure_function_spec(structure_function):
        upper_index, lower_index_a, lower_index_b = structure_function
        symbol = _structure_function_symbol_from_compact_indices(
            upper_index,
            lower_index_a,
            lower_index_b,
            strict=False,
        )
        if symbol is not None:
            return symbol
        swapped = _structure_function_symbol_from_compact_indices(
            upper_index,
            lower_index_b,
            lower_index_a,
            strict=False,
        )
        if swapped is not None:
            return swapped
        _structure_function_symbol_from_compact_indices(
            upper_index,
            lower_index_a,
            lower_index_b,
            strict=True,
        )
    if not isinstance(structure_function, tuple) or len(structure_function) != 6:
        raise ValueError(
            "structure_function must be a SymPy symbol, a symbol name string, "
            "a compact [upper, lower1, lower2] index list/tuple, or a six-tuple structure function key"
        )

    index_type_a, val_a, index_type_b, val_b, index_type_c, val_c = structure_function
    if index_type_a == "edge" and index_type_b == "vertex" and index_type_c == "vertex":
        g_k, edge_k = val_a
        g_i, vertex_i = val_b
        g_j, vertex_j = val_c
        k_src, k_tgt = edge_k
        symbol_name = f"c^{{{g_k},({k_src},{k_tgt})}}_{{({g_i},{vertex_i}),({g_j},{vertex_j})}}"
        return sp.Symbol(symbol_name)
    if index_type_a == "edge" and index_type_b == "edge" and index_type_c == "vertex":
        g_k, edge_k = val_a
        g_l, edge_l = val_b
        g_i, vertex_i = val_c
        k_src, k_tgt = edge_k
        l_src, l_tgt = edge_l
        symbol_name = f"c^{{{g_k},({k_src},{k_tgt})}}_{{({g_l},({l_src},{l_tgt})),({g_i},{vertex_i})}}"
        return sp.Symbol(symbol_name)
    if index_type_a == "vertex" and index_type_b == "vertex" and index_type_c == "vertex":
        g_l, vertex_l = val_a
        g_w, vertex_w = val_b
        g_v, vertex_v = val_c
        symbol_name = f"c^{{({g_l},{vertex_l})}}_{{({g_w},{vertex_w}),({g_v},{vertex_v})}}"
        return sp.Symbol(symbol_name)
    raise ValueError(f"Unsupported structure function key signature: {structure_function}")


def differentiate_by_structure_function(
    expr: sp.Expr,
    structure_function: sp.Symbol | str | StructureFunctionKey | Sequence[Tuple],
    *,
    order: int = 1,
) -> sp.Expr:
    """
    Differentiate an expression with respect to a chosen structure function.

    Compact ``[upper, lower1, lower2]`` input accounts for antisymmetry in the
    lower indices by also checking the swapped lower ordering on the original
    expression.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return expr
    free_symbols = getattr(expr, "free_symbols", set())
    variants = _resolve_structure_function_variants(structure_function)
    present_variants = [symbol for symbol in variants if symbol in free_symbols]
    if not present_variants:
        if _is_compact_structure_function_spec(structure_function):
            names = ", ".join(str(symbol) for symbol in variants)
            raise ValueError(
                "Neither the requested structure function nor its lower-index-swapped "
                f"form is present in the supplied expression: {names}"
            )
        symbol = variants[0]
        raise ValueError(
            f"Structure function {symbol} is not present in the supplied expression"
        )
    result = sp.Integer(0)
    for symbol in present_variants:
        result += sp.diff(expr, symbol, order)
    return result


def differentiate_sparse_coefficients(
    poly: SparsePolynomial,
    structure_function: sp.Symbol | str | StructureFunctionKey | Sequence[Tuple],
    *,
    order: int = 1,
) -> SparsePolynomial:
    """
    Differentiate sparse polynomial coefficients with respect to a structure function.

    This leaves the u-exponent vectors unchanged and differentiates only the
    coefficient data attached to each monomial.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return dict(poly)
    variants = _resolve_structure_function_variants(structure_function)
    result: SparsePolynomial = {}
    for exponents, coeff in poly.items():
        differentiated = sp.Integer(0)
        for symbol in variants:
            differentiated += sp.diff(coeff, symbol, order)
        if differentiated != 0:
            result[exponents] = differentiated
    return result


def monomial_spec_to_exponent_vector(
    monomial_spec: Dict[Tuple, int],
    u_gens: Sequence[sp.Symbol],
) -> ExponentVector:
    """Map a targeted_fas_minor monomial spec onto the current generator order."""
    name_to_index = {sym.name: idx for idx, sym in enumerate(u_gens)}
    exponents = [0] * len(u_gens)
    for key, exponent in monomial_spec.items():
        if exponent <= 0:
            continue
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(f"Invalid monomial key format: {key}")
        kind, graph_idx, local_id = key
        if kind == "vertex":
            symbol_name = f"u_{{{graph_idx},{local_id}}}"
        elif kind == "edge":
            src, tgt = local_id
            symbol_name = f"u_{{{graph_idx},({src},{tgt})}}"
        else:
            raise ValueError(f"Unknown monomial key kind: {kind}")
        try:
            idx = name_to_index[symbol_name]
        except KeyError as exc:
            raise ValueError(
                f"Monomial variable {symbol_name} is not present in the provided generator list"
            ) from exc
        exponents[idx] = exponent
    return tuple(exponents)


def enumerate_minor_u_monomials(
    char_tuples: List[Tuple[int, ...]],
    extra_row: Tuple[int, int, int],
    additional_vars: Optional[List[Tuple]] = None,
    *,
    require_p_divisor: bool = False,
    max_total_degree: Optional[int] = None,
    max_degree_per_var: Optional[int | Sequence[Optional[int]]] = None,
    max_terms: Optional[int] = None,
    return_minor: bool = False,
    return_u_gens: bool = False,
) -> SparsePolynomial | Tuple[SparsePolynomial, sp.Expr] | Tuple[SparsePolynomial, List[sp.Symbol]] | Tuple[SparsePolynomial, sp.Expr, List[sp.Symbol]]:
    """
    Compute a targeted minor and enumerate its u-monomials as a sparse polynomial.

    This uses ``compute_minor_with_p_vars`` to keep only the p-variables plus
    the requested extra variables, then expands only with respect to the kept
    u-generators.
    """
    minor, u_gens = compute_minor_with_p_vars(
        char_tuples,
        extra_row,
        additional_vars=additional_vars,
        return_u_gens=True,
    )
    poly = expr_to_sparse_u_poly(
        minor,
        u_gens,
        max_total_degree=max_total_degree,
        max_degree_per_var=max_degree_per_var,
        max_terms=max_terms,
    )
    if require_p_divisor:
        temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
        temp_det = DeterminantComputer(temp_calc)
        p_spec = temp_det.base_A_root_product_spec()
        p_exponents = monomial_spec_to_exponent_vector(p_spec, u_gens)
        poly = filter_sparse_poly(poly, must_divide=p_exponents, max_terms=max_terms)
    if return_minor and return_u_gens:
        return poly, minor, u_gens
    if return_minor:
        return poly, minor
    if return_u_gens:
        return poly, list(u_gens)
    return poly


__all__ = [
    "ExponentVector",
    "SparsePolynomial",
    "TooManyTermsError",
    "add_sparse_polynomials",
    "multiply_sparse_polynomials",
    "sparse_pow",
    "expr_to_sparse_u_poly",
    "sparse_poly_to_expr",
    "exponent_vector_to_monomial_expr",
    "filter_sparse_poly",
    "iter_sorted_terms",
    "format_sparse_terms",
    "structure_function_symbol",
    "differentiate_by_structure_function",
    "differentiate_sparse_coefficients",
    "monomial_spec_to_exponent_vector",
    "enumerate_minor_u_monomials",
]
