"""
Iterative elimination pipeline for structure function vanishing.

Automates the cascading vanishing argument for Type 2 structure functions
of the form c^{edge from Gamma_1}_{edge from Gamma_0 of form (x,v), vertex from Gamma_0}.

The cascade is ordered by root index x of the lower edge:
  Wave 0 (root x=0): prove SFs with lower edge (0,v) vanish
  Wave 1 (root x=1): inject wave-0 zeros into calculator, prove SFs with (1,v) vanish
  ...continue through all roots of Gamma_0.

Vanished structure functions are zeroed at the calculator level BEFORE row
computation to avoid carrying dead symbolic terms through the recursion.

CLI usage:
    python iterative_elimination.py \\
        --tuples "3,1,5;2,1,3" \\
        --output-dir runs --prefix elim

Library usage:
    from iterative_elimination import run_iterative_elimination
    result = run_iterative_elimination([(3, 1, 5), (2, 1, 3)])
    for wave in result.waves:
        print(f"Wave {wave.root_index}: {len(wave.vanished_sfs)} vanished")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import sympy as sp

from sparse_u_monomials import (
    differentiate_by_structure_function,
    exponent_vector_to_monomial_expr,
    expr_to_sparse_u_poly,
)
from structure_function_search import (
    build_all_vars,
    enumerate_row_complements,
    is_simple_coefficient,
)
from targeted_fas_minor import (
    ComponentGraph,
    DeterminantComputer,
    FASMinorCalculator,
    format_monomial_spec,
)


# ---------------------------------------------------------------------------
# Helpers copied from structure_function_search (private, so we duplicate)
# ---------------------------------------------------------------------------

def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tuple_to_jsonable(value):
    if isinstance(value, tuple):
        return [_tuple_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_tuple_to_jsonable(item) for item in value]
    return value


def _u_symbol_to_var_key(symbol: sp.Symbol) -> Tuple:
    name = symbol.name
    if not name.startswith("u_{") or not name.endswith("}"):
        raise ValueError(f"Unsupported u-generator symbol name: {name}")
    inner = name[3:-1]
    if ",(" in inner:
        graph_str, edge_str = inner.split(",", 1)
        graph_idx = int(graph_str)
        if not edge_str.startswith("(") or not edge_str.endswith(")"):
            raise ValueError(f"Unsupported edge u-generator symbol name: {name}")
        src_str, tgt_str = edge_str[1:-1].split(",")
        return ("edge", graph_idx, (int(src_str), int(tgt_str)))
    graph_str, vertex_str = inner.split(",")
    return ("vertex", int(graph_str), int(vertex_str))


def _selected_vars_from_monomial(
    exponents: Sequence[int],
    u_gens: Sequence[sp.Symbol],
) -> Tuple[Tuple, ...]:
    selected_vars = []
    for exponent, symbol in zip(exponents, u_gens):
        if exponent:
            selected_vars.append(_u_symbol_to_var_key(symbol))
    return tuple(selected_vars)


def _monomial_spec_from_monomial(
    exponents: Sequence[int],
    u_gens: Sequence[sp.Symbol],
) -> Dict[Tuple, int]:
    monomial_spec: Dict[Tuple, int] = {}
    for exponent, symbol in zip(exponents, u_gens):
        if not exponent:
            continue
        monomial_spec[_u_symbol_to_var_key(symbol)] = exponent
    return monomial_spec


def _monomial_cli_from_monomial(
    exponents: Sequence[int],
    u_gens: Sequence[sp.Symbol],
) -> str:
    return format_monomial_spec(_monomial_spec_from_monomial(exponents, u_gens))


def _format_char_tuples_arg(char_tuples: Sequence[Tuple[int, ...]]) -> str:
    return ";".join(",".join(str(value) for value in tup) for tup in char_tuples)


def _parse_tuples(s: str) -> List[Tuple[int, ...]]:
    tuples: List[Tuple[int, ...]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        values = tuple(int(x.strip()) for x in part.split(","))
        tuples.append(values)
    return tuples


def _is_structure_function_symbol(symbol: sp.Symbol) -> bool:
    return symbol.name.startswith("c^{")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VanishingEvidence:
    """Evidence that a structure function vanishes."""

    sf_name: str
    extra_row: Tuple[int, int, int]
    u_monomial: str
    monomial_cli: str
    coefficient: str
    classification: str  # "numeric" or "alpha_only"
    wave: int
    sub_wave: int  # 0 = initial pass, 1+ = intra-wave re-check

    def to_dict(self) -> dict:
        return {
            "sf_name": self.sf_name,
            "extra_row": list(self.extra_row),
            "u_monomial": self.u_monomial,
            "monomial_cli": self.monomial_cli,
            "coefficient": self.coefficient,
            "classification": self.classification,
            "wave": self.wave,
            "sub_wave": self.sub_wave,
        }


@dataclass
class WaveResult:
    root_index: int
    target_sfs: List[str]
    vanished_sfs: List[str]
    evidence: List[VanishingEvidence]
    unresolved_sfs: List[str]
    extra_rows_searched: int
    monomials_examined: int
    sub_waves: int

    def to_dict(self) -> dict:
        return {
            "root_index": self.root_index,
            "target_sfs": self.target_sfs,
            "vanished_sfs": self.vanished_sfs,
            "evidence": [e.to_dict() for e in self.evidence],
            "unresolved_sfs": self.unresolved_sfs,
            "extra_rows_searched": self.extra_rows_searched,
            "monomials_examined": self.monomials_examined,
            "sub_waves": self.sub_waves,
        }


@dataclass
class EliminationResult:
    char_tuples: List[Tuple[int, ...]]
    component_0_index: int
    component_1_index: int
    waves: List[WaveResult]
    all_vanished_sfs: List[str]
    all_unresolved_sfs: List[str]
    total_sfs_tested: int
    elapsed_seconds: float
    success: bool
    error_message: Optional[str] = None
    errors: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "artifact_type": "iterative_elimination",
            "generated_at": _utc_timestamp(),
            "char_tuples": [list(t) for t in self.char_tuples],
            "component_0_index": self.component_0_index,
            "component_1_index": self.component_1_index,
            "total_sfs_tested": self.total_sfs_tested,
            "total_waves": len(self.waves),
            "all_vanished_sfs": self.all_vanished_sfs,
            "total_vanished": len(self.all_vanished_sfs),
            "all_unresolved_sfs": self.all_unresolved_sfs,
            "success": self.success,
            "error_message": self.error_message,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "errors": self.errors,
            "waves": [w.to_dict() for w in self.waves],
        }


# ---------------------------------------------------------------------------
# SF enumeration
# ---------------------------------------------------------------------------

def enumerate_target_sfs(
    char_tuples: List[Tuple[int, ...]],
    comp_0: int = 0,
    comp_1: int = 1,
) -> Dict[int, List[Tuple[sp.Symbol, Tuple]]]:
    """Enumerate Type 2 SFs grouped by root index of the lower edge.

    Returns:
        Dict mapping root_index -> list of (sf_symbol, sf_key) pairs,
        where sf_key is the 6-tuple key for structure_functions_symbolic.
        Only truly symbolic SFs are included (constraints may reduce some to
        constants).
    """
    graph_0 = ComponentGraph.from_characteristic_tuple(char_tuples[comp_0])
    graph_1 = ComponentGraph.from_characteristic_tuple(char_tuples[comp_1])

    # Use a temporary calculator to check constraints
    temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)

    # Group edges of Gamma_0 by their source (root) vertex
    edges_by_root: Dict[int, List[Tuple[int, int]]] = {}
    for src, tgt in graph_0.edges:
        if src < graph_0.num_roots:
            edges_by_root.setdefault(src, []).append((src, tgt))

    result: Dict[int, List[Tuple[sp.Symbol, Tuple]]] = {}

    for root_idx in range(graph_0.num_roots):
        root_edges = edges_by_root.get(root_idx, [])
        sfs_for_root: List[Tuple[sp.Symbol, Tuple]] = []

        for edge_0 in root_edges:
            for vertex_w in sorted(graph_0.vertices):
                for edge_1 in graph_1.edges:
                    # Type 2 key: c^{edge from G1}_{edge from G0, vertex from G0}
                    sf_key = (
                        'edge', (comp_1, edge_1),
                        'edge', (comp_0, edge_0),
                        'vertex', (comp_0, vertex_w),
                    )
                    val = temp_calc._get_structure_function(sf_key)
                    if isinstance(val, sp.Symbol):
                        sfs_for_root.append((val, sf_key))

        if sfs_for_root:
            result[root_idx] = sfs_for_root

    return result


# ---------------------------------------------------------------------------
# Minor computation with SF injection
# ---------------------------------------------------------------------------

def _compute_minor_with_sf_injection(
    char_tuples: List[Tuple[int, ...]],
    extra_row: Tuple[int, int, int],
    vanished_sf_keys: List[Tuple],
    all_vars: List[Tuple],
) -> Tuple[sp.Expr, List[sp.Symbol]]:
    """Compute minor with all u-vars kept and vanished SFs zeroed at calc level.

    This is adapted from compute_minor_with_p_vars but constructs the
    calculator directly so we can inject vanished SF zeros before row
    computation.

    Args:
        char_tuples: System characteristic tuples.
        extra_row: (graph_idx, vertex, layer) for the extra row.
        vanished_sf_keys: List of 6-tuple SF keys to pre-zero.
        all_vars: All u-variable keys to keep alive.

    Returns:
        (minor_expr, u_gens)
    """
    # Build combined spec: all vars with exponent 1 (keeps everything alive)
    combined_spec: Dict[Tuple, int] = {var_key: 1 for var_key in all_vars}

    calc = FASMinorCalculator.from_characteristic_tuples(
        char_tuples,
        target_monomial_spec=combined_spec,
    )

    # Inject vanished SF zeros BEFORE any row computation
    for sf_key in vanished_sf_keys:
        calc.structure_functions_symbolic[sf_key] = 0

    det_comp = DeterminantComputer(calc)
    minor = det_comp.compute_minor_fast(*extra_row)
    u_gens = det_comp.get_u_gens()
    return minor, u_gens


# ---------------------------------------------------------------------------
# Pending coefficient storage (for sub-wave re-checking)
# ---------------------------------------------------------------------------

@dataclass
class _PendingCoeff:
    """A monomial coefficient that is not yet simple."""
    sf_name: str
    extra_row: Tuple[int, int, int]
    exponents: Tuple[int, ...]
    u_gens: List[sp.Symbol]
    coefficient: sp.Expr
    sf_symbols_in_coeff: FrozenSet[str]


# ---------------------------------------------------------------------------
# Wave processing
# ---------------------------------------------------------------------------

def _process_wave(
    char_tuples: List[Tuple[int, ...]],
    root_index: int,
    target_sfs: List[Tuple[sp.Symbol, Tuple]],
    vanished_sf_keys: List[Tuple],
    extra_rows: List[Tuple[int, int, int]],
    all_vars: List[Tuple],
    use_sparse: bool,
    max_sub_waves: int,
    live_callback: Optional[Callable[[VanishingEvidence], None]],
    progress_callback: Optional[Callable[[int, int, Tuple[int, int, int]], None]],
) -> WaveResult:
    """Process a single root group wave.

    For each extra row, computes the minor (with prior vanished SFs zeroed),
    then differentiates by each target SF and classifies coefficients.
    Runs sub-wave iterations on pending coefficients until convergence.
    """
    sf_name_to_symbol: Dict[str, sp.Symbol] = {}
    sf_name_to_key: Dict[str, Tuple] = {}
    for sf_sym, sf_key in target_sfs:
        sf_name_to_symbol[sf_sym.name] = sf_sym
        sf_name_to_key[sf_sym.name] = sf_key

    target_sf_names = list(sf_name_to_symbol.keys())

    # Track which SFs have been confirmed vanished and their evidence
    vanished_in_wave: Set[str] = set()
    evidence: List[VanishingEvidence] = []
    pending_coeffs: List[_PendingCoeff] = []
    total_monomials = 0
    errors: List[dict] = []

    # --- Pass over all extra rows ---
    for row_idx, row in enumerate(extra_rows):
        if progress_callback:
            progress_callback(row_idx + 1, len(extra_rows), row)

        try:
            minor, u_gens = _compute_minor_with_sf_injection(
                char_tuples, row, vanished_sf_keys, all_vars,
            )
        except Exception as exc:
            errors.append({"extra_row": list(row), "stage": "minor", "error": str(exc)})
            continue

        if minor == 0:
            continue

        # For each target SF, differentiate the minor and classify
        for sf_name in target_sf_names:
            if sf_name in vanished_in_wave:
                continue  # already confirmed, skip

            sf_sym = sf_name_to_symbol[sf_name]

            # Differentiate by the SF symbol directly
            try:
                diff_expr = sp.diff(minor, sf_sym)
            except Exception as exc:
                errors.append({
                    "extra_row": list(row),
                    "stage": "diff",
                    "sf": sf_name,
                    "error": str(exc),
                })
                continue

            if diff_expr == 0:
                continue

            if use_sparse:
                try:
                    sparse_poly = expr_to_sparse_u_poly(diff_expr, u_gens)
                except Exception as exc:
                    errors.append({
                        "extra_row": list(row),
                        "stage": "sparse_poly",
                        "sf": sf_name,
                        "error": str(exc),
                    })
                    continue

                if not sparse_poly:
                    continue

                for exponents, coeff in sparse_poly.items():
                    total_monomials += 1
                    simple, classification = is_simple_coefficient(coeff)
                    if simple:
                        mono_expr = exponent_vector_to_monomial_expr(exponents, u_gens)
                        mono_cli = _monomial_cli_from_monomial(exponents, u_gens)
                        ev = VanishingEvidence(
                            sf_name=sf_name,
                            extra_row=row,
                            u_monomial=str(mono_expr),
                            monomial_cli=mono_cli,
                            coefficient=str(coeff),
                            classification=classification,
                            wave=root_index,
                            sub_wave=0,
                        )
                        vanished_in_wave.add(sf_name)
                        evidence.append(ev)
                        if live_callback:
                            live_callback(ev)
                        break  # one proof suffices for this SF
                    else:
                        # Store for sub-wave re-checking
                        sf_in_coeff = frozenset(
                            s.name for s in coeff.free_symbols
                            if _is_structure_function_symbol(s)
                        )
                        pending_coeffs.append(_PendingCoeff(
                            sf_name=sf_name,
                            extra_row=row,
                            exponents=exponents,
                            u_gens=u_gens,
                            coefficient=coeff,
                            sf_symbols_in_coeff=sf_in_coeff,
                        ))
            else:
                # Dense path
                try:
                    expanded = sp.expand(diff_expr)
                    poly = sp.Poly(expanded, *u_gens, domain="EX")
                except Exception as exc:
                    errors.append({
                        "extra_row": list(row),
                        "stage": "poly",
                        "sf": sf_name,
                        "error": str(exc),
                    })
                    continue

                for monom_tuple, coeff in poly.as_dict().items():
                    total_monomials += 1
                    simple, classification = is_simple_coefficient(coeff)
                    if simple:
                        mono_expr = exponent_vector_to_monomial_expr(monom_tuple, u_gens)
                        mono_cli = _monomial_cli_from_monomial(monom_tuple, u_gens)
                        ev = VanishingEvidence(
                            sf_name=sf_name,
                            extra_row=row,
                            u_monomial=str(mono_expr),
                            monomial_cli=mono_cli,
                            coefficient=str(coeff),
                            classification=classification,
                            wave=root_index,
                            sub_wave=0,
                        )
                        vanished_in_wave.add(sf_name)
                        evidence.append(ev)
                        if live_callback:
                            live_callback(ev)
                        break
                    else:
                        sf_in_coeff = frozenset(
                            s.name for s in coeff.free_symbols
                            if _is_structure_function_symbol(s)
                        )
                        pending_coeffs.append(_PendingCoeff(
                            sf_name=sf_name,
                            extra_row=row,
                            exponents=monom_tuple,
                            u_gens=u_gens,
                            coefficient=coeff,
                            sf_symbols_in_coeff=sf_in_coeff,
                        ))

    # --- Sub-wave iterations: substitute newly vanished SFs in pending coeffs ---
    sub_wave_count = 0
    for sub_wave_num in range(1, max_sub_waves + 1):
        # Build substitution dict from SFs vanished so far in this wave
        # (these are SFs within the same root group, not prior waves)
        wave_subs: Dict[sp.Symbol, int] = {}
        for vname in vanished_in_wave:
            if vname in sf_name_to_symbol:
                wave_subs[sf_name_to_symbol[vname]] = 0

        if not wave_subs:
            break

        newly_vanished_this_sub: List[str] = []
        still_pending: List[_PendingCoeff] = []

        for pc in pending_coeffs:
            if pc.sf_name in vanished_in_wave:
                continue  # already confirmed

            # Check if this pending coeff involves any vanished SF
            if not (pc.sf_symbols_in_coeff & vanished_in_wave):
                still_pending.append(pc)
                continue

            # Substitute vanished SFs -> 0
            substituted = pc.coefficient
            for sym, val in wave_subs.items():
                substituted = substituted.subs(sym, val)

            if substituted == 0:
                still_pending.append(pc)  # coeff vanishes entirely, no info
                continue

            simple, classification = is_simple_coefficient(substituted)
            if simple:
                mono_expr = exponent_vector_to_monomial_expr(pc.exponents, pc.u_gens)
                mono_cli = _monomial_cli_from_monomial(pc.exponents, pc.u_gens)
                ev = VanishingEvidence(
                    sf_name=pc.sf_name,
                    extra_row=pc.extra_row,
                    u_monomial=str(mono_expr),
                    monomial_cli=mono_cli,
                    coefficient=str(substituted),
                    classification=classification,
                    wave=root_index,
                    sub_wave=sub_wave_num,
                )
                newly_vanished_this_sub.append(pc.sf_name)
                vanished_in_wave.add(pc.sf_name)
                evidence.append(ev)
                if live_callback:
                    live_callback(ev)
            else:
                # Update the stored coefficient and SF set
                new_sf_in_coeff = frozenset(
                    s.name for s in substituted.free_symbols
                    if _is_structure_function_symbol(s)
                )
                still_pending.append(_PendingCoeff(
                    sf_name=pc.sf_name,
                    extra_row=pc.extra_row,
                    exponents=pc.exponents,
                    u_gens=pc.u_gens,
                    coefficient=substituted,
                    sf_symbols_in_coeff=new_sf_in_coeff,
                ))

        pending_coeffs = still_pending
        sub_wave_count = sub_wave_num

        if not newly_vanished_this_sub:
            break  # convergence

    # Determine unresolved SFs
    unresolved = [name for name in target_sf_names if name not in vanished_in_wave]

    return WaveResult(
        root_index=root_index,
        target_sfs=target_sf_names,
        vanished_sfs=sorted(vanished_in_wave),
        evidence=evidence,
        unresolved_sfs=unresolved,
        extra_rows_searched=len(extra_rows),
        monomials_examined=total_monomials,
        sub_waves=sub_wave_count,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_iterative_elimination(
    char_tuples: List[Tuple[int, ...]],
    *,
    comp_0: int = 0,
    comp_1: int = 1,
    use_sparse: bool = True,
    max_sub_waves: int = 10,
    extra_rows: Optional[List[Tuple[int, int, int]]] = None,
    live_callback: Optional[Callable[[VanishingEvidence], None]] = None,
    progress_callback: Optional[Callable[[int, int, Tuple[int, int, int]], None]] = None,
    wave_callback: Optional[Callable[[WaveResult], None]] = None,
) -> EliminationResult:
    """Run iterative elimination across all root groups.

    Args:
        char_tuples: System characteristic tuples.
        comp_0: Component index for Gamma_0 (default 0).
        comp_1: Component index for Gamma_1 (default 1).
        use_sparse: Use sparse u-polynomial mode (default True).
        max_sub_waves: Max intra-wave sub-iterations (default 10).
        extra_rows: Explicit extra rows. If None, enumerates from comp_0.
        live_callback: Called for each VanishingEvidence as found.
        progress_callback: Called as (current, total, extra_row) per row.
        wave_callback: Called after each wave completes.

    Returns:
        EliminationResult with full cascade details.
    """
    t0 = time.monotonic()

    # Enumerate target SFs grouped by root
    sf_groups = enumerate_target_sfs(char_tuples, comp_0, comp_1)

    if not sf_groups:
        elapsed = time.monotonic() - t0
        return EliminationResult(
            char_tuples=list(char_tuples),
            component_0_index=comp_0,
            component_1_index=comp_1,
            waves=[],
            all_vanished_sfs=[],
            all_unresolved_sfs=[],
            total_sfs_tested=0,
            elapsed_seconds=elapsed,
            success=True,
            error_message=None,
        )

    # Build extra rows and all_vars once
    if extra_rows is None:
        extra_rows = enumerate_row_complements(char_tuples, component=comp_0)
    all_vars = build_all_vars(char_tuples)

    # Accumulate vanished SF keys across waves
    all_vanished_keys: List[Tuple] = []
    all_vanished_names: List[str] = []
    waves: List[WaveResult] = []
    total_sfs_tested = 0
    all_errors: List[dict] = []

    for root_idx in sorted(sf_groups.keys()):
        target_sfs = sf_groups[root_idx]
        total_sfs_tested += len(target_sfs)

        wave_result = _process_wave(
            char_tuples=char_tuples,
            root_index=root_idx,
            target_sfs=target_sfs,
            vanished_sf_keys=list(all_vanished_keys),
            extra_rows=extra_rows,
            all_vars=all_vars,
            use_sparse=use_sparse,
            max_sub_waves=max_sub_waves,
            live_callback=live_callback,
            progress_callback=progress_callback,
        )
        waves.append(wave_result)

        if wave_callback:
            wave_callback(wave_result)

        # Collect vanished SFs for injection in subsequent waves
        for sf_name in wave_result.vanished_sfs:
            all_vanished_names.append(sf_name)
            # Find the key for this SF name
            for sf_sym, sf_key in target_sfs:
                if sf_sym.name == sf_name:
                    all_vanished_keys.append(sf_key)
                    break

        # Check for failure: unresolved SFs in this wave
        if wave_result.unresolved_sfs:
            elapsed = time.monotonic() - t0
            unresolved_list = wave_result.unresolved_sfs
            error_msg = (
                f"Wave {root_idx} (root x={root_idx}): "
                f"{len(unresolved_list)} structure function(s) could not be "
                f"confirmed as vanishing: {', '.join(unresolved_list)}"
            )
            return EliminationResult(
                char_tuples=list(char_tuples),
                component_0_index=comp_0,
                component_1_index=comp_1,
                waves=waves,
                all_vanished_sfs=sorted(all_vanished_names),
                all_unresolved_sfs=unresolved_list,
                total_sfs_tested=total_sfs_tested,
                elapsed_seconds=elapsed,
                success=False,
                error_message=error_msg,
                errors=all_errors,
            )

    elapsed = time.monotonic() - t0
    return EliminationResult(
        char_tuples=list(char_tuples),
        component_0_index=comp_0,
        component_1_index=comp_1,
        waves=waves,
        all_vanished_sfs=sorted(all_vanished_names),
        all_unresolved_sfs=[],
        total_sfs_tested=total_sfs_tested,
        elapsed_seconds=elapsed,
        success=True,
        error_message=None,
        errors=all_errors,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stderr_progress(current: int, total: int, extra_row: Tuple[int, int, int]) -> None:
    sys.stderr.write(f"\r  [{current}/{total}] Processing extra row {extra_row}...")
    sys.stderr.flush()
    if current == total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterative elimination of Type 2 structure functions via cascading vanishing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python iterative_elimination.py \\
    --tuples "3,1,5;2,1,3"

  python iterative_elimination.py \\
    --tuples "3,1,5;2,1,3" \\
    --output-dir runs --prefix elim --quiet
""",
    )
    parser.add_argument(
        "--tuples", required=True,
        help='Characteristic tuples, semicolon-separated (e.g., "3,1,5;2,1,3")',
    )
    parser.add_argument(
        "--component-0", type=int, default=0,
        help="Component index for Gamma_0 (default: 0).",
    )
    parser.add_argument(
        "--component-1", type=int, default=1,
        help="Component index for Gamma_1 (default: 1).",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--prefix", default="elimination",
        help="Prefix for generated output files.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Use dense poly instead of sparse (default: sparse).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress and live output.",
    )
    parser.add_argument(
        "--max-sub-waves", type=int, default=10,
        help="Maximum intra-wave sub-iterations (default: 10).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    char_tuples = _parse_tuples(args.tuples)
    use_sparse = not args.dense

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"{args.prefix}_result.json")
    vanished_path = os.path.join(args.output_dir, f"{args.prefix}_vanished.txt")

    # Enumerate SFs to show scope before starting
    sf_groups = enumerate_target_sfs(char_tuples, args.component_0, args.component_1)
    graph_0 = ComponentGraph.from_characteristic_tuple(char_tuples[args.component_0])
    total_sfs = sum(len(sfs) for sfs in sf_groups.values())

    if not args.quiet:
        print(f"System: {char_tuples}")
        print(f"Gamma_0 = component {args.component_0}, "
              f"Gamma_1 = component {args.component_1}")
        print(f"Gamma_0 roots: {graph_0.num_roots}, "
              f"waves to process: {len(sf_groups)}")
        print(f"Total Type 2 SFs to test: {total_sfs}")
        for root_idx in sorted(sf_groups.keys()):
            sf_names = [sym.name for sym, _ in sf_groups[root_idx]]
            print(f"  Root x={root_idx}: {len(sf_names)} SFs")
        print()

    # Live finding callback
    finding_count = [0]

    def _live_finding(ev: VanishingEvidence) -> None:
        finding_count[0] += 1
        sub_tag = f" (sub-wave {ev.sub_wave})" if ev.sub_wave > 0 else ""
        print(
            f"  VANISHED [{finding_count[0]}]: {ev.sf_name} "
            f"via {ev.u_monomial} row={ev.extra_row} "
            f"-> {ev.coefficient} [{ev.classification}]{sub_tag}",
            flush=True,
        )

    # Wave callback
    def _wave_done(wave: WaveResult) -> None:
        status = "OK" if not wave.unresolved_sfs else "INCOMPLETE"
        print(
            f"\n[Wave {wave.root_index}, root={wave.root_index}] {status}: "
            f"{len(wave.vanished_sfs)} vanished, "
            f"{len(wave.unresolved_sfs)} unresolved, "
            f"{wave.sub_waves} sub-waves\n",
            flush=True,
        )

    progress = None if args.quiet else _stderr_progress
    live_cb = None if args.quiet else _live_finding
    wave_cb = None if args.quiet else _wave_done

    result = run_iterative_elimination(
        char_tuples,
        comp_0=args.component_0,
        comp_1=args.component_1,
        use_sparse=use_sparse,
        max_sub_waves=args.max_sub_waves,
        live_callback=live_cb,
        progress_callback=progress,
        wave_callback=wave_cb,
    )

    # Write outputs
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, indent=2)

    with open(vanished_path, "w", encoding="utf-8") as fp:
        for wave in result.waves:
            if wave.vanished_sfs:
                fp.write(f"# Wave {wave.root_index} (root x={wave.root_index})\n")
                for sf_name in wave.vanished_sfs:
                    fp.write(f"{sf_name}\n")
                fp.write("\n")

    # Final summary
    print(f"{'='*60}")
    print(f"System: {result.char_tuples}")
    print(f"Total SFs tested: {result.total_sfs_tested}")
    print(f"Total vanished: {len(result.all_vanished_sfs)}")
    print(f"Waves completed: {len(result.waves)}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
    print(f"Success: {result.success}")

    if not result.success:
        print(f"\nERROR: {result.error_message}")
        print("\nPartial results preserved in output files.")

    if result.errors:
        print(f"\nComputation errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  {err}")

    print(f"\nResult written to {result_path}")
    print(f"Vanished list written to {vanished_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
