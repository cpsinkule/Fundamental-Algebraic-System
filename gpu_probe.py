"""
GPU Probe Utilities for FAS Minors

This module provides optional GPU-accelerated numeric probing helpers that
evaluate the fast minor formula without changing the symbolic source of truth.

Usage pattern:
- Build a GPUMinorProbe with your calculator and determinant computer
- Prepare for a specific extra row (graph_idx, vertex, layer)
- Evaluate the minor numerically for a given assignment of variables

All core results remain symbolic (use DeterminantComputer APIs). These helpers
are intended for fast numeric exploration (e.g., monomial search prefilters).
"""

from typing import Dict, Tuple, Any, List, Optional

import sympy as sp
import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional
    cp = None

from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer


BackendMod = Any


class GPUMinorProbe:
    """
    Optional GPU helper to numerically evaluate compute_minor_fast without
    constructing the full symbolic minor. Keeps the symbolic path intact.

    Steps:
    - For a given extra row (graph_idx, vertex, layer), build lambdified
      functions for per-component A-block rows, the extra A row block, and b
      entries in the same component as the extra row.
    - Evaluate those on GPU (CuPy) or CPU (NumPy) and assemble the fast minor
      via block-structured Laplace expansion.
    """

    def __init__(self, calc: FASMinorCalculator, det_comp: DeterminantComputer, backend: str = 'cupy'):
        self.calc = calc
        self.det_comp = det_comp
        self.backend = backend
        self._check_backend()

        # Cache for per-extra-row lambdas and symbol ordering
        self._extra_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def _check_backend(self) -> None:
        if self.backend == 'cupy' and cp is None:
            raise RuntimeError("CuPy is not available. Install cupy to use GPU backend or set backend='numpy'.")

    @property
    def _mod(self) -> BackendMod:
        if self.backend == 'cupy' and cp is not None:
            return cp
        return np

    def _gather_component_slices(self) -> Tuple[Dict[int, int], Dict[int, int], int]:
        comp_starts: Dict[int, int] = {}
        comp_sizes: Dict[int, int] = {}
        start = 0
        for g_idx, g in enumerate(self.calc.graphs):
            sz = len(g.edges)
            comp_starts[g_idx] = start
            comp_sizes[g_idx] = sz
            start += sz
        total_edges = start
        return comp_starts, comp_sizes, total_edges

    def _prepare_for_extra_row(self, extra_row: Tuple[int, int, int]) -> None:
        """
        Build and cache lambdified row/entry functions and a shared symbol list
        for the given extra row.
        """
        if extra_row in self._extra_cache:
            return

        g_star, v_star, s_star = extra_row
        comp_starts, comp_sizes, total_edges = self._gather_component_slices()

        # Collect expressions: A-block rows for all components; extra A row for g*;
        # and b entries for rows in component g* plus the extra row.
        exprs: List[sp.Expr] = []
        a_row_exprs_by_comp: Dict[int, List[List[sp.Expr]]] = {g: [] for g in range(len(self.calc.graphs))}
        b_row_exprs: Dict[Tuple[int, int, int], sp.Expr] = {}

        # Base rows by component
        base_rows_by_comp: Dict[int, List[Tuple[int, int, int]]] = {g: [] for g in range(len(self.calc.graphs))}
        for r in self.det_comp.base_rows:
            base_rows_by_comp[r[0]].append(r)

        # A-block row exprs (per component)
        for g_idx, rows in base_rows_by_comp.items():
            if comp_sizes[g_idx] == 0:
                continue
            c0 = comp_starts[g_idx]
            c1 = c0 + comp_sizes[g_idx]
            for r in rows:
                row = self.calc.get_row(*r)
                block = row[:, 0:total_edges][:, c0:c1]  # ensure slicing from A-only then component slice
                # Store as a python list of entry expressions
                row_list = [block[0, j] for j in range(block.shape[1])]
                a_row_exprs_by_comp[g_idx].append(row_list)
                exprs.extend(row_list)

        # Extra A-block for g*
        if comp_sizes[g_star] == 0:
            raise ValueError("Component of extra row has zero edges; cannot form minor.")
        c0s = comp_starts[g_star]
        c1s = c0s + comp_sizes[g_star]
        extra_row_full = self.calc.get_row(*extra_row)
        extra_block = extra_row_full[:, 0:total_edges][:, c0s:c1s]
        extra_list = [extra_block[0, j] for j in range(extra_block.shape[1])]
        exprs.extend(extra_list)

        # b entries for rows in g* and the extra row
        rows_star = base_rows_by_comp[g_star] + [extra_row]
        for r in rows_star:
            b_expr = self.calc.build_matrix_entry(r, ('b', None, None))
            b_row_exprs[r] = b_expr
            exprs.append(b_expr)

        # Shared ordered symbol list: union of free symbols across exprs
        sym_set = set()  # type: ignore
        for e in exprs:
            sym_set.update(e.free_symbols)
        # Stable ordering by name
        var_list = sorted(sym_set, key=lambda s: s.name)

        # Build lambdified functions per row/expression
        modules = [self._mod, 'numpy']
        a_row_funcs_by_comp: Dict[int, List[Any]] = {g: [] for g in range(len(self.calc.graphs))}
        for g_idx, rows in a_row_exprs_by_comp.items():
            for row_list in rows:
                a_row_funcs_by_comp[g_idx].append(sp.lambdify(var_list, row_list, modules=modules))

        extra_block_func = sp.lambdify(var_list, extra_list, modules=modules)
        b_funcs: Dict[Tuple[int, int, int], Any] = {}
        for r, be in b_row_exprs.items():
            b_funcs[r] = sp.lambdify(var_list, be, modules=modules)

        # Cache payload
        payload = {
            'var_list': var_list,
            'comp_starts': comp_starts,
            'comp_sizes': comp_sizes,
            'total_edges': total_edges,
            'a_row_funcs_by_comp': a_row_funcs_by_comp,
            'extra_block_func': extra_block_func,
            'b_funcs': b_funcs,
            'rows_star': rows_star,
        }
        self._extra_cache[extra_row] = payload

    def _args_from_assignments(self, extra_row: Tuple[int, int, int], assignments: Dict[sp.Symbol, Any]) -> Tuple[Any, ...]:
        payload = self._extra_cache[extra_row]
        var_list: List[sp.Symbol] = payload['var_list']
        # Build positional arguments for lambdified functions
        args: List[Any] = []
        for s in var_list:
            if s not in assignments:
                raise KeyError(f"Missing assignment for symbol {s}")
            args.append(assignments[s])
        return tuple(args)

    def evaluate_minor_numeric(
        self,
        extra_row: Tuple[int, int, int],
        assignments: Dict[sp.Symbol, Any],
    ) -> Any:
        """
        Evaluate the minor numerically for the given assignments using the fast
        expansion. Returns a scalar (NumPy/CuPy) value.
        """
        self._prepare_for_extra_row(extra_row)
        payload = self._extra_cache[extra_row]
        args = self._args_from_assignments(extra_row, assignments)

        comp_starts = payload['comp_starts']
        comp_sizes = payload['comp_sizes']
        total_edges = payload['total_edges']
        a_row_funcs_by_comp = payload['a_row_funcs_by_comp']
        extra_block_func = payload['extra_block_func']
        b_funcs = payload['b_funcs']
        rows_star = payload['rows_star']

        mod = self._mod

        # Build per-component A-blocks and dets
        det_base_by_comp: Dict[int, Any] = {}
        for g_idx in range(len(self.calc.graphs)):
            e_sz = comp_sizes[g_idx]
            if e_sz == 0:
                det_base_by_comp[g_idx] = 1.0
                continue
            row_funcs = a_row_funcs_by_comp[g_idx]
            # Stack rows
            rows_numeric = [mod.asarray(row_func(*args), dtype=float).reshape(1, e_sz) for row_func in row_funcs]
            block = mod.vstack(rows_numeric)
            # det
            det_val = mod.linalg.det(block)
            det_base_by_comp[g_idx] = det_val

        # Precompute product of det(A_h) for h != g*
        g_star = extra_row[0]
        prod_other = 1.0
        for h, dv in det_base_by_comp.items():
            if h != g_star:
                prod_other = prod_other * dv

        # A* and extra row block
        e_star = comp_sizes[g_star]
        row_funcs_star = a_row_funcs_by_comp[g_star]
        rows_star_numeric = [mod.asarray(f(*args), dtype=float).reshape(1, e_star) for f in row_funcs_star]
        A_star = mod.vstack(rows_star_numeric)
        det_A_star = det_base_by_comp[g_star]
        # Guard: det_A_star should be nonzero by theory; numeric near-zero indicates ill-conditioning
        # (we do not raise here; caller may resample inputs)

        extra_block = mod.asarray(extra_block_func(*args), dtype=float).reshape(1, e_star)
        # Solve A_star^T y = extra^T
        y = mod.linalg.solve(A_star.T, extra_block.T)  # shape (e_star, 1)

        # Laplace expansion along last column: sum over rows in component g*
        # Build global row list = base_rows + [extra_row]
        all_rows = self.det_comp.base_rows + [extra_row]
        ncols = total_edges + 1

        det_total = 0.0
        # Map row in g* to its index in A_star
        index_in_star = {r: i for i, r in enumerate([r for r in self.det_comp.base_rows if r[0] == g_star])}

        for i_global, r in enumerate(all_rows):
            if r[0] != g_star:
                continue
            sign = -1.0 if ((i_global + 1 + ncols) % 2) else 1.0
            b_i = float(b_funcs[r](*args))
            if r == extra_row:
                minor_det = prod_other * det_A_star
            else:
                idx = index_in_star[r]
                minor_det = prod_other * (float(det_A_star) * float(y[idx, 0]))
            det_total += sign * b_i * float(minor_det)

        return det_total

    # ----------------------- Batched evaluation (optional) -----------------------
    def evaluate_minor_numeric_batch(
        self,
        extra_row: Tuple[int, int, int],
        assignments_list: List[Dict[sp.Symbol, Any]],
    ) -> Any:
        """
        Evaluate the minor numerically for a list of assignments. Returns an
        array (NumPy/CuPy) of shape (N,) with one value per assignment.
        """
        mod = self._mod
        vals = [self.evaluate_minor_numeric(extra_row, a) for a in assignments_list]
        return mod.asarray(vals)

    # -------------------- Monomial probing (numeric prefilter) -------------------
    def _u_symbol_sets(self) -> Tuple[set, set]:
        v_syms = set(self.calc.vertex_variables.values())
        e_syms = set(self.calc.edge_variables.values())
        return v_syms, e_syms

    def _sym_from_mono_key(self, key: Tuple[Any, ...]) -> sp.Symbol:
        kind = key[0]
        if kind == 'vertex' and len(key) == 3:
            g, v = key[1], key[2]
            sym = self.calc.vertex_variables.get((g, v))
            if sym is None:
                raise KeyError(f"Unknown vertex variable in monomial spec: {key}")
            return sym
        if kind == 'edge' and len(key) == 3:
            g, edge = key[1], key[2]
            sym = self.calc.edge_variables.get((g, edge))
            if sym is None:
                raise KeyError(f"Unknown edge variable in monomial spec: {key}")
            return sym
        raise ValueError(f"Invalid monomial key format: {key}")

    def _numeric_monomial_value(self, monomial_spec: Dict[Any, int], assignments: Dict[sp.Symbol, Any]) -> float:
        val = 1.0
        for key, exp in monomial_spec.items():
            if exp == 0:
                continue
            sym = self._sym_from_mono_key(key)
            base = float(assignments[sym])
            val *= base ** int(exp)
        return val

    def probe_monomial_in_minor(
        self,
        extra_row: Tuple[int, int, int],
        monomial_spec: Dict[Any, int],
        mode: str = 'divides',
        samples: int = 16,
        seed: Optional[int] = None,
        tol: float = 1e-9,
    ) -> Dict[str, Any]:
        """
        Numeric prefilter for monomial presence using the fast minor formula.

        mode='divides': tests if monomial divides some term in the minor by
        checking p/m at random draws (avoids m≈0 draws). If any reliable sample
        gives |p/m|>tol, likely True.

        mode='exact': sets u variables not in the monomial to 0 (masking) and
        tests if p!=0 across draws (likely True if present). This is a heuristic.

        Returns a dict with keys: likely (bool), samples (int), nonzero_count (int),
        and details (list of per-sample diagnostics).
        """
        rng = np.random.default_rng(seed)
        self._prepare_for_extra_row(extra_row)
        v_syms, e_syms = self._u_symbol_sets()
        u_syms = v_syms.union(e_syms)

        details: List[Dict[str, Any]] = []
        nonzero = 0
        attempts = 0
        i = 0
        max_attempts = samples * 3  # allow retries for degenerate draws

        while i < samples and attempts < max_attempts:
            attempts += 1
            vals = self.random_assignments(extra_row, seed=rng.integers(0, 2**31-1))

            if mode == 'divides':
                m = self._numeric_monomial_value(monomial_spec, vals)
                if abs(m) < tol:
                    continue  # resample to avoid division by ~0
                p = float(self.evaluate_minor_numeric(extra_row, vals))
                ratio = p / m
                hit = abs(ratio) > tol
                nonzero += int(hit)
                details.append({'p': p, 'm': m, 'ratio': ratio, 'hit': hit})
                i += 1
                continue

            if mode == 'exact':
                # Mask u variables not in monomial to 0
                mono_syms = {self._sym_from_mono_key(k) for k in monomial_spec.keys()}
                for s in u_syms:
                    if s not in mono_syms:
                        vals[s] = 0.0
                p = float(self.evaluate_minor_numeric(extra_row, vals))
                hit = abs(p) > tol
                nonzero += int(hit)
                details.append({'p': p, 'hit': hit})
                i += 1
                continue

            raise ValueError("mode must be 'divides' or 'exact'")

        likely = nonzero > 0
        return {
            'likely': likely,
            'samples': i,
            'nonzero_count': nonzero,
            'details': details,
        }

    # -------------------- Convenience: probe p in full minor --------------------
    def probe_p_in_minor(
        self,
        extra_row: Tuple[int, int, int],
        *,
        samples: int = 16,
        seed: Optional[int] = None,
        tol: float = 1e-9,
    ) -> Dict[str, Any]:
        """
        Fast numeric prefilter for whether the full minor contains any term
        divisible by the global base-A root product p = ∏_i p_i.

        Uses the same divides-mode probing as probe_monomial_in_minor, with the
        monomial spec built from the calculator/determinant computer state.
        """
        p_spec = self.det_comp.base_A_root_product_spec()
        return self.probe_monomial_in_minor(
            extra_row, p_spec, mode='divides', samples=samples, seed=seed, tol=tol
        )


# ------------------------- One-shot convenience APIs -------------------------
def probe_monomial_from_characteristic_tuples(
    char_tuples: List[Tuple[int, ...]],
    row: Tuple[int, int, int],
    monomial_spec: Dict[Any, int],
    *,
    mode: str = 'divides',
    samples: int = 16,
    backend: str = 'cupy',
    seed: Optional[int] = None,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """
    One-shot GPU probe for monomial presence starting from characteristic tuples.

    Constructs the calculator/determinant computer, prepares the probe for the
    given extra row, and runs the numeric prefilter.
    """
    calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
    det_comp = DeterminantComputer(calc)
    probe = GPUMinorProbe(calc, det_comp, backend=backend)
    probe._prepare_for_extra_row(row)
    return probe.probe_monomial_in_minor(row, monomial_spec, mode=mode, samples=samples, seed=seed, tol=tol)


def evaluate_minor_numeric_from_characteristic_tuples(
    char_tuples: List[Tuple[int, ...]],
    row: Tuple[int, int, int],
    *,
    backend: str = 'cupy',
    assignments: Optional[Dict[sp.Symbol, Any]] = None,
    seed: Optional[int] = None,
) -> Any:
    """
    One-shot numeric evaluation of the minor using the fast GPU probe pipeline.
    If `assignments` is None, a random assignment is generated.
    """
    calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
    det_comp = DeterminantComputer(calc)
    probe = GPUMinorProbe(calc, det_comp, backend=backend)
    probe._prepare_for_extra_row(row)
    if assignments is None:
        assignments = probe.random_assignments(row, seed=seed)
    return probe.evaluate_minor_numeric(row, assignments)

def probe_p_from_characteristic_tuples(
    char_tuples: List[Tuple[int, ...]],
    row: Tuple[int, int, int],
    *,
    samples: int = 16,
    backend: str = 'cupy',
    seed: Optional[int] = None,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """
    One-shot GPU/CPU numeric prefilter for p in the full minor, starting from
    characteristic tuples.
    """
    calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
    det_comp = DeterminantComputer(calc)
    probe = GPUMinorProbe(calc, det_comp, backend=backend)
    probe._prepare_for_extra_row(row)
    return probe.probe_p_in_minor(row, samples=samples, seed=seed, tol=tol)

    # ----------------------- Convenience helpers (optional) -----------------------
    def random_assignments(
        self,
        extra_row: Tuple[int, int, int],
        seed: Optional[int] = None,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Dict[sp.Symbol, Any]:
        """
        Generate a random assignment for all symbols needed by the extra-row
        pipeline. Use nonzero values to avoid trivial zeros.
        """
        rng = np.random.default_rng(seed)
        payload = self._extra_cache.get(extra_row)
        if payload is None:
            self._prepare_for_extra_row(extra_row)
            payload = self._extra_cache[extra_row]
        var_list: List[sp.Symbol] = payload['var_list']
        vals: Dict[sp.Symbol, Any] = {}
        for s in var_list:
            # Avoid zeros; sample from [low, high] excluding a small neighborhood of 0
            v = rng.uniform(low, high)
            while abs(v) < 1e-6:
                v = rng.uniform(low, high)
            vals[s] = v
        return vals
