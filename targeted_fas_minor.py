"""
Targeted FAS Minor Calculator with Early Variable Zeroing.

This script extends hpc_fas_minor.py to compute FAS minors efficiently
when searching for specific monomial coefficients. The key optimization:
automatically zero out variables NOT in the target monomial EARLY
(during row construction), preventing expression explosion.

Usage examples:
  # Compute coefficient of a specific monomial
  python targeted_fas_minor.py \
    --tuples "2,1,4;2,1,3" \
    --row "0,0,1" \
    --monomial "v:0,0;e:1,(0,1)" \
    --out-prefix results/run1

  # Compute coefficient of p (root product)
  python targeted_fas_minor.py \
    --tuples "2,1,4;2,1,3" \
    --row "0,0,1" \
    --coeff-p \
    --out-prefix results/run1

Monomial format:
  "v:g,vertex;e:g,(src,tgt)" - semicolon-separated terms
  "v:0,0" - vertex variable u_{0,0}
  "e:1,(0,1)" - edge variable u_{1,(0,1)}
  "v:0,0:2" - vertex variable with exponent 2

Outputs (in addition to standard hpc_fas_minor outputs):
- <prefix>.coeff.txt:       str(coefficient)
- <prefix>.coeff.srepr:     srepr(coefficient)
- <prefix>.coeff.meta.json: metadata (is_zero, free_symbols, etc.)

Dependencies: sympy>=1.9, numpy>=1.20.0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Dict, Any, Set, Optional

import sympy as sp


# ------------------------------- Graph model ---------------------------------
class ComponentGraph:
    """Represents a component graph with vertices and directed edges."""

    def __init__(self, vertices: List[int], edges: List[Tuple[int, int]], num_roots: int | None = None):
        self.vertices = set(vertices)
        self.edges = edges
        self.num_roots = num_roots if num_roots is not None else 0
        self.adjacency = self._build_adjacency()
        self.vertex_depths = self._compute_vertex_depths()
        self.edge_depths = self._compute_edge_depths()

    def _build_adjacency(self) -> Dict[int, List[int]]:
        adj = {v: [] for v in self.vertices}
        for src, tgt in self.edges:
            if src in self.vertices and tgt in self.vertices:
                adj[src].append(tgt)
        return adj

    def get_outgoing_edges(self, vertex: int) -> List[int]:
        return self.adjacency.get(vertex, [])

    def _compute_vertex_depths(self) -> Dict[int, int]:
        depths: Dict[int, int] = {}
        for i in range(self.num_roots):
            if i in self.vertices:
                depths[i] = i
        edge_depths = {}
        for src, tgt in self.edges:
            edge_depths[(src, tgt)] = depths.get(src, 0)
        for v in self.vertices:
            if v < self.num_roots:
                continue
            max_depth = 0
            found_edge = False
            for edge in self.edges:
                src, tgt = edge
                if src == v or tgt == v:
                    found_edge = True
                    max_depth = max(max_depth, edge_depths.get(edge, 0))
            depths[v] = max_depth + 1 if found_edge else 0
        return depths

    def _compute_edge_depths(self) -> Dict[Tuple[int, int], int]:
        depths: Dict[Tuple[int, int], int] = {}
        for src, tgt in self.edges:
            depths[(src, tgt)] = self.vertex_depths.get(src, 0)
        return depths

    def get_vertex_depth(self, vertex: int) -> int:
        return self.vertex_depths.get(vertex, 0)

    def get_edge_depth(self, edge: Tuple[int, int]) -> int:
        return self.edge_depths.get(edge, 0)

    @classmethod
    def from_characteristic_tuple(cls, char_tuple: Tuple[int, ...]):
        v = char_tuple[-1]
        root_degrees = char_tuple[:-1]
        num_roots = len(root_degrees)
        vertices = list(range(v))
        edges: List[Tuple[int, int]] = []
        for i, r_i in enumerate(root_degrees):
            root_vertex = i
            for j in range(1, r_i + 1):
                target_vertex = root_vertex + j
                edges.append((root_vertex, target_vertex))
        return cls(vertices, edges, num_roots=num_roots)


# --------------------------- Symbolic calculator ------------------------------
class FASMinorCalculator:
    """
    Symbolic-only calculator for rows of the principal part of [A|b].
    Vertices and edges are indexed locally per component graph.

    Supports targeted variable zeroing: when target_monomial_spec is provided,
    all variables NOT in the target monomial are set to 0 during initialization,
    dramatically simplifying downstream computations.
    """

    def __init__(
        self,
        graphs: List[ComponentGraph],
        *,
        target_monomial_spec: Optional[Dict[Tuple, int]] = None,
        use_symbolic: bool = True,
        enable_simplification: bool = True,
        simplification_threshold: int = 10000,
        use_lazy_structure_functions: bool = True,
        show_performance_warnings: bool = False,
    ):
        if not use_symbolic:
            raise ValueError("Only symbolic mode is supported; numeric computation of minors is not defined.")
        self.graphs = graphs
        self.use_symbolic = True
        self.enable_simplification = enable_simplification
        self.simplification_threshold = simplification_threshold
        self.use_lazy_structure_functions = use_lazy_structure_functions
        self.show_performance_warnings = show_performance_warnings

        self.matrix_entries: Dict[Any, Any] = {}
        self._h_action_cache = None
        self._expanded_expr_cache: Dict[int, Any] = {}
        self._q_cache: Dict[Any, Any] = {}

        # Store target monomial spec for early zeroing
        self.target_monomial_spec = target_monomial_spec
        self._target_var_keys = self._parse_monomial_keys(target_monomial_spec)

        self._initialize_symbolic_variables()
        # structure functions cache (lazy)
        self.structure_functions_symbolic: Dict[Tuple, Any] = {}

    def _parse_monomial_keys(self, spec: Optional[Dict[Tuple, int]]) -> Set[Tuple]:
        """Extract variable keys from monomial spec for filtering.

        Args:
            spec: Monomial specification dict, e.g.,
                  {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}

        Returns:
            Set of variable keys to KEEP (not zero out).
        """
        if spec is None:
            return set()  # No filtering - all variables are symbols
        keys = set()
        for key, exp in spec.items():
            if exp <= 0:
                continue
            if not isinstance(key, tuple) or len(key) < 3:
                continue
            kind = key[0]
            if kind == 'vertex':
                keys.add(('vertex', key[1], key[2]))
            elif kind == 'edge':
                keys.add(('edge', key[1], key[2]))
        return keys

    def _should_create_symbol(self, kind: str, g_idx: int, local_id: Any) -> bool:
        """Return True if this variable should be a symbol, False if it should be zero.

        Args:
            kind: 'vertex' or 'edge'
            g_idx: Graph index
            local_id: Vertex number (int) or edge tuple

        Returns:
            True to create a symbol, False to create sp.Integer(0)
        """
        if not self._target_var_keys:
            return True  # No filtering, create all symbols
        return (kind, g_idx, local_id) in self._target_var_keys

    @classmethod
    def from_characteristic_tuples(
        cls,
        char_tuples: List[Tuple[int, ...]],
        *,
        target_monomial_spec: Optional[Dict[Tuple, int]] = None,
        use_symbolic: bool = True,
        enable_simplification: bool = True,
        simplification_threshold: int = 10000,
        use_lazy_structure_functions: bool = True,
        show_performance_warnings: bool = False,
    ) -> "FASMinorCalculator":
        graphs: List[ComponentGraph] = [ComponentGraph.from_characteristic_tuple(t) for t in char_tuples]
        return cls(
            graphs,
            target_monomial_spec=target_monomial_spec,
            use_symbolic=use_symbolic,
            enable_simplification=enable_simplification,
            simplification_threshold=simplification_threshold,
            use_lazy_structure_functions=use_lazy_structure_functions,
            show_performance_warnings=show_performance_warnings,
        )

    def _initialize_symbolic_variables(self) -> None:
        """Initialize symbolic variables, setting non-target variables to zero.

        When target_monomial_spec is provided, variables NOT in the target
        monomial are set to sp.Integer(0) instead of creating symbols.
        This causes all downstream computations to automatically simplify.
        """
        self.vertex_variables: Dict[Tuple[int, int], Any] = {}
        self.edge_variables: Dict[Tuple[int, Tuple[int, int]], Any] = {}

        for g_idx, graph in enumerate(self.graphs):
            for vertex in graph.vertices:
                if self._should_create_symbol('vertex', g_idx, vertex):
                    self.vertex_variables[(g_idx, vertex)] = sp.Symbol(f'u_{{{g_idx},{vertex}}}')
                else:
                    self.vertex_variables[(g_idx, vertex)] = sp.Integer(0)

        for g_idx, graph in enumerate(self.graphs):
            for edge in graph.edges:
                src, tgt = edge
                if self._should_create_symbol('edge', g_idx, edge):
                    self.edge_variables[(g_idx, edge)] = sp.Symbol(f'u_{{{g_idx},({src},{tgt})}}')
                else:
                    self.edge_variables[(g_idx, edge)] = sp.Integer(0)

        # Only include actual symbols in the set (not zeros)
        self._vertex_symbol_set = {
            v for v in self.vertex_variables.values()
            if isinstance(v, sp.Symbol)
        }

    # ----------------------- Structure function helpers -----------------------
    def _get_structure_function(self, key: Tuple) -> Any:
        if key in self.structure_functions_symbolic:
            return self.structure_functions_symbolic[key]
        if not self.use_lazy_structure_functions:
            return 0
        return self._create_structure_function(key)

    def _create_structure_function(self, key: Tuple) -> Any:
        index_type_a, val_a, index_type_b, val_b, index_type_c, val_c = key

        if index_type_a == 'edge' and index_type_b == 'vertex' and index_type_c == 'vertex':
            # c^k_{i,j}: k edge, i vertex, j vertex
            g_k, edge_k = val_a
            g_i, vertex_i = val_b
            g_j, vertex_j = val_c
            k_src, k_tgt = edge_k
            graph_k = self.graphs[g_k]
            # Constraint 0: diagonal zero when i=j
            if g_i == g_j and vertex_i == vertex_j:
                self.structure_functions_symbolic[key] = 0
                return 0
            # Constraint 1: all in same component
            if g_i != g_j or g_i != g_k:
                self.structure_functions_symbolic[key] = 0
                return 0
            # Constraint 2: depth(k) <= min(depth(i), depth(j))
            depth_k = graph_k.get_edge_depth(edge_k)
            depth_i = graph_k.get_vertex_depth(vertex_i)
            depth_j = graph_k.get_vertex_depth(vertex_j)
            if depth_k > min(depth_i, depth_j):
                self.structure_functions_symbolic[key] = 0
                return 0
            # Constraint 3: root-depth refinement
            if depth_i == depth_j:
                i_is_root = vertex_i < graph_k.num_roots
                j_is_root = vertex_j < graph_k.num_roots
                if i_is_root or j_is_root:
                    if depth_k >= depth_i:
                        self.structure_functions_symbolic[key] = 0
                        return 0
            # Constraint 4: edge-directed constraints give ±1 or 0
            edge_ij = (vertex_i, vertex_j)
            edge_ji = (vertex_j, vertex_i)
            if edge_ij in graph_k.edges:
                if edge_k == edge_ij:
                    self.structure_functions_symbolic[key] = 1
                    return 1
                else:
                    self.structure_functions_symbolic[key] = 0
                    return 0
            elif edge_ji in graph_k.edges:
                if edge_k == edge_ji:
                    self.structure_functions_symbolic[key] = -1
                    return -1
                else:
                    self.structure_functions_symbolic[key] = 0
                    return 0
            # Otherwise symbolic parameter
            symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_i},{vertex_i}),({g_j},{vertex_j})}}'
            result = sp.Symbol(symbol_name)
            self.structure_functions_symbolic[key] = result
            return result

        if index_type_a == 'edge' and index_type_b == 'edge' and index_type_c == 'vertex':
            # c^k_{l,i}: k,l edges; i vertex (Type 2 used in q when j is edge)
            g_k, edge_k = val_a
            g_l, edge_l = val_b
            g_i, vertex_i = val_c
            k_src, k_tgt = edge_k
            l_src, l_tgt = edge_l
            symbol_name = f'c^{{{g_k},({k_src},{k_tgt})}}_{{({g_l},({l_src},{l_tgt})),({g_i},{vertex_i})}}'
            result = sp.Symbol(symbol_name)
            self.structure_functions_symbolic[key] = result
            return result

        if index_type_a == 'vertex' and index_type_b == 'vertex' and index_type_c == 'vertex':
            # c^l_{w,v}: l, w, v vertices (used in b)
            g_l, vertex_l = val_a
            g_w, vertex_w = val_b
            g_v, vertex_v = val_c
            # Diagonal zero w=v
            if g_w == g_v and vertex_w == vertex_v:
                self.structure_functions_symbolic[key] = 0
                return 0
            # Edge constraint: zero if (w,v) or (v,w) is an edge in the same component
            if g_w == g_v:
                graph_for_check = self.graphs[g_w]
                edge_wv = (vertex_w, vertex_v)
                edge_vw = (vertex_v, vertex_w)
                if edge_wv in graph_for_check.edges or edge_vw in graph_for_check.edges:
                    self.structure_functions_symbolic[key] = 0
                    return 0
            # Component locality: if w and v in same component, l must also be there
            if g_w == g_v and g_l != g_w:
                self.structure_functions_symbolic[key] = 0
                return 0
            symbol_name = f'c^{{({g_l},{vertex_l})}}_{{({g_w},{vertex_w}),({g_v},{vertex_v})}}'
            result = sp.Symbol(symbol_name)
            self.structure_functions_symbolic[key] = result
            return result

        # Default: zero for unsupported signature
        self.structure_functions_symbolic[key] = 0
        return 0

    # ----------------------------- Core operations ----------------------------
    def _compute_q(self, j, k, graph_idx: int) -> Any:
        if not self.use_symbolic:
            raise ValueError("q is only defined symbolically.")
        g_j, local_j = j
        j_type = 'vertex' if isinstance(local_j, int) else 'edge'
        g_k, local_k = k
        cache_key = (j_type, (g_j, local_j), (g_k, local_k))
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]
        result = 0
        j_val = (g_j, local_j)
        k_val = (g_k, local_k)
        for g_i, graph_i in enumerate(self.graphs):
            for vertex_i in graph_i.vertices:
                if j_type == 'vertex':
                    key = ('edge', k_val, 'vertex', (g_i, vertex_i), 'vertex', j_val)
                else:
                    key = ('edge', k_val, 'edge', j_val, 'vertex', (g_i, vertex_i))
                c_coeff = self._get_structure_function(key)
                u_i = self.vertex_variables.get((g_i, vertex_i), 0)
                if c_coeff != 0 and u_i != 0:
                    result = result + c_coeff * u_i
        self._q_cache[cache_key] = result
        return result

    def _build_h_action(self) -> Dict[sp.Symbol, Any]:
        h_action: Dict[sp.Symbol, Any] = {}
        for g_j, graph_j in enumerate(self.graphs):
            for vertex_j in graph_j.vertices:
                u_j = self.vertex_variables[(g_j, vertex_j)]
                # Skip if u_j is zero (not a symbol)
                if not isinstance(u_j, sp.Symbol):
                    continue
                result = 0
                for edge_k in graph_j.edges:
                    u_k = self.edge_variables[(g_j, edge_k)]
                    for vertex_i in graph_j.vertices:
                        u_i = self.vertex_variables[(g_j, vertex_i)]
                        key = ('edge', (g_j, edge_k), 'vertex', (g_j, vertex_i), 'vertex', (g_j, vertex_j))
                        c_coeff = self._get_structure_function(key)
                        if c_coeff != 0 and u_i != 0 and u_k != 0:
                            result = result + c_coeff * u_i * u_k
                h_action[u_j] = result
        return h_action

    def _apply_derivation(self, expr, action_map: Dict[sp.Symbol, Any]) -> Any:
        if not self.use_symbolic:
            return 0
        if expr == 0:
            return 0
        if expr.is_Add:
            return sum(self._apply_derivation(arg, action_map) for arg in expr.args)
        if expr.is_Mul:
            result = 0
            args = list(expr.args)
            for i in range(len(args)):
                term = 1
                for j in range(len(args)):
                    term = term * (self._apply_derivation(args[j], action_map) if i == j else args[j])
                result = result + term
            return result
        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            base_h = self._apply_derivation(base, action_map)
            if base_h == 0:
                return 0
            try:
                if exp.is_Number:
                    return exp * (base ** (exp - 1)) * base_h
                else:
                    return sp.diff(base ** exp, base) * base_h
            except Exception:
                return 0
        if expr.is_Symbol:
            return action_map.get(expr, 0)
        return 0

    def _apply_h(self, expr, graph_idx: int) -> Any:
        if not self.use_symbolic:
            return 0
        if expr == 0 or not getattr(expr, 'free_symbols', set()):
            return 0
        if self._h_action_cache is None:
            self._h_action_cache = self._build_h_action()
        return self._apply_derivation(expr, self._h_action_cache)

    def _get_vertex_degree(self, expr) -> int:
        if not self.use_symbolic or expr == 0:
            return 0
        degree = 0
        for symbol in expr.free_symbols:
            if symbol in self._vertex_symbol_set:
                degree += expr.as_coeff_exponent(symbol)[1]
        return degree

    def _extract_principal_part(self, expr, target_vertex_degree: int) -> Any:
        if not self.use_symbolic or expr == 0:
            return expr
        expr_expanded = sp.expand(expr)
        if expr_expanded.is_Add:
            result = 0
            for term in expr_expanded.args:
                if self._get_vertex_degree(term) == target_vertex_degree:
                    result = result + term
            return result
        else:
            if self._get_vertex_degree(expr_expanded) == target_vertex_degree:
                return expr_expanded
            return 0

    def _smart_simplify(self, expr) -> Any:
        if not self.use_symbolic or expr == 0:
            return expr
        try:
            return sp.cancel(expr)
        except Exception:
            return expr

    # --------------------------- Row/entry builders ---------------------------
    def get_row(self, graph_idx: int, vertex: int, layer: int) -> sp.Matrix:
        if layer < 1:
            raise ValueError(f"Layer must be >= 1 (got {layer})")
        if graph_idx < 0 or graph_idx >= len(self.graphs):
            raise ValueError(f"Invalid graph_idx {graph_idx} (must be 0-{len(self.graphs)-1})")
        graph = self.graphs[graph_idx]
        if vertex not in graph.vertices:
            raise ValueError(f"Vertex {vertex} not in graph {graph_idx}")
        col_specs: List[Tuple] = []
        for g_idx, g in enumerate(self.graphs):
            for edge in g.edges:
                col_specs.append(('edge', g_idx, edge))
        col_specs.append(('b', None, None))
        row = sp.Matrix.zeros(1, len(col_specs))
        row_spec = (graph_idx, vertex, layer)
        for j, col_spec in enumerate(col_specs):
            row[j] = self.build_matrix_entry(row_spec, col_spec)
        return row

    def build_matrix_entry(self, row_spec: Tuple[int, int, int], col_spec: Tuple) -> Any:
        row_graph, row_vertex, row_layer = row_spec
        if col_spec[0] == 'b':
            cache_key_b = (row_graph, row_vertex, row_layer, 'b')
            if cache_key_b in self.matrix_entries:
                return self.matrix_entries[cache_key_b]
            if row_layer == 1:
                result = 0
                alpha_v = sp.Symbol(f'α_{{{row_graph}}}')
                for g_l, graph_l in enumerate(self.graphs):
                    if g_l == row_graph:
                        continue
                    for vertex_l in graph_l.vertices:
                        alpha_l = sp.Symbol(f'α_{{{g_l}}}')
                        u_l = self.vertex_variables.get((g_l, vertex_l), 0)
                        for g_w, graph_w in enumerate(self.graphs):
                            for vertex_w in graph_w.vertices:
                                key = ('vertex', (g_l, vertex_l), 'vertex', (g_w, vertex_w), 'vertex', (row_graph, row_vertex))
                                c_coeff = self._get_structure_function(key)
                                if c_coeff != 0 and u_l != 0:
                                    u_w = self.vertex_variables.get((g_w, vertex_w), 0)
                                    if u_w != 0:
                                        term = (alpha_v**2 - alpha_l**2) * c_coeff * u_w * u_l
                                        result = result + term
                self.matrix_entries[cache_key_b] = result
                return result
            # s > 1: recursive b
            prev_layer_spec = (row_graph, row_vertex, row_layer - 1)
            b_prev = self.build_matrix_entry(prev_layer_spec, ('b', None, None))
            h2_term = self._extract_principal_part(self._apply_h(b_prev, row_graph), 2)
            sum_term = 0
            graph_v = self.graphs[row_graph]
            for edge_l in graph_v.edges:
                a_s_vl = self.build_matrix_entry(prev_layer_spec, ('edge', row_graph, edge_l))
                alpha_l = sp.Symbol(f'α_{{{row_graph}}}')
                for g_w, graph_w in enumerate(self.graphs):
                    if g_w == row_graph:
                        continue
                    for edge_w in graph_w.edges:
                        alpha_w = sp.Symbol(f'α_{{{g_w}}}')
                        q_lw = self._compute_q((row_graph, edge_l), (g_w, edge_w), row_graph)
                        u_w = self.edge_variables.get((g_w, edge_w), 0)
                        if a_s_vl != 0 and q_lw != 0 and u_w != 0:
                            term = (alpha_l**2 - alpha_w**2) * a_s_vl * q_lw * u_w
                            sum_term = sum_term + term
            result = h2_term + sum_term
            if row_layer > 2:
                result = self._smart_simplify(result)
            result = self._extract_principal_part(result, 2)
            self.matrix_entries[cache_key_b] = result
            return result

        # Edge column (A-entry)
        col_type, col_graph, edge_w = col_spec
        if row_graph != col_graph:
            return 0
        # A base and recursion
        if row_layer == 1:
            return self._compute_q((row_graph, row_vertex), (col_graph, edge_w), row_graph)
        prev = self.build_matrix_entry((row_graph, row_vertex, row_layer - 1), ('edge', col_graph, edge_w))
        h1 = self._extract_principal_part(self._apply_h(prev, row_graph), 1)
        total = h1
        for edge_l in self.graphs[row_graph].edges:
            a_prev = self.build_matrix_entry((row_graph, row_vertex, row_layer - 1), ('edge', row_graph, edge_l))
            q_lw = self._compute_q((row_graph, edge_l), (row_graph, edge_w), row_graph)
            if a_prev != 0 and q_lw != 0:
                total = total + a_prev * q_lw
        total = self._extract_principal_part(total, 1)
        if row_layer > 2:
            total = self._smart_simplify(total)
        return total


# ----------------------------- Determinants ----------------------------------
class DeterminantComputer:
    """Block-structured fast minor computation from calculator rows."""

    def __init__(self, calculator: FASMinorCalculator):
        if not isinstance(calculator, FASMinorCalculator):
            raise TypeError("calculator must be a FASMinorCalculator instance")
        self.calculator = calculator
        self.base_rows = self._generate_base_rows()
        n_edges = sum(len(g.edges) for g in calculator.graphs)
        if len(self.base_rows) != n_edges:
            raise ValueError(
                f"Base row generation produced {len(self.base_rows)} rows, expected {n_edges} (n-m)."
            )

    @staticmethod
    def minor_from_characteristic_tuples(
        char_tuples: List[Tuple[int, ...]],
        row: Tuple[int, int, int],
        fast: bool = True,
    ) -> sp.Expr:
        calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
        det_comp = DeterminantComputer(calc)
        g, v, s = row
        return det_comp.compute_minor_fast(g, v, s)

    @staticmethod
    def y_from_characteristic_tuples(
        char_tuples: List[Tuple[int, ...]],
        row: Tuple[int, int, int],
    ) -> sp.Matrix:
        calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
        det_comp = DeterminantComputer(calc)
        g, v, s = row
        return det_comp.compute_y_vector(g, v, s)

    def _generate_base_rows(self) -> List[Tuple[int, int, int]]:
        base_rows: List[Tuple[int, int, int]] = []
        for graph_idx, graph in enumerate(self.calculator.graphs):
            omega = graph.num_roots - 1
            for layer in range(1, omega + 2):
                for vertex in sorted(graph.vertices):
                    if graph.get_vertex_depth(vertex) >= layer:
                        base_rows.append((graph_idx, vertex, layer))
        return base_rows

    # Cache per-component blocks and dets
    def _ensure_base_blocks_cache(self) -> None:
        if hasattr(self, "_A_block_by_comp") and self._A_block_by_comp is not None:
            return
        comp_edge_starts: Dict[int, int] = {}
        comp_edge_sizes: Dict[int, int] = {}
        start = 0
        for g_idx, g in enumerate(self.calculator.graphs):
            sz = len(g.edges)
            comp_edge_starts[g_idx] = start
            comp_edge_sizes[g_idx] = sz
            start += sz
        self._comp_edge_starts = comp_edge_starts
        self._comp_edge_sizes = comp_edge_sizes

        base_rows_by_comp: Dict[int, List[Tuple[int, int, int]]] = {g_idx: [] for g_idx, _ in enumerate(self.calculator.graphs)}
        for r in self.base_rows:
            base_rows_by_comp[r[0]].append(r)
        self._base_rows_by_comp = base_rows_by_comp

        A_block_by_comp: Dict[int, sp.Matrix] = {}
        det_base_by_comp: Dict[int, sp.Expr] = {}
        for g_idx, rows_g in base_rows_by_comp.items():
            e_sz = comp_edge_sizes[g_idx]
            if e_sz == 0:
                A_block_by_comp[g_idx] = sp.Matrix(0, 0, [])
                det_base_by_comp[g_idx] = sp.Integer(1)
                continue
            if len(rows_g) != e_sz:
                raise ValueError(
                    f"Base rows for component {g_idx} count {len(rows_g)} does not match edge count {e_sz}."
                )
            block = None
            c0 = comp_edge_starts[g_idx]
            c1 = c0 + e_sz
            for r in rows_g:
                row = self.calculator.get_row(*r)
                row_block = row[:, c0:c1]
                block = row_block if block is None else block.col_join(row_block)
            A_block_by_comp[g_idx] = block
            det_base_by_comp[g_idx] = block.det(method='berkowitz')
        self._A_block_by_comp = A_block_by_comp
        self._det_base_by_comp = det_base_by_comp

    def get_base_rows(self) -> List[Tuple[int, int, int]]:
        return self.base_rows.copy()

    def compute_minor_fast(self, graph_idx: int, vertex: int, layer: int) -> sp.Expr:
        user_row = (graph_idx, vertex, layer)
        all_rows = self.base_rows + [user_row]

        # Build full matrix: one row per (base_rows + extra), all columns (edges + b)
        rows_list = []
        for row_spec in all_rows:
            row = self.calculator.get_row(*row_spec)
            rows_list.append(row)

        full_matrix = rows_list[0]
        for r in rows_list[1:]:
            full_matrix = full_matrix.col_join(r)

        # Use SymPy's built-in determinant (berkowitz is division-free)
        return full_matrix.det(method='berkowitz')

    def compute_y_vector(self, graph_idx: int, vertex: int, layer: int) -> sp.Matrix:
        self._ensure_base_blocks_cache()
        A_star = self._A_block_by_comp[graph_idx]
        e_star = self._comp_edge_sizes[graph_idx]
        if e_star == 0:
            raise ValueError(f"Component {graph_idx} has zero edges; no A_star block exists.")
        extra_row = self.calculator.get_row(graph_idx, vertex, layer)
        c0 = self._comp_edge_starts[graph_idx]
        extra_block = extra_row[:, c0:c0 + e_star]
        return A_star.T.LUsolve(extra_block.T)

    # ------------------------- Base-A utilities and p spec -------------------------
    def compute_base_A_determinant(self) -> sp.Expr:
        """Product of per-component base A-block determinants."""
        self._ensure_base_blocks_cache()
        prod = sp.Integer(1)
        for det_val in self._det_base_by_comp.values():
            prod *= det_val
        return prod

    def _get_u_gens(self) -> List[sp.Symbol]:
        """Stable ordered list of all u symbols: vertices then edges."""
        vv = self.calculator.vertex_variables
        ev = self.calculator.edge_variables
        # Filter out zeros (non-symbols)
        verts = sorted(
            [(k, v) for k, v in vv.items() if isinstance(v, sp.Symbol)],
            key=lambda kv: (kv[0][0], kv[0][1])
        )
        edges = sorted(
            [(k, v) for k, v in ev.items() if isinstance(v, sp.Symbol)],
            key=lambda kv: (kv[0][0], kv[0][1][0], kv[0][1][1])
        )
        return [sym for _, sym in verts] + [sym for _, sym in edges]

    def _build_monomial_from_spec(self, monomial_spec: Dict[Any, int]) -> sp.Expr:
        mono = sp.Integer(1)
        for key, exp in monomial_spec.items():
            if exp == 0:
                continue
            if not isinstance(exp, int) or exp < 0:
                raise ValueError(f"Exponent must be a non-negative integer for key {key} (got {exp})")
            if not isinstance(key, tuple) or len(key) < 3:
                raise ValueError(f"Invalid monomial key format: {key}")
            kind = key[0]
            if kind == 'vertex' and len(key) == 3:
                g, v = key[1], key[2]
                sym = self.calculator.vertex_variables.get((g, v))
                if sym is None or not isinstance(sym, sp.Symbol):
                    # Variable was zeroed or doesn't exist
                    return sp.Integer(0)
            elif kind == 'edge' and len(key) == 3:
                g, edge = key[1], key[2]
                sym = self.calculator.edge_variables.get((g, edge))
                if sym is None or not isinstance(sym, sp.Symbol):
                    # Variable was zeroed or doesn't exist
                    return sp.Integer(0)
            else:
                raise ValueError(
                    f"Invalid monomial key format: {key}. Expected ('vertex', g, v) or ('edge', g, (src, tgt))"
                )
            mono *= sym ** exp
        return mono

    def coeff_of_monomial(self, expr: sp.Expr, monomial_spec: Dict[Any, int], match: str = 'exact') -> sp.Expr:
        """
        Extract coefficient or divides-residual for a monomial via Poly.

        - match='exact': exact coefficient of the monomial.
        - match='divides': residual polynomial after factoring out monomial from matching terms.
        """
        gens = self._get_u_gens()
        if not gens:
            # No variables - just return the expression itself if monomial is 1
            if all(exp == 0 for exp in monomial_spec.values()):
                return expr
            return sp.Integer(0)

        mono = self._build_monomial_from_spec(monomial_spec)
        if mono == 0:
            # Monomial involves zeroed variables - coefficient is 0
            return sp.Integer(0)

        if match not in ('exact', 'divides'):
            raise ValueError("match must be 'exact' or 'divides'")

        # Defensive: cancel any remaining rational expressions before creating Poly
        try:
            expr = sp.cancel(expr)
        except Exception:
            # If cancel fails, try to continue anyway
            pass

        poly = sp.Poly(expr, *gens, domain='EX')
        if match == 'exact':
            return poly.coeff_monomial(mono)

        # divides
        mono_pows = mono.as_powers_dict()
        mexp = [int(mono_pows.get(s, 0)) for s in gens]
        residual = sp.Integer(0)
        for exps, coeff in poly.terms():
            if all(exps[i] >= mexp[i] for i in range(len(gens))):
                res = sp.Integer(1)
                for i, e in enumerate(exps):
                    re = e - mexp[i]
                    if re:
                        res *= gens[i] ** re
                residual += coeff * res
        return residual

    def block_root_product_spec(self, component: int) -> Dict[Any, int]:
        """Return p_i spec for component using base-row layer counts."""
        self._ensure_base_blocks_cache()
        if component < 0 or component >= len(self.calculator.graphs):
            raise IndexError(f"Invalid component index: {component}")
        rows = self._base_rows_by_comp[component]
        # Count rows per layer
        counts: Dict[int, int] = {}
        for _, _, s in rows:
            counts[s] = counts.get(s, 0) + 1
        spec: Dict[Any, int] = {}
        def acc(key: Any, inc: int) -> None:
            if inc:
                spec[key] = spec.get(key, 0) + inc
        # Vertex root var appears in all layer monomials
        for s, N_s in counts.items():
            acc(('vertex', component, 0), N_s)
            for k in range(0, max(0, s - 1)):
                edge = (k, k + 1)
                acc(('edge', component, edge), N_s)
        return spec

    def base_A_root_product_spec(self) -> Dict[Any, int]:
        combined: Dict[Any, int] = {}
        def acc(key: Any, inc: int) -> None:
            if inc:
                combined[key] = combined.get(key, 0) + inc
        for g_idx, _ in enumerate(self.calculator.graphs):
            bi = self.block_root_product_spec(g_idx)
            for k, e in bi.items():
                acc(k, e)
        return combined

    # ----------------------- p-divides coefficient in minor ----------------------
    def coeff_divides_p_in_minor(self, graph_idx: int, vertex: int, layer: int, *, expand: bool = False) -> sp.Expr:
        """
        Exact residual polynomial after factoring global p from the minor using Poly.

        Computes the minor via direct determinant, then extracts the p-divides residual.
        """
        # Compute the minor using direct determinant
        minor = self.compute_minor_fast(graph_idx, vertex, layer)

        if expand:
            minor = sp.expand(minor)

        # Poly-based divides residual of p
        p_spec = self.base_A_root_product_spec()
        return self.coeff_of_monomial(minor, p_spec, match='divides')


# -------------------------- Convenience Functions -----------------------------
def compute_monomial_coefficient(
    char_tuples: List[Tuple[int, ...]],
    extra_row: Tuple[int, int, int],
    monomial_spec: Dict[Any, int],
    match: str = 'exact',
) -> sp.Expr:
    """
    One-shot API for computing coefficient of a monomial in a FAS minor.

    Uses targeted variable zeroing for efficient computation: variables NOT
    in the target monomial are set to 0 during row construction, preventing
    expression explosion.

    IMPORTANT: This function is optimized for match='exact' mode. For
    match='divides', the residual polynomial may contain variables beyond
    those in the target monomial - these will be zeroed out, potentially
    giving incorrect results. For accurate 'divides' results, use the
    standard FASMinorCalculator without targeted zeroing.

    Args:
        char_tuples: List of characteristic tuples, e.g., [(2,1,4), (2,1,3)]
        extra_row: (graph_idx, vertex, layer) for the extra row
        monomial_spec: Target monomial specification, e.g.,
                       {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}
        match: 'exact' for exact coefficient (recommended with zeroing),
               'divides' for residual (may be inaccurate with zeroing)

    Returns:
        SymPy expression for the coefficient.

    Example:
        >>> mono = {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}
        >>> coeff = compute_monomial_coefficient(
        ...     [(2,1,4), (2,1,3)],
        ...     (0, 0, 1),
        ...     mono
        ... )
    """
    calc = FASMinorCalculator.from_characteristic_tuples(
        char_tuples,
        target_monomial_spec=monomial_spec,
    )
    det_comp = DeterminantComputer(calc)
    minor = det_comp.compute_minor_fast(*extra_row)
    return det_comp.coeff_of_monomial(minor, monomial_spec, match=match)


def compute_p_coefficient(
    char_tuples: List[Tuple[int, ...]],
    extra_row: Tuple[int, int, int],
    match: str = 'divides',
) -> sp.Expr:
    """
    Convenience function: compute coefficient of p (root product) in minor.

    Auto-generates the p_spec (base A root product monomial) from the
    characteristic tuples, then uses targeted zeroing to compute efficiently.

    WARNING: For match='divides' (the default), this uses targeted zeroing
    which may give incorrect results because the residual polynomial can
    contain variables beyond p. For accurate 'divides' results, use
    DeterminantComputer.coeff_divides_p_in_minor() instead.

    For match='exact', targeted zeroing gives correct results.

    Args:
        char_tuples: List of characteristic tuples
        extra_row: (graph_idx, vertex, layer) for the extra row
        match: 'exact' for exact coefficient,
               'divides' for residual (may be inaccurate with zeroing)

    Returns:
        SymPy expression for the coefficient of p.

    Example:
        >>> coeff = compute_p_coefficient(
        ...     [(2,1,4), (2,1,3)],
        ...     (0, 0, 1),
        ...     match='exact'
        ... )
    """
    # First, build temporary calculator to get p_spec
    temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    temp_det = DeterminantComputer(temp_calc)
    p_spec = temp_det.base_A_root_product_spec()

    # Now compute with targeted zeroing
    return compute_monomial_coefficient(char_tuples, extra_row, p_spec, match=match)


# --------------------------------- Parsers ------------------------------------
def _parse_tuples(s: str) -> List[Tuple[int, ...]]:
    """Parse characteristic tuples from semicolon-separated format.

    Args:
        s: String like "2,1,4;3,1,5" where tuples are separated by semicolons
           and tuple elements are comma-separated integers.

    Returns:
        List of tuples of integers.

    Raises:
        ValueError: If the input format is invalid.
    """
    try:
        tuples = []
        for part in s.split(';'):
            part = part.strip()
            if not part:
                continue
            elements = part.split(',')
            if not elements or any(not e.strip() for e in elements):
                raise ValueError(f"Empty element in tuple: '{part}'")
            tuple_vals = tuple(int(x.strip()) for x in elements)
            if len(tuple_vals) < 2:
                raise ValueError(f"Tuple must have at least 2 elements (got {len(tuple_vals)}): '{part}'")
            tuples.append(tuple_vals)
        if not tuples:
            raise ValueError("No valid tuples found in input")
        return tuples
    except ValueError as e:
        if "invalid literal for int()" in str(e):
            raise ValueError(
                f"Invalid characteristic tuple format: non-integer value found. "
                f"Expected format: '2,1,4;3,1,5' (integers separated by commas, tuples separated by semicolons)"
            ) from e
        raise


def _parse_monomial(s: str) -> Dict[Tuple, int]:
    """
    Parse monomial spec from string format.

    Format: "v:g,vertex;e:g,(src,tgt)" or with exponents "v:g,vertex:exp"
    Default exponent is 1 if not specified.

    Args:
        s: Monomial string, e.g., "v:0,0;e:1,(0,1)"

    Returns:
        Monomial specification dict.

    Examples:
        "v:0,0" -> {('vertex', 0, 0): 1}
        "v:0,0:2" -> {('vertex', 0, 0): 2}
        "v:0,0;e:1,(0,1)" -> {('vertex', 0, 0): 1, ('edge', 1, (0, 1)): 1}

    Raises:
        ValueError: If the input format is invalid.
    """
    if not s or not s.strip():
        raise ValueError("Empty monomial specification")

    spec: Dict[Tuple, int] = {}
    for part in s.split(';'):
        part = part.strip()
        if not part:
            continue

        if part.startswith('v:'):
            # vertex format: v:g,vertex or v:g,vertex:exp
            rest = part[2:]
            # Check if exponent is specified (look for :digit at the end)
            # Need to be careful: "0,0:2" has exponent, "0,0" does not
            last_colon = rest.rfind(':')
            if last_colon > 0:
                # Check if what comes after is just a number (exponent)
                potential_exp = rest[last_colon+1:]
                if potential_exp.isdigit():
                    coords = rest[:last_colon]
                    exp = int(potential_exp)
                else:
                    coords = rest
                    exp = 1
            else:
                coords = rest
                exp = 1

            coord_parts = coords.split(',')
            if len(coord_parts) != 2:
                raise ValueError(f"Invalid vertex spec: {part}. Expected 'v:g,vertex' format.")
            try:
                g, v = int(coord_parts[0].strip()), int(coord_parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid vertex indices in: {part}")
            spec[('vertex', g, v)] = exp

        elif part.startswith('e:'):
            # edge format: e:g,(src,tgt) or e:g,(src,tgt):exp
            rest = part[2:]
            # Find the edge tuple
            try:
                paren_start = rest.index('(')
                paren_end = rest.index(')')
            except ValueError:
                raise ValueError(f"Invalid edge spec: {part}. Expected 'e:g,(src,tgt)' format.")

            try:
                g = int(rest[:paren_start].rstrip(',').strip())
            except ValueError:
                raise ValueError(f"Invalid graph index in: {part}")

            edge_str = rest[paren_start+1:paren_end]
            try:
                edge_parts = edge_str.split(',')
                if len(edge_parts) != 2:
                    raise ValueError()
                src, tgt = int(edge_parts[0].strip()), int(edge_parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid edge tuple in: {part}")

            # Check for exponent after closing paren
            after_paren = rest[paren_end+1:].strip()
            if after_paren.startswith(':'):
                try:
                    exp = int(after_paren[1:])
                except ValueError:
                    raise ValueError(f"Invalid exponent in: {part}")
            else:
                exp = 1

            spec[('edge', g, (src, tgt))] = exp
        else:
            raise ValueError(f"Unknown monomial part type: {part}. Expected 'v:' or 'e:' prefix.")

    if not spec:
        raise ValueError("No valid monomial terms found")

    return spec


# ----------------------------------- CLI -------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Targeted FAS minor calculator with early variable zeroing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard minor computation (same as hpc_fas_minor.py)
  python targeted_fas_minor.py --tuples "2,1,4;2,1,3" --row "0,0,1"

  # Compute coefficient of a specific monomial (with variable zeroing)
  python targeted_fas_minor.py --tuples "2,1,4;2,1,3" --row "0,0,1" --monomial "v:0,0;e:1,(0,1)"

  # Compute coefficient of p (root product) with variable zeroing
  python targeted_fas_minor.py --tuples "2,1,4;2,1,3" --row "0,0,1" --coeff-p

Monomial format:
  "v:g,vertex" - vertex variable u_{g,vertex}
  "e:g,(src,tgt)" - edge variable u_{g,(src,tgt)}
  "v:g,vertex:exp" - with explicit exponent
  Separate multiple terms with semicolons.
"""
    )
    ap.add_argument("--tuples", required=True, help='Characteristic tuples, e.g. "2,1,4;2,1,3"')
    ap.add_argument("--row", required=True, help='Extra row as "graph_idx,vertex,layer", e.g. "0,0,1"')
    ap.add_argument("--out-prefix", default="minor_out", help="Output file prefix")
    ap.add_argument("--latex", action="store_true", help="Also write LaTeX output (minor.tex)")
    ap.add_argument("--monomial", help='Target monomial for coefficient extraction, e.g. "v:0,0;e:1,(0,1)"')
    ap.add_argument("--coeff-p", action="store_true", help="Compute coefficient of p (root product) with targeted zeroing")
    ap.add_argument("--coeff-mode", choices=['exact', 'divides'], default='exact',
                    help="Coefficient extraction mode: 'exact' (recommended) or 'divides'. "
                         "Note: 'divides' with --monomial/--coeff-p may be inaccurate due to zeroing")
    ap.add_argument("--coeff-p-divides", action="store_true",
                    help="Legacy: compute p-divides residual without targeted zeroing")
    ap.add_argument("--expand", action="store_true", help="Expand minor before coefficient extraction")
    args = ap.parse_args()

    # Parse characteristic tuples
    try:
        char_tuples = _parse_tuples(args.tuples)
    except ValueError as e:
        ap.error(f"Error parsing --tuples: {e}")

    # Parse row specification
    try:
        row_parts = args.row.split(",")
        if len(row_parts) != 3:
            raise ValueError(f"Expected 3 values (graph_idx, vertex, layer), got {len(row_parts)}")
        g, v, s = (int(x.strip()) for x in row_parts)
    except ValueError as e:
        ap.error(f"Error parsing --row: {e}. Expected format: '0,0,1' (graph_idx,vertex,layer)")

    # Parse monomial if provided
    monomial_spec = None
    if args.monomial:
        try:
            monomial_spec = _parse_monomial(args.monomial)
        except ValueError as e:
            ap.error(f"Error parsing --monomial: {e}")

    # Handle --coeff-p: auto-generate p_spec
    if args.coeff_p:
        if monomial_spec:
            ap.error("Cannot specify both --monomial and --coeff-p")
        # Build temporary calculator to get p_spec
        temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
        temp_det = DeterminantComputer(temp_calc)
        monomial_spec = temp_det.base_A_root_product_spec()
        # Default to 'divides' mode for p coefficient
        if args.coeff_mode == 'exact':
            args.coeff_mode = 'divides'

    # Create calculator (with monomial targeting if specified)
    calc = FASMinorCalculator.from_characteristic_tuples(
        char_tuples,
        target_monomial_spec=monomial_spec,
        use_symbolic=True,
        enable_simplification=True,
        simplification_threshold=100000,
        use_lazy_structure_functions=True,
    )
    det_comp = DeterminantComputer(calc)

    # Compute minor
    minor = det_comp.compute_minor_fast(g, v, s)

    if args.expand:
        minor = sp.expand(minor)

    # Compute y-vector (only if not using targeted zeroing, as it may not work correctly)
    y = None
    if not monomial_spec:
        try:
            y = det_comp.compute_y_vector(g, v, s)
        except Exception:
            y = None

    # Build metadata
    meta = {
        "tuples": char_tuples,
        "row": [g, v, s],
        "minor_len": len(str(minor)),
        "minor_free_symbols": sorted(str(sym) for sym in getattr(minor, 'free_symbols', [])),
        "targeted_monomial": monomial_spec is not None,
    }
    if y is not None:
        meta["y_shape"] = list(y.shape)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write minor outputs
    with open(f"{args.out_prefix}.minor.txt", "w", encoding="utf-8") as f:
        f.write(str(minor))
    with open(f"{args.out_prefix}.minor.srepr", "w", encoding="utf-8") as f:
        f.write(sp.srepr(minor))
    if y is not None:
        with open(f"{args.out_prefix}.y.txt", "w", encoding="utf-8") as f:
            f.write(str(y))
    with open(f"{args.out_prefix}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if args.latex:
        with open(f"{args.out_prefix}.minor.tex", "w", encoding="utf-8") as f:
            f.write(sp.latex(minor))

    # Extract and write coefficient if monomial was specified
    if monomial_spec:
        coeff = det_comp.coeff_of_monomial(minor, monomial_spec, match=args.coeff_mode)
        with open(f"{args.out_prefix}.coeff.txt", "w", encoding="utf-8") as f:
            f.write(str(coeff))
        with open(f"{args.out_prefix}.coeff.srepr", "w", encoding="utf-8") as f:
            f.write(sp.srepr(coeff))

        coeff_meta = {
            "monomial_spec": {str(k): v for k, v in monomial_spec.items()},
            "coeff_mode": args.coeff_mode,
            "coeff_is_zero": coeff == 0,
            "coeff_len": len(str(coeff)),
            "coeff_free_symbols": sorted(str(sym) for sym in getattr(coeff, 'free_symbols', [])),
        }
        with open(f"{args.out_prefix}.coeff.meta.json", "w", encoding="utf-8") as f:
            json.dump(coeff_meta, f, indent=2)

    # Legacy: --coeff-p-divides without targeted zeroing
    if args.coeff_p_divides and not args.coeff_p:
        res = det_comp.coeff_divides_p_in_minor(g, v, s, expand=args.expand)
        with open(f"{args.out_prefix}.coeff_p_divides.txt", "w", encoding="utf-8") as f:
            f.write(str(res))
        with open(f"{args.out_prefix}.coeff_p_divides.srepr", "w", encoding="utf-8") as f:
            f.write(sp.srepr(res))
        cmeta = {
            "p_spec": {str(k): v for k, v in det_comp.base_A_root_product_spec().items()},
            "residual_len": len(str(res)),
            "residual_free_symbols": sorted(str(sym) for sym in getattr(res, 'free_symbols', [])),
        }
        with open(f"{args.out_prefix}.coeff_p_divides.meta.json", "w", encoding="utf-8") as f:
            json.dump(cmeta, f, indent=2)


if __name__ == "__main__":
    main()
