"""
Determinant Computer Module

This module provides an interface between the FASMinorCalculator and
SymPy's determinant computation functionality. It allows computing
determinants of matrices assembled from specific rows of the fundamental system.

Usage Example:
--------------
    from fas_minor_calculator import FASMinorCalculator
    from determinant_computer import DeterminantComputer

    # Create calculator
    calc = FASMinorCalculator.from_characteristic_tuples(
        [(2, 1, 4), (2, 1, 4)],
        use_symbolic=True
    )

    # Create determinant computer
    det_comp = DeterminantComputer(calc)

    # Specify rows for the minor
    row_specs = [
        (0, 0, 1),  # Component 0, vertex 0, layer 1
        (0, 1, 1),  # Component 0, vertex 1, layer 1
        (1, 0, 1),  # Component 1, vertex 0, layer 1
        # ... continue until you have n_edges + 1 rows
    ]

    # Compute determinant
    det = det_comp.compute_determinant(row_specs)
    print(f"Determinant: {det}")
"""

import sympy as sp
from typing import List, Tuple, Dict, Any
from fas_minor_calculator import FASMinorCalculator


class DeterminantComputer:
    """
    Interface for computing determinants from FAS minor calculator rows.

    This class wraps a FASMinorCalculator instance and provides methods
    to compute determinants by specifying which rows to include. The rows
    are retrieved from the calculator, assembled into a matrix, and the
    determinant is computed using SymPy.

    Attributes:
    -----------
    calculator : FASMinorCalculator
        The calculator instance used to retrieve rows
    base_rows : List[Tuple[int, int, int]]
        Automatically generated base rows (n-m rows) for minor computation

    Methods:
    --------
    compute_determinant(row_specs):
        Compute the symbolic determinant of a matrix formed from specified rows
    compute_minor(graph_idx, vertex, layer):
        Compute a symbolic minor using base rows plus one user-specified row
    compute_y_vector(graph_idx, vertex, layer):
        Compute the theoretically significant y-vector from the adjugate system
    get_base_rows():
        Return the automatically generated base rows
    get_base_A_matrix():
        Assemble the (n-m)×(n-m) A-only matrix from base rows
    compute_base_A_determinant():
        Compute det of the base A submatrix (product of per-component dets)
    minor_from_characteristic_tuples(char_tuples, row, fast=True):
        Convenience one-shot API to compute a minor from characteristic tuples
    y_from_characteristic_tuples(char_tuples, row, return_mapping=False, simplify=False):
        One-shot API to compute the Cramer's-rule y-vector from characteristic tuples
    """

    def __init__(self, calculator: FASMinorCalculator):
        """
        Initialize the DeterminantComputer with a FASMinorCalculator instance.

        This automatically generates n-m base rows using a depth-based selection
        algorithm. For each component with omega = num_roots - 1:
        - Layer 1: select vertices with depth >= 1
        - Layer 2: select vertices with depth >= 2
        - ...
        - Layer omega+1: select vertices with depth >= omega+1

        Parameters:
        -----------
        calculator : FASMinorCalculator
            The calculator instance to use for retrieving rows

        Raises:
        -------
        TypeError
            If calculator is not a FASMinorCalculator instance
        ValueError
            If the base row generation doesn't produce exactly n-m rows
        """
        if not isinstance(calculator, FASMinorCalculator):
            raise TypeError(
                f"calculator must be a FASMinorCalculator instance, "
                f"got {type(calculator).__name__}"
            )
        self.calculator = calculator

        # Generate base rows using depth-based algorithm
        self.base_rows = self._generate_base_rows()

        # Validate that we have exactly n-m base rows
        n_edges = sum(len(g.edges) for g in calculator.graphs)
        expected_base_rows = n_edges  # n - m = total_edges

        if len(self.base_rows) != expected_base_rows:
            raise ValueError(
                f"Base row generation produced {len(self.base_rows)} rows, "
                f"but expected exactly {expected_base_rows} rows (n-m where n={calculator.n}, "
                f"m={sum(len(g.vertices) for g in calculator.graphs)}, n-m={n_edges}). "
                f"This indicates an issue with the depth-based selection algorithm."
            )

    @staticmethod
    def minor_from_characteristic_tuples(
        char_tuples: List[Tuple[int, ...]],
        row: Tuple[int, int, int],
        fast: bool = True,
    ) -> sp.Expr:
        """
        Convenience one-shot API: construct calculator from characteristic tuples
        and compute the minor for a single user-specified row.

        Args:
            char_tuples: list of per-component characteristic tuples
            row: (graph_idx, vertex, layer)
            fast: if True, use the optimized block-expansion route

        Returns:
            SymPy expression (minor determinant)
        """
        calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
        det_comp = DeterminantComputer(calc)
        g, v, s = row
        return det_comp.compute_minor_fast(g, v, s) if fast else det_comp.compute_minor(g, v, s)

    @staticmethod
    def y_from_characteristic_tuples(
        char_tuples: List[Tuple[int, ...]],
        row: Tuple[int, int, int],
        *,
        return_mapping: bool = False,
        simplify: bool = False,
    ):
        """
        One-shot helper: construct calculator from characteristic tuples and return
        the Cramer's-rule y-vector for the given extra row.

        Args:
            char_tuples: list of per-component characteristic tuples
            row: (graph_idx, vertex, layer) identifying the extra row
            return_mapping: if True, also return the ordered base rows for that component
            simplify: if True, apply sp.cancel to y entries for readability

        Returns:
            y (sp.Matrix) or (y, mapping) if return_mapping=True
        """
        calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
        det_comp = DeterminantComputer(calc)
        g, v, s = row
        return det_comp.compute_y_vector(g, v, s, return_mapping=return_mapping, simplify=simplify)

    def _generate_base_rows(self) -> List[Tuple[int, int, int]]:
        """
        Generate base rows using depth-based selection algorithm.

        For each component with omega = num_roots - 1:
        - Layer s=1: select all vertices with depth >= 1
        - Layer s=2: select all vertices with depth >= 2
        - ...
        - Layer s=omega+1: select all vertices with depth >= omega+1

        Returns:
        --------
        List[Tuple[int, int, int]]
            List of (graph_idx, vertex, layer) tuples forming the base rows
        """
        base_rows = []

        for graph_idx, graph in enumerate(self.calculator.graphs):
            # Get omega for this component
            omega = graph.num_roots - 1

            # Iterate through layers 1 to omega+1
            for layer in range(1, omega + 2):  # +2 because range is exclusive
                # Select all vertices with depth >= layer (deterministic order)
                for vertex in sorted(graph.vertices):
                    vertex_depth = graph.get_vertex_depth(vertex)
                    if vertex_depth >= layer:
                        base_rows.append((graph_idx, vertex, layer))

        return base_rows

    # Internal: build and cache per-component edge slices, base row groups,
    # A-blocks, and their determinants for reuse across computations.
    def _ensure_base_blocks_cache(self) -> None:
        if hasattr(self, "_A_block_by_comp") and self._A_block_by_comp is not None:
            return
        # Column grouping per component
        comp_edge_starts = {}
        comp_edge_sizes = {}
        start = 0
        for g_idx, g in enumerate(self.calculator.graphs):
            sz = len(g.edges)
            comp_edge_starts[g_idx] = start
            comp_edge_sizes[g_idx] = sz
            start += sz
        self._comp_edge_starts = comp_edge_starts
        self._comp_edge_sizes = comp_edge_sizes

        # Group base rows by component
        base_rows_by_comp = {g_idx: [] for g_idx, _ in enumerate(self.calculator.graphs)}
        for r in self.base_rows:
            base_rows_by_comp[r[0]].append(r)
        self._base_rows_by_comp = base_rows_by_comp

        # Build A-blocks and their determinants
        A_block_by_comp = {}
        det_base_by_comp = {}
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
        """
        Return the automatically generated base rows.

        The base rows are n-m rows selected using a depth-based algorithm.
        These rows are used in minor computation along with one user-specified row.

        Returns:
        --------
        List[Tuple[int, int, int]]
            List of (graph_idx, vertex, layer) tuples
        """
        return self.base_rows.copy()

    def compute_minor(
        self,
        graph_idx: int,
        vertex: int,
        layer: int
    ) -> sp.Expr:
        """
        Compute a minor using the base rows plus one user-specified row.

        This method combines the n-m automatically generated base rows with
        the user-specified row to form n-m+1 rows total, then computes the
        determinant of the resulting square matrix.

        Parameters:
        -----------
        graph_idx : int
            Index of the component graph (0-based)
        vertex : int
            Local vertex label within that component
        layer : int
            Layer number s >= 1

        Returns:
        --------
        sp.Expr
            The computed determinant (minor) as a SymPy expression.

        Raises:
        -------
        ValueError
            If the row specification is invalid

        Examples:
        ---------
        >>> calc = FASMinorCalculator.from_characteristic_tuples(
        ...     [(3, 1, 5)], use_symbolic=True
        ... )
        >>> det_comp = DeterminantComputer(calc)
        >>> # Compute minor with additional row (0, 0, 1)
        >>> minor = det_comp.compute_minor(0, 0, 1)
        """
        # Auto-route to fast method for the base+one-row pattern
        return self.compute_minor_fast(graph_idx, vertex, layer)

    def compute_minor_fast(
        self,
        graph_idx: int,
        vertex: int,
        layer: int
    ) -> sp.Expr:
        """
        Compute a symbolic minor using a block-structured Laplace expansion.

        This optimized method exploits that, in the principal part, A is block-
        diagonal across components (edge columns grouped by component), while b
        contributes cross-component terms only in the last column. Expanding the
        determinant along the b column yields nonzero cofactors only for rows in
        the same component as the extra row.

        Steps:
        - Build the automatic base rows (n-m rows) already available on this
          instance and append the user-specified row.
        - Group edge columns by component; for each component h compute the
          square base block A_h (rows from base rows in h; columns = edges in h)
          and its determinant det(A_h).
        - Let g* be the component of the user row. For Laplace expansion along
          the last column, only rows from g* contribute. The cofactor minors are:
            • If removing the extra row: product_{h≠g*} det(A_h) · det(A_{g*})
            • If removing a base row r in g*: product_{h≠g*} det(A_h) · det(A_{g*}
              with row r replaced by the extra row’s A-block). The latter equals
              extra_row_block · adj(A_{g*})[:, idx_r] (cofactor row i).
        - Include the Laplace sign (−1)^(i + Ncols) for each contributing row i.

        Parameters:
        - graph_idx: component index of the extra row
        - vertex: local vertex label within that component
        - layer: layer s ≥ 1

        Returns:
        - SymPy expression for the determinant (minor)
        """
        # Assemble row list in the same order as compute_minor: base + user
        user_row = (graph_idx, vertex, layer)
        all_rows = self.base_rows + [user_row]

        # Ensure cached per-component data
        self._ensure_base_blocks_cache()
        comp_edge_starts = self._comp_edge_starts
        comp_edge_sizes = self._comp_edge_sizes
        base_rows_by_comp = self._base_rows_by_comp
        A_block_by_comp = self._A_block_by_comp
        det_base_by_comp = self._det_base_by_comp

        total_edges = sum(comp_edge_sizes.values())
        ncols = total_edges + 1  # +1 for b column

        # Precompute product of det(A_h) for h != g*
        prod_other = sp.Integer(1)
        for h in det_base_by_comp:
            if h != graph_idx:
                prod_other *= det_base_by_comp[h]

        # Build A-block for g*
        A_star = A_block_by_comp[graph_idx]
        e_star = comp_edge_sizes[graph_idx]
        if e_star == 0:
            # No edges in component of the extra row: matrix shape invalid
            raise ValueError("Component of extra row has zero edges; cannot form minor.")
        det_A_star = det_base_by_comp[graph_idx]
        # Optional guard: ensure A_star is invertible per theory (p(u) present)
        try:
            det_chk = sp.simplify(det_A_star)
        except Exception:
            det_chk = det_A_star
        if det_chk == 0:
            raise ValueError(
                f"Base A block for component {graph_idx} is singular (det=0). "
                f"This violates the paper's assumption that det(A_base) contains the p(u) monomial."
            )

        # Build the extra row A-block for g*
        extra_full_row = self.calculator.get_row(*user_row)
        c0s = comp_edge_starts[graph_idx]
        c1s = c0s + e_star
        extra_block = extra_full_row[:, c0s:c1s]

        # Solve A_star^T * y = extra_block^T; then row-replacement det for row i is det(A_star) * y[i]
        y = A_star.T.LUsolve(extra_block.T)

        # Helper: map base row spec in g* to its index within A_star
        comp_rows_star = base_rows_by_comp[graph_idx]
        index_in_star = {r: i for i, r in enumerate(comp_rows_star)}

        # Laplace expansion along the last column: sum over rows in component g*
        det_total = sp.Integer(0)
        for i_global, r in enumerate(all_rows):
            if r[0] != graph_idx:
                continue  # cofactor is zero for other components

            # Compute Laplace sign (1-based indices): (-1)^(i + Ncols)
            sign = -1 if ((i_global + 1 + ncols) % 2) else 1

            # b entry for this row
            b_i = self.calculator.build_matrix_entry(r, ('b', None, None))

            # Cofactor minor determinant for this row
            if r == user_row:
                # Removing the extra row leaves A_star intact
                minor_det = prod_other * det_A_star
            else:
                # Replace the corresponding base row in A_star by extra_block
                idx = index_in_star[r]
                # det(A with row idx replaced) = det(A) * y[idx]
                replaced_det = det_A_star * y[idx, 0]
                minor_det = prod_other * replaced_det

            det_total += sign * b_i * minor_det

        return det_total

    def compute_y_vector(
        self,
        graph_idx: int,
        vertex: int,
        layer: int,
        return_mapping: bool = False,
        simplify: bool = False,
    ) -> sp.Matrix:
        """
        Compute the theoretically significant y-vector from the adjugate system.

        This method explicitly computes the vector y that appears in the Cramer's
        rule-like row-replacement determinant formula. The y-vector is the solution
        to the linear system:

            A_star^T * y = extra_block^T

        where:
        - A_star is the base A-block for the component containing the extra row
        - extra_block is the A-columns from the user-specified extra row for that
          component

        **Theoretical Significance:**

        The y-vector encodes how the extra row interacts with the base rows of the
        same component. Each entry y[i] is the cofactor coefficient used in the
        row-replacement determinant formula:

            det(A with base row i replaced by extra row) = det(A_star) * y[i]

        This is mathematically equivalent to Cramer's rule but computed via LU
        decomposition (more numerically stable). The y-vector appears in the
        Laplace expansion of the full minor:

            minor = Σ (sign * b_r * minor_cofactor_r)

        where the cofactor for each base row r in the same component as the extra
        row is proportional to y[r].

        **Connection to Cramer's Rule:**

        In classical Cramer's rule for solving Ax = b, each solution component x_i
        equals det(A_i)/det(A), where A_i has column i replaced by b. Here, we use
        the transpose system and row replacements, achieving the same mathematical
        result through LU decomposition instead of direct determinant computation.

        Parameters:
        -----------
        graph_idx : int
            Index of the component graph (0-based) for the extra row
        vertex : int
            Local vertex label within that component
        layer : int
            Layer number s >= 1

        Returns:
        --------
        sp.Matrix
            Column vector (shape e_star × 1) where e_star is the number of edges
            in the specified component. Each entry y[i] corresponds to the i-th
            base row from this component.

        Raises:
        -------
        ValueError
            - If the row specification is invalid
            - If the component has zero edges (cannot form system)
            - If the base A-block is singular (violates theoretical assumptions)

        Examples:
        ---------
        >>> calc = FASMinorCalculator.from_characteristic_tuples(
        ...     [(3, 1, 5), (3, 1, 4)], use_symbolic=True
        ... )
        >>> det_comp = DeterminantComputer(calc)
        >>> # Compute y-vector for extra row (0, 0, 1)
        >>> y = det_comp.compute_y_vector(0, 0, 1)
        >>> print(y)  # Column vector with symbolic entries
        >>> # Verify it solves the system: A_star^T * y = extra_block^T
        >>> A_star = det_comp._A_block_by_comp[0]
        >>> extra_row = calc.get_row(0, 0, 1)
        >>> c0 = det_comp._comp_edge_starts[0]
        >>> e = det_comp._comp_edge_sizes[0]
        >>> extra_block = extra_row[:, c0:c0+e]
        >>> residual = A_star.T * y - extra_block.T
        >>> print(residual)  # Should be zero (or very close)

        See Also:
        ---------
        compute_minor_fast : Uses y-vector internally for Laplace expansion
        """
        # Validate inputs (similar to compute_minor_fast)
        if not isinstance(graph_idx, int) or graph_idx < 0 or graph_idx >= len(self.calculator.graphs):
            raise ValueError(f"Invalid graph_idx {graph_idx} (must be 0-{len(self.calculator.graphs)-1})")
        if not isinstance(layer, int) or layer < 1:
            raise ValueError(f"layer must be a positive integer >= 1 (got {layer})")

        user_row = (graph_idx, vertex, layer)

        # Ensure cached per-component data
        self._ensure_base_blocks_cache()
        comp_edge_starts = self._comp_edge_starts
        comp_edge_sizes = self._comp_edge_sizes
        A_block_by_comp = self._A_block_by_comp
        det_base_by_comp = self._det_base_by_comp

        # Get A-block for the specified component
        e_star = comp_edge_sizes[graph_idx]
        if e_star == 0:
            raise ValueError(
                f"Component {graph_idx} has zero edges; cannot form y-vector system."
            )

        A_star = A_block_by_comp[graph_idx]
        det_A_star = det_base_by_comp[graph_idx]

        # Guard: ensure A_star is invertible per theory (p(u) present)
        try:
            det_chk = sp.simplify(det_A_star)
        except Exception:
            det_chk = det_A_star
        if det_chk == 0:
            raise ValueError(
                f"Base A block for component {graph_idx} is singular (det=0). "
                f"This violates the paper's assumption that det(A_base) contains "
                f"the p(u) monomial. Cannot compute y-vector for singular system."
            )

        # Build the extra row A-block for this component
        extra_full_row = self.calculator.get_row(*user_row)
        c0s = comp_edge_starts[graph_idx]
        c1s = c0s + e_star
        extra_block = extra_full_row[:, c0s:c1s]

        # Solve A_star^T * y = extra_block^T using LU decomposition
        # This is the core "Cramer's rule" step
        y = A_star.T.LUsolve(extra_block.T)

        if simplify:
            try:
                y = y.applyfunc(sp.cancel)
            except Exception:
                # Fallback: shallow simplify if cancel fails
                y = y.applyfunc(sp.simplify)

        if return_mapping:
            base_rows_by_comp = self._base_rows_by_comp
            mapping = list(base_rows_by_comp[graph_idx])
            return y, mapping  # type: ignore[return-value]

        return y

    def get_component_base_rows(self, graph_idx: int) -> List[Tuple[int, int, int]]:
        """
        Return the ordered base rows for a given component.

        The order matches the row order used to assemble the per-component A-block
        (A_star) and therefore the indexing of the y-vector returned by
        compute_y_vector(return_mapping=True).

        Args:
            graph_idx: component index (0-based)

        Returns:
            List of (graph_idx, vertex, layer) tuples for this component.
        """
        if not isinstance(graph_idx, int) or graph_idx < 0 or graph_idx >= len(self.calculator.graphs):
            raise ValueError(f"Invalid graph_idx {graph_idx} (must be 0-{len(self.calculator.graphs)-1})")
        self._ensure_base_blocks_cache()
        return list(self._base_rows_by_comp[graph_idx])

    def get_A_star_and_extra_block(
        self, graph_idx: int, vertex: int, layer: int
    ) -> Tuple[sp.Matrix, sp.Matrix, List[Tuple[int, int, int]]]:
        """
        Return (A_star, extra_block, base_rows_in_component) for transparency.

        - A_star: base A-block for the component graph_idx
        - extra_block: A-columns of the user row restricted to that component
        - base_rows_in_component: ordered base rows for this component
        """
        if not isinstance(graph_idx, int) or graph_idx < 0 or graph_idx >= len(self.calculator.graphs):
            raise ValueError(f"Invalid graph_idx {graph_idx} (must be 0-{len(self.calculator.graphs)-1})")
        if not isinstance(layer, int) or layer < 1:
            raise ValueError(f"layer must be a positive integer >= 1 (got {layer})")
        self._ensure_base_blocks_cache()
        A_star = self._A_block_by_comp[graph_idx]
        e_star = self._comp_edge_sizes[graph_idx]
        if e_star == 0:
            raise ValueError(f"Component {graph_idx} has zero edges; no A_star block exists.")
        extra_row = self.calculator.get_row(graph_idx, vertex, layer)
        c0 = self._comp_edge_starts[graph_idx]
        extra_block = extra_row[:, c0:c0 + e_star]
        base_rows_in_component = list(self._base_rows_by_comp[graph_idx])
        return A_star, extra_block, base_rows_in_component

    def get_base_A_matrix(self) -> sp.Matrix:
        """
        Assemble the (n-m) × (n-m) A-only matrix for the base rows.

        Rows: exactly the n-m base rows generated by this instance.
        Columns: all edge columns across all components (exclude the b column).

        Returns:
        - sp.Matrix of shape (n_edges, n_edges)
        """
        # Compute total number of edge columns (exclude b column)
        total_edges = sum(len(g.edges) for g in self.calculator.graphs)

        # Stack each base row, slicing out only the A columns [0:total_edges]
        mat = None
        for r in self.base_rows:
            row = self.calculator.get_row(*r)
            a_slice = row[:, 0:total_edges]
            mat = a_slice if mat is None else mat.col_join(a_slice)
        return mat

    def compute_base_A_determinant(self) -> sp.Expr:
        """
        Compute the determinant of the (n-m) × (n-m) base A submatrix.

        The principal-part A is block-diagonal by component (edge columns grouped
        per component). With base rows grouped by component, the determinant of
        the full base A submatrix equals the product of per-component block
        determinants.

        Returns:
        - SymPy expression: product_h det(A_h^base)
        """
        # Reuse cached per-component A-block determinants
        self._ensure_base_blocks_cache()
        prod_det = sp.Integer(1)
        for g_idx, det_val in self._det_base_by_comp.items():
            prod_det *= det_val
        return prod_det

    # ------------------------ Monomial Search Utilities ------------------------
    def _get_u_gens(self) -> List[sp.Symbol]:
        """Return ordered list of all u symbols (vertex first, then edge)."""
        # Ensure symbolic variables exist
        vv = self.calculator.vertex_variables
        ev = self.calculator.edge_variables
        # Stable ordering: sort by (g, idx)
        vertex_items = sorted(vv.items(), key=lambda kv: kv[0])  # ((g,v), sym)
        edge_items = sorted(ev.items(), key=lambda kv: (kv[0][0], kv[0][1]))  # (((g,(s,t)), sym))
        gens = [sym for _, sym in vertex_items] + [sym for _, sym in edge_items]
        return gens

    def _build_monomial_from_spec(self, monomial_spec: Dict[Any, int]) -> sp.Expr:
        """
        Build a SymPy monomial from a spec mapping variable keys to exponents.

        monomial_spec keys:
          - ('vertex', g, v) for vertex variable u_{g,v}
          - ('edge', g, (src, tgt)) for edge variable u_{g,(src,tgt)}
        Values are non-negative integer exponents.
        """
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
                if sym is None:
                    raise KeyError(f"Unknown vertex variable ('vertex', {g}, {v}) in monomial spec")
            elif kind == 'edge' and len(key) == 3:
                g, edge = key[1], key[2]
                sym = self.calculator.edge_variables.get((g, edge))
                if sym is None:
                    raise KeyError(f"Unknown edge variable ('edge', {g}, {edge}) in monomial spec")
            else:
                raise ValueError(
                    f"Invalid monomial key format: {key}. Expected ('vertex', g, v) or ('edge', g, (src, tgt))"
                )
            mono *= sym**exp
        return mono

    def coeff_of_monomial(self, expr: sp.Expr, monomial_spec: Dict[Any, int], match: str = 'exact') -> sp.Expr:
        """
        Extract coefficient related to a given monomial from a SymPy expression.

        - match='exact': return the exact coefficient of that monomial (0 if absent).
        - match='divides': return the sum of residual terms after factoring out the
          monomial from all terms that contain it (0 if none). Equivalently, this is
          the coefficient of the monomial as a factor, allowing extra u factors.

        monomial_spec keys:
          - ('vertex', g, v): exponent of u_{g,v}
          - ('edge', g, (src, tgt)): exponent of u_{g,(src,tgt)}
        """
        gens = self._get_u_gens()
        mono = self._build_monomial_from_spec(monomial_spec)

        if match not in ('exact', 'divides'):
            raise ValueError("match must be 'exact' or 'divides'")

        poly = sp.Poly(expr, *gens, domain='EX')

        if match == 'exact':
            return poly.coeff_monomial(mono)

        # divides: sum coefficients of all terms that contain 'mono' as a factor,
        # returning the residual polynomial after dividing those terms by 'mono'.
        # Build exponent vector for 'mono' in the 'gens' ordering.
        # Map symbol -> index
        idx = {s: i for i, s in enumerate(gens)}
        # Extract exponent of each gen in mono: use mono.as_powers_dict()
        mono_pows = mono.as_powers_dict()
        mexp = [int(mono_pows.get(s, 0)) for s in gens]

        residual = sp.Integer(0)
        for exps, coeff in poly.terms():
            if all(exps[i] >= mexp[i] for i in range(len(gens))):
                # Build residual monomial for this term
                res = sp.Integer(1)
                for i, e in enumerate(exps):
                    re = e - mexp[i]
                    if re:
                        res *= gens[i]**re
                residual += coeff * res

        return residual

    # -------------------- Root monomial product (base A) helpers --------------------
    def root_monomial_spec_for_layer(self, component: int, layer: int) -> Dict[Any, int]:
        """
        Build the root-monomial spec M_i(layer) for a given component and layer.

        Definition (homogeneity + consecutive roots):
          - M_i(1) = u_{(i,0)}
          - M_i(s) = u_{(i,0)} * u_{i,(0,1)} * u_{i,(1,2)} * ... * u_{i,(s-2,s-1)} for s >= 2

        Args:
            component: component index i
            layer: layer s (1-based)

        Returns:
            Dict mapping monomial keys to exponents (all 1's here)

        Raises:
            IndexError: if component is out of range
            ValueError: if layer is invalid for this component
            KeyError: if an expected consecutive root edge is missing
        """
        if component < 0 or component >= len(self.calculator.graphs):
            raise IndexError(f"Invalid component index: {component}")

        graph = self.calculator.graphs[component]
        omega = graph.num_roots - 1
        if layer < 1 or layer > omega + 1:
            raise ValueError(
                f"Invalid layer {layer} for component {component}; expected 1..{omega+1}"
            )

        spec: Dict[Any, int] = {}
        # Vertex root variable always present
        spec[('vertex', component, 0)] = 1

        # Include consecutive root edges (k, k+1) up to s-2
        for k in range(0, max(0, layer - 1)):
            edge = (k, k + 1)
            if edge not in graph.edges:
                # In canonical construction this should exist; guard for robustness
                raise KeyError(
                    f"Component {component} missing expected consecutive root edge {edge}"
                )
            spec[('edge', component, edge)] = 1

        return spec

    def block_root_product_spec(self, component: int) -> Dict[Any, int]:
        """
        Build p_i for a component i: product of root monomials in its base-A block.

        Let N_{i,s} = number of base rows from layer s in component i. Under the
        canonical construction (strictly decreasing root degrees), this equals the
        number of vertices with depth >= s, i.e., N_{i,s} = r_{s-1} from the
        characteristic tuple.

        Then: p_i = Π_{s=1}^{omega_i+1} M_i(s)^{N_{i,s}}.

        This routine computes exponents by accumulating N_{i,s} over the variables
        present in M_i(s).

        Returns:
            Dict mapping monomial keys to total exponents for component i.
        """
        if component < 0 or component >= len(self.calculator.graphs):
            raise IndexError(f"Invalid component index: {component}")

        graph = self.calculator.graphs[component]
        omega = graph.num_roots - 1

        # Precompute vertex depths once
        depths = graph.vertex_depths

        # Accumulate exponents
        spec: Dict[Any, int] = {}

        def acc(key: Any, inc: int) -> None:
            if inc == 0:
                return
            spec[key] = spec.get(key, 0) + inc

        for s in range(1, omega + 2):
            # N_s = count of vertices with depth >= s
            N_s = sum(1 for v, d in depths.items() if d >= s)
            if N_s == 0:
                continue
            # u_{(i,0)} appears in every layer's root monomial
            acc(('vertex', component, 0), N_s)
            # consecutive root edges up to (s-2, s-1)
            for k in range(0, s - 1):
                edge = (k, k + 1)
                if edge not in graph.edges:
                    raise KeyError(
                        f"Component {component} missing expected consecutive root edge {edge}"
                    )
                acc(('edge', component, edge), N_s)

        return spec

    def base_A_root_product_spec(self) -> Dict[Any, int]:
        """
        Build the global monomial p = Π_i p_i for the base A submatrix across
        all components, combining exponents.

        Returns:
            Dict mapping monomial keys to total exponents across all components.
        """
        combined: Dict[Any, int] = {}

        def acc(key: Any, inc: int) -> None:
            if inc == 0:
                return
            combined[key] = combined.get(key, 0) + inc

        for g_idx, _ in enumerate(self.calculator.graphs):
            bi = self.block_root_product_spec(g_idx)
            for k, e in bi.items():
                acc(k, e)

        return combined

    @staticmethod
    def base_A_root_product_spec_from_characteristic_tuples(
        char_tuples: List[Tuple[int, ...]]
    ) -> Dict[Any, int]:
        """
        One-shot helper: construct from characteristic tuples and return the
        global base-A root product monomial spec p.
        """
        calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
        det_comp = DeterminantComputer(calc)
        return det_comp.base_A_root_product_spec()

    def coeff_of_block_root_product(self, component: int, match: str = 'exact') -> sp.Expr:
        """
        Extract the coefficient of p_i (the component i root product) in det(A_i^base).

        Args:
            component: component index i
            match: 'exact' or 'divides' (exact is recommended here)

        Returns:
            SymPy expression for the coefficient (0 if absent).
        """
        if component < 0 or component >= len(self.calculator.graphs):
            raise IndexError(f"Invalid component index: {component}")
        self._ensure_base_blocks_cache()
        det_block = self._det_base_by_comp[component]
        spec = self.block_root_product_spec(component)
        return self.coeff_of_monomial(det_block, spec, match=match)

    def coeff_of_base_A_root_product(self, match: str = 'exact') -> sp.Expr:
        """
        Extract the coefficient of p = Π_i p_i in det(A_base).

        Args:
            match: 'exact' or 'divides' (exact is recommended here)

        Returns:
            SymPy expression for the coefficient (0 if absent).
        """
        det_full = self.compute_base_A_determinant()
        spec = self.base_A_root_product_spec()
        return self.coeff_of_monomial(det_full, spec, match=match)

    @staticmethod
    def coeff_of_base_A_root_product_from_characteristic_tuples(
        char_tuples: List[Tuple[int, ...]],
        *,
        match: str = 'exact',
    ) -> sp.Expr:
        """
        One-shot: build calculator from characteristic tuples and return the
        coefficient of the global root product monomial in det(A_base).
        """
        calc = FASMinorCalculator.from_characteristic_tuples(char_tuples, use_symbolic=True)
        det_comp = DeterminantComputer(calc)
        return det_comp.coeff_of_base_A_root_product(match=match)

    # ---------------------- p in full minor (symbolic exact) ---------------------
    def coeff_of_p_in_minor(self, graph_idx: int, vertex: int, layer: int) -> sp.Expr:
        """
        Extract the residual polynomial after factoring out p (global base-A
        root product) from the full minor with one extra row.

        This computes the minor (fast path) for the given extra row and returns
        coeff_divides(minor, p_spec), i.e., the sum of residual terms for which
        p divides the monomial term in the minor.

        Args:
            graph_idx: component index for the extra row
            vertex: local vertex index in that component
            layer: layer s (>=1)

        Returns:
            SymPy expression: residual polynomial (0 if no term is divisible by p).
        """
        p_spec = self.base_A_root_product_spec()
        minor = self.compute_minor_fast(graph_idx, vertex, layer)
        return self.coeff_of_monomial(minor, p_spec, match='divides')

    def has_p_in_minor(self, graph_idx: int, vertex: int, layer: int) -> bool:
        """
        Boolean existence check for whether the minor contains any term divisible
        by the global p (product of block root monomials).

        Returns:
            True if residual polynomial is not identically 0, else False.
        """
        try:
            res = self.coeff_of_p_in_minor(graph_idx, vertex, layer)
        except Exception:
            return False
        if res == 0:
            return False
        try:
            import sympy as _sp
            return not _sp.simplify(res) == 0
        except Exception:
            return True

    # ------------------- Expansion-free exact (structured) path ------------------
    def _block_root_product_coefficient_fast(self, component: int) -> sp.Expr:
        """
        Compute coeff(p_i) in det(A_i^base) without expanding determinants.

        Strategy: for each base row in component i at layer s, find the unique
        column j where the A-entry contains the root monomial M_i(s). Multiply
        those entry coefficients and apply the permutation sign from the chosen
        columns. Returns the exact symbolic coefficient (no expansion).
        """
        self._ensure_base_blocks_cache()
        rows_g = self._base_rows_by_comp[component]
        comp_edge_starts = self._comp_edge_starts
        comp_edge_sizes = self._comp_edge_sizes

        c0 = comp_edge_starts[component]
        c1 = c0 + comp_edge_sizes[component]

        coeff_prod = sp.Integer(1)
        perm: List[int] = []
        for r in rows_g:
            g, v, s = r
            row = self.calculator.get_row(g, v, s)
            arow = row[:, c0:c1]
            mi = self.root_monomial_spec_for_layer(component, s)
            found_col = None
            found_coeff = None
            for j in range(arow.shape[1]):
                entry = arow[0, j]
                coeff = self.coeff_of_monomial(entry, mi, match='exact')
                if coeff != 0:
                    if found_col is not None:
                        raise RuntimeError(
                            f"Multiple columns carry root monomial in component {component}, row {r}"
                        )
                    found_col = j
                    found_coeff = coeff
            if found_col is None:
                raise RuntimeError(
                    f"No column found for root monomial in component {component}, row {r}"
                )
            perm.append(found_col)
            coeff_prod *= found_coeff

        # Permutation parity (mapping i -> perm[i])
        n = len(perm)
        visited = [False] * n
        sign = 1
        for i in range(n):
            if not visited[i]:
                j = i
                cycle_len = 0
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_len += 1
                if cycle_len > 0:
                    sign *= (-1) ** (cycle_len - 1)

        return sign * coeff_prod

    def coeff_of_base_A_root_product_fast(self) -> sp.Expr:
        """
        Compute coeff(p) in det(A_base) without determinant expansion by
        multiplying per-component fast coefficients.
        """
        self._ensure_base_blocks_cache()
        coeff_total = sp.Integer(1)
        for g_idx, _ in enumerate(self.calculator.graphs):
            coeff_total *= self._block_root_product_coefficient_fast(g_idx)
        return coeff_total

    def coeff_of_p_in_minor_fast(self, graph_idx: int, vertex: int, layer: int) -> sp.Expr:
        """
        Expansion-free exact coefficient (divides) of p in the full minor.

        Uses the structured formula:
            minor = det(A_base) * sum_{rows r in component g*} (-1)^(i+Ncols) b_r * t_r
        where t_r = 1 if r is the extra row, else y[idx_r] with A_star^T y = extra_block^T.

        The residual after factoring p equals coeff(p in det(A_base)) times the
        bracketed sum. This avoids expanding the determinant.
        """
        # Precompute caches and blocks
        self._ensure_base_blocks_cache()
        comp_edge_starts = self._comp_edge_starts
        comp_edge_sizes = self._comp_edge_sizes
        base_rows_by_comp = self._base_rows_by_comp
        A_block_by_comp = self._A_block_by_comp

        # User and base rows
        user_row = (graph_idx, vertex, layer)
        all_rows = self.base_rows + [user_row]

        total_edges = sum(comp_edge_sizes.values())
        ncols = total_edges + 1

        # Component of extra row
        e_star = comp_edge_sizes[graph_idx]
        if e_star == 0:
            raise ValueError("Component of extra row has zero edges; cannot form minor.")

        # A_star and extra block
        A_star = A_block_by_comp[graph_idx]
        extra_full_row = self.calculator.get_row(*user_row)
        c0s = comp_edge_starts[graph_idx]
        c1s = c0s + e_star
        extra_block = extra_full_row[:, c0s:c1s]

        # y from A_star^T y = extra_block^T
        y = A_star.T.LUsolve(extra_block.T)

        # Map base row -> index within A_star
        comp_rows_star = base_rows_by_comp[graph_idx]
        index_in_star = {r: i for i, r in enumerate(comp_rows_star)}

        # Bracket sum S = sum sign * b_i * t_i
        S = sp.Integer(0)
        for i_global, r in enumerate(all_rows):
            if r[0] != graph_idx:
                continue
            sign = -1 if ((i_global + 1 + ncols) % 2) else 1
            b_i = self.calculator.build_matrix_entry(r, ('b', None, None))
            if r == user_row:
                t_i = 1
            else:
                idx = index_in_star[r]
                t_i = y[idx, 0]
            S += sign * b_i * t_i

        # coeff(p in det(A_base)) via fast per-component method
        c_base = self.coeff_of_base_A_root_product_fast()
        return c_base * S

    def find_monomial_in_minor(
        self,
        graph_idx: int,
        vertex: int,
        layer: int,
        monomial_spec: Dict[Any, int],
        match: str = 'exact',
        return_coeff: bool = True,
    ) -> Any:
        """
        Search the minor for a given monomial.

        - match='exact': exact monomial match.
        - match='divides': monomial divides term (allow extra u-factors); returns
          residual polynomial when return_coeff=True.

        Returns:
        - If return_coeff=True: SymPy expression for coefficient (exact) or residual
          polynomial (divides). 0 if not present.
        - If return_coeff=False: bool indicating existence.
        """
        minor = self.compute_minor_fast(graph_idx, vertex, layer)
        coeff = self.coeff_of_monomial(minor, monomial_spec, match=match)
        if return_coeff:
            return coeff
        # Determine existence without relying on symbolic equality to zero
        if coeff == 0:
            return False
        # For nontrivial expressions, consider non-zero if not structurally zero
        try:
            return not sp.simplify(coeff) == 0
        except Exception:
            return True

    def compute_determinant(
        self,
        row_specs: List[Tuple[int, int, int]]
    ) -> sp.Expr:
        """
        Compute the determinant of a matrix formed from specified rows.

        This method retrieves the specified rows from the calculator,
        assembles them into a square matrix, and computes the determinant
        using SymPy's det() function.

        Parameters:
        -----------
        row_specs : List[Tuple[int, int, int]]
            List of row specifications, where each specification is a tuple:
            (graph_idx, vertex, layer)
            - graph_idx: Index of the component graph (0-based)
            - vertex: Local vertex label within that component
            - layer: Layer number s >= 1

            The number of row specifications must equal the total number
            of edges plus one (n_edges + 1) to form a square matrix.

        Returns:
        --------
        sp.Expr
            The computed determinant as a SymPy expression.

        Raises:
        -------
        ValueError
            - If row_specs is empty
            - If the resulting matrix is not square
            - If any row specification is invalid
        TypeError
            - If row_specs is not a list or contains invalid tuples

        Examples:
        ---------
        >>> calc = FASMinorCalculator.from_characteristic_tuples(
        ...     [(2, 1, 4)], use_symbolic=True
        ... )
        >>> det_comp = DeterminantComputer(calc)
        >>> # For a system with 3 edges, need 4 rows
        >>> row_specs = [(0, 0, 1), (0, 1, 1), (0, 0, 2), (0, 1, 2)]
        >>> det = det_comp.compute_determinant(row_specs)
        """
        # Input validation
        if not isinstance(row_specs, list):
            raise TypeError(
                f"row_specs must be a list, got {type(row_specs).__name__}"
            )

        if not row_specs:
            raise ValueError("row_specs cannot be empty")

        # Validate each row specification
        for i, spec in enumerate(row_specs):
            if not isinstance(spec, tuple) or len(spec) != 3:
                raise TypeError(
                    f"Each row spec must be a 3-tuple (graph_idx, vertex, layer). "
                    f"row_specs[{i}] = {spec}"
                )
            graph_idx, vertex, layer = spec
            if not isinstance(graph_idx, int) or graph_idx < 0:
                raise ValueError(
                    f"graph_idx must be a non-negative integer. "
                    f"row_specs[{i}][0] = {graph_idx}"
                )
            if not isinstance(vertex, int) or vertex < 0:
                raise ValueError(
                    f"vertex must be a non-negative integer. "
                    f"row_specs[{i}][1] = {vertex}"
                )
            if not isinstance(layer, int) or layer < 1:
                raise ValueError(
                    f"layer must be a positive integer >= 1. "
                    f"row_specs[{i}][2] = {layer}"
                )

        # Retrieve all rows
        rows = []
        for graph_idx, vertex, layer in row_specs:
            try:
                row = self.calculator.get_row(graph_idx, vertex, layer)
                rows.append(row)
            except Exception as e:
                raise ValueError(
                    f"Failed to retrieve row for (graph_idx={graph_idx}, "
                    f"vertex={vertex}, layer={layer}): {str(e)}"
                ) from e

        # Enforce symbolic-only
        if not self.calculator.use_symbolic:
            raise ValueError("Determinant computation is only supported in symbolic mode.")

        # Assemble into matrix by stacking row matrices
        if len(rows) == 1:
            matrix = rows[0]
        else:
            matrix = rows[0]
            for row in rows[1:]:
                matrix = matrix.col_join(row)

        # Validate square matrix
        if matrix.rows != matrix.cols:
            n_edges = matrix.cols - 1
            raise ValueError(
                f"Matrix must be square for determinant computation. "
                f"Got {matrix.rows} rows and {matrix.cols} columns. "
                f"The system has {n_edges} edges, so you need exactly "
                f"{matrix.cols} row specifications (n_edges + 1). "
                f"You provided {len(row_specs)} row specifications."
            )

        # Compute determinant using SymPy (Berkowitz is robust for symbolic)
        det = matrix.det(method='berkowitz')
        return det

    def get_matrix(
        self,
        row_specs: List[Tuple[int, int, int]]
    ) -> sp.Matrix:
        """
        Assemble a matrix from specified rows without computing the determinant.

        This method is useful for inspecting the matrix before computing
        the determinant, or for using other matrix operations.

        Parameters:
        -----------
        row_specs : List[Tuple[int, int, int]]
            List of row specifications (same format as compute_determinant)

        Returns:
        --------
        sp.Matrix
            The assembled SymPy matrix

        Raises:
        -------
        ValueError, TypeError
            Same validation errors as compute_determinant
        """
        # Input validation (same as compute_determinant)
        if not isinstance(row_specs, list):
            raise TypeError(
                f"row_specs must be a list, got {type(row_specs).__name__}"
            )

        if not row_specs:
            raise ValueError("row_specs cannot be empty")

        # Validate each row specification
        for i, spec in enumerate(row_specs):
            if not isinstance(spec, tuple) or len(spec) != 3:
                raise TypeError(
                    f"Each row spec must be a 3-tuple (graph_idx, vertex, layer). "
                    f"row_specs[{i}] = {spec}"
                )

        # Retrieve all rows
        rows = []
        for graph_idx, vertex, layer in row_specs:
            try:
                row = self.calculator.get_row(graph_idx, vertex, layer)
                rows.append(row)
            except Exception as e:
                raise ValueError(
                    f"Failed to retrieve row for (graph_idx={graph_idx}, "
                    f"vertex={vertex}, layer={layer}): {str(e)}"
                ) from e

        # Enforce symbolic-only
        if not self.calculator.use_symbolic:
            raise ValueError("Matrix assembly is only supported in symbolic mode.")

        if len(rows) == 1:
            matrix = rows[0]
        else:
            matrix = rows[0]
            for row in rows[1:]:
                matrix = matrix.col_join(row)
        return matrix
