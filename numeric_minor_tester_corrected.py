"""
Numeric Minor Tester Module

This module provides a standalone numerical implementation of the fast minor
calculation algorithm from DeterminantComputer. It is designed to verify the
correctness of the LU decomposition + Cramer's rule logic using numerical
arrays instead of symbolic expressions.

The primary use case is testing: by evaluating the symbolic calculator with
numeric values and comparing against this direct numeric implementation, we can
verify the algorithmic correctness independent of symbolic complexity.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import sympy as sp


class NumericMinorTester:
    """
    Numerical implementation of the fast minor calculation algorithm.

    This class replicates the block-structured Laplace expansion logic from
    DeterminantComputer.compute_minor_fast() but operates on numpy arrays
    instead of sympy matrices.

    Key algorithm (from determinant_computer.py:315-433):
    1. Compute det(A_h) for each component h
    2. Build product: prod_other = ∏_{h≠g*} det(A_h)
    3. Solve A_g*^T @ y = extra_block_g*^T via LU decomposition
    4. Laplace expansion along b column:
       det = Σ sign * b_i * minor_cofactor_i
       where minor_cofactor depends on whether row i is the extra row

    Attributes
    ----------
    A_blocks : Dict[int, np.ndarray]
        Per-component A-blocks (n_h × e_h matrices)
    b_vectors : Dict[int, np.ndarray]
        Per-component b-vectors (n_h × 1)
    component_edge_sizes : Dict[int, int]
        Number of edges per component
    component_edge_starts : Dict[int, int]
        Starting column index for each component's edges
    base_rows_by_comp : Dict[int, List[Tuple[int, int, int]]]
        Base row specifications grouped by component
    """

    def __init__(
        self,
        A_blocks: Dict[int, np.ndarray],
        b_vectors: Dict[int, np.ndarray],
        component_edge_sizes: Dict[int, int],
        component_edge_starts: Dict[int, int],
        base_rows_by_comp: Dict[int, List[Tuple[int, int, int]]],
        tolerance: float = 1e-10
    ):
        """
        Initialize numeric minor tester with pre-computed matrix blocks.

        Parameters
        ----------
        A_blocks : Dict[int, np.ndarray]
            Per-component A-blocks. Keys are component indices (0-based).
            Values are numpy arrays of shape (n_h, e_h) where n_h is the
            number of base rows and e_h is the number of edges in component h.
        b_vectors : Dict[int, np.ndarray]
            Per-component b-vectors. Keys are component indices.
            Values are numpy arrays of shape (n_h, 1).
        component_edge_sizes : Dict[int, int]
            Number of edges per component. Keys are component indices.
        component_edge_starts : Dict[int, int]
            Starting column index for each component's edges in the global
            edge ordering.
        base_rows_by_comp : Dict[int, List[Tuple[int, int, int]]]
            Base row specifications grouped by component. Each row is a tuple
            (graph_idx, vertex, layer).
        tolerance : float, optional
            Numerical tolerance for comparisons and zero checks.
        """
        self.A_blocks = A_blocks
        self.b_vectors = b_vectors
        self.component_edge_sizes = component_edge_sizes
        self.component_edge_starts = component_edge_starts
        self.base_rows_by_comp = base_rows_by_comp
        self.tolerance = tolerance

        # Validate inputs
        self._validate_inputs()

        # Cache determinants of A-blocks
        self.det_A_blocks = {}
        for comp_idx, A_block in self.A_blocks.items():
            if A_block.size == 0:
                self.det_A_blocks[comp_idx] = 1.0
            else:
                self.det_A_blocks[comp_idx] = np.linalg.det(A_block)

    def _validate_inputs(self) -> None:
        """Validate that input data structures are consistent."""
        # Check all components have consistent data
        component_indices = set(self.component_edge_sizes.keys())

        if set(self.A_blocks.keys()) != component_indices:
            raise ValueError("A_blocks keys do not match component indices")
        if set(self.b_vectors.keys()) != component_indices:
            raise ValueError("b_vectors keys do not match component indices")
        if set(self.component_edge_starts.keys()) != component_indices:
            raise ValueError("component_edge_starts keys do not match component indices")
        if set(self.base_rows_by_comp.keys()) != component_indices:
            raise ValueError("base_rows_by_comp keys do not match component indices")

        # Check dimensions
        for comp_idx in component_indices:
            e_h = self.component_edge_sizes[comp_idx]
            A_block = self.A_blocks[comp_idx]
            b_vec = self.b_vectors[comp_idx]

            if e_h == 0:
                if A_block.size != 0 or b_vec.size != 0:
                    raise ValueError(
                        f"Component {comp_idx} has 0 edges but non-empty A-block or b-vector"
                    )
                continue

            if A_block.shape[1] != e_h:
                raise ValueError(
                    f"Component {comp_idx}: A-block has {A_block.shape[1]} columns, expected {e_h}"
                )
            if A_block.shape[0] != e_h:
                raise ValueError(
                    f"Component {comp_idx}: A-block has {A_block.shape[0]} rows, expected {e_h} "
                    f"(base rows must equal edge count for square block)"
                )
            if b_vec.shape[0] != e_h:
                raise ValueError(
                    f"Component {comp_idx}: b-vector has {b_vec.shape[0]} rows, expected {e_h}"
                )
            if b_vec.shape[1] != 1:
                raise ValueError(
                    f"Component {comp_idx}: b-vector should be column vector (n,1), "
                    f"got shape {b_vec.shape}"
                )

    def compute_y_vector_numeric(
        self,
        component_idx: int,
        extra_row_A_block: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the y-vector via LU decomposition (Cramer's rule equivalent).

        Solves the linear system:
            A_star^T @ y = extra_block^T

        where A_star is the base A-block for the specified component and
        extra_block is the A-entries from the extra row for that component.

        This corresponds to determinant_computer.py:397-400.

        Parameters
        ----------
        component_idx : int
            Index of the component containing the extra row
        extra_row_A_block : np.ndarray
            A-block entries from the extra row for this component.
            Shape should be (1, e_star) where e_star is the number of edges
            in the component.

        Returns
        -------
        np.ndarray
            Y-vector of shape (e_star, 1). Each entry y[i] is the cofactor
            coefficient for row-replacement determinant formula.

        Raises
        ------
        ValueError
            If component has no edges or if A-block is singular.
        """
        # Get A-block for this component
        A_star = self.A_blocks[component_idx]
        e_star = self.component_edge_sizes[component_idx]

        if e_star == 0:
            raise ValueError(
                f"Component {component_idx} has zero edges; cannot compute y-vector"
            )

        # Check A-block is not singular
        det_A_star = self.det_A_blocks[component_idx]
        if np.abs(det_A_star) < self.tolerance:
            raise ValueError(
                f"Base A-block for component {component_idx} is singular "
                f"(det = {det_A_star:.2e}). Cannot solve for y-vector."
            )

        # Validate extra_row shape
        if extra_row_A_block.shape != (1, e_star):
            raise ValueError(
                f"extra_row_A_block has shape {extra_row_A_block.shape}, "
                f"expected (1, {e_star})"
            )

        # Solve A_star^T @ y = extra_block^T
        # Note: A_star.T is (e_star, e_star), extra_block.T is (e_star, 1)
        rhs = extra_row_A_block.T  # Shape (e_star, 1)
        y = np.linalg.solve(A_star.T, rhs)

        return y

    def compute_fast_minor_numeric(
        self,
        component_idx: int,
        extra_row_A_block: np.ndarray,
        extra_row_b: float,
    ) -> float:
        """
        Compute the minor determinant using the fast block-structured algorithm.

        This replicates the exact logic from determinant_computer.py:315-433.

        Algorithm steps:
        1. Compute prod_other = ∏_{h≠g*} det(A_h)
        2. Compute y-vector: A_g*^T @ y = extra_A_block^T
        3. Laplace expansion along b column:
           - For extra row: cofactor = prod_other * det(A_g*)
           - For base row i in g*: cofactor = prod_other * det(A_g*) * y[i]
        4. Sum: det = Σ sign * b_i * cofactor_i

        Parameters
        ----------
        component_idx : int
            Component index (0-based) of the extra row
        extra_row_A_block : np.ndarray
            A-block entries from the extra row for component g*.
            Shape: (1, e_star)
        extra_row_b : float
            B-entry for the extra row

        Returns
        -------
        float
            The computed minor determinant

        Raises
        ------
        ValueError
            If component has no edges or if A-block is singular
        """
        # Step 1: Compute product of determinants for all components except g*
        prod_other = 1.0
        for h in self.det_A_blocks:
            if h != component_idx:
                prod_other *= self.det_A_blocks[h]

        # Get A-block and det for component g*
        A_star = self.A_blocks[component_idx]
        det_A_star = self.det_A_blocks[component_idx]
        e_star = self.component_edge_sizes[component_idx]

        if e_star == 0:
            raise ValueError(
                f"Component {component_idx} has zero edges; cannot form minor"
            )

        if np.abs(det_A_star) < self.tolerance:
            raise ValueError(
                f"Base A-block for component {component_idx} is singular "
                f"(det = {det_A_star:.2e}). This violates theoretical assumptions."
            )

        # Step 2: Compute y-vector
        y = self.compute_y_vector_numeric(component_idx, extra_row_A_block)

        # Step 3: Laplace expansion along b column for the component g* only.
        #
        # The full determinant factorizes as:
        #     det_full = (∏_{h≠g*} det(A_h)) * det_single_g*
        # where det_single_g* is the determinant one would obtain by restricting
        # to the rows from component g* together with the extra row, and to the
        # columns consisting of the A_g* block plus the b column. The Laplace
        # signs for this single-component subproblem depend only on the local
        # row index within g* and the local column count ncols_local = e_star+1.
        #
        # We therefore:
        #   1. Compute det_single_g* using local Laplace signs and the
        #      row-replacement formula, including the row-permutation sign
        #      needed to move the extra row to the bottom of the A_star block.
        #   2. Multiply by prod_other to obtain the full determinant.

        # Local column count for the component block plus b-column
        ncols_local = e_star + 1

        det_single = 0.0

        # Process base rows in component g* using local row indices
        base_rows_star = self.base_rows_by_comp[component_idx]
        for local_idx, _row_spec in enumerate(base_rows_star):
            # Local Laplace sign: (-1)^(i_local + Ncols_local) with 1-based i_local
            sign_local = -1 if ((local_idx + 1 + ncols_local) % 2) else 1

            # B-entry for this base row
            b_i = self.b_vectors[component_idx][local_idx, 0]

            # Cofactor: as in the single-component case, the true cofactor minor
            # differs from det(A_star) * y[local_idx] by a row-permutation sign.
            # Moving the extra row from position local_idx to the last row of the
            # block requires (e_star - 1 - local_idx) swaps.
            sign_block = -1 if ((e_star - 1 - local_idx) % 2) else 1
            replaced_det = sign_block * det_A_star * y[local_idx, 0]

            det_single += sign_local * b_i * replaced_det

        # Extra row for component g*: local index e_star (after all base rows)
        sign_extra_local = -1 if ((e_star + 1 + ncols_local) % 2) else 1
        det_single += sign_extra_local * extra_row_b * det_A_star

        # Multiply by product of determinants from other components
        det_total = prod_other * det_single

        return det_total

    def verify_y_vector_solution(
        self,
        component_idx: int,
        extra_row_A_block: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Verify that y-vector actually solves A^T @ y = extra_block^T.

        Parameters
        ----------
        component_idx : int
            Component index
        extra_row_A_block : np.ndarray
            Extra row A-block (1, e_star)
        y : np.ndarray
            Y-vector to verify (e_star, 1)

        Returns
        -------
        dict
            Contains:
            - 'is_valid': bool, True if residual is below tolerance
            - 'residual_norm': float, ||A^T @ y - extra_block^T||
            - 'residual': np.ndarray, the actual residual vector
        """
        A_star = self.A_blocks[component_idx]
        expected_rhs = extra_row_A_block.T
        actual_lhs = A_star.T @ y

        residual = actual_lhs - expected_rhs
        residual_norm = np.linalg.norm(residual)
        is_valid = residual_norm < self.tolerance

        return {
            'is_valid': is_valid,
            'residual_norm': residual_norm,
            'residual': residual
        }


def extract_numeric_blocks_from_symbolic(
    det_computer,
    symbol_values: Optional[Dict[sp.Symbol, float]] = None,
    random_seed: int = 42,
    return_symbol_values: bool = False
) -> Union[NumericMinorTester, Tuple[NumericMinorTester, Dict[sp.Symbol, float]]]:
    """
    Extract numeric blocks from a symbolic DeterminantComputer instance.

    This helper function evaluates all symbolic matrix entries using the provided
    symbol values and constructs a NumericMinorTester instance for comparison.

    If symbol_values is not provided, random values will be generated for all
    symbols found in the matrix entries.

    Parameters
    ----------
    det_computer : DeterminantComputer
        Symbolic determinant computer instance (must have cached base blocks)
    symbol_values : Dict[sp.Symbol, float], optional
        Mapping from SymPy symbols to numeric values. If None, random values
        will be generated.
    random_seed : int, optional
        Random seed for generating symbol values if symbol_values is None
    return_symbol_values : bool, optional
        If True, return tuple of (tester, symbol_values). If False, return just tester.

    Returns
    -------
    NumericMinorTester or tuple
        If return_symbol_values is False: NumericMinorTester instance
        If return_symbol_values is True: (NumericMinorTester, symbol_values dict)
    """
    # Ensure cache is built
    det_computer._ensure_base_blocks_cache()

    # If symbol_values not provided, collect all symbols and assign random values
    if symbol_values is None:
        import numpy as np
        np.random.seed(random_seed)

        all_symbols = set()
        calc = det_computer.calculator

        # Collect symbols from vertex and edge variables
        all_symbols.update(calc.vertex_variables.values())
        all_symbols.update(calc.edge_variables.values())

        # Collect symbols from A-blocks (this will trigger lazy structure function creation)
        for A_block_sym in det_computer._A_block_by_comp.values():
            if A_block_sym.shape[0] > 0:
                for i in range(A_block_sym.shape[0]):
                    for j in range(A_block_sym.shape[1]):
                        all_symbols.update(A_block_sym[i, j].free_symbols)

        # Create random assignments
        symbol_values = {sym: np.random.rand() for sym in all_symbols}

    # Extract component structure info
    component_edge_sizes = dict(det_computer._comp_edge_sizes)
    component_edge_starts = dict(det_computer._comp_edge_starts)
    base_rows_by_comp = dict(det_computer._base_rows_by_comp)

    # Convert symbolic A-blocks to numeric
    A_blocks = {}
    for comp_idx, A_block_sym in det_computer._A_block_by_comp.items():
        if A_block_sym.shape[0] == 0:  # Empty block
            A_blocks[comp_idx] = np.array([]).reshape(0, 0)
        else:
            # Evaluate each entry
            A_numeric = np.zeros(A_block_sym.shape, dtype=float)
            for i in range(A_block_sym.shape[0]):
                for j in range(A_block_sym.shape[1]):
                    expr = A_block_sym[i, j]
                    A_numeric[i, j] = float(expr.subs(symbol_values))
            A_blocks[comp_idx] = A_numeric

    # Convert symbolic b-vectors to numeric
    b_vectors = {}
    for comp_idx, rows_list in base_rows_by_comp.items():
        if len(rows_list) == 0:
            b_vectors[comp_idx] = np.array([]).reshape(0, 1)
        else:
            b_numeric = np.zeros((len(rows_list), 1), dtype=float)
            for local_idx, row_spec in enumerate(rows_list):
                # Get b-entry from symbolic calculator
                b_expr = det_computer.calculator.build_matrix_entry(
                    row_spec, ('b', None, None)
                )
                # b_expr might be a concrete value (int/float) or symbolic
                if isinstance(b_expr, (int, float)):
                    b_numeric[local_idx, 0] = float(b_expr)
                else:
                    b_numeric[local_idx, 0] = float(b_expr.subs(symbol_values))
            b_vectors[comp_idx] = b_numeric

    tester = NumericMinorTester(
        A_blocks=A_blocks,
        b_vectors=b_vectors,
        component_edge_sizes=component_edge_sizes,
        component_edge_starts=component_edge_starts,
        base_rows_by_comp=base_rows_by_comp
    )

    if return_symbol_values:
        return tester, symbol_values
    else:
        return tester


def evaluate_extra_row_numeric(
    det_computer,
    extra_row_spec: Tuple[int, int, int],
    symbol_values: Dict[sp.Symbol, float],
) -> Tuple[np.ndarray, float]:
    """
    Evaluate the extra row numerically.

    Parameters
    ----------
    det_computer : DeterminantComputer
        Symbolic determinant computer instance
    extra_row_spec : Tuple[int, int, int]
        Extra row specification (graph_idx, vertex, layer)
    symbol_values : Dict[sp.Symbol, float]
        Mapping from SymPy symbols to numeric values

    Returns
    -------
    extra_row_A_block : np.ndarray
        Numeric A-block for the component of the extra row (shape 1, e_star)
    extra_row_b : float
        Numeric b-entry for the extra row
    """
    det_computer._ensure_base_blocks_cache()

    graph_idx = extra_row_spec[0]

    # Get full symbolic extra row
    extra_row_sym = det_computer.calculator.get_row(*extra_row_spec)

    # Extract A-block for the component
    c0 = det_computer._comp_edge_starts[graph_idx]
    e_star = det_computer._comp_edge_sizes[graph_idx]
    c1 = c0 + e_star

    extra_block_sym = extra_row_sym[:, c0:c1]

    # Evaluate numerically
    extra_row_A_block = np.zeros((1, e_star), dtype=float)
    for j in range(e_star):
        expr = extra_block_sym[0, j]
        extra_row_A_block[0, j] = float(expr.subs(symbol_values))

    # Get b-entry
    b_expr = det_computer.calculator.build_matrix_entry(
        extra_row_spec, ('b', None, None)
    )
    # b_expr might be a concrete value (int/float) or symbolic
    if isinstance(b_expr, (int, float)):
        extra_row_b = float(b_expr)
    else:
        extra_row_b = float(b_expr.subs(symbol_values))

    return extra_row_A_block, extra_row_b
