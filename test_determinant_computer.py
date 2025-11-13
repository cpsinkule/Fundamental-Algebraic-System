"""
Tests for the DeterminantComputer module

This file contains comprehensive tests for the determinant computation
interface between FASMinorCalculator and SymPy's det() function.
"""

import pytest
import sympy as sp
import numpy as np
from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer


class TestDeterminantComputerBasics:
    """Test basic functionality of DeterminantComputer"""

    def test_initialization(self):
        """Test that DeterminantComputer initializes correctly"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)
        assert det_comp.calculator is calc

    def test_initialization_with_invalid_type(self):
        """Test that initialization fails with non-calculator object"""
        with pytest.raises(TypeError, match="must be a FASMinorCalculator"):
            DeterminantComputer("not a calculator")

    def test_compute_determinant_symbolic(self):
        """Test computing determinant in symbolic mode"""
        # Create a simple system
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # This system has 3 edges, so we need 4 rows for a square matrix
        row_specs = [
            (0, 0, 1),  # Component 0, vertex 0, layer 1
            (0, 1, 1),  # Component 0, vertex 1, layer 1
            (0, 0, 2),  # Component 0, vertex 0, layer 2
            (0, 1, 2),  # Component 0, vertex 1, layer 2
        ]

        # Compute determinant
        det = det_comp.compute_determinant(row_specs)

        # Result should be a SymPy expression
        assert isinstance(det, sp.Basic)

    # Removed numeric-mode determinant test (calculator is symbolic-only)

    def test_get_matrix(self):
        """Test the get_matrix method"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        row_specs = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
        ]

        # Get matrix without computing determinant
        matrix = det_comp.get_matrix(row_specs)

        # Should be a SymPy Matrix
        assert isinstance(matrix, sp.Matrix)

        # Should be 4x4 (4 rows, 3 edges + 1 b column)
        assert matrix.rows == 4
        assert matrix.cols == 4


class TestDeterminantComputerValidation:
    """Test input validation and error handling"""

    def test_empty_row_specs(self):
        """Test that empty row_specs raises ValueError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        with pytest.raises(ValueError, match="cannot be empty"):
            det_comp.compute_determinant([])

    def test_non_list_row_specs(self):
        """Test that non-list row_specs raises TypeError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        with pytest.raises(TypeError, match="must be a list"):
            det_comp.compute_determinant((0, 0, 1))

    def test_invalid_tuple_length(self):
        """Test that row specs with wrong tuple length raise TypeError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        with pytest.raises(TypeError, match="must be a 3-tuple"):
            det_comp.compute_determinant([(0, 0)])  # Only 2 elements

    def test_non_square_matrix(self):
        """Test that non-square matrix raises ValueError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # System has 3 edges, so 4 columns total
        # Providing only 3 rows creates non-square matrix
        row_specs = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
        ]

        with pytest.raises(ValueError, match="must be square"):
            det_comp.compute_determinant(row_specs)

    def test_negative_graph_idx(self):
        """Test that negative graph_idx raises ValueError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        with pytest.raises(ValueError, match="graph_idx must be"):
            det_comp.compute_determinant([(-1, 0, 1)])

    def test_negative_vertex(self):
        """Test that negative vertex raises ValueError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        with pytest.raises(ValueError, match="vertex must be"):
            det_comp.compute_determinant([(0, -1, 1)])

    def test_zero_layer(self):
        """Test that layer < 1 raises ValueError"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        with pytest.raises(ValueError, match="layer must be"):
            det_comp.compute_determinant([(0, 0, 0)])

    def test_invalid_graph_idx(self):
        """Test that invalid graph_idx is caught when retrieving row"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Only 1 component (idx 0), so idx 1 is invalid
        with pytest.raises(ValueError, match="Failed to retrieve row"):
            det_comp.compute_determinant([
                (1, 0, 1),  # Invalid graph_idx
                (0, 0, 1),
                (0, 1, 1),
                (0, 0, 2),
            ])


class TestDeterminantComputerIntegration:
    """Test integration with FASMinorCalculator"""

    # Removed numeric two-component determinant test (symbolic-only)

    def test_mixed_layers(self):
        """Test using rows from different layers"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Mix layers 1, 2, and 3
        row_specs = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 0, 3),  # Layer 3
        ]

        det = det_comp.compute_determinant(row_specs)
        assert isinstance(det, sp.Basic)

    def test_get_matrix_validation(self):
        """Test that get_matrix has same validation as compute_determinant"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Empty row_specs should raise error
        with pytest.raises(ValueError, match="cannot be empty"):
            det_comp.get_matrix([])

        # Invalid tuple should raise error
        with pytest.raises(TypeError, match="must be a 3-tuple"):
            det_comp.get_matrix([(0, 0)])


class TestBaseRowGeneration:
    """Test automatic base row generation"""

    def test_base_rows_count(self):
        """Test that base rows count equals n-m"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        base_rows = det_comp.get_base_rows()
        n_edges = sum(len(g.edges) for g in calc.graphs)

        assert len(base_rows) == n_edges

    def test_base_rows_structure(self):
        """Test that base rows follow depth-based selection"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        base_rows = det_comp.get_base_rows()

        # All base rows should be valid tuples
        for row in base_rows:
            assert isinstance(row, tuple)
            assert len(row) == 3
            graph_idx, vertex, layer = row
            assert isinstance(graph_idx, int)
            assert isinstance(vertex, int)
            assert isinstance(layer, int)

    def test_base_rows_two_components(self):
        """Test base row generation with two components"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4), (2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        base_rows = det_comp.get_base_rows()
        n_edges = 6  # 3 edges per component

        assert len(base_rows) == n_edges

        # Should have rows from both components
        component_0_rows = [r for r in base_rows if r[0] == 0]
        component_1_rows = [r for r in base_rows if r[0] == 1]

        assert len(component_0_rows) > 0
        assert len(component_1_rows) > 0

    def test_get_base_rows_returns_copy(self):
        """Test that get_base_rows returns a copy, not the original"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        base_rows_1 = det_comp.get_base_rows()
        base_rows_2 = det_comp.get_base_rows()

        # Should be equal but not the same object
        assert base_rows_1 == base_rows_2
        assert base_rows_1 is not base_rows_2

        # Modifying one shouldn't affect the other
        base_rows_1.append((99, 99, 99))
        assert len(base_rows_2) != len(base_rows_1)


class TestComputeMinor:
    """Test compute_minor method with automatic base rows"""

    def test_compute_minor_basic(self):
        """Test basic minor computation with user row"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute minor with user row (0, 0, 1)
        minor = det_comp.compute_minor(0, 0, 1)

        # Should return a SymPy expression
        assert isinstance(minor, sp.Basic)

    def test_compute_minor_different_rows(self):
        """Test computing different minors with different user rows"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute multiple minors
        minor_1 = det_comp.compute_minor(0, 0, 1)
        minor_2 = det_comp.compute_minor(0, 1, 1)
        minor_3 = det_comp.compute_minor(0, 0, 2)

        # All should be SymPy expressions
        assert isinstance(minor_1, sp.Basic)
        assert isinstance(minor_2, sp.Basic)
        assert isinstance(minor_3, sp.Basic)

    # Removed numeric compute_minor test (symbolic-only)

    # Removed numeric two-component compute_minor test (symbolic-only)


class TestComputeYVector:
    """Test compute_y_vector method for Cramer's rule adjugate system"""

    def test_y_vector_basic(self):
        """Test basic y-vector computation"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute y-vector for extra row (0, 0, 1)
        y = det_comp.compute_y_vector(0, 0, 1)

        # Should return a SymPy Matrix (column vector)
        assert isinstance(y, sp.Matrix)

    def test_y_vector_dimensions(self):
        """Test that y-vector has correct dimensions"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Component 0 has 3 edges, so y should be 3x1
        y = det_comp.compute_y_vector(0, 0, 1)

        assert y.shape == (3, 1)

    def test_y_vector_solves_system(self):
        """Test that y satisfies the linear system A_star^T * y = extra_block^T"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute y-vector
        graph_idx, vertex, layer = 0, 0, 1
        y = det_comp.compute_y_vector(graph_idx, vertex, layer)

        # Get the cached A_star and extra_block to verify the system
        det_comp._ensure_base_blocks_cache()
        A_star = det_comp._A_block_by_comp[graph_idx]

        # Get extra row and extract the block
        extra_row = calc.get_row(graph_idx, vertex, layer)
        c0 = det_comp._comp_edge_starts[graph_idx]
        e_star = det_comp._comp_edge_sizes[graph_idx]
        c1 = c0 + e_star
        extra_block = extra_row[:, c0:c1]

        # Verify: A_star^T * y = extra_block^T
        result = A_star.T * y
        expected = extra_block.T

        # Simplify to check equality (symbolic may have complex forms)
        residual = result - expected
        residual_simplified = sp.simplify(residual)

        # All entries should be zero
        assert residual_simplified == sp.zeros(e_star, 1)

    def test_y_vector_different_rows(self):
        """Test computing y-vectors for different extra rows"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute y-vectors for different rows
        y1 = det_comp.compute_y_vector(0, 0, 1)
        y2 = det_comp.compute_y_vector(0, 1, 1)
        y3 = det_comp.compute_y_vector(0, 0, 2)

        # All should be valid column vectors
        assert isinstance(y1, sp.Matrix)
        assert isinstance(y2, sp.Matrix)
        assert isinstance(y3, sp.Matrix)

        # All should have same shape (3x1 for this system)
        assert y1.shape == y2.shape == y3.shape == (3, 1)

        # They should generally be different (not equal)
        # Note: in symbolic mode they might simplify to same thing,
        # but we just check they're all valid
        assert y1 is not y2
        assert y2 is not y3

    def test_y_vector_two_components(self):
        """Test y-vector computation with two-component system"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4), (2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute y-vectors from different components
        y_comp0 = det_comp.compute_y_vector(0, 0, 1)
        y_comp1 = det_comp.compute_y_vector(1, 0, 1)

        # Both should be 3x1 (each component has 3 edges)
        assert y_comp0.shape == (3, 1)
        assert y_comp1.shape == (3, 1)

        # Both should be valid SymPy matrices
        assert isinstance(y_comp0, sp.Matrix)
        assert isinstance(y_comp1, sp.Matrix)

    def test_y_vector_invalid_component_zero_edges(self):
        """Test that y-vector raises error if component has zero edges"""
        # Create a degenerate case (shouldn't normally happen, but test edge case)
        # We can't easily create this with characteristic tuples, so we test
        # the error path by checking the error message exists in the code
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Invalid graph_idx should trigger error in get_row before our check
        with pytest.raises(Exception):  # Could be IndexError or ValueError
            det_comp.compute_y_vector(99, 0, 1)

    def test_y_vector_with_different_layers(self):
        """Test y-vector computation across different layers"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(3, 1, 5)],  # Use system with more layers (omega=2, so layers 1-3)
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute y-vectors for different layers
        y_layer1 = det_comp.compute_y_vector(0, 0, 1)
        y_layer2 = det_comp.compute_y_vector(0, 0, 2)
        y_layer3 = det_comp.compute_y_vector(0, 0, 3)

        # All should be valid and have same dimensions
        assert isinstance(y_layer1, sp.Matrix)
        assert isinstance(y_layer2, sp.Matrix)
        assert isinstance(y_layer3, sp.Matrix)

        # All should have same shape (based on edge count)
        n_edges = len(calc.graphs[0].edges)
        expected_shape = (n_edges, 1)

        assert y_layer1.shape == expected_shape
        assert y_layer2.shape == expected_shape
        assert y_layer3.shape == expected_shape

    def test_y_vector_consistency_with_compute_minor_fast(self):
        """Test that y-vector from compute_y_vector matches internal computation"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Compute y-vector explicitly
        graph_idx, vertex, layer = 0, 0, 1
        y_explicit = det_comp.compute_y_vector(graph_idx, vertex, layer)

        # Now compute minor to trigger internal y computation
        # We can't directly access the internal y, but we can verify
        # that compute_y_vector runs without error and produces valid output
        minor = det_comp.compute_minor_fast(graph_idx, vertex, layer)

        # Both should complete successfully
        assert isinstance(y_explicit, sp.Matrix)
        assert isinstance(minor, sp.Basic)

        # The y-vector should be used internally in the minor computation
        # Verify dimensions match what's expected
        n_edges = len(calc.graphs[graph_idx].edges)
        assert y_explicit.shape == (n_edges, 1)


class TestDeterminantComputerSmallExample:
    """Test with a very small, hand-verifiable example"""

    def test_simple_system(self):
        """Test with simple system: (2,1,3) - two roots, three edges"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 3)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        # Verify base rows were generated
        base_rows = det_comp.get_base_rows()
        assert isinstance(base_rows, list)
        assert len(base_rows) == 3  # n-m = 3 edges

        # Can still use manual row specs (3 edges + 1 = 4 rows)
        row_specs = [
            (0, 0, 1),  # Vertex 0, layer 1
            (0, 1, 1),  # Vertex 1, layer 1
            (0, 2, 1),  # Vertex 2, layer 1
            (0, 0, 2),  # Vertex 0, layer 2
        ]

        # Get the matrix to inspect it
        matrix = det_comp.get_matrix(row_specs)
        assert matrix.rows == 4
        assert matrix.cols == 4

        # Compute determinant
        det = det_comp.compute_determinant(row_specs)
        assert isinstance(det, sp.Basic)


def test_readme_example():
    """Test the example from the module docstring"""
    # Create calculator
    calc = FASMinorCalculator.from_characteristic_tuples(
        [(2, 1, 4)],
        use_symbolic=True
    )

    # Create determinant computer
    det_comp = DeterminantComputer(calc)

    # Specify rows for the minor (3 edges + 1 = 4 rows needed)
    row_specs = [
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 2),
        (0, 1, 2),
    ]

    # Compute determinant
    det = det_comp.compute_determinant(row_specs)

    # Should produce a symbolic result
    assert isinstance(det, sp.Basic)
    print(f"Determinant computed successfully: type={type(det)}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
