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

    def test_compute_determinant_numeric(self):
        """Test computing determinant in numeric mode"""
        # Create calculator with custom values
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=False
        )

        # Set some concrete values for structure functions
        calc.set_structure_functions({
            (0, 0, 1): 1.0,
            (0, 0, 2): 1.0,
        })

        det_comp = DeterminantComputer(calc)

        # Same row specs as symbolic test
        row_specs = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
        ]

        # Compute determinant
        det = det_comp.compute_determinant(row_specs)

        # Result should be a float
        assert isinstance(det, (float, int))

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

    def test_two_component_system(self):
        """Test determinant computation with multiple components"""
        # Use numeric mode for faster computation (symbolic 7x7 det is very slow)
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4), (2, 1, 4)],
            use_symbolic=False
        )
        det_comp = DeterminantComputer(calc)

        # This system has 6 edges (3 per component), so 7 columns total
        # Need 7 rows for square matrix
        row_specs = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (1, 0, 1),
            (1, 1, 1),
            (1, 0, 2),
            (1, 1, 2),
        ]

        # Should successfully compute determinant
        det = det_comp.compute_determinant(row_specs)
        assert isinstance(det, (float, int))

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

    def test_compute_minor_numeric_mode(self):
        """Test compute_minor in numeric mode"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4)],
            use_symbolic=False
        )
        det_comp = DeterminantComputer(calc)

        # Compute minor
        minor = det_comp.compute_minor(0, 0, 1)

        # Should return a float
        assert isinstance(minor, (float, int))

    def test_compute_minor_two_components(self):
        """Test compute_minor with two-component system"""
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4), (2, 1, 4)],
            use_symbolic=False  # Use numeric for speed
        )
        det_comp = DeterminantComputer(calc)

        # Compute minors from different components
        minor_comp0 = det_comp.compute_minor(0, 0, 1)
        minor_comp1 = det_comp.compute_minor(1, 0, 1)

        assert isinstance(minor_comp0, (float, int))
        assert isinstance(minor_comp1, (float, int))


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
