"""Quick smoke test for compute_y_vector method"""

from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer
import sympy as sp

# Test 1: Basic functionality
print("Test 1: Basic functionality...")
calc = FASMinorCalculator.from_characteristic_tuples(
    [(2, 1, 4)],
    use_symbolic=True
)
det_comp = DeterminantComputer(calc)

y = det_comp.compute_y_vector(0, 0, 1)
print(f"  y-vector shape: {y.shape}")
print(f"  y-vector type: {type(y)}")
assert isinstance(y, sp.Matrix), "y should be a SymPy Matrix"
assert y.shape == (3, 1), f"y should be 3x1, got {y.shape}"
print("  ✓ Test 1 passed!")

# Test 2: Different rows
print("\nTest 2: Different rows...")
y1 = det_comp.compute_y_vector(0, 0, 1)
y2 = det_comp.compute_y_vector(0, 1, 1)
assert y1.shape == y2.shape == (3, 1)
print("  ✓ Test 2 passed!")

# Test 3: Two component system
print("\nTest 3: Two component system...")
calc2 = FASMinorCalculator.from_characteristic_tuples(
    [(2, 1, 4), (2, 1, 4)],
    use_symbolic=True
)
det_comp2 = DeterminantComputer(calc2)

y_comp0 = det_comp2.compute_y_vector(0, 0, 1)
y_comp1 = det_comp2.compute_y_vector(1, 0, 1)
assert y_comp0.shape == (3, 1)
assert y_comp1.shape == (3, 1)
print("  ✓ Test 3 passed!")

# Test 4: Verify system (lightweight check without simplify)
print("\nTest 4: Verify linear system (no simplification)...")
det_comp._ensure_base_blocks_cache()
A_star = det_comp._A_block_by_comp[0]
extra_row = calc.get_row(0, 0, 1)
c0 = det_comp._comp_edge_starts[0]
e_star = det_comp._comp_edge_sizes[0]
c1 = c0 + e_star
extra_block = extra_row[:, c0:c1]

y_test = det_comp.compute_y_vector(0, 0, 1)
result = A_star.T * y_test
expected = extra_block.T

# Check dimensions match
assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: {expected.shape}")
print("  ✓ Test 4 passed (dimensions match)!")

print("\n" + "="*50)
print("All quick smoke tests passed! ✓")
print("="*50)
