"""
Calculate the b entry (0,0,1) for the system (5,2,9)+(3,1,5)

This computes b^1_v for graph_idx=0, vertex=0, layer=1
"""

from fas_minor_calculator import FASMinorCalculator
import sympy as sp

# Define the characteristic tuples
char_tuples = [(5, 2, 9), (3, 1, 5)]

print("=" * 80)
print("Computing b entry for (graph_idx=0, vertex=0, layer=1)")
print(f"Characteristic tuples: {char_tuples}")
print("=" * 80)

# Create the calculator
print("\nCreating FASMinorCalculator...")
calc = FASMinorCalculator.from_characteristic_tuples(
    char_tuples,
    use_symbolic=True
)

print(f"Component 0: {len(calc.graphs[0].vertices)} vertices, {len(calc.graphs[0].edges)} edges")
print(f"Component 1: {len(calc.graphs[1].vertices)} vertices, {len(calc.graphs[1].edges)} edges")

# Get the full row for (0, 0, 1)
print("\nComputing row (0, 0, 1)...")
row = calc.get_row(0, 0, 1)

# The b entry is the last column
b_entry = row[-1]

print("\n" + "=" * 80)
print("RESULT: b entry (0,0,1)")
print("=" * 80)
print(f"\nSymbolic expression:")
print(b_entry)

print(f"\n\nSimplified:")
b_simplified = sp.simplify(b_entry)
print(b_simplified)

# Check if it's zero
if b_entry == 0:
    print("\n*** The b entry is ZERO ***")
    print("\nThis is expected per CLAUDE.md: Type 3 structure functions c^l_{w,v}")
    print("(all vertex indices) are not fully implemented, causing b entries to often be zero.")
else:
    print(f"\n*** The b entry is NON-ZERO ***")
    print(f"\nNumber of terms: {len(b_entry.as_ordered_terms()) if hasattr(b_entry, 'as_ordered_terms') else 'N/A'}")

    # Try to get LaTeX representation
    try:
        print("\nLaTeX representation:")
        print(sp.latex(b_simplified))
    except Exception as e:
        print(f"Could not generate LaTeX: {e}")

print("\n" + "=" * 80)
