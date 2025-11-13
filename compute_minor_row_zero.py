#!/usr/bin/env python3
"""
Compute the minor with extra row (0,0,1) for system (2,1,4)+(2,1,3)
"""

from fas_minor_calculator import FASMinorCalculator
from determinant_computer import DeterminantComputer

def main():
    print("="*70)
    print("Computing Minor for System (2,1,4)+(2,1,3)")
    print("="*70)

    # Define characteristic tuples for the two components
    tuples = [(2, 1, 4), (2, 1, 3)]
    print(f"\nCharacteristic tuples: {tuples}")

    # Create calculator from characteristic tuples
    print("\nCreating FASMinorCalculator...")
    calc = FASMinorCalculator.from_characteristic_tuples(
        tuples,
        use_symbolic=True
    )

    # Create determinant computer
    print("Creating DeterminantComputer...")
    det_comp = DeterminantComputer(calc)

    # Get info about base rows
    base_rows = det_comp.get_base_rows()
    print(f"\nAuto-generated {len(base_rows)} base rows:")
    for g, v, s in base_rows:
        print(f"  Component {g}, vertex {v}, layer {s}")

    # Specify the extra row
    extra_row = (0, 0, 1)  # graph_idx=0, vertex=0, layer=1
    print(f"\nExtra row: Component {extra_row[0]}, vertex {extra_row[1]}, layer {extra_row[2]}")

    # Compute the minor using fast method
    print("\nComputing minor (this may take a moment)...")
    minor = det_comp.compute_minor_fast(
        extra_row[0],  # graph_idx
        extra_row[1],  # vertex
        extra_row[2]   # layer
    )

    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"\nMinor determinant:\n{minor}")
    print(f"\nSimplified:\n{minor.simplify()}")

    return minor

if __name__ == "__main__":
    result = main()
