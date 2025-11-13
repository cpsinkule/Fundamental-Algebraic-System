"""
Test script for the display widget functionality.

Run this in a Jupyter notebook or with IPython to see the interactive widgets.
In a terminal, it will just print basic info.
"""

import sys

def test_widget_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    try:
        from output_display import SymbolicExpressionWidget, WIDGETS_AVAILABLE
        print(f"✓ output_display imported successfully")
        print(f"  Widget support available: {WIDGETS_AVAILABLE}")

        from determinant_computer import DISPLAY_AVAILABLE, DeterminantComputer
        print(f"✓ determinant_computer imported successfully")
        print(f"  Display methods available: {DISPLAY_AVAILABLE}")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_simple_widget():
    """Test creating a simple widget with a basic expression."""
    print("\nTesting simple widget creation...")

    try:
        from output_display import SymbolicExpressionWidget
        import sympy as sp

        # Create a simple expression
        x, y = sp.symbols('x y')
        expr = sp.expand((x + y)**8)

        print(f"  Created expression with {len(str(expr))} characters")

        # Create widget
        widget = SymbolicExpressionWidget(expr, name="Test (x+y)^8")

        print(f"✓ Widget created successfully")
        print(f"  Expression type: {widget.expr_type}")
        print(f"  Term count: {widget.term_count}")
        print(f"  Symbol count: {widget.symbol_count}")
        print(f"  Degree: {widget.degree}")

        # Test methods
        latex = widget.get_latex(truncate=100)
        print(f"  LaTeX (truncated): {latex[:50]}...")

        return True
    except Exception as e:
        print(f"✗ Widget creation failed: {e}")
        return False

def test_calculator_integration():
    """Test integration with FASMinorCalculator and DeterminantComputer."""
    print("\nTesting calculator integration...")

    try:
        from fas_minor_calculator import FASMinorCalculator
        from determinant_computer import DeterminantComputer

        # Create a small system
        print("  Creating two-component system...")
        calc = FASMinorCalculator.from_characteristic_tuples(
            [(2, 1, 4), (2, 1, 3)],
            use_symbolic=True
        )
        det_comp = DeterminantComputer(calc)

        print(f"  System: {len(calc.graphs)} components, "
              f"{calc.total_vertices} vertices, {calc.total_edges} edges")

        # Test display method availability
        if hasattr(det_comp, 'compute_minor_display'):
            print("✓ compute_minor_display method available")

            # Try to compute (will work in notebooks with widgets installed)
            try:
                widget = det_comp.compute_minor_display(0, 0, 1)
                print(f"✓ Minor widget created successfully")
                print(f"  Minor size: {widget.expr_length:,} characters")
                print(f"  Minor terms: {widget.term_count:,}")
                return True
            except ImportError as e:
                print(f"⚠ Widget creation skipped (ipywidgets not installed)")
                print(f"  To enable widgets: pip install ipywidgets")
                return True
        else:
            print("✗ compute_minor_display method not found")
            return False

    except Exception as e:
        print(f"✗ Calculator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Display Widget Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_widget_imports()))

    # Test 2: Simple widget
    results.append(("Simple Widget", test_simple_widget()))

    # Test 3: Calculator integration
    results.append(("Calculator Integration", test_calculator_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Open examples/viewing_large_outputs.ipynb in Jupyter")
        print("2. Run the cells to see interactive widgets in action")
        print("3. Use *_display() methods in your notebooks for large outputs")
    else:
        print("\n⚠ Some tests failed. Check error messages above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
