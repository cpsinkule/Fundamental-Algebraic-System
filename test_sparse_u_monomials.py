import sympy as sp

from sparse_u_monomials import (
    differentiate_by_structure_function,
    differentiate_sparse_coefficients,
    enumerate_minor_u_monomials,
    expr_to_sparse_u_poly,
    filter_sparse_poly,
    monomial_spec_to_exponent_vector,
    structure_function_symbol,
    sparse_poly_to_expr,
)
from targeted_fas_minor import DeterminantComputer, FASMinorCalculator, compute_minor_with_p_vars


def _poly_terms_dict(expr, gens):
    poly = sp.Poly(expr, *gens, domain="EX")
    return {tuple(exponents): coeff for exponents, coeff in poly.terms()}


def test_expr_to_sparse_u_poly_matches_poly_for_small_expression():
    x, y = sp.symbols("x y")
    a = sp.Symbol("a")
    expr = (x + y) * (x + 2 * y) + a * x

    sparse = expr_to_sparse_u_poly(expr, [x, y])

    assert sparse == _poly_terms_dict(sp.expand(expr), [x, y])
    assert sp.expand(sparse_poly_to_expr(sparse, [x, y]) - sp.expand(expr)) == 0


def test_filter_sparse_poly_by_divisor_vector():
    x, y = sp.symbols("x y")
    sparse = expr_to_sparse_u_poly(x**2 + x * y + y**2, [x, y])

    filtered = filter_sparse_poly(sparse, must_divide=(1, 0))

    assert filtered == {
        (2, 0): sp.Integer(1),
        (1, 1): sp.Integer(1),
    }


def test_enumerate_minor_u_monomials_matches_poly_on_small_minor():
    char_tuples = [(1, 2), (1, 2)]
    extra_row = (0, 0, 1)
    additional_vars = [("edge", 0, (0, 1))]

    sparse, minor, u_gens = enumerate_minor_u_monomials(
        char_tuples,
        extra_row,
        additional_vars=additional_vars,
        return_minor=True,
        return_u_gens=True,
    )

    assert sparse == _poly_terms_dict(sp.expand(minor), u_gens)


def test_monomial_spec_to_exponent_vector_matches_p_generator_order():
    char_tuples = [(1, 2), (1, 2)]
    extra_row = (0, 0, 1)
    additional_vars = [("edge", 0, (0, 1))]

    _, u_gens = compute_minor_with_p_vars(
        char_tuples,
        extra_row,
        additional_vars=additional_vars,
        return_u_gens=True,
    )
    temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    temp_det = DeterminantComputer(temp_calc)
    p_spec = temp_det.base_A_root_product_spec()

    exponents = monomial_spec_to_exponent_vector(p_spec, u_gens)

    assert exponents == (1, 1, 0)


def test_structure_function_symbol_builds_expected_name():
    symbol = structure_function_symbol(
        ("vertex", (1, 0), "vertex", (1, 0), "vertex", (0, 0))
    )

    assert symbol == sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}")


def test_structure_function_symbol_accepts_compact_index_input():
    symbol = structure_function_symbol([(1, 0), (1, 0), (0, 0)])

    assert symbol == sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}")


def test_differentiate_by_structure_function_matches_sympy_diff_on_minor():
    char_tuples = [(1, 2), (1, 2)]
    extra_row = (0, 0, 1)
    additional_vars = [("edge", 0, (0, 1))]
    minor = compute_minor_with_p_vars(
        char_tuples,
        extra_row,
        additional_vars=additional_vars,
    )
    key = ("vertex", (1, 0), "vertex", (1, 0), "vertex", (0, 0))
    symbol = sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}")

    expected = sp.diff(minor, symbol)
    actual = differentiate_by_structure_function(minor, key)

    assert sp.expand(actual - expected) == 0


def test_differentiate_by_structure_function_compact_input_checks_both_orders():
    upper = (0, (0, 1))
    lower_a = (0, 0)
    lower_b = (0, 1)
    primary = sp.Symbol("c^{0,(0,1)}_{(0,0),(0,1)}")
    swapped = sp.Symbol("c^{0,(0,1)}_{(0,1),(0,0)}")
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    expr = primary * x + swapped * y

    actual = differentiate_by_structure_function(expr, [upper, lower_a, lower_b])

    assert actual == x + y


def test_differentiate_by_structure_function_compact_input_second_order():
    upper = (0, (0, 1))
    lower_a = (0, 0)
    lower_b = (0, 1)
    primary = sp.Symbol("c^{0,(0,1)}_{(0,0),(0,1)}")
    swapped = sp.Symbol("c^{0,(0,1)}_{(0,1),(0,0)}")
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    expr = primary**2 * x + swapped**2 * y

    actual = differentiate_by_structure_function(expr, [upper, lower_a, lower_b], order=2)

    assert actual == 2 * x + 2 * y


def test_differentiate_sparse_coefficients_preserves_u_exponents():
    x = sp.Symbol("u_{0,0}")
    coeff_symbol = sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}")
    poly = {
        (2,): coeff_symbol + 1,
        (1,): coeff_symbol**2,
    }

    differentiated = differentiate_sparse_coefficients(poly, coeff_symbol)

    assert differentiated == {
        (2,): sp.Integer(1),
        (1,): 2 * coeff_symbol,
    }


def test_differentiate_by_structure_function_raises_when_symbol_missing():
    x = sp.Symbol("u_{0,0}")

    try:
        differentiate_by_structure_function(
            x + 1,
            "c^{(9,9)}_{(9,9),(9,9)}",
        )
    except ValueError as exc:
        assert "is not present in the supplied expression" in str(exc)
    else:
        raise AssertionError("Expected ValueError when structure function is absent")


def test_differentiate_by_structure_function_raises_when_compact_symbol_missing():
    x = sp.Symbol("u_{0,0}")

    try:
        differentiate_by_structure_function(
            x + 1,
            [(0, (0, 1)), (0, 0), (0, 1)],
        )
    except ValueError as exc:
        assert "Neither the requested structure function nor its lower-index-swapped form" in str(exc)
    else:
        raise AssertionError("Expected ValueError when compact structure function is absent")
