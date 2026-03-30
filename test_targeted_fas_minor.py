import sympy as sp

from targeted_fas_minor import (
    compute_minor_with_p_vars,
    compute_minor_with_selected_vars,
    format_monomial_spec,
    parse_monomial_spec,
)


def test_compute_minor_with_p_vars_can_return_u_gens():
    char_tuples = [(1, 2)]
    extra_row = (0, 0, 1)
    additional_vars = [('edge', 0, (0, 1))]

    minor, u_gens = compute_minor_with_p_vars(
        char_tuples,
        extra_row,
        additional_vars=additional_vars,
        return_u_gens=True,
    )

    assert [s.name for s in u_gens] == ["u_{0,0}", "u_{0,(0,1)}"]

    poly = sp.Poly(minor, *u_gens, domain="EX")
    assert list(poly.gens) == u_gens

    other_u = [s for s in minor.free_symbols if s.name.startswith("u_{") and s not in set(u_gens)]
    assert other_u == []


def test_compute_minor_with_selected_vars_matches_explicit_p_plus_edge_keep_set():
    char_tuples = [(1, 2)]
    extra_row = (0, 0, 1)
    kept_vars = [('vertex', 0, 0), ('edge', 0, (0, 1))]

    selected_minor, selected_u_gens = compute_minor_with_selected_vars(
        char_tuples,
        extra_row,
        kept_vars=kept_vars,
        return_u_gens=True,
    )
    p_minor, p_u_gens = compute_minor_with_p_vars(
        char_tuples,
        extra_row,
        additional_vars=[('edge', 0, (0, 1))],
        return_u_gens=True,
    )

    assert [s.name for s in selected_u_gens] == [s.name for s in p_u_gens]
    assert sp.expand(selected_minor - p_minor) == 0


def test_compute_minor_with_selected_vars_keeps_only_requested_u_symbols():
    char_tuples = [(1, 2)]
    extra_row = (0, 0, 1)

    minor, u_gens = compute_minor_with_selected_vars(
        char_tuples,
        extra_row,
        kept_vars=[('edge', 0, (0, 1))],
        return_u_gens=True,
    )

    assert [s.name for s in u_gens] == ["u_{0,(0,1)}"]
    remaining_u = {s.name for s in minor.free_symbols if s.name.startswith("u_{")}
    assert remaining_u <= {"u_{0,(0,1)}"}


def test_compute_minor_with_selected_vars_empty_keep_set_zeros_all_u_variables():
    char_tuples = [(1, 2)]
    extra_row = (0, 0, 1)

    minor, u_gens = compute_minor_with_selected_vars(
        char_tuples,
        extra_row,
        kept_vars=[],
        return_u_gens=True,
    )

    assert u_gens == []
    remaining_u = [s for s in minor.free_symbols if s.name.startswith("u_{")]
    assert remaining_u == []


def test_parse_and_format_monomial_spec_round_trip():
    monomial_cli = "v:0,0:3;v:1,2;e:0,(1,3):2;e:1,(0,1)"

    parsed = parse_monomial_spec(monomial_cli)
    formatted = format_monomial_spec(parsed)

    assert parsed == {
        ("vertex", 0, 0): 3,
        ("vertex", 1, 2): 1,
        ("edge", 0, (1, 3)): 2,
        ("edge", 1, (0, 1)): 1,
    }
    assert parse_monomial_spec(formatted) == parsed


def test_parse_and_format_constant_monomial_spec():
    assert parse_monomial_spec("1") == {}
    assert format_monomial_spec({}) == "1"
