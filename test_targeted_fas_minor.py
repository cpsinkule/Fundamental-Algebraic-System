import sympy as sp

from targeted_fas_minor import compute_minor_with_p_vars, compute_minor_with_selected_vars


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
