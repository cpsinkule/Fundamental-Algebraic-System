import sympy as sp

from targeted_fas_minor import compute_minor_with_p_vars


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

