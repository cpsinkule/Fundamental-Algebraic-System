import b_entry_with_p_vars as bep
from targeted_fas_minor import DeterminantComputer, FASMinorCalculator


def _u_symbol_names(expr):
    return {
        sym.name
        for sym in expr.free_symbols
        if sym.name.startswith("u_{")
    }


def test_compute_b_entry_with_p_vars_keeps_only_p_variables():
    char_tuples = [(3, 1, 5), (2, 1, 3)]
    row = (0, 0, 2)

    temp_calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    temp_det = DeterminantComputer(temp_calc)
    p_spec = temp_det.base_A_root_product_spec()
    allowed_u = {
        temp_calc.vertex_variables[(g, local)].name
        for kind, g, local in p_spec
        if kind == "vertex"
    } | {
        temp_calc.edge_variables[(g, edge)].name
        for kind, g, edge in p_spec
        if kind == "edge"
    }

    expr = bep.compute_b_entry_with_p_vars(char_tuples, row)
    free_u = _u_symbol_names(expr)

    assert free_u <= allowed_u
    assert "u_{0,1}" not in free_u
    assert "u_{1,(0,2)}" not in free_u


def test_compute_b_entry_with_p_vars_retains_requested_exceptions():
    char_tuples = [(3, 1, 5), (2, 1, 3)]
    row = (0, 0, 2)

    expr = bep.compute_b_entry_with_p_vars(
        char_tuples,
        row,
        additional_vars=[("vertex", 0, 1)],
    )
    free_u = _u_symbol_names(expr)

    assert "u_{0,1}" in free_u
    assert "u_{1,(0,2)}" not in free_u


def test_cli_keep_flag_prints_filtered_b_entry(capsys):
    exit_code = bep.main(
        [
            "--tuples",
            "3,1,5;2,1,3",
            "--row",
            "0,0,2",
            "--keep",
            "v:0,1",
            "--show-kept-vars",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "kept_vars=" in out
    assert "u_{0,1}" in out
