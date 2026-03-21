import sympy as sp

from targeted_fas_minor import FASMinorCalculator
from structure_witness_search import (
    _build_arg_parser,
    _coefficient_matches_isolation_rule,
    _parse_structure_function_argument,
    _resolve_mixed_type2_target,
    build_full_u_gens,
    find_structure_function_witnesses,
    iter_candidate_rows_for_structure_function,
)


def test_resolve_mixed_type2_target_accepts_both_lower_orders():
    primary = _resolve_mixed_type2_target([(0, (0, 1)), (1, (0, 2)), (1, 0)])
    swapped = _resolve_mixed_type2_target([(0, (0, 1)), (1, 0), (1, (0, 2))])

    assert primary.target_symbol == sp.Symbol("c^{0,(0,1)}_{(1,(0,2)),(1,0)}")
    assert swapped.target_symbol == primary.target_symbol
    assert swapped.target_variants == primary.target_variants
    assert primary.lower_component == 1


def test_iter_candidate_rows_uses_lower_component_and_base_range():
    char_tuples = [(2, 1, 4), (2, 1, 3)]
    spec = [(0, (0, 1)), (1, (0, 2)), (1, 0)]

    rows = list(iter_candidate_rows_for_structure_function(char_tuples, spec))
    calc = FASMinorCalculator.from_characteristic_tuples(char_tuples)
    graph = calc.graphs[1]

    assert rows
    assert all(row[0] == 1 for row in rows)
    assert all(1 <= row[2] <= graph.num_roots for row in rows)
    assert all(graph.get_vertex_depth(row[1]) >= row[2] for row in rows)


def test_build_full_u_gens_matches_vertex_then_edge_order():
    calc = FASMinorCalculator.from_characteristic_tuples([(1, 2), (1, 2)])

    u_gens = build_full_u_gens(calc)

    assert [symbol.name for symbol in u_gens] == [
        "u_{0,0}",
        "u_{0,1}",
        "u_{1,0}",
        "u_{1,1}",
        "u_{0,(0,1)}",
        "u_{1,(0,1)}",
    ]


def test_coefficient_matches_linear_isolation_rule():
    target = sp.Symbol("c^{0,(0,1)}_{(1,(0,2)),(1,0)}")
    other = sp.Symbol("c^{9,(9,9)}_{(8,(8,8)),(8,0)}")
    alpha_0 = sp.Symbol("α_{0}")
    beta = sp.Symbol("beta")

    assert _coefficient_matches_isolation_rule(
        alpha_0 * (target + 1),
        target_variants=(target,),
        isolation="linear",
    )
    assert not _coefficient_matches_isolation_rule(
        alpha_0 * other,
        target_variants=(target,),
        isolation="linear",
    )
    assert not _coefficient_matches_isolation_rule(
        beta,
        target_variants=(target,),
        isolation="linear",
    )


def test_find_structure_function_witnesses_returns_empty_when_derivative_absent():
    char_tuples = [(1, 2), (1, 2)]
    spec = [(0, (0, 1)), (1, (0, 1)), (1, 0)]

    witnesses = find_structure_function_witnesses(char_tuples, spec, limit=1)

    assert witnesses == []


def test_find_structure_function_witnesses_finds_real_witness():
    char_tuples = [(2, 1, 4), (2, 1, 3)]
    spec = [(0, (0, 1)), (1, (0, 2)), (1, 0)]

    witnesses = find_structure_function_witnesses(char_tuples, spec, limit=1, max_terms=5000)

    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.extra_row[0] == 1
    assert witness.target_symbol == sp.Symbol("c^{0,(0,1)}_{(1,(0,2)),(1,0)}")
    assert witness.coefficient != 0
    coeff_symbols = witness.coefficient.free_symbols
    assert sp.Symbol("c^{0,(0,1)}_{(1,(0,2)),(1,0)}") in coeff_symbols
    assert all(
        symbol.name.startswith("α_{") or symbol == witness.target_symbol
        for symbol in coeff_symbols
    )


def test_find_structure_function_witnesses_forwards_sparse_bounds(monkeypatch):
    captured = {}

    def fake_diff(expr, structure_function, order=1):
        return sp.Symbol("u_{0,0}")

    def fake_sparse(expr, u_gens, **kwargs):
        captured["expr"] = expr
        captured["kwargs"] = kwargs
        return {tuple([0] * len(u_gens)): sp.Symbol("α_{0}")}

    monkeypatch.setattr("structure_witness_search.differentiate_by_structure_function", fake_diff)
    monkeypatch.setattr("structure_witness_search.expr_to_sparse_u_poly", fake_sparse)

    witnesses = find_structure_function_witnesses(
        [(1, 2), (1, 2)],
        [(0, (0, 1)), (1, (0, 1)), (1, 0)],
        limit=1,
        max_total_degree=3,
        max_degree_per_var=(1, None, 0, 0, 0, 0),
        max_terms=7,
    )

    assert len(witnesses) == 1
    assert captured["kwargs"] == {
        "max_total_degree": 3,
        "max_degree_per_var": (1, None, 0, 0, 0, 0),
        "max_terms": 7,
    }


def test_cli_parsing_accepts_compact_mixed_type2_format():
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--tuples",
            "2,1,4;2,1,3",
            "--structure-function",
            "[(0,(0,1)), (1,(0,2)), (1,0)]",
        ]
    )

    parsed = _parse_structure_function_argument(args.structure_function)

    assert parsed == [(0, (0, 1)), (1, (0, 2)), (1, 0)]
