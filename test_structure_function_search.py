import sympy as sp
import json

import structure_function_search as sfs


def _finding_signature(summary: sfs.SearchSummary) -> set[tuple]:
    return {
        (
            finding.extra_row,
            str(finding.u_monomial),
            finding.selected_vars,
            finding.monomial_cli,
            sp.srepr(finding.coefficient),
            finding.classification,
        )
        for finding in summary.findings
    }


def test_search_simple_coefficients_sparse_matches_dense_on_small_case():
    char_tuples = [(1, 2), (1, 2)]
    extra_rows = [(0, 0, 1)]
    target = sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}")

    dense = sfs.search_simple_coefficients(
        char_tuples,
        target,
        extra_rows=extra_rows,
        use_sparse=False,
    )
    sparse = sfs.search_simple_coefficients(
        char_tuples,
        target,
        extra_rows=extra_rows,
        use_sparse=True,
    )

    assert dense.errors == []
    assert sparse.errors == []
    assert sparse.total_extra_rows_searched == dense.total_extra_rows_searched
    assert sparse.total_monomials_examined == dense.total_monomials_examined
    assert sparse.total_simple_found == dense.total_simple_found
    assert _finding_signature(sparse) == _finding_signature(dense)


def test_search_simple_coefficients_sparse_handles_absent_target_like_dense():
    char_tuples = [(1, 2), (1, 2)]
    extra_rows = [(0, 0, 1)]
    target = "c^{(9,9)}_{(9,9),(9,9)}"

    dense = sfs.search_simple_coefficients(
        char_tuples,
        target,
        extra_rows=extra_rows,
        use_sparse=False,
    )
    sparse = sfs.search_simple_coefficients(
        char_tuples,
        target,
        extra_rows=extra_rows,
        use_sparse=True,
    )

    assert dense.errors == []
    assert sparse.errors == []
    assert dense.total_monomials_examined == 0
    assert sparse.total_monomials_examined == 0
    assert dense.total_simple_found == 0
    assert sparse.total_simple_found == 0
    assert dense.findings == []
    assert sparse.findings == []


def test_main_passes_sparse_flag_through_to_search(monkeypatch, capsys):
    captured = {}

    def fake_search_simple_coefficients(*args, **kwargs):
        captured["kwargs"] = kwargs
        return sfs.SearchSummary(
            char_tuples=[(1, 2), (1, 2)],
            target_structure_function="c^{(1,0)}_{(1,0),(0,0)}",
            diff_order=1,
            total_extra_rows_searched=1,
            total_monomials_examined=0,
            total_simple_found=0,
            findings=[
                sfs.SimpleCoefficientResult(
                    extra_row=(0, 0, 1),
                    u_monomial=sp.Symbol("u_{0,0}"),
                    selected_vars=(("vertex", 0, 0),),
                    monomial_cli="v:0,0",
                    coefficient=sp.Integer(1),
                    classification="numeric",
                )
            ],
            elapsed_seconds=0.0,
            errors=[],
        )

    monkeypatch.setattr(sfs, "search_simple_coefficients", fake_search_simple_coefficients)

    exit_code = sfs.main(
        [
            "--tuples",
            "1,2;1,2",
            "--structure-function",
            "c^{(1,0)}_{(1,0),(0,0)}",
            "--sparse",
            "--quiet",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert captured["kwargs"]["use_sparse"] is True
    assert "System: [(1, 2), (1, 2)]" in out
    assert "vars=[('vertex', 0, 0)]" in out


def test_sparse_mode_converts_differentiated_expression_not_full_minor(monkeypatch):
    captured = {}

    def fake_compute_minor_with_p_vars(*args, **kwargs):
        x = sp.Symbol("u_{0,0}")
        target = sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}")
        other = sp.Symbol("other")
        return target * x + other, [x]

    def fake_differentiate(expr, structure_function, order=1):
        captured["diff_input"] = expr
        return sp.Symbol("u_{0,0}") + 2

    def fake_sparse(expr, u_gens, **kwargs):
        captured["sparse_input"] = expr
        return {
            (1,): sp.Integer(1),
            (0,): sp.Integer(2),
        }

    monkeypatch.setattr(sfs, "compute_minor_with_p_vars", fake_compute_minor_with_p_vars)
    monkeypatch.setattr(sfs, "differentiate_by_structure_function", fake_differentiate)
    monkeypatch.setattr(sfs, "expr_to_sparse_u_poly", fake_sparse)

    summary = sfs.search_simple_coefficients(
        [(1, 2), (1, 2)],
        sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}"),
        extra_rows=[(0, 0, 1)],
        use_sparse=True,
    )

    assert captured["diff_input"] == sp.Symbol("c^{(1,0)}_{(1,0),(0,0)}") * sp.Symbol("u_{0,0}") + sp.Symbol("other")
    assert captured["sparse_input"] == sp.Symbol("u_{0,0}") + 2
    assert summary.total_monomials_examined == 2
    assert summary.total_simple_found == 2


def test_simple_coefficient_result_includes_copy_paste_selected_vars():
    result = sfs.SimpleCoefficientResult(
        extra_row=(0, 0, 1),
        u_monomial=sp.Symbol("u_{0,0}") * sp.Symbol("u_{1,(0,1)}"),
        selected_vars=(("vertex", 0, 0), ("edge", 1, (0, 1))),
        monomial_cli="v:0,0;e:1,(0,1)",
        coefficient=sp.Integer(1),
        classification="numeric",
    )

    assert result.selected_vars_literal() == "[('vertex', 0, 0), ('edge', 1, (0, 1))]"
    assert result.to_dict()["selected_vars_literal"] == "[('vertex', 0, 0), ('edge', 1, (0, 1))]"
    assert result.to_dict()["selected_vars"] == [["vertex", 0, 0], ["edge", 1, [0, 1]]]
    assert result.to_dict()["monomial_cli"] == "v:0,0;e:1,(0,1)"


def test_monomial_cli_from_monomial_preserves_exponents():
    u_gens = [
        sp.Symbol("u_{0,0}"),
        sp.Symbol("u_{0,1}"),
        sp.Symbol("u_{1,(0,1)}"),
    ]

    cli = sfs._monomial_cli_from_monomial((3, 2, 1), u_gens)

    assert cli == "v:0,0:3;v:0,1:2;e:1,(0,1)"


def test_live_targeted_command_prints_copy_paste_command(monkeypatch, capsys):
    def fake_search_simple_coefficients(*args, **kwargs):
        finding_callback = kwargs["finding_callback"]
        finding_callback(
            sfs.SimpleCoefficientResult(
                extra_row=(0, 0, 2),
                u_monomial=sp.Symbol("u_{0,0}"),
                selected_vars=(("vertex", 0, 0),),
                monomial_cli="v:0,0:3;e:1,(0,1)",
                coefficient=sp.Integer(1),
                classification="numeric",
            )
        )
        return sfs.SearchSummary(
            char_tuples=[(3, 1, 5), (2, 1, 3)],
            target_structure_function="c^{1,(0,1)}_{(0,(1,2)),(0,2)}",
            diff_order=1,
            total_extra_rows_searched=1,
            total_monomials_examined=1,
            total_simple_found=1,
            findings=[],
            elapsed_seconds=0.0,
            errors=[],
        )

    monkeypatch.setattr(sfs, "search_simple_coefficients", fake_search_simple_coefficients)

    exit_code = sfs.main(
        [
            "--tuples",
            "3,1,5;2,1,3",
            "--structure-function",
            "c^{1,(0,1)}_{(0,(1,2)),(0,2)}",
            "--live",
            "--live-targeted-command",
            "--quiet",
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert (
        'python targeted_fas_minor.py --tuples "3,1,5;2,1,3" '
        '--row "0,0,2" --monomial "v:0,0:3;e:1,(0,1)" --coeff-mode exact'
    ) in out


def test_task_output_writes_deduplicated_task_artifact(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "summary.json"
    task_path = tmp_path / "tasks.json"

    def fake_search_simple_coefficients(*args, **kwargs):
        repeated = sfs.SimpleCoefficientResult(
            extra_row=(0, 0, 2),
            u_monomial=sp.Symbol("u_{0,0}")**2,
            selected_vars=(("vertex", 0, 0),),
            monomial_cli="v:0,0:2",
            coefficient=sp.Symbol("α_{0}"),
            classification="alpha_only",
        )
        distinct = sfs.SimpleCoefficientResult(
            extra_row=(0, 1, 2),
            u_monomial=sp.Symbol("u_{0,1}"),
            selected_vars=(("vertex", 0, 1),),
            monomial_cli="v:0,1",
            coefficient=sp.Integer(2),
            classification="numeric",
        )
        return sfs.SearchSummary(
            char_tuples=[(3, 1, 5), (2, 1, 3)],
            target_structure_function="c^{1,(0,1)}_{(0,(1,2)),(0,2)}",
            diff_order=1,
            total_extra_rows_searched=2,
            total_monomials_examined=3,
            total_simple_found=3,
            findings=[repeated, repeated, distinct],
            elapsed_seconds=0.0,
            errors=[],
        )

    monkeypatch.setattr(sfs, "search_simple_coefficients", fake_search_simple_coefficients)

    exit_code = sfs.main(
        [
            "--tuples",
            "3,1,5;2,1,3",
            "--structure-function",
            "c^{1,(0,1)}_{(0,(1,2)),(0,2)}",
            "--output",
            str(summary_path),
            "--task-output",
            str(task_path),
            "--quiet",
        ]
    )
    out = capsys.readouterr().out

    with open(summary_path, "r", encoding="utf-8") as fp:
        summary_payload = json.load(fp)
    with open(task_path, "r", encoding="utf-8") as fp:
        task_payload = json.load(fp)

    assert exit_code == 0
    assert summary_payload["total_simple_found"] == 3
    assert task_payload["artifact_type"] == "monomial_pair_tasks"
    assert task_payload["source_summary_path"] == str(summary_path)
    assert task_payload["total_tasks"] == 2
    assert [task["monomial_cli"] for task in task_payload["tasks"]] == ["v:0,0:2", "v:0,1"]
    assert task_payload["tasks"][0]["source_finding_count"] == 2
    assert "Task artifact written to" in out
