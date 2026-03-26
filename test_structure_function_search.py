import sympy as sp

import structure_function_search as sfs


def _finding_signature(summary: sfs.SearchSummary) -> set[tuple]:
    return {
        (
            finding.extra_row,
            str(finding.u_monomial),
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
            findings=[],
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
    assert "No simple coefficients found." in out
