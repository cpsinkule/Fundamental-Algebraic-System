import json

import monomial_pair_search as mps
from targeted_fas_minor import compute_monomial_coefficient


def test_run_monomial_pair_search_consumes_summary_json(tmp_path):
    summary_path = tmp_path / "summary.json"
    payload = {
        "char_tuples": [[1, 2], [1, 2]],
        "target_structure_function": "c^{(1,0)}_{(1,0),(0,0)}",
        "diff_order": 1,
        "total_extra_rows_searched": 1,
        "total_monomials_examined": 1,
        "total_simple_found": 1,
        "elapsed_seconds": 0.0,
        "errors": [],
        "findings": [
            {
                "extra_row": [0, 0, 1],
                "u_monomial": "u_{0,0}*u_{1,0}**3",
                "selected_vars": [["vertex", 0, 0], ["vertex", 1, 0]],
                "selected_vars_literal": "[('vertex', 0, 0), ('vertex', 1, 0)]",
                "monomial_cli": "v:0,0;v:1,0:3",
                "coefficient": "α_{0}",
                "coefficient_srepr": "Symbol('α_{0}')",
                "classification": "alpha_only",
            }
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)

    summary = mps.run_monomial_pair_search(str(summary_path))

    expected = compute_monomial_coefficient(
        [(1, 2), (1, 2)],
        (0, 0, 1),
        {("vertex", 0, 0): 1, ("vertex", 1, 0): 3},
        match="exact",
    )
    assert summary.total_tasks == 1
    assert summary.processed_tasks == 1
    assert summary.errors == []
    assert summary.results[0].coefficient == expected


def test_run_monomial_pair_search_consumes_task_artifact_and_dedupes_upstream(tmp_path):
    task_path = tmp_path / "tasks.json"
    payload = {
        "artifact_type": "monomial_pair_tasks",
        "generated_at": "2026-03-30T00:00:00+00:00",
        "char_tuples": [[1, 2]],
        "target_structure_function": "c^{(1,0)}_{(1,0),(0,0)}",
        "diff_order": 1,
        "source_summary_path": None,
        "total_tasks": 1,
        "tasks": [
            {
                "extra_row": [0, 0, 1],
                "u_monomial": "u_{0,0}",
                "monomial_cli": "v:0,0",
                "selected_vars": [["vertex", 0, 0]],
                "source_finding_indices": [0, 3],
                "source_classifications": ["numeric"],
                "source_coefficients": ["1"],
                "source_finding_count": 2,
            }
        ],
    }
    with open(task_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)

    summary = mps.run_monomial_pair_search(str(task_path))

    expected = compute_monomial_coefficient(
        [(1, 2)],
        (0, 0, 1),
        {("vertex", 0, 0): 1},
        match="exact",
    )
    assert summary.total_tasks == 1
    assert summary.processed_tasks == 1
    assert summary.results[0].task["source_finding_count"] == 2
    assert summary.results[0].coefficient == expected


def test_main_writes_json_output(tmp_path, capsys):
    task_path = tmp_path / "tasks.json"
    output_path = tmp_path / "pair_results.json"
    payload = {
        "artifact_type": "monomial_pair_tasks",
        "generated_at": "2026-03-30T00:00:00+00:00",
        "char_tuples": [[1, 2]],
        "target_structure_function": "c^{(1,0)}_{(1,0),(0,0)}",
        "diff_order": 1,
        "source_summary_path": None,
        "total_tasks": 1,
        "tasks": [
            {
                "extra_row": [0, 0, 1],
                "u_monomial": "u_{0,0}",
                "monomial_cli": "v:0,0",
                "selected_vars": [["vertex", 0, 0]],
                "source_finding_indices": [0],
                "source_classifications": ["numeric"],
                "source_coefficients": ["1"],
                "source_finding_count": 1,
            }
        ],
    }
    with open(task_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)

    exit_code = mps.main(["--input", str(task_path), "--output", str(output_path), "--quiet"])
    out = capsys.readouterr().out

    with open(output_path, "r", encoding="utf-8") as fp:
        output_payload = json.load(fp)

    assert exit_code == 0
    assert output_payload["processed_tasks"] == 1
    assert output_payload["results"][0]["monomial_cli"] == "v:0,0"
    assert "Results written to" in out
