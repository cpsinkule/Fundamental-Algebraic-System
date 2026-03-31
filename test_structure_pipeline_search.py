import json

import sympy as sp

import monomial_pair_search as mps
import structure_pipeline_search as sps
import structure_function_search as sfs


def test_pipeline_defaults_to_sparse_and_live_and_hands_off_tasks(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_search_simple_coefficients(*args, **kwargs):
        captured["search_kwargs"] = kwargs
        finding_callback = kwargs["finding_callback"]
        finding_callback(
            sfs.SimpleCoefficientResult(
                extra_row=(0, 0, 1),
                u_monomial=sp.Symbol("u_{0,0}"),
                selected_vars=(("vertex", 0, 0),),
                monomial_cli="v:0,0",
                coefficient=sp.Integer(1),
                classification="numeric",
            )
        )
        return sfs.SearchSummary(
            char_tuples=[(1, 2)],
            target_structure_function="c^{(1,0)}_{(1,0),(0,0)}",
            diff_order=1,
            total_extra_rows_searched=1,
            total_monomials_examined=1,
            total_simple_found=1,
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

    def fake_run_from_artifact(artifact, *, input_label, progress_callback=None):
        captured["artifact"] = artifact
        captured["input_label"] = input_label
        captured["pair_progress_callback"] = progress_callback
        return mps.MonomialPairSearchSummary(
            input_path=input_label,
            char_tuples=[(1, 2)],
            total_tasks=artifact["total_tasks"],
            processed_tasks=1,
            results=[],
            elapsed_seconds=0.0,
            errors=[],
        )

    monkeypatch.setattr(sps, "search_simple_coefficients", fake_search_simple_coefficients)
    monkeypatch.setattr(sps, "run_monomial_pair_search_from_artifact", fake_run_from_artifact)

    exit_code = sps.main(
        [
            "--tuples",
            "1,2",
            "--structure-function",
            "c^{(1,0)}_{(1,0),(0,0)}",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "demo",
            "--quiet",
        ]
    )
    out = capsys.readouterr().out

    with open(tmp_path / "demo_summary.json", "r", encoding="utf-8") as fp:
        summary_payload = json.load(fp)
    with open(tmp_path / "demo_tasks.json", "r", encoding="utf-8") as fp:
        task_payload = json.load(fp)
    with open(tmp_path / "demo_pair_results.json", "r", encoding="utf-8") as fp:
        pair_payload = json.load(fp)

    assert exit_code == 0
    assert captured["search_kwargs"]["use_sparse"] is True
    assert captured["search_kwargs"]["finding_callback"] is not None
    assert captured["artifact"]["total_tasks"] == 1
    assert captured["artifact"]["tasks"][0]["monomial_cli"] == "v:0,0"
    assert captured["input_label"].endswith("demo_tasks.json")
    assert summary_payload["total_simple_found"] == 1
    assert task_payload["total_tasks"] == 1
    assert pair_payload["processed_tasks"] == 1
    assert "Sparse search: yes" in out
    assert "Live search output: yes" in out


def test_pipeline_honors_dense_and_no_live_opt_outs(tmp_path, monkeypatch):
    captured = {}

    def fake_search_simple_coefficients(*args, **kwargs):
        captured["search_kwargs"] = kwargs
        return sfs.SearchSummary(
            char_tuples=[(1, 2)],
            target_structure_function="c^{(1,0)}_{(1,0),(0,0)}",
            diff_order=1,
            total_extra_rows_searched=0,
            total_monomials_examined=0,
            total_simple_found=0,
            findings=[],
            elapsed_seconds=0.0,
            errors=[],
        )

    monkeypatch.setattr(sps, "search_simple_coefficients", fake_search_simple_coefficients)

    exit_code = sps.main(
        [
            "--tuples",
            "1,2",
            "--structure-function",
            "c^{(1,0)}_{(1,0),(0,0)}",
            "--output-dir",
            str(tmp_path),
            "--dense",
            "--no-live",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert captured["search_kwargs"]["use_sparse"] is False
    assert captured["search_kwargs"]["finding_callback"] is None


def test_pipeline_zero_hits_still_writes_all_outputs(tmp_path, monkeypatch):
    def fake_search_simple_coefficients(*args, **kwargs):
        return sfs.SearchSummary(
            char_tuples=[(1, 2)],
            target_structure_function="c^{(9,9)}_{(9,9),(9,9)}",
            diff_order=1,
            total_extra_rows_searched=1,
            total_monomials_examined=0,
            total_simple_found=0,
            findings=[],
            elapsed_seconds=0.0,
            errors=[],
        )

    monkeypatch.setattr(sps, "search_simple_coefficients", fake_search_simple_coefficients)

    exit_code = sps.main(
        [
            "--tuples",
            "1,2",
            "--structure-function",
            "c^{(9,9)}_{(9,9),(9,9)}",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "empty",
            "--quiet",
            "--no-live",
        ]
    )

    with open(tmp_path / "empty_tasks.json", "r", encoding="utf-8") as fp:
        task_payload = json.load(fp)
    with open(tmp_path / "empty_pair_results.json", "r", encoding="utf-8") as fp:
        pair_payload = json.load(fp)

    assert exit_code == 0
    assert task_payload["total_tasks"] == 0
    assert task_payload["tasks"] == []
    assert pair_payload["total_tasks"] == 0
    assert pair_payload["processed_tasks"] == 0
    assert pair_payload["results"] == []


def test_pipeline_records_exact_step_errors_without_aborting(tmp_path, monkeypatch, capsys):
    def fake_search_simple_coefficients(*args, **kwargs):
        return sfs.SearchSummary(
            char_tuples=[(1, 2)],
            target_structure_function="c^{(1,0)}_{(1,0),(0,0)}",
            diff_order=1,
            total_extra_rows_searched=1,
            total_monomials_examined=1,
            total_simple_found=1,
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

    def fake_run_from_artifact(artifact, *, input_label, progress_callback=None):
        return mps.MonomialPairSearchSummary(
            input_path=input_label,
            char_tuples=[(1, 2)],
            total_tasks=artifact["total_tasks"],
            processed_tasks=0,
            results=[],
            elapsed_seconds=0.0,
            errors=[
                {
                    "extra_row": [0, 0, 1],
                    "monomial_cli": "v:0,0",
                    "error": "boom",
                }
            ],
        )

    monkeypatch.setattr(sps, "search_simple_coefficients", fake_search_simple_coefficients)
    monkeypatch.setattr(sps, "run_monomial_pair_search_from_artifact", fake_run_from_artifact)

    exit_code = sps.main(
        [
            "--tuples",
            "1,2",
            "--structure-function",
            "c^{(1,0)}_{(1,0),(0,0)}",
            "--output-dir",
            str(tmp_path),
            "--quiet",
            "--no-live",
        ]
    )
    out = capsys.readouterr().out

    with open(tmp_path / "pipeline_pair_results.json", "r", encoding="utf-8") as fp:
        pair_payload = json.load(fp)

    assert exit_code == 0
    assert pair_payload["processed_tasks"] == 0
    assert len(pair_payload["errors"]) == 1
    assert "Exact-step errors: 1" in out
    assert "error=boom" in out
