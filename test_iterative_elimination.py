import json

import sympy as sp

import iterative_elimination as ie


def test_is_valid_exact_witness_accepts_target_powers_and_alpha_factor():
    target = "c^{1,(0,1)}_{(0,(1,2)),(0,2)}"
    target_sym = sp.Symbol(target)
    alpha0 = sp.Symbol("α_{0}")

    assert ie._is_valid_exact_witness(target_sym, target) == (True, "sole_sf_numeric")
    assert ie._is_valid_exact_witness(target_sym**2, target) == (True, "sole_sf_numeric")
    assert ie._is_valid_exact_witness(alpha0 * target_sym**3, target) == (True, "sole_sf_alpha")


def test_is_valid_exact_witness_rejects_additive_and_multi_sf_cases():
    target = "c^{1,(0,1)}_{(0,(1,2)),(0,2)}"
    target_sym = sp.Symbol(target)
    alpha0 = sp.Symbol("α_{0}")
    other_sf = sp.Symbol("c^{1,(0,1)}_{(0,(0,1)),(0,0)}")

    assert ie._is_valid_exact_witness(alpha0 + target_sym**2, target) == (False, "not_pure_target_power")
    assert ie._is_valid_exact_witness(target_sym + 1, target) == (False, "not_pure_target_power")
    assert ie._is_valid_exact_witness(target_sym * other_sf, target) == (False, "multiple_sfs")
    assert ie._is_valid_exact_witness(alpha0, target) == (False, "no_sf")


def test_process_wave_records_vanishing_evidence_from_exact_witness(monkeypatch):
    target_sym = sp.Symbol("c^{1,(0,1)}_{(0,(1,2)),(0,2)}")
    u0 = sp.Symbol("u_{0,0}")

    monkeypatch.setattr(
        ie,
        "_compute_minor_with_sf_injection",
        lambda *args, **kwargs: (sp.Symbol("minor"), [u0]),
    )
    monkeypatch.setattr(ie.sp, "diff", lambda expr, sf: u0)
    monkeypatch.setattr(ie, "expr_to_sparse_u_poly", lambda expr, u_gens: {(1,): sp.Integer(1)})
    monkeypatch.setattr(
        ie,
        "_compute_exact_coefficient_with_sf_injection",
        lambda *args, **kwargs: sp.Symbol("α_{0}") * target_sym**2,
    )

    result = ie._process_wave(
        char_tuples=[(1, 2)],
        root_index=0,
        target_sfs=[(target_sym, ("sf_key",))],
        vanished_sf_keys=[],
        extra_rows=[(0, 0, 1)],
        all_vars=[("vertex", 0, 0)],
        use_sparse=True,
        max_sub_waves=3,
        live_callback=None,
        progress_callback=None,
    )

    assert result.vanished_sfs == [target_sym.name]
    assert result.unresolved_sfs == []
    assert result.errors == []
    assert result.evidence[0].exact_classification == "sole_sf_alpha"


def test_process_wave_rejects_additive_exact_witness(monkeypatch):
    target_sym = sp.Symbol("c^{1,(0,1)}_{(0,(1,2)),(0,2)}")
    u0 = sp.Symbol("u_{0,0}")
    alpha0 = sp.Symbol("α_{0}")

    monkeypatch.setattr(
        ie,
        "_compute_minor_with_sf_injection",
        lambda *args, **kwargs: (sp.Symbol("minor"), [u0]),
    )
    monkeypatch.setattr(ie.sp, "diff", lambda expr, sf: u0)
    monkeypatch.setattr(ie, "expr_to_sparse_u_poly", lambda expr, u_gens: {(1,): sp.Integer(1)})
    monkeypatch.setattr(
        ie,
        "_compute_exact_coefficient_with_sf_injection",
        lambda *args, **kwargs: alpha0 + target_sym**2,
    )

    result = ie._process_wave(
        char_tuples=[(1, 2)],
        root_index=0,
        target_sfs=[(target_sym, ("sf_key",))],
        vanished_sf_keys=[],
        extra_rows=[(0, 0, 1)],
        all_vars=[("vertex", 0, 0)],
        use_sparse=True,
        max_sub_waves=1,
        live_callback=None,
        progress_callback=None,
    )

    assert result.vanished_sfs == []
    assert result.unresolved_sfs == [target_sym.name]


def test_run_iterative_elimination_accumulates_wave_errors(monkeypatch):
    wave = ie.WaveResult(
        root_index=0,
        target_sfs=["sf_a"],
        vanished_sfs=[],
        evidence=[],
        unresolved_sfs=["sf_a"],
        extra_rows_searched=1,
        monomials_examined=0,
        sub_waves=0,
        errors=[{"stage": "exact_coeff", "error": "boom"}],
    )

    monkeypatch.setattr(ie, "enumerate_target_sfs", lambda *args, **kwargs: {0: [(sp.Symbol("sf_a"), ("key",))]})
    monkeypatch.setattr(ie, "enumerate_row_complements", lambda *args, **kwargs: [(0, 0, 1)])
    monkeypatch.setattr(ie, "build_all_vars", lambda *args, **kwargs: [("vertex", 0, 0)])
    monkeypatch.setattr(ie, "_process_wave", lambda **kwargs: wave)

    result = ie.run_iterative_elimination([(1, 2)])

    assert result.success is False
    assert result.errors == [{"stage": "exact_coeff", "error": "boom"}]
    assert result.waves[0].errors == [{"stage": "exact_coeff", "error": "boom"}]


def test_cli_result_json_includes_wave_and_top_level_errors(tmp_path, monkeypatch, capsys):
    wave = ie.WaveResult(
        root_index=0,
        target_sfs=["sf_a"],
        vanished_sfs=[],
        evidence=[],
        unresolved_sfs=["sf_a"],
        extra_rows_searched=1,
        monomials_examined=0,
        sub_waves=0,
        errors=[{"stage": "poly", "error": "boom"}],
    )
    fake_result = ie.EliminationResult(
        char_tuples=[(1, 2)],
        component_0_index=0,
        component_1_index=1,
        waves=[wave],
        all_vanished_sfs=[],
        all_unresolved_sfs=["sf_a"],
        total_sfs_tested=1,
        elapsed_seconds=0.0,
        success=False,
        error_message="failed",
        errors=[{"stage": "poly", "error": "boom"}],
    )

    monkeypatch.setattr(ie, "enumerate_target_sfs", lambda *args, **kwargs: {0: [(sp.Symbol("sf_a"), ("key",))]})
    monkeypatch.setattr(
        ie,
        "run_iterative_elimination",
        lambda *args, **kwargs: fake_result,
    )

    exit_code = ie.main(
        [
            "--tuples",
            "1,2",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "elim",
        ]
    )
    out = capsys.readouterr().out

    with open(tmp_path / "elim_result.json", "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    assert exit_code == 1
    assert payload["errors"] == [{"stage": "poly", "error": "boom"}]
    assert payload["waves"][0]["errors"] == [{"stage": "poly", "error": "boom"}]
    assert "Computation errors (1):" in out
