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


def test_process_wave_injects_same_wave_zeroes_into_future_rows(monkeypatch):
    target_sym = sp.Symbol("c^{1,(0,1)}_{(0,(1,2)),(0,2)}")
    sf_key = ("sf_key",)
    u0 = sp.Symbol("u_{0,0}")
    row_calls = []

    def fake_compute_minor(char_tuples, row, vanished_sf_keys, all_vars):
        row_calls.append((row, tuple(vanished_sf_keys)))
        return sp.Symbol(f"minor_{row[1]}"), [u0]

    monkeypatch.setattr(ie, "_compute_minor_with_sf_injection", fake_compute_minor)
    monkeypatch.setattr(
        ie.sp,
        "diff",
        lambda expr, sf: u0 if expr == sp.Symbol("minor_0") and sf == target_sym else sp.Integer(0),
    )
    monkeypatch.setattr(
        ie,
        "expr_to_sparse_u_poly",
        lambda expr, u_gens: {(1,): sp.Integer(1)} if expr == u0 else {},
    )
    monkeypatch.setattr(
        ie,
        "_compute_exact_coefficient_with_sf_injection",
        lambda *args, **kwargs: target_sym**2,
    )

    result = ie._process_wave(
        char_tuples=[(1, 2)],
        root_index=0,
        target_sfs=[(target_sym, sf_key)],
        vanished_sf_keys=[],
        extra_rows=[(0, 0, 1), (0, 1, 1)],
        all_vars=[("vertex", 0, 0)],
        use_sparse=True,
        max_sub_waves=1,
        live_callback=None,
        progress_callback=None,
    )

    assert result.vanished_sfs == [target_sym.name]
    assert row_calls == [
        ((0, 0, 1), ()),
        ((0, 1, 1), (sf_key,)),
    ]


def test_process_wave_does_not_recompute_earlier_rows_after_same_wave_vanishing(monkeypatch):
    target_sym = sp.Symbol("c^{1,(0,1)}_{(0,(1,2)),(0,2)}")
    sf_key = ("sf_key",)
    u0 = sp.Symbol("u_{0,0}")
    row_calls = []

    def fake_compute_minor(char_tuples, row, vanished_sf_keys, all_vars):
        row_calls.append((row, tuple(vanished_sf_keys)))
        return sp.Symbol(f"minor_{row[1]}"), [u0]

    monkeypatch.setattr(ie, "_compute_minor_with_sf_injection", fake_compute_minor)
    monkeypatch.setattr(
        ie.sp,
        "diff",
        lambda expr, sf: u0 if expr in {sp.Symbol("minor_0"), sp.Symbol("minor_1")} and sf == target_sym else sp.Integer(0),
    )
    monkeypatch.setattr(
        ie,
        "expr_to_sparse_u_poly",
        lambda expr, u_gens: {(1,): sp.Integer(1)} if expr == u0 else {},
    )
    monkeypatch.setattr(
        ie,
        "_compute_exact_coefficient_with_sf_injection",
        lambda *args, **kwargs: target_sym**2,
    )

    ie._process_wave(
        char_tuples=[(1, 2)],
        root_index=0,
        target_sfs=[(target_sym, sf_key)],
        vanished_sf_keys=[],
        extra_rows=[(0, 0, 1), (0, 1, 1)],
        all_vars=[("vertex", 0, 0)],
        use_sparse=True,
        max_sub_waves=1,
        live_callback=None,
        progress_callback=None,
    )

    assert len(row_calls) == 2
    assert row_calls[0] == ((0, 0, 1), ())
    assert row_calls[1] == ((0, 1, 1), (sf_key,))


def test_process_wave_same_wave_injection_coexists_with_sub_wave_resolution(monkeypatch):
    sf_a = sp.Symbol("c^{1,(0,1)}_{(0,(1,2)),(0,2)}")
    sf_b = sp.Symbol("c^{1,(0,1)}_{(0,(0,2)),(0,1)}")
    key_a = ("sf_a_key",)
    key_b = ("sf_b_key",)
    u0 = sp.Symbol("u_{0,0}")
    alpha0 = sp.Symbol("α_{0}")
    row_calls = []

    def fake_compute_minor(char_tuples, row, vanished_sf_keys, all_vars):
        row_calls.append((row, tuple(vanished_sf_keys)))
        return sp.Symbol(f"minor_{row[1]}"), [u0]

    def fake_diff(expr, sf):
        if expr == sp.Symbol("minor_0") and sf == sf_a:
            return sp.Symbol("diff_row0_a")
        if expr == sp.Symbol("minor_1") and sf == sf_b:
            return sp.Symbol("diff_row1_b")
        return sp.Integer(0)

    def fake_sparse(expr, u_gens):
        if expr == sp.Symbol("diff_row0_a"):
            return {(1,): sf_b}
        if expr == sp.Symbol("diff_row1_b"):
            return {(1,): sp.Integer(1)}
        return {}

    def fake_exact(char_tuples, extra_row, monomial_spec, vanished_sf_keys):
        if extra_row == (0, 1, 1):
            return sf_b**2
        if extra_row == (0, 0, 1):
            return sf_a**2 * sf_b + alpha0 * sf_a**2
        raise AssertionError(f"Unexpected exact request for row {extra_row}")

    monkeypatch.setattr(ie, "_compute_minor_with_sf_injection", fake_compute_minor)
    monkeypatch.setattr(ie.sp, "diff", fake_diff)
    monkeypatch.setattr(ie, "expr_to_sparse_u_poly", fake_sparse)
    monkeypatch.setattr(ie, "_compute_exact_coefficient_with_sf_injection", fake_exact)

    result = ie._process_wave(
        char_tuples=[(1, 2)],
        root_index=0,
        target_sfs=[(sf_a, key_a), (sf_b, key_b)],
        vanished_sf_keys=[],
        extra_rows=[(0, 0, 1), (0, 1, 1)],
        all_vars=[("vertex", 0, 0)],
        use_sparse=True,
        max_sub_waves=3,
        live_callback=None,
        progress_callback=None,
    )

    assert row_calls == [
        ((0, 0, 1), ()),
        ((0, 1, 1), ()),
    ]
    assert set(result.vanished_sfs) == {sf_a.name, sf_b.name}
    assert any(ev.sf_name == sf_b.name and ev.sub_wave == 0 for ev in result.evidence)
    assert any(ev.sf_name == sf_a.name and ev.sub_wave == 1 for ev in result.evidence)


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
