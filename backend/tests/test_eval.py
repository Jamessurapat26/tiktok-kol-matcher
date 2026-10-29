"""Behavior tests for the offline evaluation harness (no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_EVAL = _REPO_ROOT / "eval" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("eval_run_eval", _RUN_EVAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ndcg_perfect_order_is_one_and_reversed_is_lower():
    run_eval = _load_run_eval()
    labels = {"alpha": 2, "beta": 1, "gamma": 0}

    perfect = run_eval.ndcg_at_10(["alpha", "beta", "gamma"], labels)
    reversed_order = run_eval.ndcg_at_10(["gamma", "beta", "alpha"], labels)

    assert perfect == pytest.approx(1.0)
    assert reversed_order < perfect


def test_example_eval_prints_a_reasonable_mean(capsys):
    run_eval = _load_run_eval()

    code = run_eval.main([])

    out = capsys.readouterr().out
    assert code == 0
    assert "brand" in out
    assert "n_pool" in out
    assert "n_labeled" in out
    assert "NDCG@10" in out
    assert "example-brand" in out

    mean_line = next(line for line in out.splitlines() if line.startswith("mean"))
    mean = float(mean_line.split()[-1])
    assert 0.0 < mean <= 1.0


def test_ablation_all_reports_each_criterion(capsys):
    run_eval = _load_run_eval()

    code = run_eval.main(["--ablate", "all"])

    out = capsys.readouterr().out
    assert code == 0
    for criterion in ("F", "E", "T", "A"):
        assert f"[{criterion}]" in out
    assert out.splitlines()[-1].startswith("candidates for deletion:")
