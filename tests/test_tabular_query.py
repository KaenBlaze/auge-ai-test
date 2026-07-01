"""Tests for deterministic tabular query handling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tabular_query import try_tabular_answer


def test_lowest_turnout_2025(tmp_path):
    csv_path = tmp_path / "historical_results.csv"
    csv_path.write_text(
        "year,region,turnout_pct,education_budget_m,health_budget_m,campaign_spend_m,institutional_trust_pct\n"
        "2025,West,49,104,99,9.7,35\n"
        "2025,Central,71,160,148,19.1,61\n",
        encoding="utf-8",
    )
    answer, fragment = try_tabular_answer(
        "Which region had the lowest turnout in 2025?",
        csv_path,
    )
    assert answer is not None
    assert "West" in answer
    assert "49" in answer
    assert "historical_results.csv" in fragment


def test_region_turnout_lookup(tmp_path):
    csv_path = tmp_path / "historical_results.csv"
    csv_path.write_text(
        "year,region,turnout_pct,education_budget_m,health_budget_m,campaign_spend_m,institutional_trust_pct\n"
        "2025,North,62,134,119,14.2,47\n",
        encoding="utf-8",
    )
    answer, _ = try_tabular_answer("What was North Region turnout in 2025?", csv_path)
    assert answer is not None
    assert "62" in answer
