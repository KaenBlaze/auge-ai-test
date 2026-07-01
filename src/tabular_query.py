"""Deterministic answers for simple CSV aggregation questions."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

YEAR_RE = re.compile(r"\b(20\d{2})\b")
REGION_ALIASES = {
    "north": "North",
    "south": "South",
    "central": "Central",
    "west": "West",
}

METRIC_ALIASES = {
    "turnout": "turnout_pct",
    "turnout_pct": "turnout_pct",
    "education budget": "education_budget_m",
    "education": "education_budget_m",
    "health budget": "health_budget_m",
    "health": "health_budget_m",
    "campaign spend": "campaign_spend_m",
    "campaign spending": "campaign_spend_m",
    "institutional trust": "institutional_trust_pct",
    "trust": "institutional_trust_pct",
}


def try_tabular_answer(question: str, csv_path: Path) -> tuple[str, str] | None:
    """Return (answer, fragment) for a supported tabular question, else None."""
    path = Path(csv_path)
    if not path.exists():
        return None

    question_lower = question.lower()
    if "historical_results" not in question_lower and not _looks_tabular(question_lower):
        return None

    df = pd.read_csv(path)
    years = [int(match) for match in YEAR_RE.findall(question)]
    year = years[0] if years else None
    region = _extract_region(question_lower)
    metric = _extract_metric(question_lower)

    if year is not None:
        df = df[df["year"] == year]
    if df.empty:
        return None

    if _asks_lowest(question_lower) and metric:
        row = df.loc[df[metric].idxmin()]
        return _format_metric_answer(row, metric, "lowest", year)

    if _asks_highest(question_lower) and metric:
        row = df.loc[df[metric].idxmax()]
        return _format_metric_answer(row, metric, "highest", year)

    if region and metric and year:
        match = df[df["region"].str.lower() == region.lower()]
        if match.empty:
            return None
        row = match.iloc[0]
        value = row[metric]
        fragment = _row_fragment(row, path.name)
        if metric == "turnout_pct":
            answer = f"{region} Region turnout in {int(row['year'])} was {int(value)}%. [source: {path.name}]"
        elif metric == "education_budget_m":
            answer = f"{region} Region education budget in {int(row['year'])} was {int(value)}. [source: {path.name}]"
        elif metric == "institutional_trust_pct":
            answer = f"{region} Region institutional trust in {int(row['year'])} was {int(value)}. [source: {path.name}]"
        else:
            label = metric.replace("_", " ")
            answer = f"{region} Region {label} in {int(row['year'])} was {value}. [source: {path.name}]"
        return answer, fragment

    if region and year and "turnout" in question_lower:
        match = df[df["region"].str.lower() == region.lower()]
        if match.empty:
            return None
        row = match.iloc[0]
        return (
            f"{region} Region turnout in {int(row['year'])} was {int(row['turnout_pct'])}%. "
            f"[source: {path.name}]",
            _row_fragment(row, path.name),
        )

    return None


def _looks_tabular(question_lower: str) -> bool:
    keywords = (
        "turnout",
        "budget",
        "campaign spend",
        "institutional trust",
        "lowest",
        "highest",
        "region",
    )
    return any(keyword in question_lower for keyword in keywords)


def _extract_region(question_lower: str) -> str | None:
    for alias, canonical in REGION_ALIASES.items():
        if alias in question_lower:
            return canonical
    return None


def _extract_metric(question_lower: str) -> str | None:
    for alias, column in sorted(METRIC_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in question_lower:
            return column
    if "turnout" in question_lower:
        return "turnout_pct"
    return None


def _asks_lowest(question_lower: str) -> bool:
    return any(word in question_lower for word in ("lowest", "minimum", "smallest", "worst"))


def _asks_highest(question_lower: str) -> bool:
    return any(word in question_lower for word in ("highest", "maximum", "largest", "best"))


def _format_metric_answer(row: pd.Series, metric: str, extreme: str, year: int | None) -> tuple[str, str]:
    region = row["region"]
    value = row[metric]
    label = metric.replace("_", " ")
    year_text = f" in {int(row['year'])}"
    if metric == "turnout_pct":
        answer = f"{region} Region had the {extreme} turnout{year_text} at {int(value)}%."
    elif metric == "education_budget_m":
        answer = f"{region} Region had the {extreme} education budget{year_text} at {int(value)}."
    elif metric == "health_budget_m":
        answer = f"{region} Region had the {extreme} health budget{year_text} at {int(value)}."
    elif metric == "campaign_spend_m":
        answer = f"{region} Region had the {extreme} campaign spend{year_text} at {value}."
    elif metric == "institutional_trust_pct":
        answer = f"{region} Region had the {extreme} institutional trust{year_text} at {int(value)}."
    else:
        answer = f"{region} Region had the {extreme} {label}{year_text} at {value}."
    answer += f" [source: historical_results.csv]"
    return answer, _row_fragment(row, "historical_results.csv")


def _row_fragment(row: pd.Series, filename: str) -> str:
    parts = [f"{column}={row[column]}" for column in row.index]
    return f"{filename} row: " + ", ".join(parts)
