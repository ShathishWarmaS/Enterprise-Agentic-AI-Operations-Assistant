"""Clean and validate tabular data before it is stored or analysed.

The goal is not to make data "perfect" but to make it *predictable*: stable
column names, real dtypes where possible, and an explicit report of everything
that looked wrong so downstream agents can cite data-quality caveats.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.schemas.documents import CleaningIssue, CleaningReport

_SNAKE_RE = re.compile(r"[^0-9a-zA-Z]+")


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningIssue]]:
    issues: list[CleaningIssue] = []
    new_names: list[str] = []
    seen: dict[str, int] = {}
    for col in frame.columns:
        base = _SNAKE_RE.sub("_", str(col).strip()).strip("_").lower() or "column"
        name = base
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
            issues.append(
                CleaningIssue(
                    severity="warning",
                    location=f"column {col!r}",
                    message=f"duplicate column name, renamed to {name!r}",
                )
            )
        else:
            seen[base] = 0
        new_names.append(name)
    frame = frame.copy()
    frame.columns = new_names
    return frame, issues


def _coerce_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    cleaned = series.str.replace(r"[,$%\s]", "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    non_null = series.notna()
    # Only accept the coercion if it does not destroy real values.
    if non_null.sum() and converted.notna().sum() / non_null.sum() >= 0.7:
        coerced = int((converted.isna() & non_null).sum())
        return converted, coerced
    return series, 0


def _coerce_datetime(series: pd.Series) -> pd.Series | None:
    sample = series.dropna().head(20)
    if sample.empty:
        return None
    parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=False)
    if parsed.notna().sum() / series.notna().sum() >= 0.9:
        return parsed
    return None


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(frame))
    frame, name_issues = normalize_columns(frame)
    report.issues.extend(name_issues)

    # Trim whitespace on every string cell.
    for col in frame.columns:
        frame[col] = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        frame[col] = frame[col].replace({"": np.nan, "null": np.nan, "NA": np.nan, "N/A": np.nan})

    # Drop rows and columns that are entirely empty.
    empty_cols = [c for c in frame.columns if frame[c].isna().all()]
    for col in empty_cols:
        report.issues.append(
            CleaningIssue(
                severity="warning", location=f"column {col}", message="column is entirely empty"
            )
        )
    frame = frame.drop(columns=empty_cols)
    before = len(frame)
    frame = frame.dropna(how="all").reset_index(drop=True)
    report.dropped_rows = before - len(frame)

    # Type coercion pass.
    for col in list(frame.columns):
        as_dt = _coerce_datetime(frame[col])
        if as_dt is not None:
            frame[col] = as_dt
            continue
        coerced_series, coerced_count = _coerce_numeric(frame[col].astype("string"))
        if coerced_count or pd.api.types.is_numeric_dtype(coerced_series):
            frame[col] = coerced_series
            report.coerced_cells += coerced_count
            if coerced_count:
                report.issues.append(
                    CleaningIssue(
                        severity="warning",
                        location=f"column {col}",
                        message=f"{coerced_count} value(s) were not numeric and became null",
                    )
                )

    # Duplicate rows.
    dup_mask = frame.duplicated(keep="first")
    if dup_mask.any():
        report.issues.append(
            CleaningIssue(
                severity="warning",
                location=f"{int(dup_mask.sum())} rows",
                message="exact duplicate rows detected (kept first occurrence)",
            )
        )
        frame = frame[~dup_mask].reset_index(drop=True)

    _flag_quality(frame, report)

    report.rows_out = len(frame)
    return frame, report


def _flag_quality(frame: pd.DataFrame, report: CleaningReport) -> None:
    for col in frame.columns:
        null_rate = frame[col].isna().mean()
        if 0 < null_rate < 1:
            severity = "error" if null_rate > 0.5 else "info"
            report.issues.append(
                CleaningIssue(
                    severity=severity,
                    location=f"column {col}",
                    message=f"{null_rate:.0%} of values are missing",
                )
            )
        if pd.api.types.is_numeric_dtype(frame[col]):
            values = frame[col].dropna()
            if len(values) >= 8 and values.std(ddof=0) > 0:
                z = (values - values.mean()).abs() / values.std(ddof=0)
                outliers = values[z > 4]
                for idx, val in outliers.items():
                    report.issues.append(
                        CleaningIssue(
                            severity="warning",
                            location=f"row {idx}, column {col}",
                            message=f"value {val} is >4 std devs from the mean (possible bad data)",
                        )
                    )
