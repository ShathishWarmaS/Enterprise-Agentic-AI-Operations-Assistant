"""Data Analysis agent: query ingested tables and surface metrics + anomalies."""

from __future__ import annotations

import re
from typing import cast

from app.agents.base import Agent
from app.data.tables import AggFunc, QuerySpec, TableStore
from app.schemas.agents import DataAnalysisResult, DataFinding, ToolCall

# matched with word boundaries so "summarise" does not read as "sum"
_AGG_BY_KEYWORD = [
    (r"\baverages?\b|\bavg\b|\bmean\b", "mean"),
    (r"\bmax(imum)?\b|\bpeak\b|\bhighest\b", "max"),
    (r"\bmin(imum)?\b|\blowest\b", "min"),
    (r"\btotals?\b|\bsum\b", "sum"),
    (r"\bcounts?\b|\bnumber of\b|\bhow many\b", "count"),
]


class DataAnalysisAgent(Agent):
    name = "data_analysis"

    def __init__(self, settings, llm, table_store: TableStore) -> None:
        super().__init__(settings, llm)
        self._store = table_store

    def analyse(self, request: str) -> tuple[DataAnalysisResult, list[ToolCall]]:
        tables = self._store.list_tables()
        if not tables:
            return DataAnalysisResult(missing_fields=["no tabular data has been ingested"]), []

        table = self._choose_table(request, tables)
        description = self._store.describe(table)
        columns = description["columns"]
        calls: list[ToolCall] = []

        findings: list[DataFinding] = [
            DataFinding(
                metric="row_count",
                value=description["rows"],
                observation=f"table {table!r} has {description['rows']} rows after cleaning",
            )
        ]

        group_by = self._detect_group_by(request, columns)
        agg = self._detect_agg(request)

        # "how many ... per service" style: count rows per group
        if group_by and (
            agg == "count" or re.search(r"how many|number of|per \w+", request.lower())
        ):
            id_col = next(iter(columns))
            call = self._run(
                QuerySpec(table=table, group_by=group_by, aggregate={id_col: "count"}, limit=50)
            )
            calls.append(call)
            if call.ok and call.result:
                for row in call.result["rows"]:
                    label = ", ".join(f"{g}={row.get(g)}" for g in group_by)
                    findings.append(
                        DataFinding(
                            metric="incident_count",
                            value=row.get(id_col),
                            observation=f"{label}: {row.get(id_col)} row(s)",
                        )
                    )

        numeric_targets = self._target_columns(request, columns)
        agg = agg if agg != "count" else "mean"
        for col in numeric_targets:
            spec = QuerySpec(table=table, group_by=group_by or None, aggregate={col: agg}, limit=20)
            call = self._run(spec)
            calls.append(call)
            if call.ok and call.result:
                findings.extend(_findings_from_result(call.result, col, agg, group_by))

        anomalies = _anomalies(columns, description["rows"])
        missing = [c for c, info in columns.items() if 0 < info["null"] and info["non_null"] > 0]

        result = DataAnalysisResult(
            table=table,
            row_count=description["rows"],
            findings=findings,
            anomalies=anomalies,
            missing_fields=missing,
        )
        return result, calls

    # -- helpers -----------------------------------------------------
    def _run(self, spec: QuerySpec) -> ToolCall:
        import time

        started = time.perf_counter()
        try:
            result = self._store.query(spec)
            return ToolCall(
                tool="query_data",
                arguments=spec.model_dump(exclude_none=True),
                ok=True,
                result=result.model_dump(),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except (KeyError, ValueError) as exc:
            return ToolCall(
                tool="query_data",
                arguments=spec.model_dump(exclude_none=True),
                ok=False,
                error=str(exc),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

    def _choose_table(self, request: str, tables: list[str]) -> str:
        words = set(re.findall(r"[a-z0-9]+", request.lower()))
        best, best_score = tables[0], -1
        for name in tables:
            cols = set(re.findall(r"[a-z0-9]+", " ".join(self._store.describe(name)["columns"])))
            score = len(words & cols) + (2 if name in words else 0)
            if score > best_score:
                best, best_score = name, score
        return best

    @staticmethod
    def _detect_agg(request: str) -> AggFunc:
        lower = request.lower()
        for pattern, agg in _AGG_BY_KEYWORD:
            if re.search(pattern, lower):
                return cast(AggFunc, agg)
        return "mean"

    @staticmethod
    def _detect_group_by(request: str, columns: dict) -> list[str]:
        match = re.search(r"\b(?:by|per|across|for each) ([a-z_]+)", request.lower())
        if not match:
            return []
        candidate = match.group(1).strip()
        for col in columns:
            if col == candidate or candidate in col.split("_") or col in candidate:
                return [col]
        return []

    @staticmethod
    def _target_columns(request: str, columns: dict) -> list[str]:
        words = set(re.findall(r"[a-z0-9_]+", request.lower()))
        numeric = [
            c for c, info in columns.items() if any(t in info["dtype"] for t in ("int", "float"))
        ]
        named = [
            c for c in numeric if c in words or any(w in c.split("_") for w in words if len(w) > 3)
        ]
        return named or numeric[:2]  # empty if the table has no numeric columns


def _findings_from_result(
    result: dict, column: str, agg: str, group_by: list[str]
) -> list[DataFinding]:
    out: list[DataFinding] = []
    for row in result.get("rows", [])[:10]:
        value = row.get(column)
        if not isinstance(value, (int, float)):
            continue
        value = round(value, 3)
        label = ", ".join(f"{g}={row.get(g)}" for g in group_by) if group_by else "overall"
        out.append(
            DataFinding(
                metric=f"{agg}({column})",
                value=value,
                observation=f"{agg} of {column} for {label} is {value}",
            )
        )
    return out


def _anomalies(columns: dict, rows: int) -> list[str]:
    notes: list[str] = []
    for col, info in columns.items():
        if rows and info["non_null"] == 0:
            notes.append(f"column {col!r} is entirely empty")
        elif rows and info["null"] / rows > 0.3:
            notes.append(f"column {col!r} is {info['null'] / rows:.0%} null")
        if {"min", "max", "mean"} <= info.keys() and info["mean"]:
            spread = (info["max"] - info["min"]) / (abs(info["mean"]) + 1e-9)
            if spread > 50:
                notes.append(
                    f"column {col!r} ranges {info['min']}–{info['max']} around mean "
                    f"{info['mean']} (possible outliers or mixed units)"
                )
            # a single value dwarfing the median is usually a bad reading / sentinel
            median = info.get("median", info["mean"])
            if info["max"] > 50 * max(abs(median), 1) and info["max"] > 100:
                notes.append(
                    f"column {col!r} has a maximum of {info['max']} versus a median of "
                    f"{median} - likely a sentinel value or instrumentation error"
                )
    return notes
