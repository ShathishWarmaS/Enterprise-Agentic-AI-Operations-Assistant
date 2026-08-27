"""Store cleaned tabular sources and expose a *safe* structured-query surface.

The Data Analysis agent and the `query_data` MCP tool never run arbitrary SQL or
`df.query` strings. They call `TableStore.query()` with a small, validated spec
(column names, a fixed set of operators, optional group-by + aggregation). This
keeps a hostile or hallucinated query from doing anything worse than returning
an empty result.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

_NAME_RE = re.compile(r"[^0-9a-z_]+")

FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "isnull", "notnull"]
AggFunc = Literal["count", "sum", "mean", "min", "max", "median", "nunique"]


class Filter(BaseModel):
    column: str
    op: FilterOp
    value: Any = None


class QuerySpec(BaseModel):
    table: str
    columns: list[str] | None = None
    filters: list[Filter] = Field(default_factory=list)
    group_by: list[str] | None = None
    aggregate: dict[str, AggFunc] | None = None
    sort_by: str | None = None
    descending: bool = True
    limit: int = Field(default=50, ge=1, le=1000)

    @field_validator("table")
    @classmethod
    def _safe_table(cls, v: str) -> str:
        norm = _NAME_RE.sub("", v.strip().lower())
        if not norm:
            raise ValueError("invalid table name")
        return norm


class QueryResult(BaseModel):
    table: str
    row_count: int
    columns: list[str]
    rows: list[dict]
    truncated: bool


class TableStore:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory) / "tables"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._dir / f"{_NAME_RE.sub('', name.lower())}.pkl"

    def register(self, name: str, frame: pd.DataFrame) -> str:
        safe = _NAME_RE.sub("", name.lower()) or "table"
        frame.to_pickle(self._path(safe))
        return safe

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def list_tables(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.pkl"))

    def load(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            raise KeyError(f"no table named {name!r}; known: {self.list_tables()}")
        return pd.read_pickle(path)

    def describe(self, name: str) -> dict:
        frame = self.load(name)
        cols = {}
        for col in frame.columns:
            series = frame[col]
            info: dict[str, Any] = {
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "null": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(series) and series.notna().any():
                info["min"] = float(series.min())
                info["max"] = float(series.max())
                info["mean"] = round(float(series.mean()), 4)
                info["median"] = round(float(series.median()), 4)
            cols[col] = info
        return {"table": name, "rows": len(frame), "columns": cols}

    def query(self, spec: QuerySpec) -> QueryResult:
        frame = self.load(spec.table)
        known = set(frame.columns)

        referenced = set(spec.columns or []) | {f.column for f in spec.filters}
        referenced |= set(spec.group_by or []) | set((spec.aggregate or {}).keys())
        if spec.sort_by:
            referenced.add(spec.sort_by)
        unknown = referenced - known
        if unknown:
            raise KeyError(f"unknown column(s) {sorted(unknown)}; available: {sorted(known)}")

        for f in spec.filters:
            frame = _apply_filter(frame, f)

        if spec.group_by and spec.aggregate:
            frame = frame.groupby(spec.group_by, dropna=False).agg(spec.aggregate).reset_index()
        elif spec.aggregate:
            agg_row = {c: _aggregate(frame[c], fn) for c, fn in spec.aggregate.items()}
            frame = pd.DataFrame([agg_row])
        elif spec.columns:
            frame = frame[spec.columns]

        if spec.sort_by and spec.sort_by in frame.columns:
            frame = frame.sort_values(spec.sort_by, ascending=not spec.descending)

        total = len(frame)
        truncated = total > spec.limit
        frame = frame.head(spec.limit)

        return QueryResult(
            table=spec.table,
            row_count=total,
            columns=list(map(str, frame.columns)),
            rows=_records(frame),
            truncated=truncated,
        )


def _apply_filter(frame: pd.DataFrame, f: Filter) -> pd.DataFrame:
    col = frame[f.column]
    if f.op == "isnull":
        return frame[col.isna()]
    if f.op == "notnull":
        return frame[col.notna()]
    if f.op == "contains":
        return frame[col.astype(str).str.contains(str(f.value), case=False, na=False)]

    target: Any = f.value
    if pd.api.types.is_numeric_dtype(col):
        target = pd.to_numeric(f.value, errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(col):
        target = pd.to_datetime(f.value, errors="coerce")

    ops = {
        "eq": col == target,
        "ne": col != target,
        "gt": col > target,
        "gte": col >= target,
        "lt": col < target,
        "lte": col <= target,
    }
    return frame[ops[f.op]]


def _aggregate(series: pd.Series, fn: AggFunc) -> Any:
    result = {
        "count": series.count,
        "sum": series.sum,
        "mean": series.mean,
        "min": series.min,
        "max": series.max,
        "median": series.median,
        "nunique": series.nunique,
    }[fn]()
    if hasattr(result, "item"):
        result = result.item()
    if isinstance(result, float):
        return round(result, 4)
    return result


def _records(frame: pd.DataFrame) -> list[dict]:
    safe = frame.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].astype(str)
    return safe.where(pd.notna(safe), None).to_dict(orient="records")
