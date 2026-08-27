"""Structured-data tools: `query_data`, `compute_metrics`, `check_schema`."""

from __future__ import annotations

from pydantic import ValidationError

from app.data.tables import QuerySpec, TableStore
from app.tools.base import Tool, ToolError


class QueryDataTool(Tool):
    name = "query_data"
    description = (
        "Run a safe, structured query against an ingested table (from CSV/JSON). "
        "Supports column projection, filters (eq/ne/gt/gte/lt/lte/contains/isnull/"
        "notnull), group-by, and aggregation (count/sum/mean/min/max/median/nunique)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "op": {
                            "type": "string",
                            "enum": [
                                "eq",
                                "ne",
                                "gt",
                                "gte",
                                "lt",
                                "lte",
                                "contains",
                                "isnull",
                                "notnull",
                            ],
                        },
                        "value": {},
                    },
                    "required": ["column", "op"],
                },
            },
            "group_by": {"type": "array", "items": {"type": "string"}},
            "aggregate": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": ["count", "sum", "mean", "min", "max", "median", "nunique"],
                },
            },
            "sort_by": {"type": "string"},
            "descending": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50},
        },
        "required": ["table"],
        "additionalProperties": False,
    }

    def __init__(self, store: TableStore) -> None:
        self._store = store

    def run(self, arguments: dict) -> dict:
        try:
            spec = QuerySpec.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"invalid query spec: {exc}") from exc
        if not self._store.exists(spec.table):
            raise ToolError(
                f"no table {spec.table!r}; ingested tables: {self._store.list_tables()}"
            )
        try:
            return self._store.query(spec).model_dump()
        except KeyError as exc:
            raise ToolError(str(exc)) from exc


class ComputeMetricsTool(Tool):
    name = "compute_metrics"
    description = (
        "Compute one or more operational metrics over a table in a single call, "
        "optionally grouped. Each metric is {name, column, agg} with an optional "
        "shared filter set. Returns a list of {metric, value} findings."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "group_by": {"type": "array", "items": {"type": "string"}},
            "filters": QueryDataTool.input_schema["properties"]["filters"],
            "metrics": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "column": {"type": "string"},
                        "agg": {
                            "type": "string",
                            "enum": ["count", "sum", "mean", "min", "max", "median", "nunique"],
                        },
                    },
                    "required": ["name", "column", "agg"],
                },
            },
        },
        "required": ["table", "metrics"],
        "additionalProperties": False,
    }

    def __init__(self, store: TableStore) -> None:
        self._store = store

    def run(self, arguments: dict) -> dict:
        table = arguments["table"]
        metrics = arguments["metrics"]
        group_by = arguments.get("group_by") or None
        filters = arguments.get("filters") or []
        if not self._store.exists(table):
            raise ToolError(f"no table {table!r}; ingested: {self._store.list_tables()}")

        findings = []
        for metric in metrics:
            spec = QuerySpec.model_validate(
                {
                    "table": table,
                    "filters": filters,
                    "group_by": group_by,
                    "aggregate": {metric["column"]: metric["agg"]},
                }
            )
            try:
                result = self._store.query(spec)
            except KeyError as exc:
                raise ToolError(str(exc)) from exc
            findings.append(
                {
                    "metric": metric["name"],
                    "agg": metric["agg"],
                    "column": metric["column"],
                    "rows": result.rows,
                }
            )
        return {"table": table, "group_by": group_by, "findings": findings}


class CheckSchemaTool(Tool):
    name = "check_schema"
    description = (
        "Validate that a table has the columns an analysis expects, flag columns "
        "that are entirely/mostly null, and check coarse types "
        "(numeric | datetime | string)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "required_columns": {"type": "array", "items": {"type": "string"}},
            "expected_types": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": ["numeric", "datetime", "string"],
                },
            },
        },
        "required": ["table", "required_columns"],
        "additionalProperties": False,
    }

    def __init__(self, store: TableStore) -> None:
        self._store = store

    def run(self, arguments: dict) -> dict:
        table = arguments["table"]
        if not self._store.exists(table):
            raise ToolError(f"no table {table!r}; ingested: {self._store.list_tables()}")
        description = self._store.describe(table)
        cols = description["columns"]

        missing = [c for c in arguments["required_columns"] if c not in cols]
        mostly_null = [
            c
            for c, info in cols.items()
            if info["non_null"] == 0 or info["null"] / max(1, info["null"] + info["non_null"]) > 0.5
        ]
        type_mismatches = []
        for col, expected in (arguments.get("expected_types") or {}).items():
            if col not in cols:
                continue
            actual = _coarse_type(cols[col]["dtype"])
            if actual != expected:
                type_mismatches.append({"column": col, "expected": expected, "actual": actual})

        return {
            "table": table,
            "row_count": description["rows"],
            "missing_columns": missing,
            "mostly_null_columns": mostly_null,
            "type_mismatches": type_mismatches,
            "ok": not (missing or type_mismatches),
        }


def _coarse_type(dtype: str) -> str:
    if "datetime" in dtype:
        return "datetime"
    if any(t in dtype for t in ("int", "float")):
        return "numeric"
    return "string"
