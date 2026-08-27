"""Run the evaluation harness against the sample dataset and print a report.

    python scripts/run_eval.py [--cases sample_data/eval_cases.json] [--json]

Exits non-zero if the pass rate drops below --min-pass (default 0.6) so it can
gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.data.database import init_db
from app.services.container import get_container
from app.services.evaluation import load_cases, run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--min-pass", type=float, default=0.6)
    args = parser.parse_args()

    init_db()
    container = get_container()
    if container.vector_store.stats()["chunks"] == 0:
        print("vector store is empty - run `python scripts/seed.py` first", file=sys.stderr)
        return 2

    cases = load_cases(args.cases) if args.cases else load_cases()
    summary = run_evaluation(container, cases)

    if args.json:
        print(json.dumps(summary.model_dump(), indent=2))
    else:
        _print_report(summary, container.llm.mode)

    if summary.pass_rate < args.min_pass:
        print(f"\nFAIL: pass rate {summary.pass_rate:.0%} < required {args.min_pass:.0%}")
        return 1
    return 0


def _print_report(summary, llm_mode: str) -> None:
    print(f"\nEvaluation over {summary.cases} cases (llm_mode={llm_mode})\n")
    metrics = [
        ("retrieval relevance (recall@k)", summary.retrieval_relevance),
        ("citation presence", summary.citation_presence),
        ("tool-selection accuracy", summary.tool_selection_accuracy),
        ("structured-output validity", summary.structured_output_validity),
        ("groundedness", summary.groundedness),
        ("missing-info handling", summary.missing_info_handling),
        ("response consistency", summary.response_consistency),
    ]
    for name, value in metrics:
        bar = "█" * int(round(value * 20))
        print(f"  {name:<32} {value:5.2f}  {bar}")
    print(f"\n  {'PASS RATE':<32} {summary.pass_rate:5.2f}\n")
    for r in summary.results:
        mark = "PASS" if r.passed else "fail"
        print(f"  [{mark}] {r.id:<22} {r.detail}")


if __name__ == "__main__":
    raise SystemExit(main())
