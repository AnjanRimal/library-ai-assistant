"""Eval runner.

    python -m evals.run retrieval        # offline, deterministic; runs in CI
    python -m evals.run agent            # live; needs ANTHROPIC_API_KEY
    python -m evals.run all

Each run prints a results table, writes evals/results/<suite>.json, and exits non-zero
when the score falls below the threshold, so a regression fails the build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from library_assistant.catalog import LocalCatalog
from library_assistant.config import get_settings
from library_assistant.embeddings import Embedder, build_embedder
from library_assistant.retrieval import VectorIndex

CASES = json.loads((Path(__file__).parent / "cases.json").read_text())
RESULTS_DIR = Path(__file__).parent / "results"

RETRIEVAL_THRESHOLDS = {"recall_at_3": 0.85, "mrr": 0.75}
AGENT_PASS_RATE_THRESHOLD = 0.85


# ---------------------------------------------------------------- retrieval


def run_retrieval(embedder: Embedder, k: int = 3) -> dict[str, Any]:
    index = VectorIndex(embedder).build(LocalCatalog().all_books())
    rows = []
    for case in CASES["retrieval"]:
        ranked = [h.book.id for h in index.search(case["query"], k=10, min_score=0.0)]
        relevant = set(case["relevant"])
        first_hit = next((i + 1 for i, b in enumerate(ranked) if b in relevant), None)
        rows.append(
            {
                "query": case["query"],
                "category": case.get("category", "keyword"),
                "top3": ranked[:k],
                "hit_at_3": bool(relevant & set(ranked[:k])),
                "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
            }
        )
    n = len(rows)
    by_category = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "cases": len(subset),
            "recall_at_3": round(sum(r["hit_at_3"] for r in subset) / len(subset), 3),
            "mrr": round(sum(r["reciprocal_rank"] for r in subset) / len(subset), 3),
        }
    return {
        "suite": "retrieval",
        "recall_at_3": round(sum(r["hit_at_3"] for r in rows) / n, 3),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 3),
        "by_category": by_category,
        "cases": rows,
    }


def retrieval_passed(result: dict[str, Any]) -> bool:
    return all(result[m] >= t for m, t in RETRIEVAL_THRESHOLDS.items())


def _print_retrieval(result: dict[str, Any]) -> None:
    for row in result["cases"]:
        mark = "PASS" if row["hit_at_3"] else "MISS"
        print(f"  {mark}  {row['category']:<10} {row['query'][:50]:<50} -> {', '.join(row['top3'])}")
    for category, stats in result["by_category"].items():
        print(
            f"\n  {category:<10} ({stats['cases']} cases): recall@3 = {stats['recall_at_3']:.3f}, MRR = {stats['mrr']:.3f}",
            end="",
        )
    print("\n")
    print(f"  recall@3 = {result['recall_at_3']:.3f}  (threshold {RETRIEVAL_THRESHOLDS['recall_at_3']})")
    print(f"  MRR      = {result['mrr']:.3f}  (threshold {RETRIEVAL_THRESHOLDS['mrr']})")


# -------------------------------------------------------------------- agent


def grade_agent_case(case: dict[str, Any], response: Any) -> list[str]:
    """Return a list of failure reasons (empty means the case passed)."""
    failures = []
    result = response.result
    used = [c.name for c in response.tool_calls]
    succeeded = [c.name for c in response.tool_calls if not c.is_error]

    if "expect_intent" in case and result.intent != case["expect_intent"]:
        failures.append(f"intent was '{result.intent}', expected '{case['expect_intent']}'")
    if "expect_intent_in" in case and result.intent not in case["expect_intent_in"]:
        failures.append(f"intent '{result.intent}' not in {case['expect_intent_in']}")
    for name in case.get("expect_tools", []):
        if name not in used:
            failures.append(f"did not call {name}")
    for name in case.get("forbid_tools", []):
        if name in used:
            failures.append(f"called forbidden tool {name}")
    for name in case.get("forbid_successful_tools", []):
        if name in succeeded:
            failures.append(f"{name} succeeded but should not have")
    if "must_cite_any" in case and not set(case["must_cite_any"]) & set(result.cited_book_ids):
        failures.append(f"cited {result.cited_book_ids}, expected one of {case['must_cite_any']}")
    if "max_recommendations" in case and len(result.recommendations) > case["max_recommendations"]:
        failures.append(f"made {len(result.recommendations)} recommendations")
    bad_flags = {"ungrounded_citation_removed", "borrow_claimed_without_tool_call"} & set(
        response.guardrail_flags
    )
    if bad_flags:
        failures.append(f"guardrail flags: {sorted(bad_flags)}")
    return failures


def run_agent() -> dict[str, Any]:
    from library_assistant.agent import AgentError, LibraryAgent, make_client

    settings = get_settings()
    client = make_client()
    rows = []
    for case in CASES["agent"]:
        # Fresh catalog per case, so a borrow in one case can't affect another.
        catalog = LocalCatalog()
        index = VectorIndex(build_embedder(settings.embedder)).build(catalog.all_books())
        agent = LibraryAgent(client, catalog, index, settings)
        start = time.perf_counter()
        try:
            response = agent.ask(case["question"], member_id=case.get("member_id"))
            failures = grade_agent_case(case, response)
            row = {
                "id": case["id"],
                "passed": not failures,
                "failures": failures,
                "intent": response.result.intent,
                "tools": [c.name for c in response.tool_calls],
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
            }
        except (AgentError, ValueError) as exc:
            row = {
                "id": case["id"],
                "passed": False,
                "failures": [f"error: {exc}"],
                "cost_usd": 0.0,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        rows.append(row)
    n = len(rows)
    return {
        "suite": "agent",
        "model": settings.model,
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 3),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        "cases": rows,
    }


def _print_agent(result: dict[str, Any]) -> None:
    for row in result["cases"]:
        mark = "PASS" if row["passed"] else "FAIL"
        detail = "" if row["passed"] else "  <- " + "; ".join(row["failures"])
        print(f"  {mark}  {row['id']:<34}{detail}")
    print(f"\n  pass rate = {result['pass_rate']:.3f}  (threshold {AGENT_PASS_RATE_THRESHOLD})")
    print(
        f"  cost      = ${result['total_cost_usd']:.4f}   mean latency = {result['mean_latency_ms']:.0f} ms"
    )


# ---------------------------------------------------------------------- cli


def _save(result: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{result['suite']}.json").write_text(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("suite", choices=["retrieval", "agent", "all"])
    args = parser.parse_args(argv)
    ok = True

    if args.suite in ("retrieval", "all"):
        print("Retrieval eval")
        result = run_retrieval(build_embedder(get_settings().embedder))
        _print_retrieval(result)
        _save(result)
        ok &= retrieval_passed(result)

    if args.suite in ("agent", "all"):
        print("\nAgent eval")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  skipped: set ANTHROPIC_API_KEY to run live agent evals")
        else:
            result = run_agent()
            _print_agent(result)
            _save(result)
            ok &= result["pass_rate"] >= AGENT_PASS_RATE_THRESHOLD

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
