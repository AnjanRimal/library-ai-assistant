"""Keep the eval harness itself honest: the offline suite must pass, and the
grader must catch the failures it claims to catch."""

from types import SimpleNamespace

from evals.run import CASES, grade_agent_case, retrieval_passed, run_retrieval
from library_assistant.embeddings import HashingEmbedder
from library_assistant.models import AssistantAnswer, ToolCallRecord


def test_retrieval_eval_meets_thresholds():
    result = run_retrieval(HashingEmbedder())
    assert retrieval_passed(result), result


def test_every_relevant_id_exists_in_the_catalog():
    from library_assistant.catalog import LocalCatalog

    ids = {b.id for b in LocalCatalog().all_books()}
    for case in CASES["retrieval"] + CASES["agent"]:
        for book_id in case.get("relevant", []) + case.get("must_cite_any", []):
            assert book_id in ids, (case, book_id)


def _response(intent="recommend", tools=(), cited=(), recs=0, flags=()):
    return SimpleNamespace(
        result=AssistantAnswer(
            answer="x",
            intent=intent,
            cited_book_ids=list(cited),
            recommendations=[
                {"book_id": "b01", "title": "t", "reason": "r", "available": True} for _ in range(recs)
            ],
        ),
        tool_calls=[ToolCallRecord(name=n, input={}, is_error=err) for n, err in tools],
        guardrail_flags=list(flags),
    )


def test_grader_passes_a_good_response():
    case = {"expect_intent": "recommend", "expect_tools": ["search_catalog"], "must_cite_any": ["b06"]}
    assert grade_agent_case(case, _response(tools=[("search_catalog", False)], cited=["b06"])) == []


def test_grader_catches_each_failure_type():
    case = {
        "expect_intent": "lookup",
        "expect_tools": ["search_catalog"],
        "forbid_tools": ["borrow_book"],
        "must_cite_any": ["b03"],
        "max_recommendations": 0,
    }
    failures = grade_agent_case(
        case,
        _response(tools=[("borrow_book", False)], recs=2, flags=["ungrounded_citation_removed"]),
    )
    assert len(failures) == 6


def test_grader_allows_a_failed_forbidden_borrow():
    case = {"forbid_successful_tools": ["borrow_book"]}
    assert grade_agent_case(case, _response(tools=[("borrow_book", True)])) == []
    assert grade_agent_case(case, _response(tools=[("borrow_book", False)])) != []
