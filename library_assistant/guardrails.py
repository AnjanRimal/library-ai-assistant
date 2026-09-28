"""Input and output guardrails.

Input checks run before any model call; they block empty or oversized questions and
flag likely prompt-injection attempts. Output checks run on the structured answer;
they drop citations to books the assistant never retrieved, and flag claims of a
borrow that no tool call actually performed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import AssistantAnswer, ToolCallRecord

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"(reveal|print|show) (me )?(your|the) (system prompt|instructions)",
    r"you are now (?!a patron)",
    r"developer mode",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class InputCheck:
    allowed: bool
    reason: str = ""
    flags: list[str] = field(default_factory=list)


def check_input(question: str, max_chars: int) -> InputCheck:
    text = question.strip()
    if not text:
        return InputCheck(False, "The question is empty.")
    if len(text) > max_chars:
        return InputCheck(False, f"The question is longer than {max_chars} characters.")
    flags = ["possible_prompt_injection"] if _INJECTION_RE.search(text) else []
    return InputCheck(True, flags=flags)


def check_output(
    answer: AssistantAnswer,
    seen_book_ids: set[str],
    tool_calls: list[ToolCallRecord],
) -> tuple[AssistantAnswer, list[str]]:
    """Return a cleaned copy of the answer plus any flags raised."""
    flags: list[str] = []

    grounded_recs = [r for r in answer.recommendations if r.book_id in seen_book_ids]
    grounded_cites = [b for b in answer.cited_book_ids if b in seen_book_ids]
    if len(grounded_recs) < len(answer.recommendations) or len(grounded_cites) < len(answer.cited_book_ids):
        flags.append("ungrounded_citation_removed")

    borrowed = any(c.name == "borrow_book" and not c.is_error for c in tool_calls)
    if (
        answer.intent == "borrow"
        and not borrowed
        and re.search(
            r"\b(you('ve| have)? borrowed|is now checked out|successfully borrowed)\b",
            answer.answer,
            re.IGNORECASE,
        )
    ):
        flags.append("borrow_claimed_without_tool_call")

    cleaned = answer.model_copy(update={"recommendations": grounded_recs, "cited_book_ids": grounded_cites})
    return cleaned, flags
