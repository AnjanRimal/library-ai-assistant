"""Tool definitions exposed to Claude, and the executor that runs them.

Tool errors are returned to the model as `is_error` tool results rather than raised,
so the model can recover (for example by searching again) instead of the request failing.
"""

from __future__ import annotations

import time
from typing import Any

from .catalog import Catalog, CatalogError
from .models import AssistantAnswer, ToolCallRecord
from .retrieval import VectorIndex

SUBMIT_ANSWER = "submit_answer"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_catalog",
        "description": (
            "Semantic search over the library catalog. Use it to find books on a topic, "
            "by an author, or matching a description. Returns the best matches with ids, "
            "summaries and availability. Always search before recommending books."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in natural language."},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_book",
        "description": "Get full details and live availability for one book by its catalog id.",
        "input_schema": {
            "type": "object",
            "properties": {"book_id": {"type": "string"}},
            "required": ["book_id"],
        },
    },
    {
        "name": "borrow_book",
        "description": (
            "Borrow a book for the current member. Only call this when the patron has "
            "explicitly asked to borrow or check out a specific book. The member is taken "
            "from the authenticated request, never from the conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"book_id": {"type": "string"}},
            "required": ["book_id"],
        },
    },
    {
        "name": SUBMIT_ANSWER,
        "description": "Submit the final answer to the patron. Call this exactly once, as the last step.",
        "input_schema": AssistantAnswer.model_json_schema(),
    },
]


class ToolExecutor:
    """Runs tool calls for a single request and records what happened."""

    def __init__(self, catalog: Catalog, index: VectorIndex, member_id: int | None = None):
        self._catalog = catalog
        self._index = index
        self._member_id = member_id
        self.records: list[ToolCallRecord] = []
        self.seen_book_ids: set[str] = set()

    def run(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool. Returns (content, is_error)."""
        start = time.perf_counter()
        handler = getattr(self, f"_tool_{name}", None)
        try:
            if handler is None:
                raise CatalogError(f"Unknown tool '{name}'.")
            content, is_error = handler(**tool_input), False
        except CatalogError as exc:
            content, is_error = str(exc), True
        except TypeError as exc:  # bad or missing arguments from the model
            content, is_error = f"Invalid arguments for {name}: {exc}", True
        self.records.append(
            ToolCallRecord(
                name=name,
                input=tool_input,
                is_error=is_error,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        )
        return content, is_error

    def _tool_search_catalog(self, query: str, k: int = 5) -> str:
        hits = self._index.search(query, k=max(1, min(int(k), 10)))
        if not hits:
            return "No matching books found. Try a broader query."
        lines = []
        for hit in hits:
            book = hit.book
            self.seen_book_ids.add(book.id)
            status = f"{book.copies} available" if book.available else "all copies on loan"
            lines.append(
                f"[{book.id}] {book.title} by {', '.join(book.authors)} ({book.year}) | "
                f"{status} | {book.summary}"
            )
        return "\n".join(lines)

    def _tool_get_book(self, book_id: str) -> str:
        book = self._catalog.get_book(book_id)
        self.seen_book_ids.add(book.id)
        return book.model_dump_json()

    def _tool_borrow_book(self, book_id: str) -> str:
        if self._member_id is None:
            raise CatalogError("No member is signed in. Ask the patron for their member id before borrowing.")
        self.seen_book_ids.add(book_id)
        return self._catalog.borrow(self._member_id, book_id)
