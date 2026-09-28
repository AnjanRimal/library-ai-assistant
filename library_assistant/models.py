"""Data models shared across the assistant."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Book(BaseModel):
    id: str
    title: str
    authors: list[str]
    isbn: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    summary: str = ""
    copies: int = 0

    @property
    def available(self) -> bool:
        return self.copies > 0

    def to_document(self) -> str:
        """Text that gets embedded for retrieval."""
        return f"{self.title}. By {', '.join(self.authors)}. Topics: {', '.join(self.genres)}. {self.summary}"


class BookRecommendation(BaseModel):
    book_id: str = Field(description="Catalog id of the book, e.g. 'b04'.")
    title: str
    reason: str = Field(description="One sentence on why this book fits the request.")
    available: bool


class AssistantAnswer(BaseModel):
    """The structured output every request ends with."""

    answer: str = Field(description="Reply to the patron in plain language.")
    intent: Literal["recommend", "lookup", "availability", "borrow", "out_of_scope"]
    recommendations: list[BookRecommendation] = Field(default_factory=list)
    cited_book_ids: list[str] = Field(
        default_factory=list,
        description="Ids of every catalog book the answer relies on.",
    )


class ToolCallRecord(BaseModel):
    name: str
    input: dict
    is_error: bool = False
    latency_ms: float = 0.0


class AskRequest(BaseModel):
    question: str
    member_id: int | None = Field(default=None, description="Needed only to borrow a book.")


class AskResponse(BaseModel):
    result: AssistantAnswer
    tool_calls: list[ToolCallRecord]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    guardrail_flags: list[str] = Field(default_factory=list)
