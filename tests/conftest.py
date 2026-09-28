"""Shared fixtures, including a scripted fake of the Anthropic client."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import pytest

from library_assistant.agent import LibraryAgent
from library_assistant.catalog import LocalCatalog
from library_assistant.config import Settings
from library_assistant.embeddings import HashingEmbedder
from library_assistant.retrieval import VectorIndex

_ids = itertools.count(1)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool(name: str, **tool_input: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=tool_input)


def answer(**fields: Any) -> SimpleNamespace:
    fields.setdefault("answer", "Here you go.")
    fields.setdefault("intent", "recommend")
    return tool("submit_answer", **fields)


def reply(*blocks: SimpleNamespace, input_tokens: int = 100, output_tokens: int = 20) -> SimpleNamespace:
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeClient:
    """Plays back scripted model responses and records every request."""

    def __init__(self, *responses: Any):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        # Snapshot the messages: the agent keeps appending to the same list.
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def settings() -> Settings:
    return Settings(model="test-model", max_agent_steps=4, max_question_chars=500)


@pytest.fixture
def catalog() -> LocalCatalog:
    return LocalCatalog()


@pytest.fixture
def index(catalog: LocalCatalog) -> VectorIndex:
    return VectorIndex(HashingEmbedder()).build(catalog.all_books())


@pytest.fixture
def make_agent(catalog, index, settings):
    def _make(*responses: Any) -> tuple[LibraryAgent, FakeClient]:
        client = FakeClient(*responses)
        return LibraryAgent(client, catalog, index, settings), client

    return _make
