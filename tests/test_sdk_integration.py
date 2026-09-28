"""Run the agent through the real Anthropic SDK against a mocked HTTP transport.

The FakeClient tests cover the agent's logic; this one proves the requests we build
are accepted by the SDK and that real response objects parse the way the agent expects.
"""

import json

import anthropic

try:  # newer SDK releases ship their own fork of httpx
    import httpx2 as httpx
except ImportError:
    import httpx

from library_assistant.agent import LibraryAgent


def _message(content, stop_reason):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


def test_agent_round_trip_through_the_sdk(catalog, index, settings):
    scripted = [
        _message(
            [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "toolu_1", "name": "search_catalog", "input": {"query": "docker"}},
            ],
            "tool_use",
        ),
        _message(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "submit_answer",
                    "input": {
                        "answer": "Docker Deep Dive.",
                        "intent": "recommend",
                        "cited_book_ids": ["b15"],
                    },
                }
            ],
            "tool_use",
        ),
    ]
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=scripted[len(requests) - 1])

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = LibraryAgent(client, catalog, index, settings).ask("Books about Docker?")

    assert response.result.cited_book_ids == ["b15"]
    assert (response.input_tokens, response.output_tokens) == (240, 60)

    first, second = requests
    assert first["tool_choice"] == {"type": "auto"}
    assert {t["name"] for t in first["tools"]} >= {"search_catalog", "submit_answer"}
    # The assistant turn and the tool result were sent back in the API's wire format.
    assistant_turn, tool_turn = second["messages"][1], second["messages"][2]
    assert assistant_turn["content"][1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "search_catalog",
        "input": {"query": "docker"},
    }
    assert tool_turn["content"][0]["tool_use_id"] == "toolu_1"
    assert "[b15]" in tool_turn["content"][0]["content"]
