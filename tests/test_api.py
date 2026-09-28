from fastapi.testclient import TestClient

from library_assistant.api import app, get_agent
from tests.conftest import answer, reply, tool


def _client(make_agent, *responses):
    agent, _ = make_agent(*responses)
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_ask_returns_structured_answer_and_updates_metrics(make_agent):
    client = _client(
        make_agent,
        reply(tool("search_catalog", query="python")),
        reply(answer(answer="Fluent Python.", cited_book_ids=["b17"])),
    )
    body = client.post("/ask", json={"question": "Python books?"}).json()
    assert body["result"]["cited_book_ids"] == ["b17"]
    assert body["tool_calls"][0]["name"] == "search_catalog"

    metrics = client.get("/metrics").json()
    assert metrics["requests"] == 1 and metrics["input_tokens"] == 200


def test_blocked_question_is_a_400(make_agent):
    client = _client(make_agent)
    resp = client.post("/ask", json={"question": "  "})
    assert resp.status_code == 400


def test_agent_failure_is_a_502(make_agent, settings):
    from tests.conftest import text

    client = _client(make_agent, *[reply(text("hmm")) for _ in range(settings.max_agent_steps)])
    assert client.post("/ask", json={"question": "hi"}).status_code == 502


def test_missing_api_key_is_a_clear_503(monkeypatch):
    from library_assistant.api import _build_agent

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _build_agent.cache_clear()
    resp = TestClient(app).post("/ask", json={"question": "hi"})
    assert resp.status_code == 503 and "ANTHROPIC_API_KEY" in resp.json()["detail"]
