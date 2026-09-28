"""Command-line chat: `python -m library_assistant "find me a book on terraform"`."""

from __future__ import annotations

import argparse
import json
import sys

from .agent import AgentError, LibraryAgent, make_client
from .catalog import build_catalog
from .config import get_settings
from .embeddings import build_embedder
from .retrieval import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the library assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--member-id", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print the full structured response.")
    args = parser.parse_args()

    settings = get_settings()
    try:
        client = make_client()
    except AgentError as exc:
        sys.exit(str(exc))
    catalog = build_catalog(settings.catalog_backend, settings.library_api_url)
    index = VectorIndex(build_embedder(settings.embedder)).build(catalog.all_books())
    agent = LibraryAgent(client, catalog, index, settings)

    try:
        response = agent.ask(args.question, member_id=args.member_id)
    except (AgentError, ValueError) as exc:
        sys.exit(f"Error: {exc}")
    if args.json:
        print(json.dumps(response.model_dump(), indent=2))
        return
    print(response.result.answer)
    for rec in response.result.recommendations:
        status = "available" if rec.available else "on loan"
        print(f"  - {rec.title} [{rec.book_id}, {status}]: {rec.reason}")
    print(
        f"\n({len(response.tool_calls)} tool calls, {response.input_tokens + response.output_tokens} "
        f"tokens, ${response.cost_usd:.4f}, {response.latency_ms:.0f} ms)"
    )


if __name__ == "__main__":
    main()
