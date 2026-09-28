"""Cost, latency and quality telemetry.

Every request emits one structured JSON log line, and `Metrics` keeps running totals
that the API exposes at GET /metrics for dashboards and alerting.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("library_assistant")


def cost_usd(input_tokens: int, output_tokens: int, in_price: float, out_price: float) -> float:
    return round(input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price, 6)


@dataclass
class RequestTrace:
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    steps: int
    tool_calls: list[str]
    tool_errors: int
    guardrail_flags: list[str]
    outcome: str  # "ok", "blocked" or "error"

    def log(self) -> None:
        logger.info(json.dumps({"event": "ask", **asdict(self)}))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return round(ordered[index], 2)


@dataclass
class Metrics:
    requests: int = 0
    errors: int = 0
    blocked: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    guardrail_flags: dict[str, int] = field(default_factory=dict)
    _latencies: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, trace: RequestTrace) -> None:
        with self._lock:
            self.requests += 1
            self.errors += trace.outcome == "error"
            self.blocked += trace.outcome == "blocked"
            self.tool_errors += trace.tool_errors
            self.input_tokens += trace.input_tokens
            self.output_tokens += trace.output_tokens
            self.cost_usd = round(self.cost_usd + trace.cost_usd, 6)
            for flag in trace.guardrail_flags:
                self.guardrail_flags[flag] = self.guardrail_flags.get(flag, 0) + 1
            self._latencies = (self._latencies + [trace.latency_ms])[-1000:]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": self.requests,
                "errors": self.errors,
                "blocked": self.blocked,
                "tool_errors": self.tool_errors,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd,
                "latency_ms_p50": _percentile(self._latencies, 50),
                "latency_ms_p95": _percentile(self._latencies, 95),
                "guardrail_flags": dict(self.guardrail_flags),
            }
