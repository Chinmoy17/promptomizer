"""Efficiency tracking: tokens, latency, cost per query."""

import time
from dataclasses import dataclass, field
from functools import wraps

from loguru import logger


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""

    system: str
    question: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class SystemMetrics:
    """Aggregated metrics for a RAG system."""

    system: str
    queries: list[QueryMetrics] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if not self.queries:
            return 0.0
        return sum(q.latency_ms for q in self.queries) / len(self.queries)

    @property
    def p95_latency_ms(self) -> float:
        if not self.queries:
            return 0.0
        latencies = sorted(q.latency_ms for q in self.queries)
        idx = int(0.95 * len(latencies))
        return latencies[min(idx, len(latencies) - 1)]

    @property
    def avg_tokens(self) -> float:
        if not self.queries:
            return 0.0
        return sum(q.total_tokens for q in self.queries) / len(self.queries)

    @property
    def total_cost_usd(self) -> float:
        return sum(q.cost_usd for q in self.queries)

    @property
    def avg_cost_usd(self) -> float:
        if not self.queries:
            return 0.0
        return self.total_cost_usd / len(self.queries)

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "system": self.system,
            "num_queries": len(self.queries),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "avg_tokens": round(self.avg_tokens, 1),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
        }


# Pricing per 1M tokens (input/output) - update as needed
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate cost in USD for a single API call.

    Args:
        model: Model name.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = MODEL_PRICING.get(model, {"input": 5.0, "output": 15.0})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class EfficiencyTracker:
    """Track efficiency metrics across all RAG systems."""

    def __init__(self):
        self._systems: dict[str, SystemMetrics] = {}

    def get_or_create(self, system: str) -> SystemMetrics:
        """Get or create metrics tracker for a system."""
        if system not in self._systems:
            self._systems[system] = SystemMetrics(system=system)
        return self._systems[system]

    def record(
        self,
        system: str,
        question: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "gpt-4o",
    ) -> QueryMetrics:
        """Record metrics for a single query.

        Returns:
            The recorded QueryMetrics.
        """
        total_tokens = prompt_tokens + completion_tokens
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        metrics = QueryMetrics(
            system=system,
            question=question,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
        )

        self.get_or_create(system).queries.append(metrics)
        return metrics

    def summary(self) -> list[dict]:
        """Get summary for all systems."""
        return [sm.summary() for sm in self._systems.values()]

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._systems.clear()


def timed_execution(func):
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Attach timing to result if it's a dict
        if isinstance(result, dict):
            result["_latency_ms"] = elapsed_ms
        return result

    return wrapper
