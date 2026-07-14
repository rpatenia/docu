"""
observability.py

Section 13 of DocuMind: lightweight latency and throughput
instrumentation for the retrieval and generation pipeline stages. These
are self-hosted open-source models — there's no per-request API cost to
track, but latency and tokens/second are the real engineering signals
for a deployment decision. Section 6's quality metrics only answer half
of "which model would you deploy"; this answers the other half.

Deliberately not a full observability stack (no OpenTelemetry, no
Prometheus/Grafana) — that would be over-engineering at this project's
scale. A dataclass and a context manager cover what's actually needed.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from generation import CitedAnswer, Generator
from retrieval import HybridRetriever

logger = logging.getLogger("documind.observability")


@dataclass
class StageTiming:
    """Wall-clock duration of one named pipeline stage, in seconds."""

    stage: str
    duration_seconds: float


@dataclass
class PipelineMetrics:
    """All timing and token data collected for a single end-to-end query."""

    query: str
    model_key: str
    stage_timings: list[StageTiming] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_seconds(self) -> float:
        return sum(t.duration_seconds for t in self.stage_timings)

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second of GENERATION time only — retrieval
        time doesn't produce tokens and would understate throughput if
        included in the denominator.
        """
        generation_time = next(
            (t.duration_seconds for t in self.stage_timings if t.stage == "generation"),
            0.0,
        )
        if generation_time <= 0:
            return 0.0
        return self.output_tokens / generation_time


@contextmanager
def timed_stage(metrics: PipelineMetrics, stage: str) -> Iterator[None]:
    """Time a block of code and record it under `stage` on `metrics`.

    Usage:
        with timed_stage(metrics, "retrieval"):
            scored_chunks = retriever.retrieve(query)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        metrics.stage_timings.append(StageTiming(stage=stage, duration_seconds=duration))
        logger.info("Stage '%s' took %.3fs", stage, duration)


def run_instrumented_query(
    query: str, retriever: HybridRetriever, generator: Generator
) -> tuple[CitedAnswer, PipelineMetrics]:
    """Run retrieval + generation for one query, instrumented with
    per-stage timing and token throughput.

    A convenience wrapper for simple call sites (a single-query CLI, an
    app.py caption). Section 6's evaluate_model uses timed_stage
    directly instead of this, since it needs to skip generation
    entirely when retrieval returns nothing — logic this wrapper
    doesn't have.
    """
    metrics = PipelineMetrics(query=query, model_key=generator.model_key)

    with timed_stage(metrics, "retrieval"):
        scored_chunks = retriever.retrieve(query)

    with timed_stage(metrics, "generation"):
        cited_answer = generator.generate(query, scored_chunks)

    metrics.input_tokens = cited_answer.input_token_count
    metrics.output_tokens = cited_answer.output_token_count

    logger.info(
        "query=%r model=%s retrieval=%.3fs generation=%.3fs tokens/sec=%.1f",
        query,
        generator.model_key,
        next((t.duration_seconds for t in metrics.stage_timings if t.stage == "retrieval"), 0.0),
        next((t.duration_seconds for t in metrics.stage_timings if t.stage == "generation"), 0.0),
        metrics.tokens_per_second,
    )

    return cited_answer, metrics
