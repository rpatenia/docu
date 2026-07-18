"""
evaluation.py

Section 6 of DocuMind: evaluates end-to-end answer quality using three
complementary metric families — RAGAS (faithfulness/relevancy/context
quality, RAG-specific), ROUGE-L (lexical overlap with a reference
answer), and BERTScore (semantic similarity with a reference answer).
Used to compare Qwen2.5-1.5B, Llama-3.2-3B, and Mistral-7B on the same
question set.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from bert_score import score as bert_score
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from rouge_score import rouge_scorer

from generation import Generator
from observability import PipelineMetrics, timed_stage
from retrieval import HybridRetriever

logger = logging.getLogger("documind.evaluation")


class EvaluationError(Exception):
    """Raised when an evaluation run is misconfigured or produces no data."""


@dataclass
class EvalQuestion:
    """One evaluation question with its known-correct answer."""

    question: str
    ground_truth: str


@dataclass
class EvalRecord:
    """A single evaluated question: what was asked, what came back, and
    what it should have been.
    """

    question: str
    ground_truth: str
    answer: str
    contexts: list[str]


@dataclass
class EvaluationReport:
    """Aggregated scores for one model over one evaluation set."""

    model_key: str
    ragas_scores: dict[str, float]
    rouge_l_f1: float
    bertscore_f1: float
    records: list[EvalRecord] = field(default_factory=list)
    avg_retrieval_seconds: float = 0.0  # Section 13
    avg_generation_seconds: float = 0.0  # Section 13
    avg_tokens_per_second: float = 0.0  # Section 13
    n_questions: int = 0
    ragas_score_cis: dict[str, tuple[float, float]] = field(default_factory=dict)
    rouge_l_f1_ci: tuple[float, float] = (float("nan"), float("nan"))
    bertscore_f1_ci: tuple[float, float] = (float("nan"), float("nan"))


def _bootstrap_ci(
    scores: list[float], n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean of `scores`.

    Resamples with replacement rather than assuming a normal
    distribution — a manual eval run is typically a handful to a few
    dozen questions, and a normal approximation is unreliable at that N.
    With fewer than ~10 questions this interval will be wide (sometimes
    uselessly so) — that width is real information about how little a
    single small run tells you, not a bug in the calculation.
    """
    n = len(scores)
    if n < 2:
        mean = scores[0] if scores else float("nan")
        return (mean, mean)

    rng = random.Random(seed)
    resample_means = []
    for _ in range(n_resamples):
        resample_means.append(sum(scores[rng.randrange(n)] for _ in range(n)) / n)
    resample_means.sort()

    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return (resample_means[lower_idx], resample_means[upper_idx])


def _rouge_l_scores(records: list[EvalRecord]) -> list[float]:
    """Per-record ROUGE-L F1 scores (not yet averaged)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r.ground_truth, r.answer)["rougeL"].fmeasure for r in records]


def _bertscore_scores(records: list[EvalRecord]) -> list[float]:
    """Per-record BERTScore F1 scores (not yet averaged)."""
    candidates = [r.answer for r in records]
    references = [r.ground_truth for r in records]
    _, _, f1 = bert_score(candidates, references, lang="en", verbose=False)
    return [float(x) for x in f1]


def _compute_ragas(
    records: list[EvalRecord], llm=None, embeddings=None
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Run RAGAS's RAG-specific metrics.

    `faithfulness` is the direct hallucination signal: it checks whether
    every claim in the answer is actually supported by the retrieved
    contexts, independent of Section 5's citation-marker validation.

    Returns both the aggregate (mean) score per metric and the
    per-question breakdown, so callers can bootstrap a confidence
    interval around the aggregate instead of trusting a single number
    computed from a handful of questions.

    NOTE (confidence flag): RAGAS's LLM-based metrics use OpenAI by
    default, which is wrong for an open-source-only stack. Pass `llm` /
    `embeddings` — LangChain-wrapped, pointed at a local model — to
    override that. I'm moderately, not fully, confident in the exact
    wrapper import paths for ragas==0.1.16 specifically (this API has
    moved across ragas releases); verify against your installed
    version's docs before relying on it. Same caveat applies to
    `result.to_pandas()` below for per-question scores. Everything else
    in this function (the metric list, the dataset shape) I'm confident
    in.
    """
    dataset = Dataset.from_dict(
        {
            "question": [r.question for r in records],
            "answer": [r.answer for r in records],
            "contexts": [r.contexts for r in records],
            "ground_truth": [r.ground_truth for r in records],
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    kwargs = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    result = ragas_evaluate(dataset, metrics=metrics, **kwargs)
    aggregate = {name: float(value) for name, value in dict(result).items()}

    per_question_df = result.to_pandas()
    per_question = {
        name: [float(v) for v in per_question_df[name].tolist()]
        for name in aggregate
        if name in per_question_df.columns
    }
    return aggregate, per_question


def evaluate_model(
    model_key: str,
    questions: list[EvalQuestion],
    retriever: HybridRetriever,
    generator: Generator,
    llm=None,
    embeddings=None,
) -> EvaluationReport:
    """Run retrieval + generation for every question, then score the
    results with RAGAS, ROUGE-L, and BERTScore — and record latency and
    token throughput alongside quality (Section 13). Quality metrics
    alone don't answer "which model would you actually deploy" — speed
    matters just as much for that decision.
    """
    if not questions:
        raise EvaluationError("Cannot evaluate against an empty question set.")

    records: list[EvalRecord] = []
    retrieval_seconds: list[float] = []
    generation_seconds: list[float] = []
    tokens_per_second_samples: list[float] = []

    for eval_question in questions:
        metrics = PipelineMetrics(query=eval_question.question, model_key=model_key)

        with timed_stage(metrics, "retrieval"):
            scored_chunks = retriever.retrieve(eval_question.question)

        if not scored_chunks:
            logger.warning(
                "No retrieval results for eval question %r — skipping.",
                eval_question.question,
            )
            continue

        with timed_stage(metrics, "generation"):
            cited_answer = generator.generate(eval_question.question, scored_chunks)

        metrics.input_tokens = cited_answer.input_token_count
        metrics.output_tokens = cited_answer.output_token_count

        retrieval_seconds.append(
            next(t.duration_seconds for t in metrics.stage_timings if t.stage == "retrieval")
        )
        generation_seconds.append(
            next(t.duration_seconds for t in metrics.stage_timings if t.stage == "generation")
        )
        tokens_per_second_samples.append(metrics.tokens_per_second)

        records.append(
            EvalRecord(
                question=eval_question.question,
                ground_truth=eval_question.ground_truth,
                answer=cited_answer.answer_text,
                contexts=[sc.chunk.text for sc in scored_chunks],
            )
        )

    if not records:
        raise EvaluationError(
            f"No question produced retrieval results for model '{model_key}' — "
            f"cannot compute scores."
        )

    if len(records) < 10:
        logger.warning(
            "Only %d question(s) evaluated — confidence intervals below will be "
            "wide and not very trustworthy. Treat them as a rough sense of "
            "uncertainty, not a rigorous statistical result; ~20-30+ questions "
            "is a more reasonable minimum for that.",
            len(records),
        )

    logger.info("Scoring %d records for model '%s' ...", len(records), model_key)
    ragas_scores, ragas_per_question = _compute_ragas(records, llm=llm, embeddings=embeddings)
    rouge_l_scores = _rouge_l_scores(records)
    bertscore_scores = _bertscore_scores(records)
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    bertscore_f1 = sum(bertscore_scores) / len(bertscore_scores)

    report = EvaluationReport(
        model_key=model_key,
        ragas_scores=ragas_scores,
        rouge_l_f1=rouge_l,
        bertscore_f1=bertscore_f1,
        records=records,
        avg_retrieval_seconds=sum(retrieval_seconds) / len(retrieval_seconds),
        avg_generation_seconds=sum(generation_seconds) / len(generation_seconds),
        avg_tokens_per_second=sum(tokens_per_second_samples) / len(tokens_per_second_samples),
        n_questions=len(records),
        ragas_score_cis={
            name: _bootstrap_ci(scores) for name, scores in ragas_per_question.items()
        },
        rouge_l_f1_ci=_bootstrap_ci(rouge_l_scores),
        bertscore_f1_ci=_bootstrap_ci(bertscore_scores),
    )
    logger.info(
        "Model '%s': faithfulness=%.3f, rouge_l_f1=%.3f, bertscore_f1=%.3f, "
        "avg_generation=%.2fs, tokens/sec=%.1f (n=%d questions)",
        model_key,
        ragas_scores.get("faithfulness", float("nan")),
        rouge_l,
        bertscore_f1,
        report.avg_generation_seconds,
        report.avg_tokens_per_second,
        report.n_questions,
    )
    return report


if __name__ == "__main__":
    from pathlib import Path

    from indexing import Embedder, HybridIndex

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    # Illustrative only — a real evaluation needs a curated set of
    # questions with verified ground-truth answers, not one example.
    eval_set = [
        EvalQuestion(
            question="What was the patient retention rate?",
            ground_truth="Patient retention exceeded 92% across all three cohorts.",
        ),
    ]

    shared_embedder = Embedder()
    loaded_index = HybridIndex.load(Path("index_store"), shared_embedder)
    shared_retriever = HybridRetriever(loaded_index, shared_embedder)

    reports = []
    for key in ["qwen2.5-1.5b", "llama-3.2-3b", "mistral-7b"]:
        model_generator = Generator(model_key=key)
        reports.append(evaluate_model(key, eval_set, shared_retriever, model_generator))
        model_generator.unload()

    for r in reports:
        print(
            f"{r.model_key}: faithfulness={r.ragas_scores.get('faithfulness'):.3f} "
            f"rouge_l={r.rouge_l_f1:.3f} bertscore={r.bertscore_f1:.3f} "
            f"avg_generation={r.avg_generation_seconds:.2f}s "
            f"tokens/sec={r.avg_tokens_per_second:.1f}"
        )
