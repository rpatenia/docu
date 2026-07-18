"""
run_eval.py

A small CLI for manually sanity-checking DocuMind against a real PDF:
build an index, ask it a set of questions, and print the pipeline's own
built-in grounding/confidence signals (valid vs. invalid citations,
security flags) for each answer -- this is how you actually see whether
it's hallucinating on your document, not a separate feature.

If every question in the input file also has a "ground_truth" answer,
this additionally runs Section 6's full evaluate_model() and prints
quantitative RAGAS/ROUGE-L/BERTScore numbers.

Usage:
    python run_eval.py <path_to_pdf> <model_key> <questions.json>

model_key is one of: qwen2.5-1.5b, llama-3.2-3b, mistral-7b

questions.json is a flat JSON list. Each entry is either a plain string:
    "What was the patient retention rate?"
or an object with a known-correct answer, to enable quantitative scoring:
    {"question": "What was the patient retention rate?",
     "ground_truth": "Patient retention exceeded 92% across all three cohorts."}

Include at least one question that is NOT answerable from the document --
that's the one that should trigger the "I don't have enough information"
refusal instead of a fabricated answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluation import EvalQuestion, evaluate_model
from generation import Generator
from indexing import Embedder, HybridIndex
from pdf_ingestion import ingest_pdf
from retrieval import HybridRetriever


def _load_questions(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for entry in raw:
        if isinstance(entry, str):
            questions.append({"question": entry, "ground_truth": None})
        else:
            questions.append({"question": entry["question"], "ground_truth": entry.get("ground_truth")})
    return questions


def _print_qualitative(question: str, retriever: HybridRetriever, generator: Generator) -> None:
    scored_chunks = retriever.retrieve(question)
    print(f"\nQ: {question}")

    if not scored_chunks:
        print("  -> No chunks retrieved at all. Nothing to generate from.")
        return

    answer = generator.generate(question, scored_chunks)
    print(f"A: {answer.answer_text}")
    print(f"  Cited sources:    {answer.cited_numbers}")
    print(f"  Uncited sources:  {answer.uncited_numbers}")
    print(f"  Invalid sources:  {answer.invalid_citations}  <- non-empty means a fabricated citation")
    print(f"  Security flags:   {answer.security_flags}")

    is_refusal = "don't have enough information" in answer.answer_text.lower()
    if answer.invalid_citations:
        print("  VERDICT: likely hallucination -- cited a source number that was never provided.")
    elif not answer.cited_numbers and not is_refusal:
        print("  VERDICT: ungrounded -- answered without citing anything and didn't refuse.")
    elif is_refusal:
        print("  VERDICT: refused -- correct behavior if the document truly doesn't answer this.")
    else:
        print("  VERDICT: grounded -- every claim traces back to a real, provided source.")


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python run_eval.py <path_to_pdf> <model_key> <questions.json>")
        sys.exit(1)

    pdf_path, model_key, questions_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

    print(f"Ingesting {pdf_path.name} ...")
    chunks = ingest_pdf(pdf_path)

    embedder = Embedder()
    index = HybridIndex(embedder)
    index.build(chunks)
    retriever = HybridRetriever(index, embedder)

    print(f"Loading generator '{model_key}' ...")
    generator = Generator(model_key=model_key)

    questions = _load_questions(questions_path)

    print("\n=== Per-question grounding check ===")
    for q in questions:
        _print_qualitative(q["question"], retriever, generator)

    if all(q["ground_truth"] for q in questions):
        print("\n=== Quantitative evaluation (RAGAS + ROUGE-L + BERTScore) ===")
        eval_questions = [
            EvalQuestion(question=q["question"], ground_truth=q["ground_truth"]) for q in questions
        ]
        report = evaluate_model(model_key, eval_questions, retriever, generator)

        def _fmt(name: str, mean: float, ci: tuple[float, float]) -> str:
            return f"{name:<19} {mean:.3f}  (95% CI: {ci[0]:.3f}-{ci[1]:.3f})"

        print(_fmt("Faithfulness:", report.ragas_scores.get("faithfulness", float("nan")), report.ragas_score_cis.get("faithfulness", (float("nan"), float("nan")))))
        print(_fmt("Answer relevancy:", report.ragas_scores.get("answer_relevancy", float("nan")), report.ragas_score_cis.get("answer_relevancy", (float("nan"), float("nan")))))
        print(_fmt("Context precision:", report.ragas_scores.get("context_precision", float("nan")), report.ragas_score_cis.get("context_precision", (float("nan"), float("nan")))))
        print(_fmt("Context recall:", report.ragas_scores.get("context_recall", float("nan")), report.ragas_score_cis.get("context_recall", (float("nan"), float("nan")))))
        print(_fmt("ROUGE-L F1:", report.rouge_l_f1, report.rouge_l_f1_ci))
        print(_fmt("BERTScore F1:", report.bertscore_f1, report.bertscore_f1_ci))
        print(f"Avg generation:     {report.avg_generation_seconds:.2f}s")
        print(f"Tokens/sec:         {report.avg_tokens_per_second:.1f}")
        print(f"(n = {report.n_questions} question(s) — intervals are wide/unreliable below ~10)")
    else:
        print(
            "\n(Skipping quantitative RAGAS/ROUGE-L/BERTScore scoring -- not every "
            "question in the input file has a ground_truth. Add one to every entry "
            "to get numeric scores instead of just the qualitative check above.)"
        )

    generator.unload()


if __name__ == "__main__":
    main()
