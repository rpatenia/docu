"""
app.py

Section 7 of DocuMind: the Streamlit application that ties Sections 2-5
together into the user-facing product — upload a PDF, ask questions in
plain English, get cited answers. Amended by Section 8 (persistent
document library), Section 11 (prompt-injection warnings), Section 13
(latency/throughput captions), and Section 15 (upload size limit).

Run with: streamlit run app.py
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

from config import MAX_UPLOAD_SIZE_MB
from generation import CitedAnswer, GenerationError, Generator, MODEL_REGISTRY
from indexing import Embedder, HybridIndex, IndexingError
from observability import run_instrumented_query
from pdf_ingestion import Chunk, PDFIngestionError, ingest_pdf
from retrieval import HybridRetriever, RetrievalError
from security import scan_text_for_injection
from storage import StorageError, list_indexes, load_index, save_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("documind.app")

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_embedder() -> Embedder:
    """Loaded once per server process, not once per rerun.

    st.cache_resource is for non-serializable, expensive-to-create
    objects like ML models — st.cache_data (for plain data) would try
    to hash/serialize this and is the wrong tool.
    """
    return Embedder()


@st.cache_resource(show_spinner=False)
def load_generator(model_key: str) -> Generator:
    """One cached Generator per model_key — cheap to call repeatedly
    once a given model_key has been loaded once.
    """
    return Generator(model_key=model_key, use_4bit=False)


def get_generator(model_key: str) -> Generator:
    """Ensure only one generation model is resident in GPU memory at a time.

    load_generator() caches per model_key, but nothing stops two cached
    models from piling up in GPU memory if the user switches models
    mid-session — a single T4 can't hold Qwen + Llama + Mistral at once.
    This unloads the previous model and clears the whole cache before
    loading a new one, rather than relying on per-key cache eviction.
    """
    previous_key = st.session_state.get("loaded_model_key")
    if previous_key is not None and previous_key != model_key:
        load_generator(previous_key).unload()
        load_generator.clear()

    generator = load_generator(model_key)
    st.session_state.loaded_model_key = model_key
    return generator


def build_index_and_retriever(uploaded_files: list) -> tuple[HybridIndex, HybridRetriever]:
    """Ingest every uploaded PDF, build one combined hybrid index, and
    build the retriever for it once — not per query.
    """
    all_chunks: list[Chunk] = []
    for uploaded_file in uploaded_files:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            st.error(
                f"'{uploaded_file.name}' is {size_mb:.1f} MB, which exceeds the "
                f"{MAX_UPLOAD_SIZE_MB} MB limit — skipped. (Streamlit's own "
                f"server.maxUploadSize should normally reject this before it "
                f"even reaches this check — see Section 15 if you're seeing "
                f"large files get this far.)"
            )
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            all_chunks.extend(ingest_pdf(tmp_path))
        except PDFIngestionError as exc:
            st.error(f"Could not process '{uploaded_file.name}': {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

    if not all_chunks:
        raise IndexingError("No usable text extracted from any uploaded PDF.")

    flagged = [
        (chunk.chunk_id, scan_text_for_injection(chunk.text)) for chunk in all_chunks
    ]
    flagged = [(chunk_id, flags) for chunk_id, flags in flagged if flags]
    if flagged:
        st.warning(
            f"{len(flagged)} chunk(s) in the uploaded document(s) contain text "
            f"resembling prompt-injection attempts. DocuMind will still answer "
            f"questions, but treat answers touching these sections with extra "
            f"caution."
        )
        logger.warning("Flagged chunks at ingestion: %s", flagged)

    embedder = load_embedder()
    index = HybridIndex(embedder)
    index.build(all_chunks)
    retriever = HybridRetriever(index, embedder)
    return index, retriever


def render_answer(cited_answer: CitedAnswer) -> None:
    """Render a generated answer, its sources, and any grounding or
    security warnings.
    """
    if cited_answer.security_flags:
        st.error(
            "Possible prompt injection detected in this response — "
            "treat it with extra caution:\n"
            + "\n".join(f"- {flag}" for flag in cited_answer.security_flags)
        )

    st.markdown(cited_answer.answer_text)

    if cited_answer.invalid_citations:
        st.warning(
            f"This answer references source number(s) "
            f"{cited_answer.invalid_citations} that weren't actually provided — "
            f"treat it as unverified."
        )
    elif not cited_answer.cited_numbers:
        st.warning("This answer doesn't cite any source — treat it as unverified.")

    with st.expander("Sources"):
        for number, scored_chunk in cited_answer.sources.items():
            marker = "cited" if number in cited_answer.cited_numbers else "not cited"
            st.markdown(
                f"**[{number}] ({marker})** {scored_chunk.chunk.source}, "
                f"page {scored_chunk.chunk.page_number}"
            )
            st.caption(scored_chunk.chunk.text[:300] + "...")


def main() -> None:
    st.title("📄 DocuMind")
    st.caption("Upload a PDF, ask questions in plain English, get cited answers.")

    if "index" not in st.session_state:
        st.session_state.index = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Document")
        uploaded_files = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files and st.button("Build index"):
            with st.spinner("Reading and indexing document(s)..."):
                try:
                    index, retriever = build_index_and_retriever(uploaded_files)
                    st.session_state.index = index
                    st.session_state.retriever = retriever
                    st.session_state.messages = []
                    st.success(f"Indexed {len(index.chunks)} chunks.")

                    default_name = "-".join(
                        f.name.rsplit(".", 1)[0] for f in uploaded_files
                    )[:60]
                    try:
                        save_index(index, default_name)
                        st.caption(f"Saved to library as '{default_name}'.")
                    except StorageError:
                        timestamped_name = f"{default_name}-{int(time.time())}"
                        save_index(index, timestamped_name)
                        st.caption(
                            f"Saved to library as '{timestamped_name}' "
                            f"(name collision resolved)."
                        )
                except (PDFIngestionError, IndexingError) as exc:
                    st.error(str(exc))

        st.header("Document Library")
        try:
            saved = list_indexes()
        except StorageError as exc:
            saved = []
            st.caption(f"Could not read document library: {exc}")

        if saved:
            options = {f"{e.name} ({e.chunk_count} chunks)": e.name for e in saved}
            selected_label = st.selectbox("Load a saved document", list(options.keys()))
            if st.button("Load"):
                with st.spinner("Loading saved index..."):
                    try:
                        embedder = load_embedder()
                        index = load_index(options[selected_label], embedder)
                        st.session_state.index = index
                        st.session_state.retriever = HybridRetriever(index, embedder)
                        st.session_state.messages = []
                        st.success(f"Loaded '{options[selected_label]}'.")
                    except StorageError as exc:
                        st.error(str(exc))
        else:
            st.caption("No saved documents yet.")

        st.header("Model")
        model_key = st.selectbox("Generation model", list(MODEL_REGISTRY.keys()))

    if st.session_state.index is None:
        st.info("Upload a PDF and click 'Build index' to get started.")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["cited_answer"])
            else:
                st.markdown(message["content"])

    query = st.chat_input("Ask a question about your document...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving and generating with {model_key}..."):
            try:
                generator = get_generator(model_key)
                cited_answer, metrics = run_instrumented_query(
                    query, st.session_state.retriever, generator
                )
                if not cited_answer.sources:
                    st.warning("No relevant content found in the document for this question.")
                else:
                    render_answer(cited_answer)
                    retrieval_time = next(
                        (t.duration_seconds for t in metrics.stage_timings if t.stage == "retrieval"),
                        0.0,
                    )
                    generation_time = next(
                        (t.duration_seconds for t in metrics.stage_timings if t.stage == "generation"),
                        0.0,
                    )
                    st.caption(
                        f"Retrieved in {retrieval_time:.2f}s, generated in {generation_time:.2f}s "
                        f"({metrics.tokens_per_second:.1f} tok/s)"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "cited_answer": cited_answer}
                    )
            except (RetrievalError, GenerationError) as exc:
                st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
