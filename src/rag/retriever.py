"""
SentinelAI — RAG cybersecurity knowledge base.

Pipeline: documents (data/knowledge_base/*.md) -> chunking (by ## section
headers) -> embeddings (TF-IDF, see note below) -> FAISS -> retrieval.

This module provides general cybersecurity knowledge for investigation
(attack descriptions, indicators, investigation questions, defensive
recommendations). It never has access to telemetry and must never be
used to override ML evidence -- the SOC Copilot (Stage 11) is
responsible for keeping retrieved knowledge clearly separate from
ML/historical evidence in its responses.

NOTE ON EMBEDDINGS: TF-IDF vectors (scikit-learn) are used instead of a
neural sentence-transformer model, since this environment's network
does not include huggingface.co, and a demo shouldn't depend on
downloading model weights at showtime. To upgrade to neural embeddings
when internet access is available, replace `embed_texts()` below with
a sentence-transformers encode() call and refit the index -- everything
else (chunking, FAISS indexing, retrieval interface) stays the same.
"""

import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg


# ---------------------------------------------------------------------------
# Document loading + chunking
# ---------------------------------------------------------------------------
def load_documents(kb_dir: str = cfg.KNOWLEDGE_BASE_DIR) -> list:
    docs = []
    for filename in sorted(os.listdir(kb_dir)):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(kb_dir, filename), "r") as f:
            docs.append({"filename": filename, "text": f.read()})
    return docs


def chunk_document(doc: dict) -> list:
    """Split a markdown document into chunks by '## ' section headers.
    The top-level '# Title' becomes part of the first chunk's context."""
    text = doc["text"]
    title_match = re.match(r"^#\s+(.+)", text)
    doc_title = title_match.group(1).strip() if title_match else doc["filename"]

    sections = re.split(r"\n##\s+", text)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        if i == 0:
            # This is the "# Title" preamble -- skip if it has no real content
            # beyond the title line itself.
            body = re.sub(r"^#\s+.+", "", section).strip()
            if not body:
                continue
            section_title = "Overview"
            section_text = body
        else:
            lines = section.split("\n", 1)
            section_title = lines[0].strip()
            section_text = lines[1].strip() if len(lines) > 1 else ""

        if not section_text:
            continue
        chunks.append({
            "source_document": doc["filename"],
            "document_title": doc_title,
            "section_title": section_title,
            "text": section_text,
        })
    return chunks


def build_chunks(kb_dir: str = cfg.KNOWLEDGE_BASE_DIR) -> list:
    all_chunks = []
    for doc in load_documents(kb_dir):
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# ---------------------------------------------------------------------------
# Embeddings (TF-IDF -- see module docstring)
# ---------------------------------------------------------------------------
def fit_vectorizer(texts: list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    vectorizer.fit(texts)
    return vectorizer


def embed_texts(vectorizer, texts: list) -> np.ndarray:
    """Returns L2-normalized dense float32 vectors, ready for FAISS inner-
    product (cosine similarity) search."""
    vectors = vectorizer.transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------------------------------------------------------------------
# FAISS index build + persistence
# ---------------------------------------------------------------------------
def build_index(vectors: np.ndarray):
    import faiss
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(vectors)
    return index


def build_and_save_knowledge_base():
    chunks = build_chunks()
    # Embed title + body together so a query matching words that only
    # appear in a section's TITLE (e.g. "false positive" matching the
    # "False Positive Considerations" heading) can still retrieve it --
    # the displayed `text` field itself stays pure body content.
    embedding_texts = [f"{c['document_title']}. {c['section_title']}. {c['text']}" for c in chunks]
    vectorizer = fit_vectorizer(embedding_texts)
    vectors = embed_texts(vectorizer, embedding_texts)
    index = build_index(vectors)

    import faiss
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    faiss.write_index(index, cfg.RAG_INDEX_PATH)
    with open(cfg.RAG_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(cfg.RAG_CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    return chunks, vectorizer, index


def load_knowledge_base():
    import faiss
    index = faiss.read_index(cfg.RAG_INDEX_PATH)
    with open(cfg.RAG_VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(cfg.RAG_CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    return chunks, vectorizer, index


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def search_security_knowledge(query: str, chunks: list, vectorizer, index,
                               top_k: int = cfg.RAG_TOP_K) -> list:
    """Returns the top_k most relevant knowledge-base chunks for `query`,
    each with a similarity score. This function only ever returns
    GENERAL cybersecurity knowledge -- it has no access to telemetry and
    should never be treated as evidence about a specific entity/event."""
    query_vec = embed_texts(vectorizer, [query])
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        results.append({
            "source_document": chunk["source_document"],
            "document_title": chunk["document_title"],
            "section_title": chunk["section_title"],
            "text": chunk["text"],
            "relevance_score": round(float(score), 4),
        })
    return results
