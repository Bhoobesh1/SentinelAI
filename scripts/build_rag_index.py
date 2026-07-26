"""
Run this script to build the RAG knowledge base index and sanity-check
retrieval quality with representative SOC analyst questions.

Usage:
    python scripts/build_rag_index.py

Requires:
    data/knowledge_base/*.md

Outputs:
    models/rag_faiss.index
    models/rag_vectorizer.pkl
    models/rag_chunks.json
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as cfg
from src.rag.retriever import build_and_save_knowledge_base, search_security_knowledge

TEST_QUERIES = [
    "Why would repeated failed logins from the same IP be dangerous?",
    "What does it mean when a login comes from a location far from normal?",
    "How should I investigate a device that has never been seen before?",
    "What is low and slow data exfiltration?",
    "How do I tell if behavioral drift is a legitimate role change or an insider threat?",
    "How is the risk score calculated?",
]


def main():
    print("=" * 70)
    print("SENTINELAI — RAG KNOWLEDGE BASE INDEXING")
    print("=" * 70)

    chunks, vectorizer, index = build_and_save_knowledge_base()
    print(f"Documents loaded: {len(set(c['source_document'] for c in chunks))}")
    print(f"Chunks indexed  : {len(chunks)}")
    print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")

    print("\n" + "-" * 70)
    print("RETRIEVAL SANITY CHECK")
    print("-" * 70)
    for query in TEST_QUERIES:
        print(f"\nQuery: \"{query}\"")
        results = search_security_knowledge(query, chunks, vectorizer, index, top_k=2)
        for r in results:
            print(f"  [{r['relevance_score']}] {r['document_title']} -> {r['section_title']}")
            print(f"      {r['text'][:120]}...")

    # ---------------- Sanity checks ----------------
    assert len(chunks) >= 20, "Suspiciously few chunks -- check chunking logic / knowledge base files!"
    for query in TEST_QUERIES:
        results = search_security_knowledge(query, chunks, vectorizer, index, top_k=1)
        assert len(results) > 0, f"No results retrieved for query: {query}"
        assert results[0]["relevance_score"] > 0, f"Zero relevance score for query: {query}"

    print("\n[OK] Knowledge base has a reasonable number of chunks.")
    print("[OK] Every test query retrieves at least one relevant chunk with positive relevance.")
    print("=" * 70)
    print(f"\nWrote: {cfg.RAG_INDEX_PATH}")
    print(f"Wrote: {cfg.RAG_VECTORIZER_PATH}")
    print(f"Wrote: {cfg.RAG_CHUNKS_PATH}")


if __name__ == "__main__":
    main()
