"""
Retriever Module
================

Implements two retrieval strategies for the comparison:

1. NAÏVE — pure cosine similarity over MiniLM embeddings.
2. HYBRID — vector + BM25 sparse retrieval, fused with Reciprocal Rank Fusion (RRF),
           with optional metadata pre-filtering on stage / sector.

The hybrid design follows the recommendation in Ng, Matsuba & Zhang (NEJM AI,
2024) that pure vector search struggles with the relevance-vs-credibility trade-
off, and that combining dense + sparse + metadata is the standard fix.
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
COLLECTION_NAME = "toronto_ecosystem"

# Minimum cosine similarity for a chunk to count as "relevant."
# Below this, the question is treated as off-topic and the retriever
# returns []. Maps to Zhang's "accuracy" pillar: refuse to answer when
# the corpus has nothing relevant rather than confidently returning
# the least-bad of a bad batch.
# For all-MiniLM-L6-v2 embeddings, ~0.15 separates "loosely related"
# from "unrelated" pretty cleanly.
MIN_VECTOR_SIMILARITY = 0.15


def _tokenize(text: str) -> list:
    """Lowercase + simple word-level tokenization for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class Retriever:
    """
    Wraps the ChromaDB collection and adds a BM25 index over the same
    documents for hybrid retrieval.
    """

    def __init__(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )

        # Pull all documents once at init for BM25. With 12 programs this is
        # trivial; at production scale (10k+ programs) BM25 would move to a
        # dedicated index (e.g., Elasticsearch / OpenSearch) but the API
        # contract stays the same.
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self._all_ids = all_docs["ids"]
        self._all_docs = all_docs["documents"]
        self._all_metas = all_docs["metadatas"]
        self._bm25 = BM25Okapi([_tokenize(d) for d in self._all_docs])

    # ---- Naïve retrieval --------------------------------------------------

    def retrieve_naive(self, query: str, k: int = 4) -> list:
        """
        Pure dense vector search. Simple, fast, the baseline for comparison.
        Filters out chunks below MIN_VECTOR_SIMILARITY so off-topic queries
        return [] instead of low-scoring noise.
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        formatted = self._format_results(
            ids=results["ids"][0],
            docs=results["documents"][0],
            metas=results["metadatas"][0],
            distances=results["distances"][0],
            method="naive_vector",
        )
        # Relevance gate
        return [hit for hit in formatted if hit["score"] >= MIN_VECTOR_SIMILARITY]

    # ---- Hybrid retrieval -------------------------------------------------

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 4,
        stage_filter: str = None,
        sector_filter: str = None,
    ) -> list:
        """
        Hybrid retrieval: union the top-k from dense vector search and BM25
        sparse search, then re-rank using Reciprocal Rank Fusion (RRF).

        RRF score for a document d is: sum over rankers r of 1 / (60 + rank_r(d))
        where 60 is the standard constant from Cormack et al. (2009).

        Optionally pre-filters the candidate set by stage or sector metadata
        before fusion — this is the cheapest form of "structural search with
        metadata filtering" that Zhang's paper recommends.
        """
        # 1. Dense candidates (over-fetch so RRF has material to work with).
        # When metadata filters are active, expand the candidate pool to the
        # whole corpus — otherwise filters can leave us with zero results
        # because the pre-filter top-k didn't contain any matches.
        if stage_filter or sector_filter:
            dense_n = len(self._all_ids)
        else:
            dense_n = min(2 * k, len(self._all_ids))
        dense_results = self.collection.query(query_texts=[query], n_results=dense_n)
        dense_ranking = list(dense_results["ids"][0])

        # Relevance gate. RRF scores are rank-based and don't reflect actual
        # semantic similarity — even an off-topic query will get a respectable
        # RRF score because *something* ranks #1. So we check the underlying
        # cosine similarity BEFORE fusion: if even the closest chunk is below
        # threshold, the query is off-topic and we refuse to answer.
        best_distance = dense_results["distances"][0][0]
        best_similarity = 1.0 - best_distance
        if best_similarity < MIN_VECTOR_SIMILARITY:
            return []

        # 2. Sparse (BM25) candidates over the same corpus
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        # Pair each score with the matching document id
        sparse_ranking = [
            doc_id
            for _, doc_id in sorted(
                zip(bm25_scores, self._all_ids), key=lambda x: x[0], reverse=True
            )
        ][:dense_n]

        # 3. RRF fusion
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
        for rank, doc_id in enumerate(sparse_ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Apply optional metadata filters AFTER fusion (so we keep semantic
        # quality even when filters would otherwise exclude a near-miss).
        id_to_idx = {doc_id: i for i, doc_id in enumerate(self._all_ids)}
        filtered = []
        for doc_id, score in fused:
            meta = self._all_metas[id_to_idx[doc_id]]
            if stage_filter and stage_filter.lower() not in meta["stage"].lower():
                continue
            if sector_filter and sector_filter.lower() not in meta["sector"].lower():
                continue
            filtered.append((doc_id, score))
            if len(filtered) >= k:
                break

        # Format
        out = []
        for doc_id, score in filtered:
            idx = id_to_idx[doc_id]
            out.append(
                {
                    "id": doc_id,
                    "document": self._all_docs[idx],
                    "metadata": self._all_metas[idx],
                    "score": float(score),
                    "method": "hybrid_rrf",
                }
            )
        return out

    # ---- Helpers ----------------------------------------------------------

    @staticmethod
    def _format_results(ids, docs, metas, distances, method):
        out = []
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "document": docs[i],
                    "metadata": metas[i],
                    # Lower distance = closer in cosine space; convert to a
                    # similarity-style score (higher = better) for display.
                    "score": float(1.0 - distances[i]),
                    "method": method,
                }
            )
        return out


if __name__ == "__main__":
    # Smoke test
    r = Retriever()
    q = "I'm a clinician with a digital health idea, where should I start?"
    print(f"\n--- NAIVE ---\nQuery: {q}\n")
    for hit in r.retrieve_naive(q):
        print(f"  [{hit['score']:.3f}] {hit['metadata']['name']}")
    print(f"\n--- HYBRID ---\nQuery: {q}\n")
    for hit in r.retrieve_hybrid(q, sector_filter="digital health"):
        print(f"  [{hit['score']:.4f}] {hit['metadata']['name']}")
