"""
Document Ingestion Module
=========================

Following the three-module decomposition from Ng, Matsuba & Zhang (NEJM AI, 2024),
this is the Document Ingestion module: it takes structured source data, chunks
it, embeds each chunk, and writes the result to a vector database with metadata
attached.

Design choice: STRUCTURE-AWARE CHUNKING
---------------------------------------
For ecosystem data, fixed-window chunking (split every N tokens) would separate
a program's "deadline" from its "eligibility" and destroy retrieval quality. So
each program is treated as one chunk, with structured metadata stored alongside
the text. The retriever can then filter on metadata before semantic search.

Metadata captured per chunk:
- id, name, organization, url (provenance)
- stage, sector, funding_type, geography (filter fields)
- last_verified (FRESHNESS — surfaces Zhang's "recency" pillar in the UI)
- source_note (traceability — surfaces Zhang's "traceability" pillar)
"""

import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
SEED_FILE = DATA_DIR / "seed_programs.json"

COLLECTION_NAME = "toronto_ecosystem"


def build_chunk_text(program: dict) -> str:
    """
    Compose the searchable text for one program.

    The composition order matters: we put the name and one-line description
    first because dense retrievers weight early tokens more heavily. Eligibility
    and distinctive features come next because those are what user queries
    semantically match on.
    """
    parts = [
        f"Program: {program['name']}",
        f"Run by: {program['organization']}",
        f"Stage focus: {', '.join(program['stage'])}",
        f"Sector focus: {', '.join(program['sector'])}",
        f"Funding type: {', '.join(program['funding_type'])}",
        f"Geography: {program['geography']}",
        f"Eligibility: {program['eligibility_summary']}",
        f"Description: {program['description']}",
        f"What makes it distinctive: {program['what_makes_it_distinctive']}",
    ]
    return "\n".join(parts)


def build_metadata(program: dict) -> dict:
    """
    Metadata stored in ChromaDB alongside the embedding. ChromaDB metadata
    values must be primitive types (str / int / float / bool), so lists are
    joined into comma-separated strings.
    """
    return {
        "id": program["id"],
        "name": program["name"],
        "organization": program["organization"],
        "url": program["url"],
        "stage": ", ".join(program["stage"]),
        "sector": ", ".join(program["sector"]),
        "funding_type": ", ".join(program["funding_type"]),
        "geography": program["geography"],
        "last_verified": program["last_verified"],
        "source_note": program["source_note"],
    }


def ingest():
    """
    Build (or rebuild) the vector index from seed data.

    Idempotent: deletes the existing collection first so re-running gives a
    clean state. This matters when iterating on chunking strategy during
    development.
    """
    print(f"[ingest] Loading seed data from {SEED_FILE}")
    with open(SEED_FILE, "r") as f:
        programs = json.load(f)
    print(f"[ingest] Loaded {len(programs)} programs.")

    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset the collection to ensure a clean rebuild.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[ingest] Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    # ChromaDB's default embedding function uses a small ONNX MiniLM model
    # (all-MiniLM-L6-v2). Runs locally, ~80MB download on first use. Good
    # enough for a demo; production would benchmark against OpenAI
    # text-embedding-3-small or a domain-tuned model.
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    docs, metadatas, ids = [], [], []
    for program in programs:
        docs.append(build_chunk_text(program))
        metadatas.append(build_metadata(program))
        ids.append(program["id"])

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"[ingest] Indexed {len(docs)} chunks into '{COLLECTION_NAME}'.")
    print(f"[ingest] Vector store written to {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
