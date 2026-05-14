# Toronto Health-Tech Ecosystem RAG

A retrieval-augmented generation system that lets prospective health-tech founders
query Toronto's accelerator, grant, and funding ecosystem and get **grounded, cited,
date-aware** answers about programs like H2i, MaRS, JLABS, Creative Destruction Lab,
OBIO, and more.

---

## Why this architecture (mapping to the research framing)

This demo is structured around the four limitations of standalone LLMs that
**Ng, Matsuba, and Zhang (NEJM AI, 2024)** identify as the motivation for RAG:

| Limitation | How this demo addresses it |
|---|---|
| **Recency** — training data is a static snapshot | Each chunk carries a `last_verified` date; UI surfaces it next to every answer |
| **Private / domain data** — LLMs can't see what they weren't trained on | Ingestion pipeline takes structured JSON, PDFs, and HTML; vectorizes locally |
| **Traceability** — black-box outputs erode trust | Every claim in the answer is inline-cited to a specific source chunk |
| **Accuracy / hallucination** — ungrounded answers in high-stakes domains | Generator is prompted to refuse rather than guess when retrieval is weak |

The pipeline follows the three-module decomposition from Figure 1 of that paper:
**Document Ingestion → Retriever → Generator.**

---

## What's in this demo (and what isn't)

**Implemented:**
- Document Ingestion: structured JSON → chunked, embedded, indexed in ChromaDB
- Retriever: vector search (Naïve mode) **and** hybrid vector + BM25 + metadata filter (Hybrid mode)
- Generator: Claude (Anthropic API) with grounded-citation prompting, plus an offline fallback for demos without keys
- Evaluator: lightweight context-precision and groundedness checks on a 6-question test set
- UI: Streamlit app with mode toggle, inline citations, retrieved-chunk inspector, and freshness indicators

**Deliberately scoped out (next steps for the real project):**
- **CRAG** (Corrective RAG with web-search fallback) — natural fit because Toronto programs get discontinued; CRAG's evaluator-then-fallback pattern is built for exactly that. See `ARCHITECTURE_NOTES.md`.
- **Self-Reflective RAG** — reflection tokens for groundedness self-check; addresses the "relevance vs. credibility" failure mode the NEJM AI paper flags.
- **Live semantic scraping** with Crawl4AI on a weekly cron — `src/scrape_skeleton.py` shows the intended structure.
- Re-ranker model (cross-encoder) — Zhang's paper specifically recommends this; current demo uses cosine similarity only.

The Naïve-vs-Hybrid comparison in this demo is a **proof of methodology** for the
larger CRAG vs. Self-Reflective vs. Naïve comparison that's the project's research
contribution.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. (optional) Set Anthropic key — without this, demo runs in offline mode
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-...

# 3. Build the index from seed data
python src/ingest.py

# 4. Launch the UI
streamlit run src/app.py
```

Try these queries:
- *"I'm a first-time founder building a digital health app. What accelerators in Toronto should I look at?"*
- *"I have a regenerative medicine company at the seed stage. Who funds that here?"*
- *"What's the deadline for the next Lab2Market cohort?"* (intentionally tests freshness)
- *"Compare H2i and MaRS for an early-stage clinician founder."*

Toggle between **Naïve** and **Hybrid** mode in the sidebar to see retrieval quality differences.

---

## Architecture notes

**Embedding model:** ChromaDB's default ONNX MiniLM (runs locally, ~80MB, no API
calls). Production would benchmark this against `text-embedding-3-small` and
domain-tuned alternatives.

**Chunking:** structure-aware — each program description is treated as an atomic
unit with metadata (stage, sector, funding type, last_verified date) attached
as ChromaDB metadata fields. This is intentional: naive fixed-window chunking
would split "deadline" from "eligibility" and destroy retrieval quality on
ecosystem data.

**Hybrid retrieval:** the Hybrid mode runs vector search and BM25 in parallel,
takes the union of top-k from each, then re-ranks by reciprocal rank fusion
(RRF). This is the cheapest version of what Zhang's paper recommends as the
fix for the "relevance vs. credibility" problem.

**Metadata filtering:** Hybrid mode also lets the user filter by stage
(pre-seed, seed, Series A+) and sector (digital health, devices, biotech), which
front-loads relevance before retrieval rather than relying on the embedding to
capture it.

**Generation prompt:** the generator is told to (a) only use retrieved context,
(b) cite each claim with the source program name, and (c) explicitly say *"I
don't have current information on this"* rather than guess. This is the
hallucination guardrail.

---

## Research design (what I'd actually evaluate, given more time)

For the eventual three-way comparison (Naïve vs. CRAG vs. Self-RAG), I'd
evaluate on:

**Quantitative (RAG triad, via RAGAs):**
- Context Relevance — are retrieved chunks topically relevant to the query?
- Groundedness / Faithfulness — does the answer follow from the chunks?
- Answer Relevance — does the answer address the question asked?

**Quantitative (domain-specific):**
- **Freshness accuracy** — when a program has changed since the index was built,
  does the system surface uncertainty or confidently hallucinate? Built a
  10-query "stale data" probe set.
- **Citation coverage** — % of claims in the answer that map to a retrieved chunk.

**Qualitative (founder survey):**
- Task completion: did the founder leave knowing what to do next?
- Trust calibration: did they over-trust or under-trust the output?

The NEJM AI paper notes RAGAs metrics "do not fully capture the complexity" of
its target domain. The entrepreneurship version of that gap is that **answer
relevance can be high while task usefulness is low** — e.g., correctly listing
five accelerators when the founder needed help narrowing to one. The qualitative
survey is what would distinguish those cases.

---

## File map

```
toronto-ecosystem-rag/
├── README.md                       <- you are here
├── ARCHITECTURE_NOTES.md           <- CRAG / Self-RAG implementation sketches
├── requirements.txt
├── .env.example
├── data/
│   └── seed_programs.json          <- 12 real Toronto ecosystem programs
├── src/
│   ├── ingest.py                   <- Document Ingestion module
│   ├── retriever.py                <- Retriever module (Naïve + Hybrid)
│   ├── generator.py                <- Generator module + offline fallback
│   ├── evaluator.py                <- Lightweight eval harness
│   ├── scrape_skeleton.py          <- Outline of weekly scraper
│   └── app.py                      <- Streamlit UI
└── notebooks/
    └── rag_comparison.ipynb        <- Naïve vs. Hybrid eval, plottable
```
