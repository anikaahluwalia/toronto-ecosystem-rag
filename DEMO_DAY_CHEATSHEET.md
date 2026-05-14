# Demo-Day Cheat Sheet

How to walk Peter Zhang through this in the interview. Read once before; don't
read from it during.

---

## Before the call

1. **Re-run `python src/ingest.py`** in the morning so the index is fresh and
   the first query doesn't trigger a model download in front of him.
2. **Run `streamlit run src/app.py` once** to confirm the UI loads. Leave a tab
   open.
3. **Have the GitHub repo URL ready in your clipboard.** You want to be able to
   paste it the moment he asks.
4. **Pre-stage one good query** in the input box. *"I'm a U of T med student
   with a digital health idea. Where do I start?"* is the strongest opening —
   it surfaces H2i, MaRS, and Lab2Market, all of which he knows well.

## The 5-minute walkthrough (if he gives you the floor)

**Minute 1 — Frame it using HIS paper.**
"Before I built anything, I re-read the NEJM AI piece you co-authored on RAG in
healthcare. The four pillars you describe — recency, private data, traceability,
accuracy — translate almost one-to-one onto the Toronto ecosystem problem, so I
used those as the design principles. Same with the three-module decomposition
from your Figure 1 — ingestion, retriever, generator. Want me to walk through
the UI?"

**Minute 2 — Run the query.**
Type the prepared query. Let it render. Point at:
- The inline `[H2i]` citations — *"that's the traceability pillar made literal."*
- The "last verified" date on the right panel — *"that's the recency pillar — every chunk carries its own freshness, and the prompt is told to flag time-sensitive claims."*
- The "verify on website" notes the model adds — *"that's the accuracy guardrail. The system prompt tells it to refuse before guessing."*

**Minute 3 — Flip to Hybrid mode and re-run.**
"This is the comparison I built into the demo. Naïve is pure vector. Hybrid is
vector + BM25 + metadata filtering with reciprocal rank fusion — which is the
fix you specifically recommend in the paper for the relevance-vs-credibility
problem."

Show the chunk list shift. If precision is visibly better, point at it. If
it's not different on this query, say so — *"on this query both modes find the
same top-3; the difference shows up on queries with rare keywords like 'IRAP'
or 'CCRM' where BM25 catches what the embedding misses. The evaluator notebook
quantifies it across the test set."*

**Minute 4 — Show ARCHITECTURE_NOTES.md.**
"I deliberately scoped CRAG and Self-RAG out of this demo because half-
implementing them would be worse than not implementing them. But I sketched
how I'd add each. CRAG's web fallback is the natural answer to the staleness
problem you flag in the limitations section. Self-RAG's reflection tokens are
the natural answer to the relevance-vs-credibility problem. The three-way
comparison the project's research question asks for is something I'd build out
in the first month."

**Minute 5 — End on the research question.**
"The piece I'm most curious about is the evaluation design. The RAGAs metrics
give you faithfulness and context precision, but your paper notes they don't
fully capture clinical complexity. The entrepreneurship analogue is that
answer relevance can be high while task usefulness is low — five accelerators
listed correctly when the founder needed help picking one. I think that gap is
what the qualitative survey side of the eval has to capture, and I'd want to
co-design that with H2i mentors. Does that match how you've been thinking
about it?"

That last question turns the demo back into a conversation.

---

## If something breaks

- **WiFi flakes / API down** → the UI shows "Offline mode" badge automatically.
  Don't apologize. Say: *"This is the offline fallback I built for exactly
  this case — it shows structured retrieval output instead of generated prose.
  The retrieval layer is what's interesting anyway, since that's where the
  research comparison lives."*
- **ChromaDB takes 30 seconds on first query** → first run downloads the
  embedding model. *"First-query latency in production would be hidden by a
  warm pool — that's a deployment concern, not an architecture one."*
- **He asks a question I can't answer** → *"I don't know — would you walk me
  through how you'd think about it?"* He literally wrote the paper. Let him
  teach you something. That's a feature, not a bug.

---

## Lines worth memorizing

These will land:

> *"Structure-aware chunking — one program = one chunk — because fixed-window
> chunking would separate a program's deadline from its eligibility and
> destroy retrieval quality on this kind of data."*

> *"The retriever doesn't decide whether to trust a chunk. It returns
> candidates. Trust is the generator's job, and the prompt is what enforces
> it."*

> *"I'd default to Anthropic for the demo because Claude is what I work with at
> LangPal, but the production choice depends on data residency. If H2i needs
> founder query data to stay in Canada or inside the U of T tenant, Azure
> OpenAI or self-hosted Ollama become the right answer, not OpenAI direct."*

> *"The strongest evidence the comparison is worth running is that the three
> architectures fail in different ways. CRAG fails when web fallback retrieves
> noise. Self-RAG fails when the reflection model is miscalibrated. Naïve
> fails on staleness. Picking a winner overall is the wrong question — picking
> a winner per query type is the right one."*

---

## Things NOT to say

- Don't oversell. *"I built a full RAG system"* is too strong for 12 programs
  and ~400 lines. *"I built a demo MVP to make the architecture decisions
  concrete"* is right.
- Don't apologize for what's missing. He knows it's a demo. Frame everything
  not done as a deliberate scope decision, with a reason.
- Don't tell him what his paper says. He wrote it. Reference it, build on it,
  but don't paraphrase it back at him for more than a sentence.
- Don't say "I think your paper is great." If he wants compliments, his mom
  is around.
