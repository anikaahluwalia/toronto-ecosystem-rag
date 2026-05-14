# Architecture Notes — Beyond the MVP

This demo implements Naïve RAG and Hybrid RAG. The two architectures the JD
specifically names as comparison targets — **CRAG** and **Self-Reflective RAG** —
are outlined here. Both are tractable extensions of the existing pipeline.

---

## CRAG (Corrective RAG)

**The problem CRAG solves for Toronto ecosystem data:** programs get
discontinued, deadlines change, new cohorts launch. Static vector retrieval will
confidently return a chunk about a program that no longer exists. CRAG adds a
**retrieval evaluator** that grades each retrieved chunk as Correct, Ambiguous,
or Incorrect, and falls back to **live web search** when retrieval is judged
insufficient.

**Implementation sketch:**

```
User query
   │
   ▼
Retriever (existing Hybrid module)
   │
   ▼
┌──────────────────────────────┐
│ Retrieval Evaluator          │   <- new module
│ - small LLM call per chunk   │
│ - score: {Correct, Ambig.,   │
│   Incorrect}                 │
└──────────────────────────────┘
   │
   ├── all Correct ────────► Generator (existing)
   │
   ├── all Incorrect ──────► Web search fallback (Tavily / SerpAPI)
   │                          → re-chunk → Generator
   │
   └── Ambiguous ──────────► Combine local + web → Generator
```

**Practical notes:**
- The evaluator can be a lightweight model (Haiku, GPT-4o-mini) — doesn't need
  to be the full generation model.
- Web fallback needs source-credibility weighting (a hit on `mars.ca` should
  outrank a Medium post), which is exactly the "relevance vs. credibility" issue
  Zhang's paper flags.
- Caching: web-fallback results should write back into the vector store with a
  short TTL so we don't re-fetch on every query.

---

## Self-Reflective RAG (Self-RAG)

**The problem Self-RAG solves:** sometimes retrieval isn't needed (general
knowledge question), sometimes it returns irrelevant junk, and sometimes the
generator hallucinates beyond what was retrieved. Self-RAG uses reflection
tokens — small classifier outputs from the LLM — to make these decisions
explicit at each step.

**Four reflection token types (from Asai et al., 2023):**

| Token | Decides |
|---|---|
| `[Retrieve]` | Is retrieval needed for this query at all? |
| `[IsRel]` | Is each retrieved passage relevant? |
| `[IsSup]` | Is the generated answer supported by the passages? |
| `[IsUse]` | Is the answer useful to the user? |

**Implementation sketch:**

```
User query
   │
   ▼
LLM emits [Retrieve] token
   │
   ├── No ──► Generate from parametric memory only
   │
   └── Yes ─► Retriever
                │
                ▼
              For each chunk:
              LLM emits [IsRel]
                │
                ▼
              Keep only Relevant chunks
                │
                ▼
              Generate draft answer
                │
                ▼
              LLM emits [IsSup] (groundedness)
                │
                ├── Fully supported ──► Return
                │
                └── Partially / not ──► Re-retrieve OR
                                         flag uncertainty in output
```

**Practical notes for entrepreneurship data:**
- The `[Retrieve]` decision matters here because some founder questions ("what
  is a SAFE?") don't need ecosystem-specific retrieval — they need general LLM
  knowledge. Naïve RAG over-retrieves.
- The `[IsSup]` check is the strongest guardrail against the failure mode of
  recommending a discontinued program. If the chunk is dated and the answer
  asserts the program is "currently accepting applications," `[IsSup]` should
  catch it.
- Self-RAG was originally trained with reflection tokens as part of the
  vocabulary. The cheap version is to **prompt** the model to emit them as JSON
  fields — works adequately with Claude / GPT-4 class models without fine-tuning.

---

## Comparing the three architectures

| Dimension | Naïve | CRAG | Self-RAG |
|---|---|---|---|
| Latency | Lowest | Medium (extra eval call) | Highest (multiple reflection calls) |
| Cost per query | Lowest | Medium | Highest |
| Handles stale index | Poorly | Well (web fallback) | Medium (flags but doesn't fix) |
| Handles ambiguous retrieval | Poorly | Well | Well |
| Implementation complexity | Low | Medium | High |
| Where it wins | Speed-critical paths | High-churn information | High-stakes / liability-sensitive answers |

**Hypothesis going into the comparison:**
- For *factual lookup* queries ("when is the next CDL cohort"), CRAG wins
  because freshness dominates.
- For *advisory* queries ("which accelerator should I apply to as a clinician
  founder"), Self-RAG wins because groundedness and uncertainty handling matter
  more than freshness.
- Naïve is the speed baseline and the cost baseline — it should win neither
  axis but will set the latency budget the other two must compete against.

This is the actual research question, and it's not obvious what the answer is —
which is what makes it worth running the experiment.
