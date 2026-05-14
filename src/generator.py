"""
Generator Module
================

Final stage of the RAG pipeline: takes the retrieved chunks and the user query
and produces a grounded, cited answer.

Design choices:

1. ANTI-HALLUCINATION PROMPT — the model is told to (a) use only the retrieved
   context, (b) cite each claim with the program name, and (c) explicitly say
   "I don't have current information on this" rather than guess. Maps to
   Zhang's "accuracy" pillar.

2. FRESHNESS-AWARE PROMPT — the prompt includes each chunk's last_verified date
   and instructs the model to flag any time-sensitive claim (deadlines, current
   acceptance status) with a "verify on website" note. Maps to Zhang's
   "recency" pillar.

3. INLINE CITATIONS — every claim is tagged with [Program Name]. The UI parses
   these and renders them as clickable references. Maps to Zhang's
   "traceability" pillar.

4. OFFLINE FALLBACK — if no API key is set, returns a template-based answer
   built directly from the retrieved chunks. This means the demo works on a
   plane with no WiFi. Critical for interview reliability.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


SYSTEM_PROMPT = """You are an expert advisor on the Toronto health-tech \
entrepreneurship ecosystem. Your job is to help prospective founders \
understand which accelerators, grants, and funding programs are relevant to \
them.

You will be given a user question and a set of CONTEXT chunks, each describing \
one program. You must follow these rules without exception:

1. GROUND EVERY CLAIM. Use only information from the CONTEXT chunks. If the \
CONTEXT does not contain the information needed to answer, say so explicitly: \
"I don't have current information on this — please check the program's \
website." Do not guess.

2. CITE EVERY CLAIM INLINE. After each factual statement, add the source in \
square brackets, e.g., "[H2i]" or "[MaRS Discovery District]". Use the exact \
program name from the CONTEXT.

3. FLAG TIME-SENSITIVE CLAIMS. If you mention a deadline, current cohort \
status, or "accepting applications", append "(verify on website — last \
verified [date])". The last_verified date is in each CONTEXT chunk.

4. BE CONCISE AND ACTIONABLE. The user is a founder, not a researcher. Give \
them a clear next step. If multiple programs fit, briefly compare them on the \
dimensions the user cares about (stage, sector, equity model, etc.).

5. DO NOT INVENT URLS, PEOPLE, OR DOLLAR AMOUNTS that are not in the CONTEXT."""


def _format_context(retrieved: list) -> str:
    """
    Build the CONTEXT block fed to the LLM. We include the metadata fields
    (especially last_verified) inline so the model can cite freshness.
    """
    blocks = []
    for i, hit in enumerate(retrieved, start=1):
        meta = hit["metadata"]
        block = (
            f"--- CONTEXT {i} ---\n"
            f"Program name: {meta['name']}\n"
            f"Organization: {meta['organization']}\n"
            f"URL: {meta['url']}\n"
            f"Stage: {meta['stage']}\n"
            f"Sector: {meta['sector']}\n"
            f"Funding type: {meta['funding_type']}\n"
            f"Last verified: {meta['last_verified']}\n"
            f"Full description:\n{hit['document']}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def _offline_answer(query: str, retrieved: list) -> str:
    """
    Template-based fallback for when no API key is set. Lets the UI demo
    work without external dependencies. Output is intentionally less polished
    than the LLM version — the demo person should be able to tell the
    difference at a glance.
    """
    if not retrieved:
        return (
            "**[Offline mode — no Anthropic API key detected.]**\n\n"
            "No relevant programs were retrieved from the index for this query."
        )

    lines = [
        "**[Offline mode — no Anthropic API key detected. Showing structured retrieval output.]**\n",
        f"Based on your question, here are the top {len(retrieved)} matching programs:\n",
    ]
    for i, hit in enumerate(retrieved, 1):
        m = hit["metadata"]
        lines.append(f"### {i}. {m['name']} [{m['name']}]")
        lines.append(f"- **Organization:** {m['organization']}")
        lines.append(f"- **Stage:** {m['stage']}")
        lines.append(f"- **Sector:** {m['sector']}")
        lines.append(f"- **Funding type:** {m['funding_type']}")
        lines.append(f"- **URL:** {m['url']}")
        lines.append(
            f"- *(verify on website — last verified {m['last_verified']})*"
        )
        lines.append("")
    return "\n".join(lines)


def generate(query: str, retrieved: list) -> dict:
    """
    Run the generator. Returns a dict with the answer text and a flag
    indicating whether the LLM or offline fallback was used.
    """
    context = _format_context(retrieved)

    user_message = (
        f"User question: {query}\n\n"
        f"CONTEXT (retrieved program records):\n{context}\n\n"
        f"Answer the user's question following the rules in the system prompt."
    )

    if not ANTHROPIC_API_KEY:
        return {
            "answer": _offline_answer(query, retrieved),
            "mode": "offline_fallback",
        }

    try:
        # Anthropic import is inside the try so the offline path doesn't need
        # the SDK installed
        from anthropic import Anthropic

        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text
        return {"answer": answer_text, "mode": "anthropic_claude"}
    except Exception as e:
        # Graceful degradation: if the API call fails (network, rate limit,
        # bad key), fall through to offline mode rather than crashing the UI.
        return {
            "answer": (
                f"**[API call failed — falling back to offline mode. Error: {e}]**\n\n"
                + _offline_answer(query, retrieved)
            ),
            "mode": "fallback_after_error",
        }


if __name__ == "__main__":
    # Smoke test (only meaningful if a retriever has indexed seed data)
    from retriever import Retriever

    r = Retriever()
    q = "I'm a U of T med student with a digital health idea. Where do I start?"
    hits = r.retrieve_hybrid(q, k=3)
    result = generate(q, hits)
    print(f"Mode: {result['mode']}\n")
    print(result["answer"])
