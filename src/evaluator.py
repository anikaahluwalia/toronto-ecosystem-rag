"""
Evaluator
=========

A lightweight evaluation harness inspired by the RAGAs framework (Es et al., 2024)
that Ng, Matsuba & Zhang (NEJM AI, 2024) cite. We compute approximations of:

- Context Precision (proxy): % of retrieved chunks judged relevant by an LLM rubric
- Groundedness (proxy): % of answer claims that map to a retrieved chunk

Plus a domain-specific metric:
- Freshness Awareness: does the answer flag time-sensitive claims with a
  "verify on website" note?

This is a DEMO evaluator on 6 hand-crafted queries with expected sources. The
real project would build a larger query set from H2i mentor logs and founder
interviews — that's part of the research design.

Run:
    python src/evaluator.py
"""

import re
from retriever import Retriever
from generator import generate


# Test queries with expected source program IDs.
# In production these would be authored by domain experts (H2i mentors) and
# the ground truth would be curated, not inferred.
TEST_SET = [
    {
        "query": "I'm a U of T grad student wanting to commercialize my health research. What's a good first program?",
        "expected_sources": ["lab2market", "h2i"],
    },
    {
        "query": "Where can I get non-dilutive funding for a Canadian health-tech R&D project?",
        "expected_sources": ["nrc-irap", "mitacs-accelerate"],
    },
    {
        "query": "I'm building a cell therapy company. Who can help with manufacturing?",
        "expected_sources": ["ccrm"],
    },
    {
        "query": "I'm a clinician with a digital health app. Which accelerator should I look at first?",
        "expected_sources": ["h2i", "mars-discovery"],
    },
    {
        "query": "Wet lab space for a biotech startup in Toronto without giving up equity?",
        "expected_sources": ["jlabs-toronto"],
    },
    {
        "query": "Ontario-specific support for a Series A-ready health-science company?",
        "expected_sources": ["obio", "feddev-ontario"],
    },
]


def context_precision(retrieved_ids: list, expected_ids: list) -> float:
    """% of retrieved chunks that are in the expected set."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids if r in expected_ids)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids: list, expected_ids: list) -> float:
    """% of expected chunks that were retrieved."""
    if not expected_ids:
        return 0.0
    hits = sum(1 for e in expected_ids if e in retrieved_ids)
    return hits / len(expected_ids)


def citation_coverage(answer_text: str, retrieved_names: list) -> float:
    """
    Proxy for groundedness: count how many cited names in the answer appear in
    the retrieved set. (A real groundedness check would parse claims and verify
    each against the source text — this is the cheap version.)
    """
    # Find all bracketed citations like [H2i] or [MaRS Discovery District]
    citations = re.findall(r"\[([^\]]+)\]", answer_text)
    if not citations:
        return 0.0
    matched = sum(
        1 for c in citations if any(c.strip().lower() in n.lower() for n in retrieved_names)
    )
    return matched / len(citations)


def has_freshness_flag(answer_text: str) -> bool:
    """Does the answer surface freshness uncertainty?"""
    flags = ["verify on website", "last verified", "check the program"]
    return any(f in answer_text.lower() for f in flags)


def evaluate_mode(retriever_fn, mode_name: str) -> dict:
    """Run the test set against a given retriever function (naive vs hybrid)."""
    rows = []
    for case in TEST_SET:
        retrieved = retriever_fn(case["query"], k=3)
        retrieved_ids = [r["id"] for r in retrieved]
        retrieved_names = [r["metadata"]["name"] for r in retrieved]

        gen = generate(case["query"], retrieved)
        answer = gen["answer"]

        rows.append(
            {
                "query": case["query"],
                "mode": mode_name,
                "retrieved_ids": retrieved_ids,
                "expected_ids": case["expected_sources"],
                "context_precision": context_precision(
                    retrieved_ids, case["expected_sources"]
                ),
                "recall_at_3": recall_at_k(
                    retrieved_ids, case["expected_sources"]
                ),
                "citation_coverage": citation_coverage(answer, retrieved_names),
                "freshness_flag": has_freshness_flag(answer),
                "gen_mode": gen["mode"],
            }
        )

    avg_precision = sum(r["context_precision"] for r in rows) / len(rows)
    avg_recall = sum(r["recall_at_3"] for r in rows) / len(rows)
    avg_citation = sum(r["citation_coverage"] for r in rows) / len(rows)
    pct_freshness = sum(1 for r in rows if r["freshness_flag"]) / len(rows)

    return {
        "mode": mode_name,
        "rows": rows,
        "avg_context_precision": avg_precision,
        "avg_recall_at_3": avg_recall,
        "avg_citation_coverage": avg_citation,
        "pct_with_freshness_flag": pct_freshness,
    }


def main():
    r = Retriever()
    print("Running evaluation across Naïve and Hybrid modes...\n")

    naive_results = evaluate_mode(r.retrieve_naive, "naive")
    hybrid_results = evaluate_mode(r.retrieve_hybrid, "hybrid")

    print("=" * 60)
    print(f"{'Metric':<32} {'Naïve':>12} {'Hybrid':>12}")
    print("=" * 60)
    for key, label in [
        ("avg_context_precision", "Context Precision @ 3"),
        ("avg_recall_at_3", "Recall @ 3"),
        ("avg_citation_coverage", "Citation Coverage"),
        ("pct_with_freshness_flag", "% Answers w/ Freshness Flag"),
    ]:
        print(
            f"{label:<32} {naive_results[key]:>12.2%} {hybrid_results[key]:>12.2%}"
        )
    print("=" * 60)
    print(f"\nGeneration mode used: {naive_results['rows'][0]['gen_mode']}")
    print(
        "(Citation Coverage and Freshness Flag are only meaningful when an "
        "LLM is generating answers, not in offline mode.)"
    )


if __name__ == "__main__":
    main()
