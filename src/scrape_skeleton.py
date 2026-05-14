"""
Weekly Scraper — SKELETON
==========================

The JD requires a semantic scraper that runs on a routine basis (e.g., weekly)
to keep the index fresh. Live scraping is intentionally NOT wired into the demo
— sites change, blockers fire, and the demo should never break in an interview.

This file documents the intended structure. Production would:

1. Iterate over a configured list of source URLs (anchored to the seed data).
2. Fetch using Crawl4AI (mentioned by name in the JD) — its LLM-friendly
   structured-extraction output is far better than raw HTML for this task.
3. Validate the extracted JSON against the seed schema.
4. Diff against the existing index and write only changed records.
5. Bump last_verified on unchanged records (so freshness reflects the check,
   not just the last edit).

Why Crawl4AI specifically:
- Returns clean markdown by default, which embeds far better than raw HTML.
- Supports structured extraction with Pydantic-style schemas — useful when
  scraping accelerator pages with consistent fields (program name, deadline,
  eligibility).
- Async-first, which matters when sweeping 50+ sources weekly.

To actually wire this up:
- `pip install crawl4ai`
- Schedule via cron, GitHub Actions on a schedule trigger, or a Cloud Run job.
- Write to a staging collection, then atomically swap collections after
  validation passes — so a broken scrape never replaces a working index.
"""

import json
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = PROJECT_ROOT / "data" / "seed_programs.json"


# Schema each scraped record must conform to before ingestion. Real
# implementation would use Pydantic for validation.
REQUIRED_FIELDS = {
    "id",
    "name",
    "organization",
    "url",
    "stage",
    "sector",
    "funding_type",
    "geography",
    "eligibility_summary",
    "description",
    "what_makes_it_distinctive",
    "last_verified",
    "source_note",
}


def list_source_urls() -> list:
    """Pull source URLs from the existing seed data — those become the scrape targets."""
    with open(SEED_FILE, "r") as f:
        programs = json.load(f)
    return [{"id": p["id"], "url": p["url"]} for p in programs]


def validate_record(record: dict) -> tuple:
    """
    Validate a scraped record against the schema. Returns (is_valid, errors).
    Returning a structured result means failures can be logged for review
    rather than silently dropped.
    """
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        return False, [f"Missing required field: {f}" for f in missing]
    if not record["url"].startswith("http"):
        return False, ["URL is malformed"]
    return True, []


def scrape_one(url: str) -> dict:
    """
    STUB. Real implementation:

        from crawl4ai import AsyncWebCrawler
        from crawl4ai.extraction_strategy import LLMExtractionStrategy

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=LLMExtractionStrategy(
                    provider="anthropic/claude-haiku-4-5",
                    schema=ProgramSchema.model_json_schema(),
                    instruction="Extract this program's details using the schema.",
                ),
            )
            return json.loads(result.extracted_content)
    """
    raise NotImplementedError(
        "Live scraping is intentionally not wired in the demo. "
        "See module docstring for production design."
    )


def run_weekly_refresh(dry_run: bool = True):
    """
    Outline of the weekly refresh job. Dry-run is the default so this file
    can be imported without side effects.
    """
    sources = list_source_urls()
    print(f"[scrape] Would refresh {len(sources)} sources on {date.today()}.")
    if dry_run:
        print("[scrape] Dry run — no actual fetches performed.")
        return
    # Real implementation: loop, fetch, validate, diff, write.


if __name__ == "__main__":
    run_weekly_refresh(dry_run=True)
