"""
TruthLens AI — Factual Claim Extraction & Decomposition
========================================================

Takes a news text, user claim, or draft answer and decomposes it into atomic,
independently verifiable factual claims. Generates search queries for each claim
to power the retrieval stage.
"""

import argparse
import json
import re
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from claimverify.contracts import Claim, Draft
from claimverify.llm import LLM
from claimverify.settings import add_config_argument, load_settings


@runtime_checkable
class Decomposer(Protocol):
    """Interface: splits text/draft into atomic claims."""

    def decompose(self, draft: Draft) -> list[Claim]: ...


class LLMDecomposer:
    """Implementation: an LLM in JSON mode that decomposes claims and generates search queries."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def decompose(self, draft: Draft) -> list[Claim]:
        return decompose_into_claims(self.llm, draft)


DECOMPOSITION_SYSTEM_PROMPT = """You are an expert news fact-checker and claim extractor for TruthLens AI.
Your task is to decompose a news article, claim, or text into a list of atomic, verifiable factual claims.

Strict rules:
1. Each claim must be a SINGLE verifiable factual assertion — no compound sentences.
   If a sentence contains multiple facts, split them into distinct atomic claims.
2. Each claim must be completely self-contained: resolve pronouns and relative references ("they", "this company", "it")
   to the exact named entities based on the context.
3. Ignore subjective opinions, pure rhetoric, transition words ("Furthermore", "In conclusion"), or vague commentary.
4. For each claim, provide 1-2 targeted search queries in "search_queries" to look up supporting or refuting evidence on the web.
5. Label each claim's "kind":
   - "topic": verifiable assertion about the real-world event, finding, person, policy, or data.
   - "meta": meta-commentary about the text itself ("The article doesn't state...", "Sources say").
6. If the text itself quotes a specific source (e.g. "[reuters.com]", "[paper.pdf]"), include it in "cited_source". Otherwise null.

Respond ONLY with valid JSON in this exact structure:
{
  "claims": [
    {
      "id": "c1",
      "claim": "Full self-contained factual assertion.",
      "cited_source": null,
      "kind": "topic",
      "search_queries": ["targeted search query 1", "search query 2"]
    }
  ]
}
"""


def decompose_into_claims(llm: LLM, draft: Draft) -> list[Claim]:
    """Sends the draft/news to the LLM and returns validated atomic claims."""
    # If the text is extremely short (e.g. a single direct sentence), we can still run decomposition
    raw_text = llm.chat(
        [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": f"News / Claim text to decompose:\n\n{draft.text}"},
        ],
        json_mode=True,
    )

    try:
        raw = raw_text.strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        clean_json = match.group(0) if match else raw
        parsed = json.loads(clean_json)
    except Exception:
        # Fallback to single atomic claim from the draft
        return [
            Claim(
                id="c1",
                claim=draft.question.strip() if draft.question else draft.text.strip(),
                cited_source=None,
                kind="topic",
                search_queries=[(draft.question or draft.text).strip()[:100]],
            )
        ]

    claims_data = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(claims_data, list) or not claims_data:
        # If no claims extracted, create one from the original text
        return [
            Claim(
                id="c1",
                claim=draft.text.strip(),
                cited_source=None,
                kind="topic",
                search_queries=[draft.text.strip()[:100]],
            )
        ]

    # Validate and normalize IDs
    validated: list[Claim] = []
    for idx, c in enumerate(claims_data, start=1):
        if isinstance(c, dict):
            if "id" not in c or not c["id"]:
                c["id"] = f"c{idx}"
            if "search_queries" not in c:
                c["search_queries"] = [c.get("claim", "")[:100]]
            try:
                validated.append(Claim.model_validate(c))
            except ValidationError:
                continue

    return validated or [
        Claim(
            id="c1",
            claim=draft.text.strip(),
            cited_source=None,
            kind="topic",
            search_queries=[draft.text.strip()[:100]],
        )
    ]


def print_claims(claims: list[Claim]):
    print(f"\n{len(claims)} claims extracted:\n")
    for c in claims:
        source = c.cited_source or "(no source cited)"
        label = "" if c.kind == "topic" else " (meta: not verified)"
        print(f"  [{c.id}] {c.claim}{label}")
        print(f"        queries: {c.search_queries}")
        print(f"        source:  {source}\n")


def main():
    from claimverify.factory import build_decomposer

    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--draft_file", type=str, required=True,
                        help="Draft as JSON (output of drafting --save_to).")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Optional path to save the claims as JSON.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    decomposer = build_decomposer(settings)

    with open(args.draft_file, "r", encoding="utf-8") as f:
        draft = Draft.model_validate_json(f.read())

    print(f"Decomposing into atomic claims ({settings.answering.decomposer.model})...")
    claims = decomposer.decompose(draft)
    print_claims(claims)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"claims": [c.model_dump() for c in claims]}, f, ensure_ascii=False, indent=2)
        print(f"Claims saved to {args.save_json}")


if __name__ == "__main__":
    main()
