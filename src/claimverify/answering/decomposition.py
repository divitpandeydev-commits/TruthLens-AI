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
2. Identify the "importance" of each claim relative to the headline/topic:
   - "central": The core, pivotal assertion of the headline (e.g. "A nuclear weapon was used in the strike"). Proving or refuting this directly confirms or debunks the headline.
   - "secondary": Contextual, supporting, or background facts (e.g. "An explosion occurred", "Kyiv is in Ukraine").
   At least one claim MUST be marked "central".
3. Each claim must be completely self-contained: resolve pronouns and relative references ("they", "this company", "it")
   to the exact named entities based on the context.
4. Ignore subjective opinions, pure rhetoric, transition words, or vague commentary.
5. For each claim, provide 1-2 targeted search queries in "search_queries" to look up specific supporting or refuting evidence on the web.
6. Label each claim's "kind":
   - "topic": verifiable assertion about the real-world event, finding, person, policy, or data.
   - "meta": meta-commentary about the text itself ("The article doesn't state...", "Sources say").
7. If the text itself quotes a specific source, include it in "cited_source". Otherwise null.

Respond ONLY with valid JSON in this exact structure:
{
  "claims": [
    {
      "id": "c1",
      "claim": "Full self-contained factual assertion.",
      "cited_source": null,
      "kind": "topic",
      "importance": "central",
      "search_queries": ["targeted search query 1", "search query 2"]
    }
  ]
}
"""


def decompose_into_claims(llm: LLM, draft: Draft) -> list[Claim]:
    """Sends the draft/news to the LLM and returns validated atomic claims."""
    parent_q = (draft.question or draft.text).strip()
    raw_text = llm.chat(
        [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Headline / Claim text to decompose:\n\n{draft.text}"},
        ],
        json_mode=True,
    )

    try:
        raw = raw_text.strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        clean_json = match.group(0) if match else raw
        parsed = json.loads(clean_json)
    except Exception:
        return [
            Claim(
                id="c1",
                claim=parent_q,
                cited_source=None,
                kind="topic",
                importance="central",
                parent_headline=parent_q,
                search_queries=[parent_q[:100]],
            )
        ]

    claims_data = parsed.get("claims") if isinstance(parsed, dict) else None
    if claims_data is not None and isinstance(claims_data, list) and len(claims_data) == 0:
        return []

    if not isinstance(claims_data, list) or not claims_data:
        return [
            Claim(
                id="c1",
                claim=parent_q,
                cited_source=None,
                kind="topic",
                importance="central",
                parent_headline=parent_q,
                search_queries=[parent_q[:100]],
            )
        ]

    validated: list[Claim] = []
    has_central = False
    for idx, c in enumerate(claims_data, start=1):
        if isinstance(c, dict):
            if "id" not in c or not c["id"]:
                c["id"] = f"c{idx}"
            if "search_queries" not in c or not c["search_queries"]:
                c["search_queries"] = [c.get("claim", parent_q)[:100]]
            if c.get("importance") not in ("central", "secondary"):
                c["importance"] = "central" if idx == 1 else "secondary"
            if c.get("importance") == "central":
                has_central = True
            c["parent_headline"] = parent_q
            try:
                validated.append(Claim.model_validate(c))
            except ValidationError:
                continue

    # Ensure at least one claim is central
    if validated and not has_central:
        validated[0].importance = "central"

    return validated or [
        Claim(
            id="c1",
            claim=parent_q,
            cited_source=None,
            kind="topic",
            importance="central",
            parent_headline=parent_q,
            search_queries=[parent_q[:100]],
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
