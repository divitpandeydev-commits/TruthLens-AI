"""
Drafting an answer / news summary — Step B
==========================================

Flow: news/question -> retrieval PER DOCUMENT -> a grounded answer or summary
written by the drafter, citing the sources it uses.
"""

import argparse
from typing import Protocol, runtime_checkable

from claimverify.answering.retrieval import format_evidence
from claimverify.config import add_db_url_argument, get_db_connection
from claimverify.contracts import Draft, Passage
from claimverify.llm import LLM
from claimverify.settings import add_config_argument, load_settings


@runtime_checkable
class Drafter(Protocol):
    """Interface: writes an answer/summary to the question from the passages."""

    def draft(self, question: str, passages: list[Passage]) -> Draft: ...


class LLMDrafter:
    """Implementation: an LLM (role "draft" of llm.py) with SYSTEM_PROMPT below."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def draft(self, question: str, passages: list[Passage]) -> Draft:
        return generate_draft_answer(self.llm, question, passages)


SYSTEM_PROMPT = """You are an objective news research assistant for TruthLens AI. Your task is to
synthesize a clear, grounded factual summary answering the user query or evaluating the claim
using ONLY the provided source passages.

Rules:
1. Ground every substantive statement in the provided passages.
2. Cite sources using bracketed references like [Source: filename | chunk #N] where applicable.
3. If the provided passages do not contain sufficient evidence or if sources disagree, explicitly state so.
4. Keep the summary objective, analytical, and concise.
"""


def generate_draft_answer(llm: LLM, question: str, passages: list[Passage]) -> Draft:
    user_prompt = f"""Question / Claim to investigate:
{question}

Source passages:

{format_evidence(passages)}
"""
    answer_text = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return Draft(question=question, text=answer_text, passages=passages)


def main():
    from claimverify.factory import build_drafter, build_embedder, build_retriever

    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    add_db_url_argument(parser)
    parser.add_argument("--query", type=str, required=True,
                        help="Question or claim to investigate.")
    parser.add_argument("--save_to", type=str, default=None,
                        help="Optional path to save the Draft as JSON.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    drafter = build_drafter(settings)

    conn = get_db_connection(args.db_url)
    retriever = build_retriever(settings.answering.retriever, build_embedder(settings), conn,
                                "answering.retriever")
    passages = retriever.retrieve(args.query)
    if hasattr(conn, "close"):
        conn.close()

    docs_covered = sorted({p.filename for p in passages})
    print(f"\n{len(passages)} chunks retrieved from {len(docs_covered)} source(s):")
    for doc in docs_covered:
        print(f"  - {doc}")

    print(f"\nWriting the draft summary ({settings.answering.drafter.model})...\n")
    draft = drafter.draft(args.query, passages)

    print("=" * 100)
    print(f"QUERY: {args.query}")
    print("=" * 100)
    print(draft.text)
    print("=" * 100)

    if args.save_to:
        with open(args.save_to, "w", encoding="utf-8") as f:
            f.write(draft.model_dump_json(indent=2))
        print(f"\nDraft saved to {args.save_to}")


if __name__ == "__main__":
    main()
