"""
TruthLens AI — Full Verification Pipeline
==========================================

Runs the complete 11-step verification flow:
    1. Input news article or claim
    2. Extract important factual assertions
    3. Break complex assertions into atomic claims
    4. Generate targeted search queries
    5. Retrieve relevant external evidence (local corpus + Tavily web search)
    6. Identify supporting and contradicting evidence
    7. Remove duplicate evidence
    8. Rank evidence by relevance
    9. RAG context assembly
    10. Evidence-aware LLM verification with confidence scoring
    11. Synthesize overall verdict: Likely True, Likely False, or Uncertain
"""

import argparse
import json
import time

from claimverify.config import add_db_url_argument, get_db_connection
from claimverify.contracts import (
    TRUTHLENS_VERDICTS,
    VERDICT_ICONS,
    VERDICT_LABELS,
    ArticleVerification,
    claims_to_verify,
    normalize_verdict,
)
from claimverify.events import (
    ClaimsExtracted,
    ClaimVerified,
    DraftWritten,
    LLMWaiting,
    PassagesRetrieved,
    QuestionFinished,
    QuestionStarted,
    RunFinished,
    RunStarted,
)
from claimverify.factory import AnsweringStages, build_answering
from claimverify.reporting import Run
from claimverify.settings import LLMStageSettings, Settings, add_config_argument, load_settings


def synthesize_overall_verdict(verdicts: list, claims: list | None = None) -> tuple[str, float, str]:
    """Calculates overall article verdict, weighted confidence, and explanation,
    prioritizing central assertions so minor true context cannot mask a false core claim.
    """
    if not verdicts:
        return "uncertain", 0.50, "No verifiable factual claims were found in the input."

    # Build claim lookup by claim_id
    claim_importance: dict[str, str] = {}
    if claims:
        for c in claims:
            cid = c.id if hasattr(c, "id") else c.get("id", "")
            imp = getattr(c, "importance", "central") if hasattr(c, "importance") else c.get("importance", "central")
            if cid:
                claim_importance[cid] = imp

    central_verdicts = []
    secondary_verdicts = []

    for v in verdicts:
        cid = getattr(v, "claim_id", "") if not isinstance(v, dict) else v.get("claim_id", "")
        imp = claim_importance.get(cid, "central")
        raw_verdict = getattr(v, "verdict", "uncertain") if not isinstance(v, dict) else v.get("verdict", "uncertain")
        norm_v = normalize_verdict(raw_verdict)
        raw_conf = getattr(v, "confidence", 0.70) if not isinstance(v, dict) else v.get("confidence", 0.70)
        conf = float(raw_conf)
        item = {"verdict": norm_v, "confidence": conf, "raw": v}

        if imp == "central":
            central_verdicts.append(item)
        else:
            secondary_verdicts.append(item)

    # If no claims were marked central, treat all as central
    if not central_verdicts:
        central_verdicts = secondary_verdicts
        secondary_verdicts = []

    # Weighted confidence: central claims weight 3.0, secondary weight 1.0
    weighted_sum = sum(x["confidence"] * 3.0 for x in central_verdicts) + sum(x["confidence"] * 1.0 for x in secondary_verdicts)
    total_weights = (len(central_verdicts) * 3.0) + (len(secondary_verdicts) * 1.0)
    avg_conf = round(weighted_sum / total_weights, 2) if total_weights > 0 else 0.50

    central_false = sum(1 for x in central_verdicts if x["verdict"] == "likely_false")
    central_true = sum(1 for x in central_verdicts if x["verdict"] == "likely_true")
    central_uncertain = sum(1 for x in central_verdicts if x["verdict"] == "uncertain")

    secondary_false = sum(1 for x in secondary_verdicts if x["verdict"] == "likely_false")
    secondary_true = sum(1 for x in secondary_verdicts if x["verdict"] == "likely_true")

    # RULE 1: Core claim refutation dominates
    if central_false > 0 and central_true > 0 and not claims and central_false == central_true:
        verdict = "uncertain"
        summary = (
            f"Mixed evidence: {central_true} claim(s) supported and {central_false} claim(s) refuted. "
            "Contradictory evidence prevents a definitive verdict."
        )
    elif central_false > 0:
        verdict = "likely_false"
        if secondary_true > 0:
            summary = (
                f"Core assertion is refuted by verified evidence ({central_false} central claim(s) false). "
                f"Minor true context ({secondary_true} secondary fact(s)) does not validate the false central claim."
            )
        else:
            summary = f"Core statement is refuted by verified evidence ({central_false} claim(s) contradicted)."

    # RULE 2: Central assertion supported
    elif central_true > 0 and central_false == 0 and central_uncertain == 0:
        if secondary_false > 0:
            verdict = "uncertain"
            summary = (
                f"Central assertion is substantiated by verified sources, but {secondary_false} secondary detail(s) "
                "were contradicted, making the overall presentation misleading or contested."
            )
        else:
            verdict = "likely_true"
            summary = f"All {central_true} central assertion(s) were substantiated and supported by verified sources."

    # RULE 3: Central assertion partially supported or mixed
    elif central_true > 0 and central_uncertain > 0:
        verdict = "uncertain"
        summary = (
            f"Central assertion has mixed or incomplete verification ({central_true} supported, "
            f"{central_uncertain} uncertain). Insufficient conclusive evidence to confirm entire claim."
        )

    # RULE 4: Central assertion uncertain / inconclusive
    else:
        verdict = "uncertain"
        summary = "Evidence regarding the core assertion is insufficient, inconclusive, or conflicting across verified sources."

    return verdict, avg_conf, summary


def run_single_question(query: str, stages: AnsweringStages, run: Run,
                        question_index: int = 1) -> dict:
    """Runs the complete verification flow for an input news article or claim."""
    timings = {}
    t0 = time.perf_counter()
    stages.meter.take()

    stages.meter.on_wait = lambda w: run.emit(
        LLMWaiting, question_index=question_index, role=w.role, model=w.model,
        seconds=w.seconds, attempt=w.attempt, reason=w.reason
    )
    run.emit(QuestionStarted, question_index=question_index, question=query)

    # 1. Broad retrieval for context
    passages = stages.retriever.retrieve(query)
    docs_covered = sorted({p.filename for p in passages})
    t1 = time.perf_counter()
    run.emit(PassagesRetrieved, question_index=question_index, passages=passages, seconds=t1 - t0,
             usage=stages.meter.take())

    # 2. Grounded draft / news summary
    draft = stages.drafter.draft(query, passages)
    t2 = time.perf_counter()
    run.emit(DraftWritten, question_index=question_index, draft=draft, seconds=t2 - t1,
             usage=stages.meter.take())
    timings["retrieval_draft_seconds"] = round(t1 - t0, 2)
    timings["generation_draft_seconds"] = round(t2 - t1, 2)

    # 3. Extraction & decomposition into atomic verifiable claims
    claims = stages.decomposer.decompose(draft)
    t3 = time.perf_counter()
    run.emit(ClaimsExtracted, question_index=question_index, claims=claims, seconds=t3 - t2,
             usage=stages.meter.take())
    timings["decomposition_seconds"] = round(t3 - t2, 2)

    # 4. Evidence-aware verification loop for topic claims
    to_verify = claims_to_verify(claims)
    verdicts = []
    for position, verdict in enumerate(stages.verifier.verify(to_verify), 1):
        verdicts.append(verdict)
        run.emit(ClaimVerified, question_index=question_index, position=position,
                 total=len(to_verify), verdict=verdict, usage=stages.meter.take())
    t4 = time.perf_counter()
    timings["verification_seconds"] = round(t4 - t3, 2)
    timings["total_seconds"] = round(t4 - t0, 2)

    counts = dict.fromkeys(VERDICT_LABELS, 0)
    for v in verdicts:
        counts[v.verdict] += 1

    overall_verdict, overall_conf, overall_summary = synthesize_overall_verdict(verdicts, claims)

    run.emit(QuestionFinished, question_index=question_index, verdict_counts=counts,
             seconds=t4 - t0)

    return {
        "query": query,
        "overall_verdict": overall_verdict,
        "overall_confidence": overall_conf,
        "overall_summary": overall_summary,
        "draft_answer_docs_covered": docs_covered,
        "draft_answer": draft.text,
        "claims": [c.model_dump() for c in claims],
        "verdicts": [v.model_dump() for v in verdicts],
        "verdict_counts": counts,
        "timings": timings,
    }


def load_questions(args) -> list[str]:
    if args.query:
        return [args.query]

    with open(args.questions_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#")]


def build_markdown_report(run_results: list[dict], run_metadata: dict) -> str:
    lines = [
        f"# TruthLens AI Verification Report — {run_metadata['timestamp']}",
        "",
        f"- Verifier model: `{run_metadata['models']['verify']}`",
        f"- Embedding model: `{run_metadata['embedding_model']}`",
        f"- Questions/Claims evaluated: {len(run_results)}",
        "",
    ]

    for idx, r in enumerate(run_results, start=1):
        lines += [
            f"## Claim/Article #{idx}: {r['query']}",
            f"**Overall Verdict**: **{r.get('overall_verdict', 'uncertain').upper()}** "
            f"(Confidence: {r.get('overall_confidence', 0.5):.0%})",
            f"*{r.get('overall_summary', '')}*",
            "",
            "### Verified Atomic Claims:",
        ]
        for v in r["verdicts"]:
            icon = VERDICT_ICONS.get(v["verdict"], "•")
            lines.append(f"- [{icon}] **{v['verdict']}** ({v.get('confidence', 0.5):.0%}): {v['claim']}")
            lines.append(f"  *Reasoning*: {v['justification']}")
            if v.get("supporting_sources"):
                lines.append(f"  *Supporting sources*: {', '.join(v['supporting_sources'])}")
            if v.get("contradicting_sources"):
                lines.append(f"  *Contradicting sources*: {', '.join(v['contradicting_sources'])}")
            if v.get("abstention_reason"):
                lines.append(f"  *Abstention reason*: {v['abstention_reason']}")
        lines.append("")

    return "\n".join(lines)


def describe_llm_stage(settings: Settings, stage: LLMStageSettings) -> str:
    if stage.type == "mock":
        return "mock (offline mode)"
    p = settings.providers[stage.provider]
    return f"{stage.model} @ {stage.provider} ({p.base_url})"


def main():
    parser = argparse.ArgumentParser(description="TruthLens AI Verification Runner")
    add_config_argument(parser)
    add_db_url_argument(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="News claim or article to verify.")
    group.add_argument("--questions_file", type=str, help="File with one claim/query per line.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    a = settings.answering
    models = {
        "draft": describe_llm_stage(settings, a.drafter),
        "decompose": describe_llm_stage(settings, a.decomposer),
        "verify": describe_llm_stage(settings, a.verifier.judge),
    }
    for role, desc in models.items():
        print(f"TruthLens AI {role:<9}: {desc}")

    questions = load_questions(args)

    conn = get_db_connection(args.db_url)
    stages = build_answering(settings, conn)

    run = Run("answering")
    start = time.perf_counter()
    run.emit(RunStarted, config=settings.model_dump(),
             inputs={"config_file": str(args.config), "questions": questions})

    run_results = []
    try:
        for i, question in enumerate(questions, 1):
            run_results.append(run_single_question(question, stages, run, question_index=i))
    finally:
        if hasattr(conn, "close"):
            conn.close()

    run_metadata = {
        "timestamp": run.run_id,
        "config_file": str(args.config),
        "models": models,
        "embedding_model": settings.embedding.model,
        "config": settings.model_dump(),
        "usage": {role: u.model_dump() for role, u in stages.meter.totals_by_role().items()},
    }
    with open(run.dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": run_metadata, "results": run_results}, f, ensure_ascii=False, indent=2)
    with open(run.dir / "report.md", "w", encoding="utf-8") as f:
        f.write(build_markdown_report(run_results, run_metadata))

    run.emit(RunFinished, seconds=time.perf_counter() - start, summary={
        "questions": len(questions), "run_dir": str(run.dir),
    })
    print(f"\nVerification finished! Results saved to: {run.dir}")


if __name__ == "__main__":
    main()
