"""
TruthLens AI — Judge Evaluation on Benchmarks & Gold Set
=========================================================

Evaluates verifier performance against ground truth benchmark claims.
Computes standard NLP classification metrics ported from Evidence-Aware-LLM:
- Accuracy
- Precision (per-class & macro)
- Recall (per-class & macro)
- Macro-F1
- Evidence Relevance
- Uncertainty / Abstention Rate
- Confusion Matrix
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from claimverify.contracts import (
    TRUTHLENS_VERDICTS,
    VERDICT_LABELS,
    Verdict,
    normalize_verdict,
)
from claimverify.evaluation import CATEGORIES, GoldClaim
from claimverify.indexing.cleaning import content_signature

COMPLEMENTARY = "complementary"


@dataclass(frozen=True)
class Scored:
    gold: GoldClaim
    verdict: Verdict
    correct: bool
    proof_seen: bool | None
    sources_ok: bool


def proof_seen(gold: GoldClaim, verdict: Verdict) -> bool | None:
    if not gold.evidence:
        return None
    read = {p.filename: [] for p in verdict.evidence}
    for p in verdict.evidence:
        read[p.filename].append(content_signature(p.text))
    return all(
        any(content_signature(e.quote) in text for text in read.get(e.filename, []))
        for e in gold.evidence
    )


def sources_ok(gold: GoldClaim, verdict: Verdict) -> bool:
    pro = {e.filename for e in gold.evidence if e.stance == "supports"}
    con = {e.filename for e in gold.evidence if e.stance == "contradicts"}
    exp_norm = normalize_verdict(gold.expected)
    if exp_norm == "uncertain":
        return True
    return pro <= set(verdict.supporting_sources) and con <= set(verdict.contradicting_sources)


def score(gold: list[GoldClaim], verdicts: dict[str, Verdict]) -> list[Scored]:
    scored = []
    for g in gold:
        if g.id in verdicts:
            v = verdicts[g.id]
            is_match = (v.verdict == g.expected) or (normalize_verdict(v.verdict) == normalize_verdict(g.expected))
            scored.append(
                Scored(
                    gold=g,
                    verdict=v,
                    correct=is_match,
                    proof_seen=proof_seen(g, v),
                    sources_ok=sources_ok(g, v),
                )
            )
    return scored


def compute_evaluation_metrics(scored: list[Scored]) -> dict[str, Any]:
    """Calculates Accuracy, Precision, Recall, Macro-F1, and Uncertainty Rate."""
    total = len(scored)
    if total == 0:
        return {}

    correct = sum(1 for s in scored if s.correct)
    accuracy = round(correct / total, 4)

    classes = ["likely_true", "likely_false", "uncertain"]
    class_metrics = {}

    for c in classes:
        tp = sum(1 for s in scored if normalize_verdict(s.gold.expected) == c and normalize_verdict(s.verdict.verdict) == c)
        fp = sum(1 for s in scored if normalize_verdict(s.gold.expected) != c and normalize_verdict(s.verdict.verdict) == c)
        fn = sum(1 for s in scored if normalize_verdict(s.gold.expected) == c and normalize_verdict(s.verdict.verdict) != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    macro_precision = round(sum(m["precision"] for m in class_metrics.values()) / len(classes), 4)
    macro_recall = round(sum(m["recall"] for m in class_metrics.values()) / len(classes), 4)
    macro_f1 = round(sum(m["f1"] for m in class_metrics.values()) / len(classes), 4)

    uncertain_count = sum(1 for s in scored if normalize_verdict(s.verdict.verdict) == "uncertain")
    uncertainty_rate = round(uncertain_count / total, 4)

    scores_list = [p.score for s in scored for p in s.verdict.evidence if hasattr(p, "score")]
    avg_relevance = round(sum(scores_list) / len(scores_list), 4) if scores_list else 0.0

    return {
        "total_cases": total,
        "correct_cases": correct,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "uncertainty_rate": uncertainty_rate,
        "evidence_relevance": avg_relevance,
        "class_breakdown": class_metrics,
    }


def cause(s: Scored) -> str:
    if s.correct:
        return "–"
    if s.verdict.verdict == "error":
        return "judge answer unusable"
    if s.proof_seen is False:
        return "retrieval: proof not read"
    return "judge"


def report(scored: list[Scored], judge: str, usage_note: str) -> str:
    metrics = compute_evaluation_metrics(scored)
    n = len(scored)

    lines = [
        "# TruthLens AI — Verifier Evaluation",
        "",
        f"Verifier / Judge: `{judge}`. Evaluated across {n} claims.",
        f"**Accuracy: {metrics.get('accuracy', 0):.2%} ({metrics.get('correct_cases', 0)}/{n})**",
        f"**Macro-F1: {metrics.get('macro_f1', 0):.4f}** | **Macro-Precision: {metrics.get('macro_precision', 0):.4f}** | **Macro-Recall: {metrics.get('macro_recall', 0):.4f}**",
        f"**Uncertainty Rate: {metrics.get('uncertainty_rate', 0):.2%}** | **Average Evidence Relevance: {metrics.get('evidence_relevance', 0):.4f}**",
        "",
        "## Performance by Class",
        "",
        "| Verdict Class | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
    ]

    for c, m in metrics.get("class_breakdown", {}).items():
        lines.append(f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")

    lines += [
        "",
        "## Confusion Matrix (rows: expected, columns: predicted)",
        "",
        "| expected \\ predicted | likely_true | likely_false | uncertain | error |",
        "|---|---|---|---|---|",
    ]

    classes = ["likely_true", "likely_false", "uncertain"]
    all_preds = ["likely_true", "likely_false", "uncertain", "error"]
    counts = Counter((normalize_verdict(s.gold.expected), normalize_verdict(s.verdict.verdict)) for s in scored)

    for exp in classes:
        row = [str(counts.get((exp, pred), "0")) for pred in all_preds]
        lines.append(f"| {exp} | " + " | ".join(row) + " |")

    lines += ["", "## Detailed Claim Results", ""]
    lines += [
        "| Id | Expected | Verdict | Confidence | Proof Read | Cause | Justification |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in scored:
        seen = {True: "yes", False: "**no**", None: "–"}.get(s.proof_seen, "–")
        just = s.verdict.justification.replace("|", "/").replace("\n", " ")
        conf = getattr(s.verdict, "confidence", 0.5)
        lines.append(
            f"| {s.gold.id} | {s.gold.expected} | "
            f"{'✓' if s.correct else '✗'} {s.verdict.verdict} | {conf:.0%} | {seen} | "
            f"{cause(s)} | {just[:150]} |"
        )

    return "\n".join(lines) + "\n"
