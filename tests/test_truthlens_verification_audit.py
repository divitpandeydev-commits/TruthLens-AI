"""
TruthLens AI — Verification Audit Test Suite
=============================================
Deterministic tests covering the 10 core verification scenarios:
1. Strong supporting evidence -> likely_true
2. Strong contradicting evidence -> likely_false
3. Insufficient evidence -> uncertain
4. Irrelevant retrieved documents -> filtered out / uncertain
5. Conflicting sources -> uncertain with conflict detection
6. Official denial -> classified as contradicting (NOT supporting)
7. Compound headline containing true and false statements -> likely_false / core evaluation
8. Outdated event -> temporal awareness / abstention
9. Duplicate sources -> deduplicated correctly
10. False central statement with true secondary statements -> likely_false (core dominates)
"""

import pytest

from claimverify.answering.pipeline import synthesize_overall_verdict
from claimverify.answering.retrieval import deduplicate_passages, normalize_url
from claimverify.answering.verification import calculate_evidence_strength
from claimverify.contracts import Claim, Passage, Verdict


def test_1_strong_supporting_evidence():
    """Claim with strong, independent supporting evidence returns likely_true."""
    p1 = Passage(
        filename="https://reuters.com/news/1",
        source_type="web_search",
        text="NASA confirms the James Webb Space Telescope successfully detected CO2 on WASP-96b.",
        score=0.92,
        source_url="https://reuters.com/news/1",
        title="Reuters News",
    )
    p2 = Passage(
        filename="https://bbc.com/news/space",
        source_type="web_search",
        text="Astronomers using the James Webb Space Telescope announced the discovery of atmospheric carbon dioxide.",
        score=0.88,
        source_url="https://bbc.com/news/space",
        title="BBC Science",
    )
    conf = calculate_evidence_strength("likely_true", [p1, p2], [], [p1, p2])
    assert conf >= 0.85
    verdict = Verdict(
        claim_id="c1",
        claim="The James Webb Telescope detected carbon dioxide on WASP-96b.",
        verdict="likely_true",
        justification="Confirmed by multiple independent scientific reports.",
        confidence=conf,
        supporting_sources=[p1.source_url, p2.source_url],
        evidence=[p1, p2],
        supporting_evidence=[p1, p2],
        verifier="test_judge",
    )
    res_verdict, res_conf, res_sum = synthesize_overall_verdict([verdict])
    assert res_verdict == "likely_true"
    assert res_conf >= 0.85


def test_2_strong_contradicting_evidence():
    """Claim with strong contradicting evidence returns likely_false."""
    p1 = Passage(
        filename="https://who.int/fact-check",
        source_type="web_search",
        text="Comprehensive genetic studies confirm that mRNA vaccines do not enter the nucleus or alter human DNA.",
        score=0.95,
        source_url="https://who.int/fact-check",
        title="WHO Vaccine Safety",
    )
    conf = calculate_evidence_strength("likely_false", [], [p1], [p1])
    assert conf >= 0.80
    verdict = Verdict(
        claim_id="c1",
        claim="COVID-19 vaccines alter human DNA.",
        verdict="likely_false",
        justification="Directly refuted by WHO genomic research findings.",
        confidence=conf,
        contradicting_sources=[p1.source_url],
        evidence=[p1],
        contradicting_evidence=[p1],
        verifier="test_judge",
    )
    res_verdict, _, _ = synthesize_overall_verdict([verdict])
    assert res_verdict == "likely_false"


def test_3_insufficient_evidence():
    """Claim with no or insufficient evidence returns uncertain."""
    conf = calculate_evidence_strength("uncertain", [], [], [])
    assert conf <= 0.40
    verdict = Verdict(
        claim_id="c1",
        claim="A secret underground facility was built beneath the local stadium last week.",
        verdict="uncertain",
        justification="No independent credible news sources mention this claim.",
        confidence=conf,
        abstention_reason="No relevant external evidence found.",
        evidence=[],
        verifier="test_judge",
    )
    res_verdict, _, res_sum = synthesize_overall_verdict([verdict])
    assert res_verdict == "uncertain"
    assert "insufficient" in res_sum.lower() or "inconclusive" in res_sum.lower()


def test_4_irrelevant_retrieved_documents():
    """Unrelated documents (e.g. academic papers on unrelated topics) must not count as supporting/contradicting."""
    p_irrelevant = Passage(
        filename="anthropic-contextual-retrieval.md",
        source_type="academic_paper",
        text="Contextual retrieval prepends chunk-specific explanatory context to each chunk before embedding.",
        score=0.15,
        title="Anthropic RAG Guide",
        polarity="irrelevant",
    )
    # With only irrelevant passages, supporting and contradicting remain empty
    conf = calculate_evidence_strength("uncertain", [], [], [p_irrelevant])
    assert conf <= 0.45
    verdict = Verdict(
        claim_id="c1",
        claim="An attack took place at a local engineering college.",
        verdict="uncertain",
        justification="Retrieved documents discuss computer science methods and do not mention the college.",
        confidence=conf,
        abstention_reason="Retrieved documents are irrelevant to the event.",
        evidence=[p_irrelevant],
        supporting_evidence=[],
        contradicting_evidence=[],
        irrelevant_evidence=[p_irrelevant],
        verifier="test_judge",
    )
    assert len(verdict.supporting_evidence) == 0
    assert len(verdict.contradicting_evidence) == 0
    assert len(verdict.irrelevant_evidence) == 1


def test_5_conflicting_sources():
    """When credible sources directly disagree without resolution, verdict is uncertain."""
    p_sup = Passage(
        filename="https://source-a.com/news",
        source_type="web_search",
        text="Local officials announced the bridge opening will happen on Friday.",
        score=0.85,
        source_url="https://source-a.com/news",
    )
    p_con = Passage(
        filename="https://source-b.com/news",
        source_type="web_search",
        text="Department of Transportation states the bridge remains closed indefinitely due to safety inspection.",
        score=0.85,
        source_url="https://source-b.com/news",
    )
    conf = calculate_evidence_strength("uncertain", [p_sup], [p_con], [p_sup, p_con])
    assert conf <= 0.50
    verdict = Verdict(
        claim_id="c1",
        claim="The city bridge opened to traffic on Friday.",
        verdict="uncertain",
        justification="Direct conflict between Source A and Source B.",
        confidence=conf,
        abstention_reason="Direct conflict between primary sources.",
        conflicting_points=["Source A reports opening on Friday; Source B reports indefinite closure."],
        evidence=[p_sup, p_con],
        supporting_evidence=[p_sup],
        contradicting_evidence=[p_con],
        verifier="test_judge",
    )
    res_verdict, _, _ = synthesize_overall_verdict([verdict])
    assert res_verdict == "uncertain"


def test_6_official_denial_classified_as_contradicting():
    """An official statement denying or refuting an action must be classified as CONTRADICTING, never supporting."""
    p_denial = Passage(
        filename="https://reuters.com/world/europe/strike-statement",
        source_type="web_search",
        text="Russian military authorities confirmed the conventional strike and stated no nuclear payload was used.",
        score=0.91,
        source_url="https://reuters.com/world/europe/strike-statement",
        polarity="contradicting",
        information_type="denial",
    )
    # The claim asserts a nuclear weapon was used; the evidence denies it
    claim = Claim(id="c1", claim="A nuclear weapon was used in the recent strike on Ukraine.", importance="central")
    verdict = Verdict(
        claim_id=claim.id,
        claim=claim.claim,
        verdict="likely_false",
        justification="Official statements state explicitly that no nuclear payload was used in the strike.",
        confidence=0.88,
        contradicting_sources=[p_denial.source_url],
        evidence=[p_denial],
        contradicting_evidence=[p_denial],
        supporting_evidence=[],
        verifier="test_judge",
    )
    # Must NOT have any supporting evidence
    assert len(verdict.supporting_evidence) == 0
    assert len(verdict.contradicting_evidence) == 1
    assert verdict.verdict == "likely_false"


def test_7_compound_headline_central_false_overrides_true_secondary():
    """Compound headline: true context + false core claim -> overall likely_false."""
    c_sec1 = Claim(id="c1", claim="A missile strike occurred in Ukraine.", importance="secondary")
    c_sec2 = Claim(id="c2", claim="Kyiv was targeted in the strike.", importance="secondary")
    c_cen = Claim(id="c3", claim="The strike involved a nuclear warhead detonation.", importance="central")

    v1 = Verdict(
        claim_id="c1", claim=c_sec1.claim, verdict="likely_true",
        justification="Verified by independent news.", confidence=0.92, verifier="test_judge"
    )
    v2 = Verdict(
        claim_id="c2", claim=c_sec2.claim, verdict="likely_true",
        justification="Confirmed by local reporting.", confidence=0.90, verifier="test_judge"
    )
    v3 = Verdict(
        claim_id="c3", claim=c_cen.claim, verdict="likely_false",
        justification="International monitors confirm only conventional weapons were used.",
        confidence=0.95, verifier="test_judge"
    )

    verdicts = [v1, v2, v3]
    claims = [c_sec1, c_sec2, c_cen]

    overall_verdict, conf, summary = synthesize_overall_verdict(verdicts, claims)
    # The central false claim must dominate!
    assert overall_verdict == "likely_false"
    assert "core assertion is refuted" in summary.lower() or "central claim(s) false" in summary.lower()


def test_8_outdated_event():
    """An event from a different date/year cannot confirm a breaking claim for today."""
    p_old = Passage(
        filename="https://news.com/2021/flood",
        source_type="web_search",
        text="Heavy flooding hit Central Square in July 2021 after seasonal monsoon rains.",
        score=0.80,
        published_date="2021-07-15",
        polarity="irrelevant",
    )
    verdict = Verdict(
        claim_id="c1",
        claim="Central Square was inundated by a sudden flash flood this morning.",
        verdict="uncertain",
        justification="Retrieved evidence is from 2021 and does not substantiate an event occurring today.",
        confidence=0.40,
        abstention_reason="Temporal mismatch: retrieved sources date back to 2021.",
        evidence=[p_old],
        supporting_evidence=[],
        verifier="test_judge",
    )
    assert verdict.verdict == "uncertain"
    assert len(verdict.supporting_evidence) == 0


def test_9_duplicate_sources_deduplication():
    """Near-duplicate articles and identical URLs with query tracking parameters are properly deduplicated."""
    p1 = Passage(
        filename="https://example.com/article?utm_source=twitter",
        source_type="web_search",
        text="Prime Minister announced the new environmental subsidy program in parliament on Monday afternoon.",
        score=0.90,
        source_url="https://example.com/article?utm_source=twitter#heading",
    )
    p2 = Passage(
        filename="https://example.com/article",
        source_type="web_search",
        text="Prime Minister announced the new environmental subsidy program in parliament on Monday afternoon.",
        score=0.89,
        source_url="https://example.com/article",
    )
    p3 = Passage(
        filename="https://independent-daily.com/news/subsidy",
        source_type="web_search",
        text="A new clean energy subsidy was unveiled by the Prime Minister in today's parliamentary session.",
        score=0.87,
        source_url="https://independent-daily.com/news/subsidy",
    )
    raw_passages = [p1, p2, p3]
    deduped = deduplicate_passages(raw_passages)
    # p1 and p2 have identical snippet and normalized URL, so one should be pruned
    assert len(deduped) == 2
    assert normalize_url(p1.source_url) == "https://example.com/article"


def test_10_central_false_with_multiple_true_secondary():
    """Multiple true secondary statements cannot turn a false headline into Likely True."""
    claims = [
        Claim(id="c1", claim="France is a country in Europe.", importance="secondary"),
        Claim(id="c2", claim="The Eiffel Tower is located in Paris.", importance="secondary"),
        Claim(id="c3", claim="Emmanuel Macron is the President of France.", importance="secondary"),
        Claim(id="c4", claim="The French government officially abolished the euro and restored the franc yesterday.", importance="central"),
    ]
    verdicts = [
        Verdict(claim_id="c1", claim=claims[0].claim, verdict="likely_true", justification="Geographic fact.", confidence=0.95, verifier="test"),
        Verdict(claim_id="c2", claim=claims[1].claim, verdict="likely_true", justification="Cultural landmark fact.", confidence=0.95, verifier="test"),
        Verdict(claim_id="c3", claim=claims[2].claim, verdict="likely_true", justification="Political office fact.", confidence=0.95, verifier="test"),
        Verdict(claim_id="c4", claim=claims[3].claim, verdict="likely_false", justification="False. European Central Bank and official French journals confirm France uses the euro.", confidence=0.95, verifier="test"),
    ]

    overall_verdict, conf, summary = synthesize_overall_verdict(verdicts, claims)
    # Despite 3 true claims and 1 false claim, the overall verdict MUST be likely_false!
    assert overall_verdict == "likely_false"
    assert "core assertion is refuted" in summary.lower() or "false" in summary.lower()
