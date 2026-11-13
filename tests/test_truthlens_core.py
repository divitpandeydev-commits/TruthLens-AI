"""
TruthLens AI Core Tests — runs without API keys or PostgreSQL.
Run from the claim-verify-rag-main directory:

    $env:PYTHONPATH="src"
    python tests/test_truthlens_core.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_contracts():
    from claimverify.contracts import Verdict, normalize_verdict, VERDICT_DISPLAY_LABELS
    # TruthLens verdicts — round-trip
    v = Verdict(claim_id="1", claim="test", verdict="likely_true",
                justification="ok", confidence=0.9, verifier="test")
    assert normalize_verdict(v.verdict) == "likely_true"
    # Legacy label mapping
    assert normalize_verdict("supported") == "likely_true"
    assert normalize_verdict("Supported") == "likely_true"
    assert normalize_verdict("contradicted") == "likely_false"
    assert normalize_verdict("Refuted") == "likely_false"
    assert normalize_verdict("contested") == "uncertain"
    assert normalize_verdict("unverifiable") == "uncertain"
    assert normalize_verdict("Uncertain") == "uncertain"
    # Display labels
    assert VERDICT_DISPLAY_LABELS["likely_true"] == "Likely True"
    assert VERDICT_DISPLAY_LABELS["likely_false"] == "Likely False"
    assert VERDICT_DISPLAY_LABELS["uncertain"] == "Uncertain"
    print("✅ test_contracts passed")


def test_local_embedder():
    from claimverify.components.embedding import LocalEmbedder
    emb = LocalEmbedder()
    # embed_query API
    vec = emb.embed_query("hello world")
    assert len(vec) == 768, f"Expected 768-dim, got {len(vec)}"
    # Convenience alias
    vec2 = emb.embed("hello world")
    assert vec == vec2, "embed() should alias embed_query()"
    # Unit normalised
    norm = sum(x**2 for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4, f"Not unit-normalised: {norm}"
    # Same text → same vector (deterministic)
    vec3 = emb.embed_query("hello world")
    assert vec == vec3, "Same text should yield same hash-based vector"
    # Different text → different vector
    vec_other = emb.embed_query("the earth orbits the sun")
    assert vec != vec_other, "Different texts should yield different vectors"
    print("✅ test_local_embedder passed")


def test_local_store():
    from claimverify.components.store import LocalStore
    from claimverify.contracts import Chunk

    store = LocalStore.__new__(LocalStore)  # skip _load() to avoid file I/O
    store.path = None
    store.chunks = []

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", filename="test.md",
              source_type="md", chunk_index=0,
              text="vaccines do not alter dna",
              embedding=[0.9] + [0.0] * 767),
        Chunk(chunk_id="c2", doc_id="d1", filename="test.md",
              source_type="md", chunk_index=1,
              text="the earth is flat",
              embedding=[0.0, 0.9] + [0.0] * 766),
    ]
    # Manually populate chunks (bypass save)
    store.chunks = [
        {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "filename": c.filename,
            "source_type": c.source_type, "chunk_index": c.chunk_index,
            "text": c.text, "embedding": c.embedding
        }
        for c in chunks
    ]
    # nearest() should return c1 (closest to [0.9, 0, ...])
    results = store.nearest([1.0] + [0.0] * 767, top_k=1)
    assert len(results) == 1, "Expected 1 result"
    assert results[0].chunk_id == "c1", f"Expected c1, got {results[0].chunk_id}"
    # cosine_similarity works
    sim = store.cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 1e-6
    print("✅ test_local_store passed")


def test_mock_retrieval():
    from claimverify.answering.retrieval import MockSearchRetriever
    r = MockSearchRetriever()
    passages = r.retrieve("vaccine DNA safety")
    assert len(passages) > 0, "Expected at least one mock passage"
    assert all("[MOCK" in p.text for p in passages), "All passages should be labelled as MOCK"
    print(f"✅ test_mock_retrieval passed ({len(passages)} passages)")


def test_deduplication():
    from claimverify.answering.retrieval import deduplicate_passages
    from claimverify.contracts import Passage

    p1 = Passage(filename="a.txt", source_type="txt", chunk_index=0,
                 text="vaccines are safe and effective per WHO guidelines", score=0.9)
    p2 = Passage(filename="b.txt", source_type="txt", chunk_index=0,
                 text="vaccines are safe and effective per WHO guidelines", score=0.8)  # duplicate
    p3 = Passage(filename="c.txt", source_type="txt", chunk_index=0,
                 text="new research on mRNA technology published in Nature", score=0.7)

    deduped = deduplicate_passages([p1, p2, p3])
    assert len(deduped) == 2, f"Expected 2 after dedup, got {len(deduped)}"
    print("✅ test_deduplication passed")


def test_settings_load():
    from claimverify.settings import load_settings
    s = load_settings(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    assert s.answering.verifier.type == "evidence_aware"
    assert s.embedding.type == "local"
    # Drafter/decomposer use provider field
    assert s.answering.drafter.provider in ("gemini", "groq")
    print("✅ test_settings_load passed — verifier:", s.answering.verifier.type)


def test_factory_build():
    from claimverify.settings import load_settings
    from claimverify.factory import build_answering
    s = load_settings(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    stages = build_answering(s, conn="local")
    assert stages.verifier is not None
    assert type(stages.verifier).__name__ == "EvidenceAwareVerifier"
    print("✅ test_factory_build passed — verifier:", type(stages.verifier).__name__)


def test_verdict_synthesis():
    from claimverify.answering.pipeline import synthesize_overall_verdict

    class V:
        def __init__(self, verdict, confidence):
            self.verdict = verdict
            self.confidence = confidence

    # All likely_true → likely_true
    verdict, conf, summary = synthesize_overall_verdict([V("likely_true", 0.9), V("likely_true", 0.8)])
    assert verdict == "likely_true", f"Expected likely_true, got {verdict}"

    # Mix of true and false → uncertain (contradictory evidence)
    verdict2, conf2, summary2 = synthesize_overall_verdict([V("likely_true", 0.9), V("likely_false", 0.8)])
    assert verdict2 == "uncertain", f"Expected uncertain (mixed), got {verdict2}"

    # All uncertain → uncertain
    verdict3, conf3, summary3 = synthesize_overall_verdict([V("uncertain", 0.5)])
    assert verdict3 == "uncertain", f"Expected uncertain, got {verdict3}"

    # Only likely_false → likely_false
    verdict4, conf4, summary4 = synthesize_overall_verdict([V("likely_false", 0.85)])
    assert verdict4 == "likely_false", f"Expected likely_false, got {verdict4}"

    # Empty → uncertain
    verdict5, conf5, _ = synthesize_overall_verdict([])
    assert verdict5 == "uncertain", f"Expected uncertain (empty), got {verdict5}"

    print("✅ test_verdict_synthesis passed")


def test_evaluation_metrics():
    from claimverify.judge_eval import compute_evaluation_metrics

    class FakeGold:
        expected = "supported"  # will be normalized to "likely_true"

    class FakeVerdictObj:
        verdict = "supported"   # same
        evidence = []
        confidence = 0.9

    class FakeScored:
        gold = FakeGold()
        verdict = FakeVerdictObj()
        correct = True
        proof_seen = None
        sources_ok = True

    metrics = compute_evaluation_metrics([FakeScored()])
    assert metrics["accuracy"] == 1.0, f"Expected accuracy=1.0, got {metrics['accuracy']}"
    # Macro-F1 with 3 classes and only 1 sample in 'likely_true':
    # likely_true: P=1, R=1, F1=1; likely_false: P=0, R=0, F1=0; uncertain: P=0, R=0, F1=0
    # macro_f1 = (1 + 0 + 0) / 3 ≈ 0.3333
    assert metrics["macro_f1"] == round(1/3, 4), f"Expected macro_f1≈0.3333, got {metrics['macro_f1']}"
    # With all 3 classes represented, macro_f1 should be 1.0
    class FakeGold2:
        expected = "likely_false"
    class FakeVerdictObj2:
        verdict = "likely_false"; evidence = []; confidence = 0.9
    class FakeScored2:
        gold = FakeGold2(); verdict = FakeVerdictObj2()
        correct = True; proof_seen = None; sources_ok = True
    class FakeGold3:
        expected = "uncertain"
    class FakeVerdictObj3:
        verdict = "uncertain"; evidence = []; confidence = 0.5
    class FakeScored3:
        gold = FakeGold3(); verdict = FakeVerdictObj3()
        correct = True; proof_seen = None; sources_ok = True
    metrics2 = compute_evaluation_metrics([FakeScored(), FakeScored2(), FakeScored3()])
    assert metrics2["accuracy"] == 1.0
    assert metrics2["macro_f1"] == 1.0, f"Expected macro_f1=1.0 with all classes correct, got {metrics2['macro_f1']}"
    print("✅ test_evaluation_metrics passed — single-class macro_f1:", metrics["macro_f1"],
          " all-class macro_f1:", metrics2["macro_f1"])


def test_mock_llm():
    """Test the mock LLM path without real API credentials."""
    from claimverify.llm import LLM
    import openai
    # Build a mock client (won't be called)
    client = openai.OpenAI(api_key="NONE", base_url="http://localhost:9999/v1")
    llm = LLM(role="verifier", model="mock", base_url="http://localhost:9999/v1",
              client=client, is_mock=True)
    response = llm._mock_response([{"role": "user", "content": "decompose: climate change is real"}], json_mode=True)
    assert isinstance(response, str), "Mock response should return string"
    assert len(response) > 0, "Mock response should not be empty"
    print("✅ test_mock_llm passed")


if __name__ == "__main__":
    print("=" * 60)
    print("TruthLens AI — Core Test Suite")
    print("=" * 60)
    errors = []
    tests = [
        test_contracts,
        test_local_embedder,
        test_local_store,
        test_mock_retrieval,
        test_deduplication,
        test_settings_load,
        test_factory_build,
        test_verdict_synthesis,
        test_evaluation_metrics,
        test_mock_llm,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"❌ {t.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            errors.append(t.__name__)

    print("=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} test(s): {', '.join(errors)}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED ✅")
        print("TruthLens AI is ready for development!")
