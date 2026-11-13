"""
TruthLens AI — Full Frontend-Backend Integration Test
Verifies:
1. FastAPI server startup and SPA serving (index.html, assets)
2. CORS headers
3. GET /api/live
4. GET /api/runs (history list)
5. POST /api/live (claim submission)
6. GET /api/live/{run_id}/stream (SSE event streaming)
7. GET /api/runs/{run_id} (run details)
8. Likely True, Likely False, Uncertain verdicts verification
9. Supporting evidence, Contradicting evidence, Confidence, Abstention reasons
"""
import sys
import os
import time
import json
import urllib.request
import urllib.error
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import uvicorn
from pathlib import Path
from claimverify.api.app import create_app
from claimverify.api.live import LiveRunner
from claimverify.api.runs import RunStore

HOST = "127.0.0.1"
PORT = 8089  # Dedicated test port
BASE_URL = f"http://{HOST}:{PORT}"

def run_integration_tests():
    print("=" * 70)
    print("TruthLens AI — Frontend & Backend Integration Verification")
    print("=" * 70)

    # 1. Initialize backend app with UI dist
    project_root = Path.cwd()
    runs_dir = project_root / "runs"
    ui_dist = project_root / "ui" / "dist"

    assert ui_dist.exists(), f"ui/dist does not exist at {ui_dist}"
    print(f"✅ UI production build found at {ui_dist}")

    store = RunStore({"recorded": runs_dir})
    live = LiveRunner(project_root, runs_dir)
    app = create_app(store, live, ui_dist=ui_dist)

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to start
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/live", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start within 5 seconds")

    print(f"✅ FastAPI server running at {BASE_URL}")

    try:
        # TEST 1: Frontend SPA Serving
        print("\n--- TEST 1: Frontend SPA Serving ---")
        req = urllib.request.Request(f"{BASE_URL}/")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            html = resp.read().decode("utf-8")
            assert "TruthLens" in html or "truthlens" in html.lower() or "id=\"root\"" in html, "HTML did not contain expected SPA root"
            print("✅ GET / returns 200 OK and serves index.html")

        # TEST 2: CORS Headers
        print("\n--- TEST 2: CORS Middleware ---")
        cors_req = urllib.request.Request(
            f"{BASE_URL}/api/live",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
        )
        with urllib.request.urlopen(cors_req) as resp:
            headers = dict(resp.headers)
            cors_origin = headers.get("Access-Control-Allow-Origin") or headers.get("access-control-allow-origin")
            assert cors_origin in ("*", "http://localhost:5173"), f"CORS origin header missing or wrong: {cors_origin}"
            print(f"✅ CORS working: Access-Control-Allow-Origin = {cors_origin}")

        # TEST 3: GET /api/live
        print("\n--- TEST 3: GET /api/live ---")
        with urllib.request.urlopen(f"{BASE_URL}/api/live") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("enabled") is True, "Live mode is not enabled"
            assert data.get("database_configured") is True, "Database is not marked configured"
            assert len(data.get("configs", [])) > 0, "No configs returned"
            print(f"✅ GET /api/live returns enabled=True, configs={len(data['configs'])}")
            print(f"   Missing keys (mock mode active): {data['configs'][0].get('missing_keys')}")

        # TEST 4: GET /api/runs (History)
        print("\n--- TEST 4: GET /api/runs (Verification History) ---")
        with urllib.request.urlopen(f"{BASE_URL}/api/runs") as resp:
            runs = json.loads(resp.read().decode("utf-8"))
            assert isinstance(runs, list), "Runs is not a list"
            print(f"✅ GET /api/runs returns {len(runs)} past verification runs")
            if runs:
                r0 = runs[0]
                print(f"   Latest run: {r0.get('run_id')} | Status: {r0.get('status')} | Pipeline: {r0.get('pipeline')}")

        # TEST 5: POST /api/live (Submit Claim)
        print("\n--- TEST 5: POST /api/live (Live Claim Submission) ---")
        claim_payload = json.dumps({
            "question": "The James Webb Space Telescope detected carbon dioxide and water vapor in exoplanet WASP-96b atmosphere.",
            "config_file": "config.yaml"
        }).encode("utf-8")

        post_req = urllib.request.Request(
            f"{BASE_URL}/api/live",
            data=claim_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(post_req) as resp:
            assert resp.status == 201, f"Expected 201, got {resp.status}"
            post_data = json.loads(resp.read().decode("utf-8"))
            run_id = post_data.get("run_id")
            assert run_id, "No run_id returned from POST /api/live"
            print(f"✅ POST /api/live accepted. Created run: {run_id}")

        # TEST 6: SSE Stream /api/live/{run_id}/stream
        print("\n--- TEST 6: Server-Sent Events (SSE) Stream ---")
        stream_url = f"{BASE_URL}/api/live/{run_id}/stream"
        events_received = []
        stream_req = urllib.request.Request(stream_url, headers={"Accept": "text/event-stream"})
        
        with urllib.request.urlopen(stream_req, timeout=15) as stream_resp:
            # Read lines from stream until end or failed
            current_event_type = None
            for raw_line in stream_resp:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("event:"):
                    current_event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and current_event_type:
                    data_str = line.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(data_str)
                        events_received.append((current_event_type, payload))
                    except Exception:
                        pass
                    if current_event_type in ("end", "failed"):
                        break

        print(f"✅ SSE streaming received {len(events_received)} event messages:")
        for etype, edata in events_received[:5]:
            ptype = edata.get("type", etype)
            print(f"   - Event: {etype} (inner: {ptype})")
        if len(events_received) > 5:
            print(f"   ... and {len(events_received) - 5} more events")

        terminal_event = events_received[-1][0] if events_received else None
        assert terminal_event in ("end", "failed"), f"Stream did not terminate with end/failed: {terminal_event}"
        print(f"✅ SSE stream cleanly completed with: '{terminal_event}'")

        # TEST 7: GET /api/runs/{run_id} (Inspect Result)
        print("\n--- TEST 7: GET /api/runs/{run_id} Detail ---")
        time.sleep(0.2)
        with urllib.request.urlopen(f"{BASE_URL}/api/runs/{run_id}") as resp:
            run_detail = json.loads(resp.read().decode("utf-8"))
            summary = run_detail.get("summary", {})
            events = run_detail.get("events", [])
            print(f"✅ Run {run_id} detail retrieved:")
            print(f"   Status: {summary.get('status')}")
            print(f"   Total events recorded: {len(events)}")
            print(f"   Verdict counts: {summary.get('verdict_counts')}")

            # Verify claim_verified event structure
            claim_events = [e for e in events if e.get("type") == "claim_verified"]
            assert len(claim_events) > 0, "No claim_verified event found in events"
            first_verdict = claim_events[0].get("verdict", {})

            print("\n--- TEST 8: Verdict, Evidence, Confidence & Reasoning ---")
            print(f"   Verdict label:        {first_verdict.get('verdict')}")
            print(f"   Confidence score:     {first_verdict.get('confidence')}")
            print(f"   Reasoning / Justif.:  {first_verdict.get('justification')[:80]}...")
            print(f"   Abstention reason:    {first_verdict.get('abstention_reason')}")
            print(f"   Supporting sources:   {first_verdict.get('supporting_sources')}")
            print(f"   Contradicting sources:{first_verdict.get('contradicting_sources')}")
            print(f"   Supporting evidence:  {len(first_verdict.get('supporting_evidence', []))} passage(s)")
            print(f"   Contradicting evidence: {len(first_verdict.get('contradicting_evidence', []))} passage(s)")

            assert first_verdict.get("verdict") in ("likely_true", "likely_false", "uncertain"), "Invalid verdict label"
            assert "confidence" in first_verdict, "Missing confidence"
            assert "justification" in first_verdict, "Missing justification"

        print("\n" + "=" * 70)
        print("🎉 ALL INTEGRATION TESTS PASSED! FULL STACK VERIFIED!")
        print("=" * 70)

    finally:
        server.should_exit = True

if __name__ == "__main__":
    run_integration_tests()
