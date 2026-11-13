"""
Live runs — running the real pipeline from the UI
=================================================

The pipeline runs in a thread, with the existing code: factory.build_answering
and pipeline.run_single_question. One more sink (BroadcastSink) sends each
event to the subscribed browsers (Server-Sent Events, see app.py).
"""

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai

from claimverify.answering.pipeline import build_markdown_report, describe_llm_stage, run_single_question
from claimverify.components.embedding import Embedder
from claimverify.config import get_db_connection
from claimverify.events import Event, RunFinished, RunStarted
from claimverify.factory import build_answering, build_embedder
from claimverify.llm import LLMCallError
from claimverify.reporting import ConsoleSink, RecorderSink, Run
from claimverify.settings import Settings, load_settings

DEFAULT_TOKENS_PER_QUESTION = 15_000
DAILY_LIMIT_NOTE = "TruthLens AI supports Google Gemini free tier, Groq free tier, and built-in offline simulation mode."


class LiveBusy(RuntimeError):
    """A live run is already in progress."""


class LiveUnavailable(RuntimeError):
    """The run cannot start: a message for the user."""


@dataclass
class LiveRun:
    run_id: str
    question: str
    status: str = "running"
    error: dict | None = None
    backlog: list[str] = field(default_factory=list)
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def publish(self, kind: str, data: dict) -> None:
        payload = f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        with self.lock:
            self.backlog.append(payload)
            dead = []
            for loop, queue in self.subscribers:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, payload)
                except RuntimeError:
                    dead.append((loop, queue))
            for item in dead:
                self.subscribers.remove(item)

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[list[str], asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue()
        with self.lock:
            backlog = list(self.backlog)
            self.subscribers.append((loop, queue))
        return backlog, queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self.lock:
            self.subscribers = [s for s in self.subscribers if s[1] is not queue]

    @property
    def done(self) -> bool:
        return self.status in ("finished", "failed")


class BroadcastSink:
    def __init__(self, run: LiveRun):
        self.run = run

    def handle(self, event: Event) -> None:
        self.run.publish("pipeline", event.model_dump(mode="json"))

    def emit(self, event: Event) -> None:
        self.handle(event)


class LiveRunner:
    def __init__(self, project_root: Path, runs_dir: Path):
        self.project_root = project_root
        self.runs_dir = runs_dir
        self.db_url = os.getenv("DB_URL", "local")
        self.runs: dict[str, LiveRun] = {}
        self._embedders: dict[tuple, Embedder] = {}
        self._lock = threading.Lock()

    def config_files(self) -> list[str]:
        files = [p for p in [self.project_root / "config.yaml"] if p.exists()]
        files += sorted((self.project_root / "experiments").glob("*.yaml"))
        return [str(p.relative_to(self.project_root)) for p in files]

    def info(self, recorded_summaries: list[dict]) -> dict:
        configs = []
        for name in self.config_files():
            try:
                settings = load_settings(self.project_root / name)
            except Exception as e:
                configs.append({"file": name, "error": str(e).splitlines()[0]})
                continue
            configs.append({
                "file": name,
                "models": _models(settings),
                "missing_keys": _missing_keys(settings)
            })
        running = next((r for r in self.runs.values() if not r.done), None)
        return {
            "configs": configs,
            "database_configured": True,  # Always available via PostgreSQL or local vector store fallback
            "estimate": {
                "tokens_per_question": DEFAULT_TOKENS_PER_QUESTION,
                "basis_runs": len(recorded_summaries),
                "daily_limit_note": DAILY_LIMIT_NOTE,
            },
            "running": running.run_id if running else None,
        }

    def start(self, question: str, config_file: str) -> LiveRun:
        question = question.strip()
        if not question:
            raise LiveUnavailable("Type a news headline, article, or claim first.")
        if config_file not in self.config_files():
            raise LiveUnavailable(f"Unknown config file: {config_file}")

        with self._lock:
            if any(not r.done for r in self.runs.values()):
                raise LiveBusy("A live verification run is already in progress. Wait for it to finish.")

            settings = load_settings(self.project_root / config_file)
            conn = get_db_connection(self.db_url)

            try:
                stages = build_answering(settings, conn, embedder=self._embedder(settings))
            except Exception as e:
                if hasattr(conn, "close"):
                    conn.close()
                raise LiveUnavailable(f"Failed to initialize verification stages: {e}") from e

            run = Run("answering", runs_dir=self.runs_dir, sinks=[])
            live = LiveRun(run.run_id, question)
            run.sinks = [ConsoleSink(), RecorderSink(run.events_path), BroadcastSink(live)]
            self.runs[run.run_id] = live

        threading.Thread(
            target=self._run,
            name=f"live-{run.run_id}",
            daemon=True,
            args=(live, run, stages, settings, config_file, conn),
        ).start()
        return live

    def _run(self, live: LiveRun, run: Run, stages, settings: Settings, config_file: str, conn) -> None:
        start = time.perf_counter()
        try:
            run.emit(RunStarted, config=settings.model_dump(),
                     inputs={"config_file": config_file, "questions": [live.question]})
            result = run_single_question(live.question, stages, run, question_index=1)
            metadata = _report_metadata(run, settings, config_file, stages)
            with open(run.dir / "report.json", "w", encoding="utf-8") as f:
                json.dump({"metadata": metadata, "results": [result]}, f, ensure_ascii=False, indent=2)
            with open(run.dir / "report.md", "w", encoding="utf-8") as f:
                f.write(build_markdown_report([result], metadata))
            run.emit(RunFinished, seconds=time.perf_counter() - start,
                     summary={"questions": 1, "run_dir": str(run.dir)})
            live.status = "finished"
            live.publish("end", {"status": "finished"})
        except Exception as e:
            live.error = describe_failure(e)
            live.status = "failed"
            live.publish("failed", live.error)
        finally:
            stages.meter.on_wait = None
            if hasattr(conn, "close"):
                conn.close()

    def _embedder(self, settings: Settings) -> Embedder:
        key = (settings.embedding.type, settings.embedding.model, settings.embedding.device)
        if key not in self._embedders:
            self._embedders[key] = build_embedder(settings)
        return self._embedders[key]


def _models(settings: Settings) -> dict[str, str]:
    a = settings.answering
    return {
        "draft": a.drafter.model,
        "decompose": a.decomposer.model,
        "verify": a.verifier.judge.model,
    }


def _missing_keys(settings: Settings) -> list[str]:
    needed = set()
    a = settings.answering
    for stage in (a.drafter, a.decomposer, a.verifier.judge):
        if stage.type != "mock":
            needed.add(settings.providers[stage.provider].api_key_env)
    return [name for name in sorted(needed) if not os.getenv(name)]


def _report_metadata(run: Run, settings: Settings, config_file: str, stages) -> dict:
    a = settings.answering
    return {
        "timestamp": run.run_id,
        "config_file": config_file,
        "models": {
            "draft": describe_llm_stage(settings, a.drafter),
            "decompose": describe_llm_stage(settings, a.decomposer),
            "verify": describe_llm_stage(settings, a.verifier.judge),
        },
        "embedding_model": settings.embedding.model,
        "config": settings.model_dump(),
        "usage": {role: u.model_dump() for role, u in stages.meter.totals_by_role().items()},
    }


def describe_failure(error: Exception) -> dict:
    if isinstance(error, LLMCallError):
        return {"kind": "llm_call_error", "message": str(error)}
    if isinstance(error, openai.AuthenticationError):
        return {"kind": "auth_error", "message": "Invalid API key. Check your .env file."}
    return {"kind": type(error).__name__, "message": str(error)}
