"""
TruthLens AI — LLM Client & Mock Provider
==========================================

Supports:
1. Google Gemini API (via official OpenAI-compatible endpoint:
   https://generativelanguage.googleapis.com/v1beta/openai/)
2. Groq, Mistral, Ollama, OpenRouter, and any OpenAI-compatible provider.
3. Clearly labelled MockLLM / Offline Mode:
   Allows testing the entire verification and frontend pipeline without
   requiring API keys or paid services.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import openai
from openai import OpenAI

from claimverify.events import Usage


class LLMCallError(RuntimeError):
    """A call that is pointless (or impossible) to retry: the run stops."""


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    max_wait_seconds: float = 30
    timeout_seconds: float = 60


@dataclass(frozen=True)
class CallRecord:
    role: str
    tokens_in: int
    tokens_out: int
    attempts: int
    waited_seconds: float
    seconds: float


@dataclass(frozen=True)
class WaitNotice:
    role: str
    model: str
    seconds: float
    attempt: int
    reason: str


class UsageMeter:
    def __init__(self):
        self.records: list[CallRecord] = []
        self._cursor = 0
        self.on_wait: Callable[[WaitNotice], None] | None = None

    def record(self, call: CallRecord) -> None:
        self.records.append(call)

    def take(self) -> Usage:
        new, self._cursor = self.records[self._cursor:], len(self.records)
        return _sum(new)

    def totals_by_role(self) -> dict[str, Usage]:
        roles = dict.fromkeys(r.role for r in self.records)
        return {role: _sum([r for r in self.records if r.role == role]) for role in roles}


def _sum(records: list[CallRecord]) -> Usage:
    return Usage(
        calls=len(records),
        tokens_in=sum(r.tokens_in for r in records),
        tokens_out=sum(r.tokens_out for r in records),
        retries=sum(r.attempts - 1 for r in records),
        waited_seconds=round(sum(r.waited_seconds for r in records), 2),
        seconds=round(sum(r.seconds for r in records), 2),
    )


def _retry_after(error: openai.APIStatusError) -> float | None:
    header = error.response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error))
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


@cache
def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key or "mock-key", base_url=base_url, max_retries=0, timeout=timeout)


@dataclass(frozen=True)
class LLM:
    """A ready-to-use model for a given pipeline role."""
    role: str
    model: str
    base_url: str
    client: OpenAI
    retry: RetryPolicy = RetryPolicy()
    meter: UsageMeter = field(default_factory=UsageMeter)
    sleep: Callable[[float], None] = time.sleep
    is_mock: bool = False

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        if self.is_mock or not getattr(self.client, "api_key", None) or self.client.api_key == "mock-key":
            return self._mock_response(messages, json_mode)

        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        start, attempts, waited = time.perf_counter(), 0, 0.0
        while True:
            attempts += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
                break
            except openai.APIStatusError as e:
                # If provider (like Groq) fails on json_object response_format, fallback to plain text request
                if e.status_code == 400 and json_mode and "response_format" in kwargs:
                    kwargs = {}
                    continue
                if e.status_code == 413:
                    raise LLMCallError(
                        f"{self.role} ({self.model}): request larger than provider limit."
                    ) from e
                if e.status_code == 401:
                    raise LLMCallError(
                        f"{self.role} ({self.model}): Authentication failed (HTTP 401). Check your API key in .env."
                    ) from e
                if e.status_code != 429 and e.status_code < 500:
                    raise LLMCallError(
                        f"{self.role} ({self.model}): API error HTTP {e.status_code}: {e.message}"
                    ) from e
                wait = _retry_after(e)
                error, reason = e, "rate_limit" if e.status_code == 429 else "server_error"
            except openai.APIConnectionError as e:
                wait, error, reason = None, e, "connection"

            if wait is None:
                wait = min(2 ** (attempts - 1), self.retry.max_wait_seconds)
            if attempts > self.retry.max_retries:
                raise LLMCallError(
                    f"{self.role} ({self.model}): Connection/API failed after {attempts} attempts ({reason}): {error}"
                ) from error
            if wait > self.retry.max_wait_seconds:
                raise LLMCallError(
                    f"{self.role} ({self.model}): wait time {wait:.0f}s exceeds limit."
                ) from error

            if self.meter.on_wait:
                self.meter.on_wait(WaitNotice(self.role, self.model, wait, attempts, reason))
            self.sleep(wait)
            waited += wait

        usage = getattr(response, "usage", None)
        self.meter.record(CallRecord(
            role=self.role,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            attempts=attempts,
            waited_seconds=waited,
            seconds=time.perf_counter() - start,
        ))
        return response.choices[0].message.content

    def _mock_response(self, messages: list[dict], json_mode: bool) -> str:
        """Deterministic, clearly labelled mock responses for testing."""
        prompt = messages[-1].get("content", "")
        role = self.role.lower()

        # Simulate a small execution time and token usage
        time.sleep(0.05)
        self.meter.record(CallRecord(
            role=self.role,
            tokens_in=len(prompt) // 4,
            tokens_out=64,
            attempts=1,
            waited_seconds=0.0,
            seconds=0.05,
        ))

        if "decompos" in role or "atomic claims" in prompt.lower():
            # Decomposition mock - extract user assertion
            extracted = "The article presents verifiable factual assertions."
            for line in prompt.splitlines():
                stripped = line.strip()
                if stripped and not stripped.lower().startswith("news / claim text to decompose") and len(stripped) > 5:
                    extracted = stripped
                    break
            return json.dumps({
                "claims": [
                    {
                        "claim": extracted,
                        "cited_source": None,
                        "kind": "topic",
                        "search_queries": [extracted[:100]],
                    }
                ]
            })

        if "verif" in role or "judge" in role or "verdict" in prompt.lower():
            # Verification judge mock
            # Extract the actual claim line if present
            claim_text = ""
            for line in prompt.splitlines():
                stripped = line.strip()
                if stripped.startswith('"') and stripped.endswith('"'):
                    claim_text = stripped.strip('"').lower()
                    break
            if not claim_text:
                claim_text = prompt.lower()

            # Discriminate based on the claim semantics
            if any(k in claim_text for k in ["draining", "cooling", "flat earth", "5g", "conspiracy", "hoax", "fake"]):
                verdict = "likely_false"
                reason = "[MOCK / OFFLINE VERIFIER] Evidence refutes the claim; scientific consensus directly contradicts this assertion."
                con_src = ["https://reuters.com/fact-check/solar-energy-grid"]
                sup_src = []
                conf = 0.94
                abstain = None
            elif any(k in claim_text for k in ["james webb", "wasp-96b", "exoplanet", "vaccine", "semantic", "rag", "retrieval"]):
                verdict = "likely_true"
                reason = "[MOCK / OFFLINE VERIFIER] Verified against peer-reviewed documentation and observation data; corroborated."
                sup_src = ["https://nasa.gov/jwst/exoplanet-atmosphere"]
                con_src = []
                conf = 0.92
                abstain = None
            else:
                verdict = "uncertain"
                reason = "[MOCK / OFFLINE VERIFIER] Retrieved evidence is insufficient or inconclusive to confirm or deny the statement."
                sup_src = []
                con_src = []
                conf = 0.45
                abstain = "Insufficient corroborating evidence found in offline test corpus."

            return json.dumps({
                "verdict": verdict,
                "confidence": conf,
                "justification": f"[MOCK VERIFICATION - OFFLINE SIMULATION] {reason}",
                "supporting_sources": sup_src,
                "contradicting_sources": con_src,
                "abstention_reason": abstain,
            })

        # Drafting mock - preserve question
        question = ""
        for line in prompt.splitlines():
            if "question / claim to investigate:" in line.lower():
                idx = prompt.lower().find("question / claim to investigate:")
                rest = prompt[idx:].splitlines()
                if len(rest) > 1:
                    question = rest[1].strip()
                break
        if not question:
            question = prompt.splitlines()[0] if prompt.splitlines() else "Submitted claim"
        return f"{question}. Independent investigation and factual cross-referencing analyze this statement against primary records."

    def describe(self) -> str:
        mode = " (Mock / Offline Mode)" if self.is_mock else ""
        return f"{self.model} @ {self.base_url}{mode}"


def make_llm(role: str, model: str, base_url: str, api_key: str,
             retry: RetryPolicy | None = None, meter: UsageMeter | None = None,
             is_mock: bool = False) -> LLM:
    retry = retry or RetryPolicy()
    meter = meter or UsageMeter()
    client = _client(base_url, api_key, retry.timeout_seconds)
    return LLM(role=role, model=model, base_url=base_url, client=client, retry=retry, meter=meter, is_mock=is_mock)
