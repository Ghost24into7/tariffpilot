"""The only LLM entry point: rate limiting, cache/replay, retries, circuit breaker, router, budgets."""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from ..obs.tracing import Tracer
from ..prompts import PromptRegistry
from .base import (CircuitOpen, LLMClient, LLMError, QuotaExhausted, ReplayMiss,
                   TransientLLMError, Usage)


class RateLimiter:
    """Sliding-window RPM plus a per-day cap (day boundary is UTC; adjust if your quota resets elsewhere)."""
    def __init__(self, rpm: int, rpd: int, clock: Callable[[], float] = time.time):
        self.rpm, self.rpd, self.clock = rpm, rpd, clock
        self._w: deque[float] = deque()
        self._day: str | None = None
        self._n = 0
        self._lock = threading.Lock()

    def reserve(self) -> float:
        with self._lock:
            now = self.clock()
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if day != self._day:
                self._day, self._n = day, 0
            if self._n >= self.rpd:
                raise QuotaExhausted(f"daily request cap reached ({self.rpd})")
            while self._w and now - self._w[0] >= 60:
                self._w.popleft()
            wait = 0.0
            if len(self._w) >= self.rpm:
                wait = max(60 - (now - self._w[0]), 0.0)
            self._w.append(now + wait)
            self._n += 1
            return wait

    @property
    def remaining_today(self) -> int:
        return max(self.rpd - self._n, 0)


class DiskCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> str | None:
        p = self.dir / f"{key}.json"
        return p.read_text() if p.exists() else None

    def put(self, key: str, value: str) -> None:
        (self.dir / f"{key}.json").write_text(value)


class CallBudget:
    def __init__(self, max_calls: int):
        self.left = max_calls

    def spend(self) -> None:
        if self.left <= 0:
            raise QuotaExhausted("per-request LLM call budget exhausted")
        self.left -= 1


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    return json.loads(m.group(1) if m else raw)


class Gateway:
    def __init__(self, client: LLMClient | None, *, prompts: PromptRegistry, models: dict[str, str],
                 limiter: RateLimiter, cache: DiskCache | None = None, tracer: Tracer | None = None,
                 replay: bool = False, max_retries: int = 4, backoff: float = 1.0, max_wait: float = 30.0,
                 breaker_threshold: int = 5, breaker_cooldown: float = 30.0,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 rand: Callable[[], float] = random.random):
        self.client, self.prompts, self.models, self.limiter, self.cache = client, prompts, models, limiter, cache
        self.tracer = tracer or Tracer()
        self.replay, self.max_retries, self.backoff, self.max_wait = replay, max_retries, backoff, max_wait
        self.threshold, self.cooldown = breaker_threshold, breaker_cooldown
        self.clock, self.sleep, self.rand = clock, sleep, rand
        self._fails, self._open_until = 0, 0.0
        self.stats = {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0}

    def _call(self, model: str, system: str, user: str, schema: dict | None, meta: dict,
              budget: CallBudget | None) -> str:
        assert self.client is not None
        for attempt in range(self.max_retries + 1):
            if self.clock() < self._open_until:
                raise CircuitOpen("LLM circuit breaker open")
            if budget:
                budget.spend()
            wait = self.limiter.reserve()
            if wait > self.max_wait:
                raise QuotaExhausted(f"rate limit wait {wait:.0f}s exceeds max_wait")
            if wait:
                self.sleep(wait)
            try:
                raw, usage = self.client.complete(model=model, system=system, user=user, schema=schema, meta=meta)
            except TransientLLMError as e:
                self._fails += 1
                if self._fails >= self.threshold:
                    self._open_until = self.clock() + self.cooldown
                if attempt == self.max_retries:
                    raise
                self.sleep((e.retry_after or self.backoff * 2 ** attempt) + self.rand() * self.backoff)
                continue
            self._fails = 0
            self.stats["calls"] += 1
            self.stats["tokens_in"] += usage.tokens_in
            self.stats["tokens_out"] += usage.tokens_out
            return raw
        raise LLMError("unreachable")

    def generate(self, prompt_id: str, variables: dict, *, tier: str | None = None,
                 schema: dict | None = None, validator: Callable[[Any], Any] = lambda x: x,
                 budget: CallBudget | None = None, meta: dict | None = None) -> Any:
        p = self.prompts.get(prompt_id)
        model = self.models[tier or p.tier]
        system, user = p.render(variables)
        key = hashlib.sha256(f"{model}|{p.id}@{p.version}|{system}|{user}".encode()).hexdigest()
        with self.tracer.span("llm.generate", prompt=f"{p.id}@v{p.version}:{p.hash}", model=model) as sp:
            if self.cache and (hit := self.cache.get(key)) is not None:
                try:
                    out = validator(_parse_json(hit))
                    self.stats["cache_hits"] += 1
                    sp.attrs["cache_hit"] = True
                    return out
                except (ValueError, json.JSONDecodeError):
                    pass  # stale or invalid cache entry: fall through
            sp.attrs["cache_hit"] = False
            if self.replay or self.client is None:
                raise ReplayMiss(f"no recorded response for {p.id}")
            m = dict(meta or {})
            m["prompt_id"] = prompt_id
            err: Exception | None = None
            for repair in range(2):
                u = user if err is None else (
                    f"{user}\n\nYour previous reply was invalid ({err}). Return corrected JSON only.")
                raw = self._call(model, system, u, schema, m, budget)
                try:
                    out = validator(_parse_json(raw))
                    if self.cache:
                        self.cache.put(key, raw)
                    sp.attrs["repaired"] = repair > 0
                    return out
                except (ValueError, json.JSONDecodeError) as e:
                    err = e
            raise LLMError(f"model output failed validation after repair: {err}")
