"""ReAct agent: bounded loop with duplicate detection, clarification interrupt and verification."""
from __future__ import annotations

import json
from typing import Any, Callable

from ..governance.policy import Policy
from ..guardrails.output import verify_final
from ..llm.base import LLMError, QuotaExhausted
from ..llm.gateway import CallBudget, Gateway
from ..obs.tracing import Tracer
from ..rag.corpus import Corpus
from ..schemas import (DECISION_SCHEMA, ClassificationResult, Decision, GuardEvent)
from ..skills import SkillLoader
from .tools import ToolRegistry, duty_for

Event = Callable[[dict[str, Any]], None]


def fit_obs(obs: str, limit: int) -> str:
    """Shrink an observation to `limit` chars while keeping it valid JSON (drop trailing list items)."""
    if len(obs) <= limit:
        return obs
    try:
        data = json.loads(obs)
    except json.JSONDecodeError:
        return obs[:limit]
    if isinstance(data, list):
        while data and len(json.dumps(data, ensure_ascii=False)) > limit:
            data.pop()
        return json.dumps(data, ensure_ascii=False)
    return json.dumps({"truncated": obs[:limit - 20]})


class Agent:
    def __init__(self, gateway: Gateway, tools: ToolRegistry, corpus: Corpus, skills: SkillLoader,
                 policy: Policy, tracer: Tracer):
        self.gw, self.tools, self.corpus, self.skills, self.policy, self.tracer = (
            gateway, tools, corpus, skills, policy, tracer)

    @staticmethod
    def _scratch(items: list[dict], keep_full: int = 2) -> str:
        """Token-aware scratchpad: old observations are compressed, the last few stay verbatim."""
        lines = []
        for i, it in enumerate(items):
            obs = it["obs"] if i >= len(items) - keep_full else it["obs"][:160] + " ..."
            lines.append(f"[{i + 1}] {it['thought']} | {it['tool']}({json.dumps(it['args'])}) -> {obs}")
        return "\n".join(lines) or "(empty)"

    def run(self, product: str, clarifications: str = "", on_event: Event | None = None) -> ClassificationResult:
        emit = on_event or (lambda e: None)
        pol, items, seen = self.policy, [], set()
        events: list[GuardEvent] = []
        budget = CallBudget(pol.llm_calls_per_request)
        self.tools.calls = 0
        repeats, verify_failures, last_errors = 0, 0, []
        with self.tracer.span("agent.run") as root:
            tid = root.trace_id
            for step in range(1, pol.max_steps + 1):
                try:
                    with self.tracer.span("agent.step", step=step):
                        d: Decision = self.gw.generate(
                            "react_step", {
                                "skills": self.skills.catalog(), "tools": self.tools.describe(),
                                "product": product, "clarifications": clarifications or "(none)",
                                "scratchpad": self._scratch(items), "step": step, "max_steps": pol.max_steps},
                            schema=DECISION_SCHEMA, validator=Decision.from_dict, budget=budget,
                            meta={"product": product, "observations": items, "clarifications": clarifications})
                except QuotaExhausted as e:
                    return self._abstain(tid, f"LLM quota exhausted: {e}", step, events)
                except LLMError as e:
                    return self._abstain(tid, f"LLM unavailable: {e}", step, events)
                emit({"type": "thought", "step": step, "text": d.thought})
                if d.kind == "ask_user":
                    return ClassificationResult("needs_info", question=d.question, trace_id=tid, steps=step,
                                                guard_events=events, abstain_reason="more detail needed")
                if d.kind == "tool":
                    sig = (d.tool, json.dumps(d.args, sort_keys=True))
                    if sig in seen:
                        repeats += 1
                        obs = json.dumps({"error": "duplicate call; use a different action or finalize"})
                        events.append(GuardEvent("loop-duplicate", "warn", "low", d.tool))
                        if repeats >= 2:
                            return self._abstain(tid, "agent looped on duplicate tool calls", step, events)
                    else:
                        seen.add(sig)
                        obs = self.tools.call(d.tool, d.args)
                    emit({"type": "tool", "step": step, "tool": d.tool, "args": d.args})
                    items.append({"thought": d.thought, "tool": d.tool, "args": d.args,
                                  "obs": fit_obs(obs, pol.max_obs_chars)})
                    continue
                final = d.final
                assert final is not None
                if not final.hts_code:
                    return self._abstain(tid, final.abstain_reason or "no confident classification", step, events)
                with self.tracer.span("agent.verify"):
                    ver = verify_final(final, self.corpus)
                events.extend(ver.events)
                if ver.ok:
                    return self._finish(final, ver, product, tid, step, events)
                verify_failures, last_errors = verify_failures + 1, ver.errors
                items.append({"thought": "verification failed", "tool": "verify", "args": {},
                              "obs": json.dumps({"errors": ver.errors})[:pol.max_obs_chars]})
                if verify_failures >= 2:
                    break
            reason = ("output failed grounding checks: " + "; ".join(last_errors)) if last_errors \
                else "step budget exhausted without a final answer"
            return self._abstain(tid, reason, pol.max_steps, events)

    def _abstain(self, tid: str, reason: str, steps: int, events: list[GuardEvent]) -> ClassificationResult:
        return ClassificationResult("abstained", abstain_reason=reason, trace_id=tid, steps=steps,
                                    needs_human_review=True, guard_events=events)

    def _finish(self, final, ver, product: str, tid: str, steps: int, events: list[GuardEvent]) -> ClassificationResult:
        flags = self.policy.restrictions_for(product)
        status, review = "classified", False
        if final.confidence < self.policy.confidence_floor:
            status, review = "needs_review", True
            events.append(GuardEvent("out-low-confidence", "review", "medium", f"{final.confidence:.2f}"))
        if flags:
            status, review = "needs_review", True
            events += [GuardEvent(f"restricted-{r['id']}", "review", "medium", r["message"]) for r in flags]
        return ClassificationResult(
            status, final.hts_code, final.confidence, final.gri_path, final.citations, final.alternatives,
            duty_for(self.corpus, final.hts_code), [r["message"] for r in flags], review, None, None,
            tid, steps, events)
