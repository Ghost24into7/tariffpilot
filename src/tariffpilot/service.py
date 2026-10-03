"""Orchestration: input guardrails -> agent -> audit -> review queue. Also the composition root."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from .agent.loop import Agent
from .agent.tools import build_tools
from .config import Settings
from .governance.audit import AuditLog
from .governance.policy import Policy
from .governance.review import ReviewQueue
from .guardrails.input import check_input
from .learning.precedents import PrecedentStore
from .llm.fake import HeuristicAgentLLM
from .llm.gateway import DiskCache, Gateway, RateLimiter
from .obs.tracing import InMemoryExporter, JsonlExporter, Tracer
from .prompts import PromptRegistry
from .rag.corpus import Corpus
from .schemas import ClassificationResult, GuardEvent
from .skills import SkillLoader


@dataclass
class Service:
    settings: Settings
    policy: Policy
    corpus: Corpus
    gateway: Gateway
    agent: Agent
    audit: AuditLog
    review: ReviewQueue
    precedents: PrecedentStore
    tracer: Tracer
    prompts: PromptRegistry

    def classify(self, text: str, clarifications: str = "",
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> ClassificationResult:
        with self.tracer.span("service.classify") as root:
            verdict = check_input(text)
            if not verdict.allowed:
                res = ClassificationResult("blocked", abstain_reason="input rejected by guardrails",
                                           trace_id=root.trace_id, guard_events=verdict.events)
                self._audit(res, "", verdict.events)
                return res
            product = verdict.text
            clar = check_input(clarifications).text if clarifications else ""
            self.corpus.set_precedents([p for p in self.precedents.list("active")])
            res = self.agent.run(product, clar, on_event)
            res.trace_id = root.trace_id
            res.guard_events = verdict.events + res.guard_events
            self._audit(res, product, verdict.events)
            if res.status in ("needs_review", "abstained"):
                self.review.enqueue(res.trace_id, product, res.to_dict())
            return res

    def _audit(self, res: ClassificationResult, product: str, input_events: list[GuardEvent]) -> None:
        self.audit.append(res.trace_id, {
            "input_sha256": hashlib.sha256(product.encode()).hexdigest(),
            "versions": {"policy": self.policy.version, "prompts": self.prompts.versions(),
                         "models": self.gateway.models, "mode": self.settings.mode},
            "guard_events": [e.__dict__ for e in res.guard_events],
            "output": {k: v for k, v in res.to_dict().items() if k not in ("guard_events",)},
        })


def build_service(settings: Settings | None = None, exporters: list | None = None) -> Service:
    s = settings or Settings.from_env()
    policy = Policy.load(s.policy_path)
    for m in (s.model_flash, s.model_lite):
        if s.mode == "gemini" and m not in policy.allowed_models:
            raise ValueError(f"model '{m}' is not in policy.allowed_models")
    s.state_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(exporters if exporters is not None else [JsonlExporter(s.state_dir / "traces.jsonl")])
    prompts, skills = PromptRegistry(s.prompts_dir), SkillLoader(s.skills_dir)
    embedder = None
    if s.mode == "gemini":
        from .rag.index import GeminiEmbedder
        embedder = GeminiEmbedder(s.gemini_api_key, s.embed_model)
    corpus = Corpus.load(s.data_dir, embedder)
    client = None
    if s.mode == "gemini":
        from .llm.gemini import GeminiClient
        client = GeminiClient(s.gemini_api_key)
    elif s.mode == "offline":
        client = HeuristicAgentLLM()
    gw = Gateway(client, prompts=prompts, models={"flash": s.model_flash, "lite": s.model_lite},
                 limiter=RateLimiter(s.rpm, s.rpd) if s.mode == "gemini" else RateLimiter(10**9, 10**9), cache=DiskCache(s.cache_dir) if s.mode != "offline" else None,
                 tracer=tracer, replay=(s.mode == "replay"))
    skills_, tools = skills, build_tools(corpus, skills, policy, tracer)
    precedents = PrecedentStore(s.state_dir / "precedents.db", policy)
    review = ReviewQueue(s.state_dir / "review.db", policy, precedents)
    agent = Agent(gw, tools, corpus, skills_, policy, tracer)
    return Service(s, policy, corpus, gw, agent, AuditLog(s.audit_db), review, precedents, tracer, prompts)
