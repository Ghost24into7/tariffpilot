import json
import tempfile
from pathlib import Path

from tariffpilot.config import Settings
from tariffpilot.llm.gateway import DiskCache, Gateway, RateLimiter
from tariffpilot.obs.tracing import InMemoryExporter, Tracer
from tariffpilot.prompts import PromptRegistry
from tariffpilot.rag.corpus import Corpus
from tariffpilot.skills import SkillLoader
from tariffpilot.governance.policy import Policy
from tariffpilot.agent.tools import build_tools
from tariffpilot.agent.loop import Agent

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Settings(root=ROOT)
POLICY = Policy.load(SETTINGS.policy_path)


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t, self.slept = t, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make_gateway(client, **kw):
    clock = kw.pop("clock", Clock())
    tr = Tracer([InMemoryExporter()])
    gw = Gateway(client, prompts=PromptRegistry(SETTINGS.prompts_dir), models={"flash": "m-flash", "lite": "m-lite"},
                 limiter=kw.pop("limiter", RateLimiter(1000, 1000, clock)), tracer=tr, clock=clock,
                 sleep=clock.sleep, rand=lambda: 0.0, **kw)
    return gw, clock


def make_agent(client, **kw):
    gw, clock = make_gateway(client, **kw)
    corpus = Corpus.load(SETTINGS.data_dir)
    skills = SkillLoader(SETTINGS.skills_dir)
    tools = build_tools(corpus, skills, POLICY, gw.tracer)
    return Agent(gw, tools, corpus, skills, POLICY, gw.tracer), corpus, gw


def tool_call(tool, **args):
    return {"kind": "tool", "thought": "t", "tool": tool, "args_json": json.dumps(args)}


def final(code, conf=0.9, cites=None, quote=None):
    cites = cites if cites is not None else [{"type": "hts", "id": code or "x", "quote": quote or ""}]
    return {"kind": "final", "thought": "done", "final": {"hts_code": code, "confidence": conf, "gri_path": ["GRI 1"],
            "citations": cites, "alternatives": [], "abstain_reason": None}}
