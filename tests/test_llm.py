import json
import tempfile
import unittest

from tariffpilot.llm.base import CircuitOpen, LLMError, QuotaExhausted, ReplayMiss, TransientLLMError
from tariffpilot.llm.fake import ScriptedLLM
from tariffpilot.llm.gateway import CallBudget, DiskCache, RateLimiter
from tariffpilot.schemas import Decision
from helpers import Clock, make_gateway

VARS = {"skills": "s", "tools": "t", "product": "p", "clarifications": "c", "scratchpad": "x", "step": 1, "max_steps": 6}
OK = {"kind": "tool", "thought": "x", "tool": "search_hts", "args_json": "{\"query\": \"a\"}"}


def gen(gw, **kw):
    return gw.generate("react_step", VARS, validator=Decision.from_dict, **kw)


class LimiterTests(unittest.TestCase):
    def test_rpm_waits_then_frees(self):
        c = Clock(); rl = RateLimiter(2, 100, c)
        self.assertEqual(rl.reserve(), 0); self.assertEqual(rl.reserve(), 0)
        self.assertAlmostEqual(rl.reserve(), 60, delta=0.01)

    def test_daily_cap(self):
        rl = RateLimiter(100, 2, Clock()); rl.reserve(); rl.reserve()
        with self.assertRaises(QuotaExhausted):
            rl.reserve()

    def test_daily_reset(self):
        c = Clock(); rl = RateLimiter(100, 1, c); rl.reserve()
        c.t += 86400; rl.reserve()


class GatewayTests(unittest.TestCase):
    def test_retry_on_transient_with_retry_after(self):
        llm = ScriptedLLM([TransientLLMError("429", retry_after=7), OK])
        gw, clock = make_gateway(llm)
        self.assertEqual(gen(gw).tool, "search_hts")
        self.assertIn(7, clock.slept)

    def test_gives_up_after_max_retries(self):
        gw, _ = make_gateway(ScriptedLLM([TransientLLMError("500")] * 10), max_retries=2)
        with self.assertRaises(TransientLLMError):
            gen(gw)

    def test_schema_repair_once(self):
        llm = ScriptedLLM(["not json", OK])
        gw, _ = make_gateway(llm)
        self.assertEqual(gen(gw).kind, "tool")
        self.assertIn("previous reply was invalid", llm.calls[1]["user"])

    def test_fails_after_repair(self):
        gw, _ = make_gateway(ScriptedLLM(["bad", "bad"]))
        with self.assertRaises(LLMError):
            gen(gw)

    def test_cache_hit_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM([OK])
            gw, _ = make_gateway(llm, cache=DiskCache(d))
            gen(gw); gen(gw)
            self.assertEqual(len(llm.calls), 1); self.assertEqual(gw.stats["cache_hits"], 1)
            rgw, _ = make_gateway(None, cache=DiskCache(d), replay=True)
            self.assertEqual(gen(rgw).tool, "search_hts")

    def test_replay_miss(self):
        with tempfile.TemporaryDirectory() as d:
            gw, _ = make_gateway(None, cache=DiskCache(d), replay=True)
            with self.assertRaises(ReplayMiss):
                gen(gw)

    def test_call_budget(self):
        gw, _ = make_gateway(ScriptedLLM([OK, OK]))
        b = CallBudget(1); gen(gw, budget=b)
        with self.assertRaises(QuotaExhausted):
            gw.generate("react_step", {**VARS, "step": 2}, validator=Decision.from_dict, budget=b)

    def test_rate_wait_too_long_is_quota_error(self):
        c = Clock(); rl = RateLimiter(1, 100, c)
        gw, _ = make_gateway(ScriptedLLM([OK, OK]), limiter=rl, clock=c, max_wait=10)
        gen(gw)
        with self.assertRaises(QuotaExhausted):
            gw.generate("react_step", {**VARS, "step": 2}, validator=Decision.from_dict)

    def test_circuit_breaker_opens(self):
        gw, _ = make_gateway(ScriptedLLM([TransientLLMError("x")] * 20), max_retries=10, breaker_threshold=3)
        with self.assertRaises(CircuitOpen):
            gen(gw)

    def test_records_spans_with_prompt_version(self):
        gw, _ = make_gateway(ScriptedLLM([OK]))
        gen(gw)
        sp = gw.tracer.exporters[0].spans[-1]
        self.assertTrue(sp.attrs["prompt"].startswith("react_step@v1:"))


class SchemaTests(unittest.TestCase):
    def test_decision_validation(self):
        with self.assertRaises(ValueError): Decision.from_dict({"kind": "nope"})
        with self.assertRaises(ValueError): Decision.from_dict({"kind": "tool", "tool": "a", "args_json": "{bad"})
        with self.assertRaises(ValueError): Decision.from_dict({"kind": "final", "final": {"confidence": 2}})
        d = Decision.from_dict({"kind": "final", "final": {"hts_code": "1.1", "confidence": 0.5,
                                                          "citations": [{"type": "hts", "id": "1.1"}]}})
        self.assertEqual(d.final.hts_code, "1.1")
