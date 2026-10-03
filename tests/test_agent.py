import unittest

from tariffpilot.llm.fake import ScriptedLLM
from helpers import final, make_agent, tool_call


class AgentTests(unittest.TestCase):
    def run_agent(self, script, product="cotton t-shirt knitted"):
        agent, corpus, gw = make_agent(ScriptedLLM(script))
        return agent.run(product), gw

    def test_happy_path(self):
        r, _ = self.run_agent([tool_call("search_hts", query="cotton t-shirt"),
                               final("6109.10.00", 0.9, quote="T-shirts, singlets and tank tops, knitted, of cotton")])
        self.assertEqual((r.status, r.hts_code), ("classified", "6109.10.00"))
        self.assertEqual(r.duty_estimate.ad_valorem_pct, 16.5)
        self.assertTrue(r.trace_id)

    def test_ask_user_interrupt(self):
        r, _ = self.run_agent([{"kind": "ask_user", "thought": "?", "question": "Material?"}])
        self.assertEqual((r.status, r.question), ("needs_info", "Material?"))

    def test_hallucinated_code_is_repaired_then_accepted(self):
        r, _ = self.run_agent([final("9999.99.99"), final("6109.10.00", 0.9)])
        self.assertEqual(r.hts_code, "6109.10.00")

    def test_hallucinated_code_twice_abstains(self):
        r, _ = self.run_agent([final("9999.99.99"), final("9999.99.99")])
        self.assertEqual(r.status, "abstained")
        self.assertIsNone(r.hts_code)
        self.assertIn("grounding", r.abstain_reason)

    def test_non_leaf_rejected(self):
        r, _ = self.run_agent([final("6109"), final("6109")])
        self.assertEqual(r.status, "abstained")

    def test_fabricated_quote_rejected(self):
        r, _ = self.run_agent([final("6109.10.00", quote="made up text"), final("6109.10.00", quote="made up text")])
        self.assertEqual(r.status, "abstained")

    def test_duplicate_loop_detected(self):
        same = tool_call("search_hts", query="x")
        r, _ = self.run_agent([same, same, same, same])
        self.assertEqual(r.status, "abstained")
        self.assertIn("looped", r.abstain_reason)

    def test_step_budget(self):
        script = [tool_call("search_hts", query=f"q{i}") for i in range(10)]
        r, _ = self.run_agent(script)
        self.assertEqual(r.status, "abstained")

    def test_disallowed_tool_becomes_observation(self):
        agent, corpus, gw = make_agent(ScriptedLLM([tool_call("rm_rf", path="/"), final("6109.10.00")]))
        r = agent.run("cotton t-shirt knitted")
        self.assertEqual(r.status, "classified")
        self.assertIn("not allowed", agent.tools.call("rm_rf", {}))

    def test_bad_tool_args_are_observations(self):
        agent, *_ = make_agent(ScriptedLLM([]))
        self.assertIn("must be str", agent.tools.call("search_hts", {"query": 5}))
        self.assertIn("unknown code", agent.tools.call("get_node", {"code": "0000"}))

    def test_low_confidence_needs_review(self):
        r, _ = self.run_agent([final("6109.10.00", 0.2)])
        self.assertEqual((r.status, r.needs_human_review), ("needs_review", True))

    def test_restricted_item_needs_review(self):
        agent, *_ = make_agent(ScriptedLLM([final("8507.60.00", 0.95)]))
        r = agent.run("lithium ion power bank 10000 mAh")
        self.assertEqual(r.status, "needs_review")
        self.assertTrue(r.restrictions)

    def test_llm_quota_degrades_gracefully(self):
        from tariffpilot.llm.base import QuotaExhausted
        r, _ = self.run_agent([QuotaExhausted("no more")])
        self.assertIn(r.status, ("abstained",))

    def test_per_request_call_budget(self):
        script = [tool_call("search_hts", query=f"q{i}") for i in range(20)]
        agent, *_ = make_agent(ScriptedLLM(script))
        agent.policy = type(agent.policy)(**{**agent.policy.__dict__, "llm_calls_per_request": 2, "max_steps": 6})
        r = agent.run("shirt")
        self.assertEqual(r.status, "abstained")
        self.assertIn("quota", r.abstain_reason.lower())

    def test_scratchpad_compresses_old_observations(self):
        items = [{"thought": "t", "tool": "x", "args": {}, "obs": "A" * 1000} for _ in range(4)]
        s = type(make_agent(ScriptedLLM([]))[0])._scratch(items)
        self.assertLess(len(s), 2 * 1000 + 3 * 250 + 100)
