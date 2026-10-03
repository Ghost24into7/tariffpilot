import tempfile
import unittest
from pathlib import Path

from tariffpilot.governance.audit import AuditLog
from tariffpilot.governance.policy import Policy
from tariffpilot.governance.review import ReviewQueue
from tariffpilot.learning.precedents import GateResult, PrecedentError, PrecedentStore, regression_gate
from helpers import POLICY, SETTINGS


class AuditTests(unittest.TestCase):
    def test_chain_verifies_and_detects_tamper(self):
        a = AuditLog(":memory:")
        for i in range(3):
            a.append(f"t{i}", {"n": i})
        self.assertEqual(a.verify_chain(), (True, None))
        a.db.execute("UPDATE audit SET payload='{\"n\": 99}' WHERE seq=2"); a.db.commit()
        self.assertEqual(a.verify_chain(), (False, 2))

    def test_deleting_a_row_breaks_chain(self):
        a = AuditLog(":memory:")
        for i in range(3):
            a.append(f"t{i}", {"n": i})
        a.db.execute("DELETE FROM audit WHERE seq=2"); a.db.commit()
        self.assertFalse(a.verify_chain()[0])

    def test_retention_purge_keeps_chain_valid(self):
        a = AuditLog(":memory:")
        a.append("old", {"x": 1}, ts=1000.0); a.append("new", {"x": 2}, ts=2_000_000_000.0)
        self.assertEqual(a.purge_older_than(90, now=2_000_000_000.0), 1)
        self.assertEqual(a.get("old"), []); self.assertEqual(a.get("new"), [{"x": 2}])
        self.assertTrue(a.verify_chain()[0])

    def test_erasure_by_trace(self):
        a = AuditLog(":memory:"); a.append("t", {"x": 1})
        self.assertEqual(a.delete_trace_payload("t"), 1)
        self.assertTrue(a.verify_chain()[0])


class PolicyTests(unittest.TestCase):
    def test_loads(self):
        self.assertEqual(POLICY.confidence_floor, 0.55)

    def test_invalid_policy_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.yaml"
            p.write_text(SETTINGS.policy_path.read_text().replace("confidence_floor: 0.55", "confidence_floor: 5"))
            with self.assertRaises(ValueError):
                Policy.load(p)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.s = PrecedentStore(":memory:", POLICY)

    def test_unauthorized_reviewer(self):
        with self.assertRaises(PermissionError):
            self.s.add("shirt", "6109.10.00", "", "mallory")

    def test_duplicates_collapse_and_start_quarantined(self):
        a = self.s.add("Cotton Shirt!", "6109.10.00", "", "reviewer-1")
        b = self.s.add("cotton shirt", "6109.10.00", "", "reviewer-1")
        self.assertEqual(a, b); self.assertEqual(self.s.get(a)["state"], "quarantine")

    def test_gate_blocks_harmful_precedent(self):
        pid = self.s.add("shirt", "6109.10.00", "", "reviewer-1")
        run = lambda precs: {"accuracy_6": 0.5, "hallucination_rate": 0.0}  # noqa: E731
        gate = regression_gate(run, {"accuracy_6": 0.9}, self.s)
        self.assertFalse(self.s.promote(pid, "reviewer-1", gate).passed)
        self.assertEqual(self.s.get(pid)["state"], "rejected")

    def test_gate_allows_good_precedent_and_cannot_repromote(self):
        pid = self.s.add("shirt", "6109.10.00", "", "reviewer-1")
        gate = regression_gate(lambda p: {"accuracy_6": 0.95, "hallucination_rate": 0.0}, {"accuracy_6": 0.9}, self.s)
        self.assertTrue(self.s.promote(pid, "reviewer-1", gate).passed)
        self.assertEqual([p["id"] for p in self.s.list("active")], [pid])
        with self.assertRaises(PrecedentError):
            self.s.promote(pid, "reviewer-1", gate)

    def test_daily_limit(self):
        pol = Policy(**{**POLICY.__dict__, "feedback_per_reviewer_per_day": 2})
        s = PrecedentStore(":memory:", pol)
        s.add("a one", "1", "", "reviewer-1"); s.add("b two", "2", "", "reviewer-1")
        with self.assertRaises(PrecedentError):
            s.add("c three", "3", "", "reviewer-1")

    def test_review_queue_flow(self):
        q = ReviewQueue(":memory:", POLICY, self.s)
        i = q.enqueue("tr", "cotton shirt", {"hts_code": "6109.90.10"})
        self.assertEqual(len(q.pending()), 1)
        with self.assertRaises(PermissionError):
            q.resolve(i, "mallory", "approve")
        pid = q.resolve(i, "reviewer-1", "correct", "6109.10.00", "cotton")
        self.assertEqual(self.s.get(pid)["code"], "6109.10.00")
        self.assertEqual(q.pending(), [])
        with self.assertRaises(KeyError):
            q.resolve(i, "reviewer-1", "approve")
