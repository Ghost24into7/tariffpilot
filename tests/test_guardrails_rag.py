import json
import unittest

from tariffpilot.guardrails.input import check_input
from tariffpilot.rag.corpus import Corpus, parse_usitc, split_by_date
from tariffpilot.rag.index import HashEmbedder, tokenize
from helpers import ROOT, SETTINGS


class InputGuardTests(unittest.TestCase):
    def test_redteam_cases(self):
        cases = [json.loads(x) for x in (ROOT / "data/redteam/input_cases.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(cases), 40)
        for c in cases:
            v = check_input(c["text"])
            if c["expect"] == "block":
                self.assertFalse(v.allowed, c["text"])
            elif c["expect"] == "allow":
                self.assertTrue(v.allowed, c["text"])
                self.assertFalse([e for e in v.events if e.action == "redact"], c["text"])
            else:
                self.assertTrue(v.allowed, c["text"])
                self.assertIn(c["rule"], [e.rule_id for e in v.events], c["text"])

    def test_benign_false_positive_rate_is_zero(self):
        cases = [json.loads(x) for x in (ROOT / "data/redteam/input_cases.jsonl").read_text().splitlines()]
        benign = [c for c in cases if c["expect"] == "allow"]
        self.assertEqual(sum(not check_input(c["text"]).allowed for c in benign), 0)

    def test_redaction_removes_pii_from_text(self):
        v = check_input("Shirt, mail jane@x.com card 4111 1111 1111 1111")
        self.assertNotIn("jane@x.com", v.text); self.assertNotIn("4111", v.text)

    def test_non_luhn_digits_not_flagged_as_card(self):
        v = check_input("Pallet code 1234 5678 9012 3456 cotton shirt")
        self.assertNotIn("pii-card", [e.rule_id for e in v.events])

    def test_empty_and_long(self):
        self.assertFalse(check_input("   ").allowed)
        self.assertEqual(len(check_input("a " * 5000).text) <= 2000, True)


class RagTests(unittest.TestCase):
    def setUp(self):
        self.c = Corpus.load(SETTINGS.data_dir)

    def test_hierarchy_and_leaves(self):
        n = self.c.nodes["6109.10.00"]
        self.assertEqual(n.parent, "6109.10")
        self.assertTrue(n.is_leaf); self.assertFalse(self.c.nodes["6109"].is_leaf)
        self.assertIn("Of cotton", n.path); self.assertEqual(n.path[0].split(",")[0], "T-shirts")

    def test_search_finds_expected_leaf(self):
        top = self.c.hts_index.search("bluetooth earbuds with microphone", k=3)
        self.assertEqual(top[0].doc.id, "8518.30.20")

    def test_where_filter(self):
        hits = self.c.hts_index.search("shirt", k=5, where=lambda d: d.meta["leaf"])
        self.assertTrue(all(h.doc.meta["leaf"] for h in hits))

    def test_rulings_search(self):
        h = self.c.rulings_index.search("polyester athletic top", k=1)[0]
        self.assertEqual(h.doc.meta["hts_code"], "6109.90.10")

    def test_no_results_for_gibberish(self):
        self.assertEqual(self.c.hts_index.search("zzzxqv", k=3), [])

    def test_embedder_deterministic_and_normalized(self):
        a = HashEmbedder().embed(["cotton shirt"])[0]
        self.assertEqual(a, HashEmbedder().embed(["cotton shirt"])[0])
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=6)

    def test_chronological_split_has_no_leakage(self):
        rows = [{"id": i, "date": f"20{10 + i}-01-01"} for i in range(10)]
        p = split_by_date(rows)
        self.assertLess(max(r["date"] for r in p["train"]), min(r["date"] for r in p["dev"]))
        self.assertLess(max(r["date"] for r in p["dev"]), min(r["date"] for r in p["test"]))

    def test_parse_usitc_context_rows(self):
        nodes = parse_usitc([{"htsno": "", "indent": 0, "description": "Chapter"},
                             {"htsno": "1234", "indent": 1, "description": "Heading:"},
                             {"htsno": "1234.56.00", "indent": 2, "description": "Leaf"}])
        self.assertEqual(nodes["1234.56.00"].path, ["Chapter", "Heading", "Leaf"])

    def test_tokenize(self):
        self.assertEqual(tokenize("The shirts of cotton"), ["shirt", "cotton"])
