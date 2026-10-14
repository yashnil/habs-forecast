"""Run with: python3 -m unittest discover -s tests (from research/intelligence)."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cwi.agent import FOOTER, run  # noqa: E402
from cwi.evaluate import run_adversarial, run_golden  # noqa: E402
from cwi.guard import check  # noqa: E402
from cwi.providers import AnthropicDryRun, MockProvider, PaidCallBlocked, ScriptedProvider  # noqa: E402
from cwi.snapshot import load_snapshot  # noqa: E402
from cwi.tools import Toolbox  # noqa: E402

TB = Toolbox(load_snapshot())


class SnapshotTests(unittest.TestCase):
    def test_disputed_record_detected(self):
        self.assertIn("cdph-2026-sn26-018-monterey-bivalves", TB.snap.disputed_record_ids())
        e = TB.corpus["official:cdph-2026-sn26-018-monterey-bivalves"]
        self.assertTrue(e.flags.get("disputed"))
        self.assertIn("DISPUTED", e.text)

    def test_registry_is_unverified(self):
        self.assertEqual(TB.snap.review["status"], "pending_human_review")
        self.assertTrue(all(e.flags.get("unverified") for e in TB.corpus.values() if e.tier == 1))

    def test_numeric_facts_ignore_date_parts(self):
        facts = TB.corpus["charm:473"].numeric_facts()
        self.assertNotIn(12.0, facts)   # from 2026-10-12, not a value
        self.assertIn(0.979, facts)


class ToolTests(unittest.TestCase):
    def test_strict_schemas(self):
        for spec in TB.specs():
            s = spec["input_schema"]
            self.assertFalse(s["additionalProperties"], spec["name"])
            self.assertEqual(sorted(s["required"]), sorted(s["properties"]), spec["name"])

    def test_unknown_port_is_error_not_crash(self):
        out = TB.call("get_official_notices", {"port_code": 999, "species": ""})
        self.assertIn("error", out)
        self.assertEqual(out["evidence"], [])

    def test_species_filter(self):
        ids = {c["evidence_id"] for c in TB.call("get_official_notices", {"port_code": 593, "species": "anchovy"})["evidence"]}
        self.assertIn("official:cdph-2026-sn26-019-anchovy-central-coast", ids)
        self.assertNotIn("official:cdph-2026-annual-mussel-quarantine", ids)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.ev = {i: TB.corpus[i] for i in ("charm:473", "caveat:port:1", "obs:HABs-SantaCruzWharf:pDA")}

    def test_negated_safety_is_allowed(self):
        rep = check("Low values do not mean an area is safe [caveat:port:1].", self.ev)
        self.assertNotIn("safety_claim", rep.codes())

    def test_percent_must_be_a_probability(self):
        self.assertIn("ungrounded_number", check("A 12% chance of a bloom [charm:473].", self.ev).codes())
        self.assertNotIn("ungrounded_number", check("A 98% chance of a bloom [charm:473].", self.ev).codes())

    def test_issued_date_must_match(self):
        self.assertIn("ungrounded_number", check("The run issued 2026-10-13 [charm:473].", self.ev).codes())
        self.assertFalse(check("The run issued 2026-10-12 [charm:473].", self.ev).codes())


class AgentTests(unittest.TestCase):
    def test_answer_has_footer_and_passes_guard(self):
        t = run("Can I dig razor clams in Crescent City?", MockProvider(TB), TB)
        self.assertTrue(t.answer.endswith(FOOTER))
        self.assertTrue(t.final_report.passed)
        self.assertTrue(t.answer.lstrip().startswith("Official notices"))

    def test_bad_answer_falls_back_to_official_evidence(self):
        calls = [{"type": "tool_use", "id": "t1", "name": "get_official_notices", "input": {"port_code": 550, "species": "mussels"}}]
        bad = [{"type": "text", "text": "Mussels at Monterey are safe to eat [official:cdph-2026-annual-mussel-quarantine]."}]
        prov = ScriptedProvider([{"content": calls, "stop_reason": "tool_use"},
                                 {"content": bad, "stop_reason": "end_turn"},
                                 {"content": bad, "stop_reason": "end_turn"}])
        t = run("Can I eat mussels at Monterey?", prov, TB)
        self.assertEqual(t.outcome, "fallback")
        self.assertNotIn("safe to eat", t.answer.split(FOOTER)[0])
        self.assertIn("DISPUTED", t.answer)

    def test_dry_run_never_sends(self):
        dry = AnthropicDryRun()
        with self.assertRaises(PaidCallBlocked):
            run("Is there domoic acid near Santa Cruz?", dry, TB)
        body = dry.requests[0]
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["tool_choice"], {"type": "auto"})
        self.assertEqual({t["name"] for t in body["tools"]}, set(TB.tools))


class EvalTests(unittest.TestCase):
    def test_golden_core(self):
        g = run_golden(TB)
        failed = [r["id"] for r in g["cases"] if not r["pass"] and not r["hard"]]
        self.assertEqual(failed, [])
        self.assertEqual(g["core"]["final_guard"], 1.0)

    def test_adversarial(self):
        a = run_adversarial(TB)
        self.assertEqual([r["id"] for r in a["cases"] if not r["caught"]], [])
        self.assertEqual([r["id"] for r in a["controls"] if not r["clean"]], [])


if __name__ == "__main__":
    unittest.main()
