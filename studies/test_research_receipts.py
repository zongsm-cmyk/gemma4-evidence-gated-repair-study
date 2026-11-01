"""Invariant tests for the published deterministic Gemma retrieval-fault receipts.

These validate arithmetic, provenance metadata, and limitations of a simulation;
they are NOT tests of Gemma inference or code-repair benchmark performance.
"""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("postgres_repo_proxy.json", "kaggle_cli_proxy.json")
RATES = (0.0, 0.10, 0.25, 0.50, 0.75)


class ReceiptInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = [json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
                    for name in FILES]

    def test_distinct_source_corpora(self):
        a, b = self.data
        self.assertNotEqual(a["source_commit"], b["source_commit"])
        self.assertNotEqual(a["repository"], b["repository"])
        self.assertNotEqual(a["selected_ground_truth"], b["selected_ground_truth"])

    def test_symbol_uniqueness_and_task_accounting(self):
        for report in self.data:
            selected = report["selected_ground_truth"]
            self.assertEqual(len(selected), 48)
            self.assertEqual(len({s["symbol"] for s in selected}), 48)
            rows = report["results_by_fault_rate"]
            self.assertEqual([r["injected_graph_failure_rate"] for r in rows], list(RATES))
            self.assertEqual(report["total_decisions"], 2400)
            self.assertEqual(sum(r["tasks"] for r in rows), 2400)
            self.assertTrue(all(r["tasks"] == 480 for r in rows))

    def test_policy_counts_match_fault_injection(self):
        for report in self.data:
            for row in report["results_by_fault_rate"]:
                n = row["tasks"]
                injected = row["injected_failures"]
                misleading = row["misleading_high_confidence"]
                self.assertLessEqual(misleading, injected)
                self.assertLessEqual(injected, n)
                g = row["policies"]["graph_only"]
                c = row["policies"]["confidence_gated"]
                u = row["policies"]["unconditional_fallback"]
                self.assertEqual(g["correct"], n - injected)
                self.assertEqual(c["correct"], n - misleading)
                self.assertEqual(u["correct"], n)
                for policy, expected_calls in (
                    (g, 1.0),
                    (u, 2.0),
                    (c, 1 + (injected - misleading) / n),
                ):
                    self.assertAlmostEqual(policy["mean_retrieval_calls"], expected_calls, places=4)
                    self.assertAlmostEqual(policy["localization_accuracy"],
                                           policy["correct"] / n, places=4)
                    low, high = policy["wilson_95pct"]
                    self.assertLessEqual(low, policy["localization_accuracy"] + 0.001)
                    self.assertGreaterEqual(high, policy["localization_accuracy"] - 0.001)

    def test_corpora_share_simulated_fault_schedule(self):
        a, b = self.data
        for ra, rb in zip(a["results_by_fault_rate"], b["results_by_fault_rate"]):
            for key in ("tasks", "injected_failures", "misleading_high_confidence", "policies"):
                self.assertEqual(ra[key], rb[key],
                                 "Repeated seeded fault policy is not an independent retriever test")

    def test_no_model_or_score_claim(self):
        for report in self.data:
            desc = json.dumps(report["not"]).lower()
            self.assertIn("gemma model inference", desc)
            self.assertIn("patch acceptance", desc)
            self.assertIn("swe-bench score", desc)


if __name__ == "__main__":
    unittest.main()


