"""Tests for the two independent-seed *simulated* retrieval-fault receipts."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("postgres_holdout_seed_20261111.json",
         "kaggle_cli_holdout_seed_20261112.json")


class HoldoutReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = [json.loads((ROOT / "results" / n).read_text(encoding="utf-8"))
                       for n in NAMES]

    def test_distinct_random_schedules(self):
        a, b = self.reports
        self.assertEqual({a["seed"], b["seed"]}, {20261111, 20261112})
        self.assertNotEqual(a["source_commit"], b["source_commit"])
        self.assertNotEqual(a["results_by_fault_rate"], b["results_by_fault_rate"])

    def test_all_4800_additional_decisions_accounted(self):
        self.assertEqual(sum(r["total_decisions"] for r in self.reports), 4800)
        for report in self.reports:
            self.assertEqual(report["symbols_per_replicate"], 48)
            self.assertEqual(report["repeats_per_rate"], 10)
            self.assertEqual(len(report["results_by_fault_rate"]), 5)
            self.assertEqual(len(report["selected_ground_truth"]), 48)

    def test_mathematical_consistency(self):
        for report in self.reports:
            for row in report["results_by_fault_rate"]:
                n, fail, misleading = (row[k] for k in
                    ("tasks", "injected_failures", "misleading_high_confidence"))
                graph, gate, uncond = (row["policies"][k] for k in
                    ("graph_only", "confidence_gated", "unconditional_fallback"))
                self.assertEqual(n, 480)
                self.assertEqual(graph["correct"], n - fail)
                self.assertEqual(gate["correct"], n - misleading)
                self.assertEqual(uncond["correct"], n)
                self.assertAlmostEqual(gate["mean_retrieval_calls"],
                                       1 + (fail - misleading) / n, places=4)

    def test_explicit_non_model_claim(self):
        for report in self.reports:
            self.assertIn("Gemma model inference", report["not"])
            self.assertIn("SWE-bench score", report["not"])


if __name__ == "__main__":
    unittest.main()
