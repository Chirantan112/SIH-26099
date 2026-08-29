"""Regression tests for the scale and realism benchmark harness."""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts.scale_realism_benchmark import build_cases, load_records

ROOT = Path(__file__).resolve().parents[1]


class ScaleRealismBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(ROOT / "data" / "demo" / "material_master.csv")

    def test_benchmark_preserves_ground_truth_and_requested_size(self):
        cases = build_cases(self.records, size=100, level=4, seed=26099)
        self.assertEqual(len(cases), 100)
        self.assertTrue(all(case.ground_truth for case in cases))

    def test_noise_levels_are_deterministic_for_same_seed(self):
        first = build_cases(self.records, size=25, level=3, seed=26099)
        second = build_cases(self.records, size=25, level=3, seed=26099)
        self.assertEqual(first, second)

    def test_higher_noise_can_change_descriptions_without_changing_truth(self):
        clean = build_cases(self.records, size=25, level=0, seed=26099)
        noisy = build_cases(self.records, size=25, level=4, seed=26099)
        self.assertEqual(
            [case.ground_truth for case in clean],
            [case.ground_truth for case in noisy],
        )
        self.assertTrue(any(a.description != b.description for a, b in zip(clean, noisy)))


if __name__ == "__main__":
    unittest.main()
