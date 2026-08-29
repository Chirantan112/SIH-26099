"""Regression tests for the reproducible synthetic stress benchmark."""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts.benchmark import build_cases, evaluate_mapping
from src.demo_pipeline import load_demo_catalog


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_demo_catalog(ROOT / "data" / "demo" / "material_master.csv")

    def test_case_generation_is_deterministic(self):
        first = build_cases(self.catalog, 8)
        second = build_cases(self.catalog, 8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(self.catalog) * 8)

    def test_small_stress_set_has_no_wrong_mappings(self):
        cases = build_cases(self.catalog, 8)
        result = evaluate_mapping(cases, self.catalog)
        self.assertEqual(result["wrong"], 0)
        self.assertEqual(result["matched"], result["cases"])
        self.assertEqual(result["uncertain"], 0)


if __name__ == "__main__":
    unittest.main()
