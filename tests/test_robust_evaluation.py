"""Regression tests for hard-negative and retrieval evaluation."""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from scripts.robust_evaluation import (
    build_evaluation_pairs,
    evaluate_pairs,
    load_legacy_records,
    retrieval_recall_at_k,
)
from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.attribute_extraction import MaterialAttributes
from src.catalog_mapping import CatalogRecord


ROOT = Path(__file__).resolve().parents[1]


class RobustEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_legacy_records(ROOT / "data" / "demo" / "material_master.csv")

    def test_hard_negative_evaluation_has_no_wrong_decisions(self):
        pairs = build_evaluation_pairs(self.records, max_hard_negatives=100)
        result = evaluate_pairs(pairs, self.records)
        self.assertGreater(result.hard_negative_count, 0)
        self.assertEqual(result.wrong, 0)
        self.assertEqual(result.hard_negative_count, result.hard_negative_rejections)

    def test_hard_negatives_are_same_category_and_different_ground_truth(self):
        pairs = build_evaluation_pairs(self.records, max_hard_negatives=25)
        by_code = {code: ground_truth for code, _, ground_truth in self.records}
        self.assertTrue(pairs)
        for pair in pairs:
            if pair.hardness == "hard-negative":
                self.assertNotEqual(by_code[pair.left_code], by_code[pair.right_code])
                self.assertEqual(pair.expected, "DIFFERENT")

    def test_retrieval_recall_at_k_counts_expected_id(self):
        catalog = (
            CatalogRecord("VAL-001", MaterialAttributes(category="Valve", valve_type="Gate")),
            CatalogRecord("VAL-002", MaterialAttributes(category="Valve", valve_type="Globe")),
            CatalogRecord("VAL-003", MaterialAttributes(category="Valve", valve_type="Ball")),
        )

        class FakeAdapter:
            def status(self):
                return AdapterStatus("local_nlp", True, "test")

            def retrieve(self, raw_description, normalized_description, attributes, catalog):
                return (
                    CandidateSuggestion("VAL-002", 0.9, "test", "top"),
                    CandidateSuggestion("VAL-001", 0.8, "test", "second"),
                    CandidateSuggestion("VAL-003", 0.7, "test", "third"),
                )

        cases = (("VAL-001", "test valve", MaterialAttributes(category="Valve", valve_type="Gate")),)
        adapter = FakeAdapter()
        self.assertEqual(retrieval_recall_at_k(cases, catalog, adapter, 1), 0.0)
        self.assertEqual(retrieval_recall_at_k(cases, catalog, adapter, 2), 1.0)
        self.assertEqual(retrieval_recall_at_k(cases, catalog, adapter, 5), 1.0)


if __name__ == "__main__":
    unittest.main()
