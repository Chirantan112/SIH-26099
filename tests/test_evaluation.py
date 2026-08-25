"""Tests for LEGO #6 deterministic evaluation metrics."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import unittest

from src.evaluation import evaluate_pairs


ROOT = Path(__file__).resolve().parents[1]
PAIR_FIELDS = (
    "pair_id",
    "record_a_code",
    "record_b_code",
    "record_a_cpse",
    "record_b_cpse",
    "category",
    "label",
)
MASTER_FIELDS = ("legacy_material_code", "raw_description")


class InMemoryCSV:
    """Small Path-like CSV source for tests that need no filesystem writes."""

    def __init__(self, fieldnames: tuple[str, ...], rows: list[dict[str, str]]):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        self._content = stream.getvalue()

    def open(self, **_kwargs):
        return io.StringIO(self._content)


class EvaluationTests(unittest.TestCase):
    """Verify counts, metrics, policies, and reproducibility."""

    def _fixture(self, pairs: list[dict[str, str]], descriptions: dict[str, str]):
        pairs_path = InMemoryCSV(PAIR_FIELDS, pairs)
        master_path = InMemoryCSV(
            MASTER_FIELDS,
            [
                {"legacy_material_code": code, "raw_description": description}
                for code, description in descriptions.items()
            ],
        )
        return pairs_path, master_path

    def test_confusion_metrics_and_all_decisions(self):
        descriptions = {
            "same-a": "CS GATE VLV 50MM FLG CL150",
            "same-b": "50 mm Carbon Steel gate valve class 150 FLANGED",
            "different": "CS GATE VLV 80MM FLG CL150",
            "partial": "CS GATE VLV 50MM CL150",
        }
        pairs = [
            {"pair_id": "same", "record_a_code": "same-a", "record_b_code": "same-b", "label": "SAME"},
            {"pair_id": "different", "record_a_code": "same-a", "record_b_code": "different", "label": "DIFFERENT"},
            {"pair_id": "uncertain-same", "record_a_code": "same-a", "record_b_code": "partial", "label": "SAME"},
            {"pair_id": "uncertain-different", "record_a_code": "same-a", "record_b_code": "partial", "label": "DIFFERENT"},
        ]
        pairs_path, master_path = self._fixture(pairs, descriptions)

        result = evaluate_pairs(pairs_path, master_path)

        self.assertEqual(result.as_tuple(), (1, 1, 1, 1))
        self.assertEqual(result.decision_counts, {"SAME": 1, "DIFFERENT": 1, "UNCERTAIN": 2})
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1, 0.5)

    def test_zero_denominators_and_empty_input(self):
        descriptions = {
            "a": "CS GATE VLV 50MM FLG CL150",
            "b": "CS GATE VLV 80MM FLG CL150",
        }
        pairs = [{"pair_id": "negative", "record_a_code": "a", "record_b_code": "b", "label": "DIFFERENT"}]
        pairs_path, master_path = self._fixture(pairs, descriptions)

        negative = evaluate_pairs(pairs_path, master_path)
        self.assertEqual(negative.as_tuple(), (0, 1, 0, 0))
        self.assertEqual(negative.accuracy, 1.0)
        self.assertEqual(negative.precision, 0.0)
        self.assertEqual(negative.recall, 0.0)
        self.assertEqual(negative.f1, 0.0)

        empty_pairs = InMemoryCSV(PAIR_FIELDS, [])
        empty = evaluate_pairs(empty_pairs, master_path)
        self.assertEqual(empty.as_tuple(), (0, 0, 0, 0))
        self.assertEqual(empty.decision_counts, {"SAME": 0, "DIFFERENT": 0, "UNCERTAIN": 0})
        self.assertEqual((empty.accuracy, empty.precision, empty.recall, empty.f1), (0.0, 0.0, 0.0, 0.0))

    def test_repeated_evaluation_is_deterministic(self):
        pairs = [{"pair_id": "one", "record_a_code": "a", "record_b_code": "b", "label": "SAME"}]
        descriptions = {"a": "CS GATE VLV 50MM FLG CL150", "b": "CS GATE VLV 50MM FLG CL150"}
        pairs_path, master_path = self._fixture(pairs, descriptions)
        self.assertEqual(evaluate_pairs(pairs_path, master_path), evaluate_pairs(pairs_path, master_path))

    def test_real_material_pairs_integration_and_default_master_path(self):
        result = evaluate_pairs(ROOT / "data" / "evaluation" / "material_pairs.csv")
        self.assertEqual(result.as_tuple(), (21, 10, 0, 6))
        self.assertEqual(result.decision_counts, {"SAME": 21, "DIFFERENT": 13, "UNCERTAIN": 3})
        for actual, expected in zip(
            (result.accuracy, result.precision, result.recall, result.f1),
            (31 / 37, 1.0, 21 / 27, 0.875),
        ):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
