"""High-value edge cases used for the SIH internal demo readiness check."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.catalog_mapping import LegacyRecord, map_records
from src.demo_pipeline import load_demo_catalog
from src.normalization import normalize_description

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "demo" / "material_master.csv"


class InternalReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_demo_catalog(CATALOG_PATH)

    def test_known_abbreviated_valve_matches(self):
        result = map_records(
            [LegacyRecord("TEST-1", "CS GATE VLV 50MM FLG CL150")], self.catalog
        )[0]
        self.assertEqual(result.decision, "MATCHED")
        self.assertEqual(result.canonical_material_id, "VAL-001")

    def test_missing_required_connection_is_uncertain(self):
        result = map_records(
            [LegacyRecord("TEST-2", "GATE VLV CS 50MM CL150")], self.catalog
        )[0]
        self.assertEqual(result.decision, "UNCERTAIN")
        self.assertIsNone(result.canonical_material_id)

    def test_hard_pressure_class_conflict_is_not_matched(self):
        result = map_records(
            [LegacyRecord("TEST-3", "CS GATE VLV 50MM FLG CL999")], self.catalog
        )[0]
        self.assertEqual(result.decision, "NEW_CANDIDATE")
        self.assertIsNone(result.canonical_material_id)

    def test_unknown_pipe_dimensions_become_new_candidate(self):
        result = map_records(
            [LegacyRecord("TEST-4", "PIPE CS OD 999 MM THK 3 MM SCH-40 PLAIN END")], self.catalog
        )[0]
        self.assertEqual(result.decision, "NEW_CANDIDATE")
        self.assertIsNone(result.canonical_material_id)

    def test_normalization_is_deterministic(self):
        text = "50 mm Carbon Steel gate valve, class 150 FLANGED"
        first = normalize_description(text)
        second = normalize_description(text)
        self.assertEqual(first, second)
        self.assertIn("valve gate carbon steel", first.normalized_text)

    def test_empty_input_is_safe(self):
        result = map_records([LegacyRecord("TEST-5", "")], self.catalog)[0]
        self.assertEqual(result.decision, "NEW_CANDIDATE")
        self.assertIsNone(result.canonical_material_id)


if __name__ == "__main__":
    unittest.main()
