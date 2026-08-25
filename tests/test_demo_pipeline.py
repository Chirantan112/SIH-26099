"""Tests for the LEGO #7 offline demo integration façade."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import unittest

from src.demo_pipeline import MAX_CANDIDATE_EVIDENCE, load_demo_catalog, run_demo


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "data" / "demo" / "material_master.csv"


class InMemoryCSV:
    """Path-like immutable CSV source for validation tests without file writes."""

    def __init__(self, fieldnames: list[str], rows: list[dict[str, str]]):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        self._content = stream.getvalue()

    def open(self, **_kwargs):
        return io.StringIO(self._content)


class DemoPipelineTests(unittest.TestCase):
    """Verify safe catalog setup and deterministic UI-ready demo results."""

    @classmethod
    def setUpClass(cls):
        with MASTER_PATH.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            cls.fieldnames = reader.fieldnames or []
            cls.rows = list(reader)
        cls.catalog = load_demo_catalog(MASTER_PATH)

    def test_real_fixture_loads_as_immutable_unique_catalog_without_mutation(self):
        before = hashlib.sha256(MASTER_PATH.read_bytes()).hexdigest()
        catalog = load_demo_catalog(MASTER_PATH)
        after = hashlib.sha256(MASTER_PATH.read_bytes()).hexdigest()

        self.assertEqual(len(catalog), 21)
        self.assertEqual(len({record.canonical_material_id for record in catalog}), 21)
        self.assertEqual(catalog, tuple(sorted(catalog, key=lambda record: record.canonical_material_id)))
        self.assertEqual(before, after)
        with self.assertRaises(TypeError):
            catalog[0] = catalog[0]  # type: ignore[index]

    def test_required_column_validation(self):
        fields = [field for field in self.fieldnames if field != "pipe_end"]
        invalid = InMemoryCSV(fields, [{field: row[field] for field in fields} for row in self.rows[:1]])
        with self.assertRaisesRegex(ValueError, "missing required columns: pipe_end"):
            load_demo_catalog(invalid)  # type: ignore[arg-type]

    def test_blank_canonical_id_rejected(self):
        row = dict(self.rows[0])
        row["ground_truth_material_id"] = ""
        with self.assertRaisesRegex(ValueError, "canonical material ID"):
            load_demo_catalog(InMemoryCSV(self.fieldnames, [row]))  # type: ignore[arg-type]

    def test_conflicting_canonical_specification_rejected(self):
        rows = [dict(row) for row in self.rows if row["ground_truth_material_id"] == "VAL-001"][:2]
        rows[1]["size_mm"] = "999"
        with self.assertRaisesRegex(ValueError, "conflicting specifications"):
            load_demo_catalog(InMemoryCSV(self.fieldnames, rows))  # type: ignore[arg-type]

    def test_matched_input_exposes_normalization_and_attributes(self):
        result = run_demo("CS GATE VLV 50MM FLG CL150", self.catalog, "DEMO-MATCH")

        self.assertEqual(result.legacy_material_code, "DEMO-MATCH")
        self.assertEqual(result.normalized_description, "valve gate carbon steel size 50 mm flanged class 150")
        self.assertTrue(result.normalization_transformations)
        self.assertEqual(result.attributes.category, "Valve")
        self.assertEqual(result.attributes.valve_type, "gate")
        self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_uncertain_input_preserves_existing_safe_outcome(self):
        result = run_demo("GATE VLV CS 50MM CL150", self.catalog)
        self.assertEqual(result.mapping_result.decision, "UNCERTAIN")
        self.assertIsNone(result.mapping_result.canonical_material_id)

    def test_new_candidate_input_preserves_existing_safe_outcome(self):
        result = run_demo("PIPE CS OD 999 MM THK 3 MM SCH-40 PLAIN END", self.catalog)
        self.assertEqual(result.mapping_result.decision, "NEW_CANDIDATE")
        self.assertIsNone(result.mapping_result.canonical_material_id)

    def test_candidate_evidence_is_bounded_serializable_and_deterministic(self):
        first = run_demo("CS GATE VLV 50MM FLG CL150", self.catalog)
        second = run_demo("CS GATE VLV 50MM FLG CL150", self.catalog)

        self.assertLessEqual(len(first.candidate_evidence), MAX_CANDIDATE_EVIDENCE)
        self.assertEqual(first.candidate_evidence, second.candidate_evidence)
        self.assertEqual(first, second)
        json.dumps([asdict(candidate) for candidate in first.candidate_evidence])
        for candidate in first.candidate_evidence:
            self.assertIsInstance(candidate.canonical_material_id, str)
            self.assertIsInstance(candidate.decision, str)
            self.assertIsInstance(candidate.score, float)
            self.assertIsInstance(candidate.explanation, str)

    def test_missing_none_and_blank_input_are_safe_new_candidates(self):
        for raw_description in (None, "", "   "):
            with self.subTest(raw_description=raw_description):
                result = run_demo(raw_description, self.catalog)
                self.assertEqual(result.mapping_result.decision, "NEW_CANDIDATE")
                self.assertIsNone(result.mapping_result.canonical_material_id)
                self.assertEqual(result.candidate_evidence, ())


if __name__ == "__main__":
    unittest.main()
