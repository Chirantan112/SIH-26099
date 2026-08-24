"""Safety and full-fixture coverage for LEGO #2 normalization."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import unittest

from src.normalization import NormalizationResult, normalize_description

ROOT = Path(__file__).resolve().parents[1]


def fixture_data():
    with (ROOT / "data/demo/material_master.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (ROOT / "data/evaluation/material_pairs.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    return {row["legacy_material_code"]: row for row in rows}, pairs


class NormalizationTests(unittest.TestCase):
    def test_case_whitespace_and_complete_token_expansion(self):
        self.assertEqual(normalize_description("  CS   VLV  ").normalized_text, "carbon steel valve")
        self.assertEqual(normalize_description("CSS VLV").normalized_text, "css valve")
        self.assertEqual(normalize_description("SS316 PIPE").normalized_text, "stainless steel 316 pipe")

    def test_pressure_schedule_units_and_known_spelling(self):
        self.assertEqual({normalize_description(v).normalized_text for v in ("CL150", "CL 150", "CLASS 150", "150#")}, {"class 150"})
        self.assertEqual({normalize_description(v).normalized_text for v in ("SCH-40", "SCH 40", "SCHEDULE 40")}, {"schedule 40"})
        self.assertEqual(normalize_description("50MM").normalized_text, "50 mm")
        self.assertEqual(normalize_description("FLNGD BRG").normalized_text, "flanged bearing")

    def test_valve_format_preserves_type_material_size_connection_and_class(self):
        normal = normalize_description("CS GATE VLV 50MM FLG CL150").normalized_text
        self.assertEqual(normal, normalize_description("50 mm Carbon Steel gate valve class 150 FLANGED").normalized_text)
        self.assertEqual(normal, "valve gate carbon steel size 50 mm flanged class 150")
        for different in ("CS GLOBE VLV 50MM FLG CL150", "SS GATE VLV 50MM FLG CL150", "CS GATE VLV 80MM FLG CL150", "CS GATE VLV 50MM FLG CL300"):
            with self.subTest(different=different): self.assertNotEqual(normal, normalize_description(different).normalized_text)

    def test_pipe_groups_preserve_od_thickness_schedule_and_end(self):
        normal = normalize_description("CS PIPE 60.3 MM OD X 3.91MM SCH-40").normalized_text
        self.assertEqual(normal, normalize_description("Carbon Steel pipe 60.3mm x 3.91 mm, schedule 40").normalized_text)
        self.assertEqual(normal, "pipe carbon steel od 60.3 mm thk 3.91 mm schedule 40")
        self.assertNotEqual(normal, normalize_description("CS PIPE 3.91 MM OD X 60.3MM SCH-40").normalized_text)
        self.assertNotEqual(normal, normalize_description("CS PIPE 60.3 MM OD X 5.54MM SCH-80").normalized_text)
        self.assertIn("end plain", normalize_description("PIPE CS OD 60.3 MM THK 3.91 MM SCH 40 PLAIN END").normalized_text)

    def test_explicit_pipe_labels_override_numeric_position(self):
        standard = normalize_description("PIPE CS OD 60.3 MM THK 3.91 MM SCH-40").normalized_text
        reversed_roles = normalize_description("PIPE CS THK 60.3 MM OD 3.91 MM SCH-40").normalized_text
        reordered_labels = normalize_description("PIPE CS THK 3.91 MM OD 60.3 MM SCH-40").normalized_text
        self.assertNotEqual(standard, reversed_roles)
        self.assertEqual(standard, reordered_labels)
        self.assertNotEqual(standard, normalize_description("PIPE CS OD 3.91 MM THK 60.3 MM SCH-40").normalized_text)

    def test_bearing_tuple_is_ordered_and_unitless_status_is_preserved(self):
        explicit = normalize_description("DEEP GROOVE BLL BRG, 20x47x14mm").normalized_text
        self.assertEqual(explicit, "bearing deep groove ball dimensions 20 x 47 x 14 mm")
        self.assertNotEqual(explicit, normalize_description("DEEP GROOVE BALL BEARING 20 X 47 X 12 MM").normalized_text)
        self.assertNotEqual(explicit, normalize_description("BRG DEEP GROOVE BALL 20 X 47 X 14").normalized_text)
        self.assertNotEqual(normalize_description("20x47x14").normalized_text, normalize_description("14x47x20").normalized_text)

    def test_unknown_identifier_hyphens_and_slashes_are_not_deleted(self):
        self.assertEqual(normalize_description("X-65 PIPE").normalized_text, "x-65 pipe")
        self.assertNotEqual(normalize_description("X-65 PIPE").normalized_text, normalize_description("X 65 PIPE").normalized_text)
        self.assertNotEqual(normalize_description("A/B").normalized_text, normalize_description("A B").normalized_text)

    def test_material_grades_schedules_and_partial_pipes_remain_safe(self):
        self.assertNotEqual(normalize_description("SS316").normalized_text, normalize_description("SS304").normalized_text)
        self.assertNotEqual(normalize_description("SCH-40").normalized_text, normalize_description("SCH-80").normalized_text)
        self.assertEqual(normalize_description("PIPE CS OD").normalized_text, "pipe carbon steel od")

    def test_empty_invalid_result_and_determinism(self):
        self.assertEqual(normalize_description(None).normalized_text, "")
        self.assertEqual(normalize_description(" ").normalized_text, "")
        with self.assertRaises(TypeError): normalize_description(1)  # type: ignore[arg-type]
        result = normalize_description("CS GATE VLV 50MM FLG CL150")
        self.assertIsInstance(result, NormalizationResult)
        self.assertEqual(result, normalize_description("CS GATE VLV 50MM FLG CL150"))
        self.assertTrue(result.transformations)

    def test_all_same_pairs_are_evaluated_with_safe_ambiguities_recorded(self):
        rows, pairs = fixture_data(); same = [pair for pair in pairs if pair["label"] == "SAME"]
        self.assertEqual(len(same), 27)
        # These descriptions omit a unit or end connection. The normalizer
        # preserves that unknown rather than inferring a technical value.
        ambiguous = {"PAIR-010", "PAIR-014", "PAIR-016", "PAIR-019", "PAIR-025", "PAIR-027"}
        for pair in same:
            left = normalize_description(rows[pair["record_a_code"]]["raw_description"]).normalized_text
            right = normalize_description(rows[pair["record_b_code"]]["raw_description"]).normalized_text
            with self.subTest(pair=pair["pair_id"]):
                (self.assertNotEqual if pair["pair_id"] in ambiguous else self.assertEqual)(left, right)

    def test_all_different_pairs_remain_distinguishable(self):
        rows, pairs = fixture_data(); different = [pair for pair in pairs if pair["label"] == "DIFFERENT"]
        self.assertEqual(len(different), 10)
        for pair in different:
            left = normalize_description(rows[pair["record_a_code"]]["raw_description"]).normalized_text
            right = normalize_description(rows[pair["record_b_code"]]["raw_description"]).normalized_text
            with self.subTest(pair=pair["pair_id"]): self.assertNotEqual(left, right)

    def test_master_collisions_never_cross_ground_truth(self):
        rows, _ = fixture_data(); groups = defaultdict(list)
        for row in rows.values(): groups[normalize_description(row["raw_description"]).normalized_text].append(row)
        for normalized, group in groups.items():
            with self.subTest(normalized=normalized): self.assertEqual(len({row["ground_truth_material_id"] for row in group}), 1)


if __name__ == "__main__": unittest.main()
