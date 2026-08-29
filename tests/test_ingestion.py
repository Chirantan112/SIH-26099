import unittest
from pathlib import Path

from src.ingestion import load_csv, map_columns, onboard_rows


class IngestionTests(unittest.TestCase):
    def test_maps_common_cpse_aliases(self) -> None:
        result = onboard_rows(
            [
                {
                    "Company": "CPCL",
                    "Material Code": "CP-001",
                    "Material Description": "Carbon Steel Gate Valve",
                }
            ]
        )
        self.assertEqual(
            result.column_mapping.mapping,
            {
                "Company": "cpse",
                "Material Code": "legacy_material_code",
                "Material Description": "raw_description",
            },
        )
        self.assertTrue(result.quality.ready_for_harmonization)
        self.assertEqual(result.records[0]["legacy_material_code"], "CP-001")

    def test_reports_missing_description(self) -> None:
        result = onboard_rows(
            [{"cpse": "CPCL", "material_code": "CP-001", "description": ""}]
        )
        self.assertEqual(result.quality.total_rows, 1)
        self.assertEqual(result.quality.missing_descriptions, 1)
        self.assertFalse(result.quality.ready_for_harmonization)

    def test_reports_duplicate_codes(self) -> None:
        result = onboard_rows(
            [
                {"cpse": "CPCL", "material_code": "CP-001", "description": "Valve A"},
                {"cpse": "IOCL", "material_code": "CP-001", "description": "Valve B"},
            ]
        )
        self.assertEqual(result.quality.duplicate_codes, 1)
        self.assertTrue(result.quality.ready_for_harmonization)

    def test_unknown_columns_are_reported_without_blocking_known_fields(self) -> None:
        result = onboard_rows(
            [
                {
                    "cpse": "CPCL",
                    "item_code": "CP-001",
                    "item_description": "Valve A",
                    "cost_center": "100",
                }
            ]
        )
        self.assertEqual(result.quality.unknown_columns, ("cost_center",))
        self.assertTrue(result.quality.ready_for_harmonization)

    def test_missing_required_source_column_is_not_ready(self) -> None:
        result = onboard_rows([{"cpse": "CPCL", "item_code": "CP-001"}])
        self.assertIn("raw_description", result.quality.canonical_columns_missing)
        self.assertFalse(result.quality.ready_for_harmonization)

    def test_column_mapping_is_deterministic(self) -> None:
        columns = ["CPSE", "legacy-code", "DESCRIPTION"]
        self.assertEqual(map_columns(columns), map_columns(columns))

    def test_demo_csv_is_ready_without_modifying_the_dataset(self) -> None:
        result = load_csv(Path("data/demo/material_master.csv"))
        self.assertEqual(result.quality.total_rows, 69)
        self.assertEqual(result.quality.missing_descriptions, 0)
        self.assertTrue(result.quality.ready_for_harmonization)


if __name__ == "__main__":
    unittest.main()
