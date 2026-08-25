"""Tests for LEGO #5 batch catalog mapping and entity resolution."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
import unittest

from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, map_records

ROOT = Path(__file__).resolve().parents[1]


def row_to_attributes(row: dict[str, str]) -> MaterialAttributes:
    """Map a raw master CSV row to complete canonical MaterialAttributes."""
    category = row["category"]
    if category == "Valve":
        return MaterialAttributes(
            category="Valve",
            valve_type=row["valve_type"].lower() if row["valve_type"] else None,
            material=row["material"].lower() if row["material"] else None,
            size_mm=Decimal(row["size_mm"]) if row["size_mm"] else None,
            pressure_class=int(row["pressure_class"]) if row["pressure_class"] else None,
            connection=row["end_connection"].lower() if row["end_connection"] else None,
        )
    elif category == "Bearing":
        dims = None
        if row["bearing_inner_diameter_mm"] and row["bearing_outer_diameter_mm"] and row["bearing_width_mm"]:
            dims = (
                Decimal(row["bearing_inner_diameter_mm"]),
                Decimal(row["bearing_outer_diameter_mm"]),
                Decimal(row["bearing_width_mm"]),
            )
        # Default unit status to None for standard catalog specs
        return MaterialAttributes(
            category="Bearing",
            bearing_family=row["bearing_family"].lower() if row["bearing_family"] else None,
            dimensions=dims,
            dimension_unit_present=None,
        )
    elif category == "Pipe":
        return MaterialAttributes(
            category="Pipe",
            material=row["pipe_material"].lower() if row["pipe_material"] else None,
            od_mm=Decimal(row["pipe_od_mm"]) if row["pipe_od_mm"] else None,
            thickness_mm=Decimal(row["pipe_thickness_mm"]) if row["pipe_thickness_mm"] else None,
            schedule=int(row["pipe_schedule"]) if row["pipe_schedule"] else None,
            end=row["pipe_end"].lower() if row["pipe_end"] else None,
        )
    return MaterialAttributes()


class CatalogMappingTests(unittest.TestCase):
    """Verify batch entity resolution, tie-breaking, and mapping safeties."""

    @classmethod
    def setUpClass(cls):
        # Create standard catalog fixture
        cls.catalog = [
            CatalogRecord(
                canonical_material_id="VAL-001",
                attributes=extract_attributes("CS GATE VLV 50MM FLG CL150").attributes,
            ),
            CatalogRecord(
                canonical_material_id="VAL-002",
                attributes=extract_attributes("CS GATE VLV 50MM FLG CL300").attributes,
            ),
            CatalogRecord(
                canonical_material_id="VAL-003",
                attributes=extract_attributes("CS GATE VLV 80MM FLG CL150").attributes,
            ),
            CatalogRecord(
                canonical_material_id="VAL-004",
                attributes=extract_attributes("SS GATE VLV 50MM FLG CL150").attributes,
            ),
            CatalogRecord(
                canonical_material_id="PIP-001",
                attributes=extract_attributes("PIPE CS OD 60.3 MM THK 3.91 MM SCH-40 PLAIN END").attributes,
            ),
            CatalogRecord(
                canonical_material_id="PIP-002",
                attributes=extract_attributes("PIPE CS OD 60.3 MM THK 5.54 MM SCH-80 PLAIN END").attributes,
            ),
            CatalogRecord(
                canonical_material_id="PIP-003",
                attributes=extract_attributes("PIPE SS304 OD 60.3 MM THK 3.91 MM SCH-40 BEVELLED END").attributes,
            ),
            CatalogRecord(
                canonical_material_id="PIP-004",
                attributes=extract_attributes("PIPE SS316 OD 60.3 MM THK 3.91 MM SCH-40 BEVELLED END").attributes,
            ),
            CatalogRecord(
                canonical_material_id="BRG-001",
                attributes=extract_attributes("DEEP GROOVE BLL BRG, 20x47x14mm").attributes,
            ),
        ]

    def test_single_record_maps_to_correct_canonical(self):
        legacy = [LegacyRecord("LEG-01", "valve gate carbon steel size 50 mm flanged class 150")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "VAL-001")

    def test_multiple_cpse_records_map_to_same_canonical(self):
        legacy = [
            LegacyRecord("LEG-CPCL", "VALVE-GATE; CS; FLANGED; 50 mm; CL 150"),
            LegacyRecord("LEG-IOCL", "CS GATE VALVE 50 MM 150# FLG"),
        ]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "VAL-001")
        self.assertEqual(res[1].decision, "MATCHED")
        self.assertEqual(res[1].canonical_material_id, "VAL-001")

    def test_different_canonical_materials_remain_separate(self):
        legacy = [LegacyRecord("LEG-SS", "SS GATE VLV 50MM FLG CL150")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "VAL-004")

    def test_size_difference_does_not_collapse(self):
        legacy = [LegacyRecord("LEG-SIZE", "valve gate carbon steel size 80 mm flanged class 150")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "VAL-003")

    def test_pressure_class_difference_does_not_collapse(self):
        legacy = [LegacyRecord("LEG-CLASS", "CS GATE VLV 50MM FLG CL300")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "VAL-002")

    def test_material_grades_do_not_collapse(self):
        legacy = [LegacyRecord("LEG-SS316", "PIPE SS316 OD 60.3 MM THK 3.91 MM SCH-40 BEVELLED END")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "PIP-004")

    def test_schedules_do_not_collapse(self):
        legacy = [LegacyRecord("LEG-SCH80", "PIPE CS OD 60.3 MM THK 5.54 MM SCH-80 PLAIN END")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "MATCHED")
        self.assertEqual(res[0].canonical_material_id, "PIP-002")

    def test_pipe_od_thickness_conflict_produces_new_candidate(self):
        # Outside diameter differs from all pipes in the catalog
        legacy = [LegacyRecord("LEG-NEW-PIPE", "PIPE CS OD 88.9 MM THK 3.91 MM SCH-40")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "NEW_CANDIDATE")
        self.assertIsNone(res[0].canonical_material_id)

    def test_missing_required_attributes_produce_uncertain(self):
        # Missing 'connection' for Valve
        legacy = [LegacyRecord("LEG-UNCERTAIN", "GATE VLV CS 50MM CL150")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "UNCERTAIN")
        self.assertIsNone(res[0].canonical_material_id)

    def test_no_candidate_produces_new_candidate(self):
        # No compatible candidates in catalog (different category or types)
        legacy = [LegacyRecord("LEG-NEW", "BUTTERFLY VALVE CAST IRON 150MM WAFER CL150")]
        res = map_records(legacy, self.catalog)
        self.assertEqual(res[0].decision, "NEW_CANDIDATE")
        self.assertIsNone(res[0].canonical_material_id)

    def test_two_equally_plausible_candidates_produce_uncertain(self):
        # Create a catalog with duplicate attributes but different IDs
        double_catalog = [
            CatalogRecord("VAL-001-A", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
            CatalogRecord("VAL-001-B", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
        ]
        legacy = [LegacyRecord("LEG-TIE", "CS GATE VLV 50MM FLG CL150")]
        res = map_records(legacy, double_catalog)
        self.assertEqual(res[0].decision, "UNCERTAIN")
        self.assertIsNone(res[0].canonical_material_id)
        self.assertIn("Ambiguous match", res[0].explanation)

    def test_candidate_selection_is_deterministic(self):
        legacy = [LegacyRecord("LEG-DET", "valve gate carbon steel size 50 mm flanged class 150")]
        res_1 = map_records(legacy, self.catalog)
        res_2 = map_records(legacy, self.catalog)
        self.assertEqual(res_1, res_2)

    def test_full_dataset_integration(self):
        # Load unique canonical specifications from the synthetic master file
        master_csv = ROOT / "data" / "demo" / "material_master.csv"
        self.assertTrue(master_csv.exists(), f"Mock master CSV missing at {master_csv}")

        unique_specs: dict[str, CatalogRecord] = {}
        with master_csv.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gt_id = row["ground_truth_material_id"]
                if gt_id not in unique_specs:
                    unique_specs[gt_id] = CatalogRecord(
                        canonical_material_id=gt_id,
                        attributes=row_to_attributes(row),
                    )

        # Run mapping on all rows in master CSV
        legacy_list: list[LegacyRecord] = []
        with master_csv.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                legacy_list.append(
                    LegacyRecord(
                        legacy_material_code=row["legacy_material_code"],
                        raw_description=row["raw_description"],
                    )
                )

        catalog_entries = list(unique_specs.values())
        results = map_records(legacy_list, catalog_entries)

        matched_count = 0
        uncertain_count = 0
        for i, res in enumerate(results):
            legacy_item = legacy_list[i]
            # Find the original ground truth row
            with master_csv.open(encoding="utf-8") as handle:
                row = next(r for r in csv.DictReader(handle) if r["legacy_material_code"] == legacy_item.legacy_material_code)
            
            gt_id = row["ground_truth_material_id"]

            if res.decision == "MATCHED":
                matched_count += 1
                self.assertEqual(res.canonical_material_id, gt_id)
            else:
                self.assertEqual(res.decision, "UNCERTAIN")
                uncertain_count += 1

        # We expect a substantial portion of standard matches to resolve successfully
        self.assertGreater(matched_count, 0)


if __name__ == "__main__":
    unittest.main()
