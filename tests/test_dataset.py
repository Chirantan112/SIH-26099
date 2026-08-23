import csv
import hashlib
import io
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generate_dataset import CATEGORIES, CPSES, MASTER_FIELDS, PAIR_FIELDS, create_pairs, create_records

MASTER_PATH = ROOT / "data" / "demo" / "material_master.csv"
PAIR_PATH = ROOT / "data" / "evaluation" / "material_pairs.csv"
REQUIRED = {"Valve":("valve_type","material","pressure_class","end_connection","size_mm"), "Bearing":("bearing_family","bearing_inner_diameter_mm","bearing_outer_diameter_mm","bearing_width_mm"), "Pipe":("pipe_material","pipe_od_mm","pipe_thickness_mm","pipe_schedule","pipe_end")}

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); return list(reader), reader.fieldnames

class GeneratedArtifactTests(unittest.TestCase):
    """Validate published artifacts without regenerating them."""
    @classmethod
    def setUpClass(cls):
        cls.records, cls.master_fields = read_csv(MASTER_PATH); cls.pairs, cls.pair_fields = read_csv(PAIR_PATH)
        cls.by_code = {row["legacy_material_code"]: row for row in cls.records}

    def test_required_columns_exist(self): self.assertEqual(self.master_fields, MASTER_FIELDS); self.assertEqual(self.pair_fields, PAIR_FIELDS)
    def test_records_have_valid_core_fields(self): self.assertTrue(all(r["cpse"] in CPSES and r["category"] in CATEGORIES and r["ground_truth_material_id"] for r in self.records))
    def test_category_specific_canonical_attributes_exist(self):
        for row in self.records:
            with self.subTest(code=row["legacy_material_code"]): self.assertTrue(all(row[field] for field in REQUIRED[row["category"]]))
    def test_legacy_material_codes_are_unique(self):
        codes = [r["legacy_material_code"] for r in self.records]; self.assertEqual(len(codes), len(set(codes)))
    def test_ground_truth_ids_map_to_one_canonical_specification(self):
        fields = ("category",) + tuple(field for group in REQUIRED.values() for field in group); specs = defaultdict(set)
        for row in self.records: specs[row["ground_truth_material_id"]].add(tuple(row[field] for field in fields))
        self.assertTrue(all(len(values) == 1 for values in specs.values()))
    def test_evaluation_references_match_master_records(self):
        for pair in self.pairs:
            with self.subTest(pair=pair["pair_id"]):
                self.assertIn(pair["record_a_code"], self.by_code); self.assertIn(pair["record_b_code"], self.by_code)
                left, right = self.by_code[pair["record_a_code"]], self.by_code[pair["record_b_code"]]
                self.assertEqual(pair["record_a_cpse"], left["cpse"]); self.assertEqual(pair["record_b_cpse"], right["cpse"])
                self.assertEqual(pair["record_a_ground_truth_material_id"], left["ground_truth_material_id"]); self.assertEqual(pair["record_b_ground_truth_material_id"], right["ground_truth_material_id"])
    def test_pair_labels_match_master_ground_truth(self):
        for pair in self.pairs:
            left, right = self.by_code[pair["record_a_code"]], self.by_code[pair["record_b_code"]]
            if pair["label"] == "SAME": self.assertEqual(left["ground_truth_material_id"], right["ground_truth_material_id"])
            else: self.assertEqual(pair["label"], "DIFFERENT"); self.assertNotEqual(left["ground_truth_material_id"], right["ground_truth_material_id"])
    def test_coverage_and_within_cpse_duplicates_exist(self):
        self.assertEqual({r["cpse"] for r in self.records}, set(CPSES)); self.assertEqual({r["category"] for r in self.records}, set(CATEGORIES))
        coverage = {(p["record_a_cpse"], p["record_b_cpse"]) for p in self.pairs if p["label"] == "SAME"}
        self.assertTrue({("CPCL","IOCL"),("CPCL","NTPC"),("IOCL","NTPC")}.issubset(coverage))
        self.assertTrue(any(p["label"] == "DIFFERENT" and p["record_a_cpse"] == p["record_b_cpse"] == "NTPC" for p in self.pairs))
        self.assertTrue(any(n > 1 for n in Counter((r["cpse"],r["ground_truth_material_id"]) for r in self.records).values()))
    def test_valve_description_connection_consistency(self):
        terms = {"Flanged":("FLG","FLANGED","FLNGD"), "Threaded":("THD","THREADED"), "Wafer":("WFR","WAFER")}
        for row in self.records:
            if row["category"] == "Valve":
                self.assertTrue(any(term in row["raw_description"].upper() for term in terms[row["end_connection"]]), row["legacy_material_code"])

class GeneratorDeterminismTests(unittest.TestCase):
    @staticmethod
    def digest(rows, fields):
        output = io.StringIO(newline=""); writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        return hashlib.sha256(output.getvalue().encode()).hexdigest()
    def test_generator_is_deterministic_for_fixed_seed(self):
        first, second = create_records(), create_records()
        self.assertEqual(self.digest(first, MASTER_FIELDS), self.digest(second, MASTER_FIELDS))
        self.assertEqual(self.digest(create_pairs(first), PAIR_FIELDS), self.digest(create_pairs(second), PAIR_FIELDS))
