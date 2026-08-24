"""Tests for LEGO #4 record linkage and similarity scoring."""

from __future__ import annotations

from decimal import Decimal
import unittest

from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.record_linkage import compare_records


class RecordLinkageTests(unittest.TestCase):
    """Test deterministic record linkage and scoring logic."""

    def test_valves_identical(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "SAME")
        self.assertEqual(res.score, 1.0)
        self.assertEqual(res.matching_fields, ("category", "connection", "material", "pressure_class", "size_mm", "valve_type"))
        self.assertEqual(res.conflicting_fields, ())
        self.assertEqual(res.unresolved_fields, ())

    def test_valves_class_mismatch(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=300,
            connection="flanged",
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertLess(res.score, 1.0)
        self.assertIn("pressure_class", res.conflicting_fields)

    def test_valves_type_mismatch(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Valve",
            valve_type="globe",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("valve_type", res.conflicting_fields)

    def test_valves_material_mismatch(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="stainless steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("material", res.conflicting_fields)

    def test_valves_size_mismatch(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("80"),
            pressure_class=150,
            connection="flanged",
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("size_mm", res.conflicting_fields)

    def test_valves_missing_attribute_uncertain(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection=None,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "UNCERTAIN")
        self.assertIn("connection", res.unresolved_fields)

    def test_pipes_identical(self):
        a = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
            end="plain",
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
            end="plain",
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "SAME")
        self.assertEqual(res.score, 1.0)

    def test_pipes_od_mismatch(self):
        a = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("3.91"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("od_mm", res.conflicting_fields)

    def test_pipes_thickness_mismatch(self):
        a = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("5.54"),
            schedule=40,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("thickness_mm", res.conflicting_fields)

    def test_pipes_schedule_mismatch(self):
        a = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=80,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("schedule", res.conflicting_fields)

    def test_pipes_reversed_labels(self):
        # Even if values match diagonally, OD vs OD and THK vs THK must conflict
        a = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("3.91"),
            thickness_mm=Decimal("60.3"),
            schedule=40,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("od_mm", res.conflicting_fields)
        self.assertIn("thickness_mm", res.conflicting_fields)

    def test_pipes_missing_end_uncertain(self):
        a = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
            end="plain",
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
            end=None,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "UNCERTAIN")
        self.assertIn("end", res.unresolved_fields)

    def test_bearings_identical(self):
        a = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=True,
        )
        b = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=True,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "SAME")
        self.assertEqual(res.score, 1.0)

    def test_bearings_ordered_dimensions_mismatch(self):
        a = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=True,
        )
        b = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("14"), Decimal("47"), Decimal("20")),
            dimension_unit_present=True,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("dimensions", res.conflicting_fields)

    def test_bearings_family_mismatch(self):
        a = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=True,
        )
        b = MaterialAttributes(
            category="Bearing",
            bearing_family="taper roller",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=True,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("bearing_family", res.conflicting_fields)

    def test_bearings_missing_unit_uncertain(self):
        a = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=True,
        )
        b = MaterialAttributes(
            category="Bearing",
            bearing_family="deep groove ball",
            dimensions=(Decimal("20"), Decimal("47"), Decimal("14")),
            dimension_unit_present=None,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "UNCERTAIN")
        self.assertIn("dimension_unit_present", res.unresolved_fields)

    def test_general_category_conflict(self):
        a = MaterialAttributes(
            category="Valve",
            valve_type="gate",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
            connection="flanged",
        )
        b = MaterialAttributes(
            category="Pipe",
            material="carbon steel",
            od_mm=Decimal("60.3"),
            thickness_mm=Decimal("3.91"),
            schedule=40,
        )
        res = compare_records(a, b)
        self.assertEqual(res.decision, "DIFFERENT")
        self.assertIn("category", res.conflicting_fields)

    def test_general_incomplete_uncertain(self):
        a = MaterialAttributes(category="Pipe", material="carbon steel")
        b = MaterialAttributes(category="Pipe", material="carbon steel", od_mm=Decimal("60.3"))
        res = compare_records(a, b)
        self.assertEqual(res.decision, "UNCERTAIN")
        self.assertIn("od_mm", res.unresolved_fields)

    def test_determinism(self):
        a = extract_attributes("CS GATE VLV 50MM FLG CL150").attributes
        b = extract_attributes("50 mm Carbon Steel gate valve class 150 FLANGED").attributes
        res_first = compare_records(a, b)
        res_second = compare_records(a, b)
        self.assertEqual(res_first, res_second)


if __name__ == "__main__":
    unittest.main()
