"""Tests for LEGO #3 deterministic structured attribute extraction."""

from __future__ import annotations

from decimal import Decimal
import unittest

from src.attribute_extraction import ExtractionResult, extract_attributes


class AttributeExtractionTests(unittest.TestCase):
    def test_representative_valve_attributes(self):
        result = extract_attributes("CS GATE VLV 50MM FLG CL150")
        self.assertEqual(result.attributes.category, "Valve")
        self.assertEqual(result.attributes.valve_type, "gate")
        self.assertEqual(result.attributes.material, "carbon steel")
        self.assertEqual(result.attributes.size_mm, Decimal("50"))
        self.assertEqual(result.attributes.pressure_class, 150)
        self.assertEqual(result.attributes.connection, "flanged")

    def test_representative_bearing_attributes_preserve_order_and_units(self):
        result = extract_attributes("DEEP GROOVE BLL BRG, 20x47x14mm")
        self.assertEqual(result.attributes.category, "Bearing")
        self.assertEqual(result.attributes.bearing_family, "deep groove ball")
        self.assertEqual(result.attributes.dimensions, (Decimal("20"), Decimal("47"), Decimal("14")))
        self.assertTrue(result.attributes.dimension_unit_present)
        self.assertNotEqual(result.attributes.dimensions, extract_attributes("DEEP GROOVE BALL BEARING 14x47x20 MM").attributes.dimensions)

    def test_unitless_bearing_tuple_does_not_gain_a_unit(self):
        result = extract_attributes("BRG DEEP GROOVE BALL 20 X 47 X 14")
        self.assertEqual(result.attributes.dimensions, (Decimal("20"), Decimal("47"), Decimal("14")))
        self.assertFalse(result.attributes.dimension_unit_present)

    def test_representative_pipe_attributes_and_explicit_labels(self):
        result = extract_attributes("PIPE CS OD 60.3 MM THK 3.91 MM SCH-40 PLAIN END")
        self.assertEqual(result.attributes.category, "Pipe")
        self.assertEqual(result.attributes.material, "carbon steel")
        self.assertEqual(result.attributes.od_mm, Decimal("60.3"))
        self.assertEqual(result.attributes.thickness_mm, Decimal("3.91"))
        self.assertEqual(result.attributes.schedule, 40)
        self.assertEqual(result.attributes.end, "plain")
        reversed_roles = extract_attributes("PIPE CS THK 60.3 MM OD 3.91 MM SCH-40").attributes
        self.assertEqual(reversed_roles.od_mm, Decimal("3.91"))
        self.assertEqual(reversed_roles.thickness_mm, Decimal("60.3"))

    def test_numeric_differences_remain_explicit(self):
        valve_150 = extract_attributes("CS GATE VLV 50MM FLG CL150").attributes
        valve_300 = extract_attributes("CS GATE VLV 80MM FLG CL300").attributes
        self.assertNotEqual(valve_150.pressure_class, valve_300.pressure_class)
        self.assertNotEqual(valve_150.size_mm, valve_300.size_mm)
        pipe_40 = extract_attributes("PIPE CS OD 60.3 MM THK 3.91 MM SCH-40").attributes
        pipe_80 = extract_attributes("PIPE CS OD 60.3 MM THK 5.54 MM SCH-80").attributes
        self.assertNotEqual(pipe_40.od_mm, extract_attributes("PIPE CS OD 88.9 MM THK 3.91 MM SCH-40").attributes.od_mm)
        self.assertNotEqual(pipe_40.thickness_mm, pipe_80.thickness_mm)
        self.assertNotEqual(pipe_40.schedule, pipe_80.schedule)

    def test_material_grades_remain_distinct(self):
        grade_304 = extract_attributes("PIPE SS304 OD 60.3 MM THK 3.91 MM SCH-40").attributes
        grade_316 = extract_attributes("PIPE SS316 OD 60.3 MM THK 3.91 MM SCH-40").attributes
        self.assertEqual(grade_304.material, "stainless steel 304")
        self.assertEqual(grade_316.material, "stainless steel 316")
        self.assertNotEqual(grade_304.material, grade_316.material)

    def test_missing_partial_and_unknown_descriptions_fail_safely(self):
        partial = extract_attributes("PIPE CS OD")
        self.assertEqual(partial.attributes.category, "Pipe")
        self.assertEqual(partial.attributes.material, "carbon steel")
        self.assertIsNone(partial.attributes.od_mm)
        self.assertIsNone(partial.attributes.thickness_mm)
        unknown = extract_attributes("PUMP A/B X-65")
        self.assertIsNone(unknown.attributes.category)
        self.assertEqual(unknown.attributes, extract_attributes("PUMP A/B X-65").attributes)

    def test_none_blank_and_deterministic_execution(self):
        self.assertIsNone(extract_attributes(None).attributes.category)
        self.assertIsNone(extract_attributes("   ").attributes.category)
        result = extract_attributes("CS GATE VLV 50MM FLG CL150")
        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result, extract_attributes("CS GATE VLV 50MM FLG CL150"))
        self.assertTrue(result.extraction_notes)

    def test_no_unsafe_inference_for_incomplete_pipe(self):
        result = extract_attributes("PIPE CS 60.3 MM SCH-40")
        self.assertEqual(result.attributes.category, "Pipe")
        # An incomplete normalized pipe description does not contain labelled
        # groups, so the extractor must not manufacture OD or thickness.
        self.assertIsNone(result.attributes.od_mm)
        self.assertIsNone(result.attributes.thickness_mm)


if __name__ == "__main__":
    unittest.main()
