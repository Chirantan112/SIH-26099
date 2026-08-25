"""Tests for pure display helpers in the judge-facing Streamlit app."""

from __future__ import annotations

from decimal import Decimal
import unittest

from app import attribute_rows, candidate_rows
from src.attribute_extraction import MaterialAttributes
from src.demo_pipeline import CandidateEvidence


class AppPresentationTests(unittest.TestCase):
    """Verify that presentation helpers do not expose internal value objects."""

    def test_attribute_rows_render_missing_values_and_decimals_cleanly(self):
        attributes = MaterialAttributes(
            category="Valve",
            material="carbon steel",
            size_mm=Decimal("50"),
            pressure_class=150,
        )
        rows = attribute_rows(attributes)
        values = {row["Attribute"]: row["Extracted value"] for row in rows}

        self.assertEqual(values["Category"], "Valve")
        self.assertEqual(values["Size (mm)"], "50")
        self.assertEqual(values["Connection"], "Not provided")

    def test_candidate_rows_keep_only_ui_safe_evidence_fields(self):
        rows = candidate_rows(
            (
                CandidateEvidence("VAL-001", "SAME", 1.0, "All required attributes match."),
            )
        )
        self.assertEqual(
            rows,
            [
                {
                    "Canonical material ID": "VAL-001",
                    "Decision": "SAME",
                    "Score": 1.0,
                    "Explanation": "All required attributes match.",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
