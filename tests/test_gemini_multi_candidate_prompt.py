"""Regression coverage for Gemini's bounded multi-candidate advisory prompt."""

from __future__ import annotations

import unittest

from src.gemini_llm import GeminiLLMAdapter
from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord


CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
    CatalogRecord("VAL-003", extract_attributes("CS GATE VLV 80MM FLG CL150").attributes),
)


class GeminiMultiCandidatePromptTests(unittest.TestCase):
    def test_prompt_requests_multiple_plausible_candidates_without_padding(self):
        prompt = GeminiLLMAdapter._build_prompt(
            "GATE VLV CS 50MM FLG",
            "GATE VLV CS 50MM FLG",
            extract_attributes("GATE VLV CS 50MM FLG").attributes,
            CATALOG,
        )
        self.assertIn("normally return the 3 strongest distinct plausible candidates", prompt)
        self.assertIn("Do not pad the list with weak or fabricated candidates", prompt)
        self.assertIn("returning only that candidate is acceptable", prompt)
        self.assertIn("Order candidates from strongest to weakest", prompt)

    def test_prompt_requires_catalog_ids_and_preserves_advisory_boundary(self):
        prompt = GeminiLLMAdapter._build_prompt(
            "CS GATE VLV 50MM FLG CL150",
            "CS GATE VLV 50MM FLG CL150",
            extract_attributes("CS GATE VLV 50MM FLG CL150").attributes,
            CATALOG,
        )
        self.assertIn("Return only supplied canonical material IDs", prompt)
        self.assertIn("Gemini is advisory only", prompt)
        self.assertIn("deterministic MappingResult remains authoritative", prompt)
        self.assertIn("Do not fabricate candidates or scores", prompt)


if __name__ == "__main__":
    unittest.main()
