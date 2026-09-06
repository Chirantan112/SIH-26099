"""Focused regression tests for Gemini live-response diagnostics."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord
from src.gemini_llm import GeminiLLMAdapter

CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


class FakeResponse:
    def __init__(self, output_text: str, status: str = "completed"):
        self.output_text = output_text
        self.status = status


class GeminiLiveDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise the adapter's response/diagnostic behavior with
        # injected fake clients; they must not depend on the developer's real
        # environment having GEMINI_API_KEY configured.
        self._api_key_patch = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        self._api_key_patch.start()
        self.addCleanup(self._api_key_patch.stop)

    def _adapter(self, output_text: str):
        response = FakeResponse(output_text)
        return GeminiLLMAdapter(client_factory=lambda _key: _FakeClient(response))

    def test_real_compatibility_score_is_preserved_exactly(self):
        adapter = self._adapter(
            '{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.731,"reason":"Technical match"}]}'
        )
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].score, 0.731)
        self.assertIn("Gemini Compatibility Score", result[0].explanation)

    def test_missing_score_is_rejected_without_positional_fallback(self):
        adapter = self._adapter(
            '{"candidates":[{"canonical_material_id":"VAL-001","reason":"Missing score"}]}'
        )
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertTrue(adapter.status().available)
        detail = adapter.status().detail
        self.assertIn("missing or invalid compatibility_score", detail)
        for forbidden in ("1.0", "0.8", "0.6", "0.4", "0.2"):
            self.assertNotIn(forbidden, detail)

    def test_invalid_score_unknown_id_duplicate_and_malformed_candidates_are_diagnosed(self):
        adapter = self._adapter(
            '{"candidates":['
            '{"canonical_material_id":"VAL-001","compatibility_score":1.2,"reason":"bad"},'
            '{"canonical_material_id":"UNKNOWN","compatibility_score":0.9,"reason":"unknown"},'
            '{"canonical_material_id":"VAL-001","compatibility_score":0.9,"reason":"first"},'
            '{"canonical_material_id":"VAL-001","compatibility_score":0.8,"reason":"duplicate"},'
            '"not-a-candidate"'
            ']}'
        )
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([item.canonical_material_id for item in result], ["VAL-001"])
        detail = adapter.status().detail
        self.assertIn("compatibility_score outside [0,1]", detail)
        self.assertIn("unknown catalog ID", detail)
        self.assertIn("duplicate catalog ID", detail)
        self.assertIn("candidate must be an object", detail)

    def test_successful_empty_response_is_distinguishable_from_failure(self):
        adapter = self._adapter('{"candidates":[]}')
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertTrue(adapter.status().available)
        self.assertIn("Gemini returned zero candidates.", adapter.status().detail)

    def test_api_failure_is_not_reported_as_zero_candidates(self):
        class FailingClient:
            class interactions:
                @staticmethod
                def create(**_kwargs):
                    raise RuntimeError("request failed")

        adapter = GeminiLLMAdapter(client_factory=lambda _key: FailingClient())
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)
        self.assertIn("Gemini interpretation failed", adapter.status().detail)
        self.assertNotIn("zero candidates", adapter.status().detail.lower())

    def test_streamlit_uses_explicit_advisory_score_labels(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn('"Cosine Similarity"', app_source)
        self.assertIn('"Gemini Compatibility Score"', app_source)
        self.assertNotIn('"Advisory Rank"', app_source)


class _FakeClient:
    def __init__(self, response: FakeResponse):
        self.interactions = _FakeInteractions(response)


class _FakeInteractions:
    def __init__(self, response: FakeResponse):
        self.response = response

    def create(self, **_kwargs):
        return self.response


if __name__ == "__main__":
    unittest.main()
