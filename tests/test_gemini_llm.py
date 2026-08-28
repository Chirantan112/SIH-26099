"""Dependency-free tests for the optional LEGO #10 Gemini adapter."""

from __future__ import annotations

import inspect
import os
import unittest
from unittest.mock import patch

from src.ai_retrieval import CandidateSuggestion
from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, map_records
from src.gemini_llm import DEFAULT_GEMINI_MODEL, GeminiLLMAdapter
from src.hybrid_pipeline import run_hybrid_pipeline

CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


class FakeResponse:
    def __init__(self, output_text: str, status: str = "completed"):
        self.output_text = output_text
        self.status = status


class FakeInteractions:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, interactions):
        self.interactions = interactions


class GeminiLLMAdapterTests(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "placeholder"

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.previous_key

    def _adapter(self, payload, factory_calls=None, error=None, status="completed"):
        interactions = FakeInteractions(
            FakeResponse(payload, status=status) if payload is not None else None,
            error=error,
        )

        def factory(api_key):
            if factory_calls is not None:
                factory_calls.append(api_key)
            return FakeClient(interactions)

        return GeminiLLMAdapter(client_factory=factory), interactions

    def test_missing_api_key_is_unavailable(self):
        os.environ.pop("GEMINI_API_KEY", None)
        adapter = GeminiLLMAdapter(client_factory=lambda _key: None)
        self.assertFalse(adapter.status().available)
        self.assertIn("API key", adapter.status().detail)
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_missing_sdk_is_unavailable(self):
        with patch("src.gemini_llm.find_spec", return_value=None):
            status = GeminiLLMAdapter().status()
        self.assertFalse(status.available)
        self.assertIn("google-genai", status.detail)

    def test_status_makes_no_network_or_client_request(self):
        calls = []
        adapter = GeminiLLMAdapter(client_factory=lambda key: calls.append(key))
        self.assertTrue(adapter.status().available)
        self.assertEqual(calls, [])

    def test_client_initialization_is_lazy(self):
        calls = []
        adapter, interactions = self._adapter('{"candidates": []}', factory_calls=calls)
        self.assertEqual(calls, [])
        adapter.status()
        self.assertEqual(calls, [])
        adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(calls, ["placeholder"])
        self.assertEqual(len(interactions.calls), 1)

    def test_successful_structured_response(self):
        adapter, interactions = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.94,"reason":"Matching valve specification."}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].canonical_material_id, "VAL-001")
        self.assertEqual(result[0].source, "gemini")
        self.assertEqual(result[0].score, 0.94)
        self.assertIn("NON-PROBABILISTIC", result[0].explanation)
        self.assertEqual(len(interactions.calls), 1)
        call = interactions.calls[0]
        self.assertEqual(call["model"], DEFAULT_GEMINI_MODEL)
        response_format = call["response_format"]
        self.assertIsInstance(response_format, dict)
        self.assertEqual(response_format["type"], "text")
        self.assertEqual(response_format["mime_type"], "application/json")
        schema = response_format["schema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "object")
        self.assertIn("candidates", schema["properties"])
        candidate_schema = schema["properties"]["candidates"]["items"]
        self.assertIn("canonical_material_id", candidate_schema["properties"])
        self.assertIn("compatibility_score", candidate_schema["properties"])
        self.assertIn("reason", candidate_schema["properties"])
        self.assertEqual(candidate_schema["required"], ["canonical_material_id", "compatibility_score", "reason"])

    def test_successful_fenced_json_response_shape(self):
        payload = """```json
{
  "candidates": [
    {
      "canonical_material_id": "VAL-001",
      "compatibility_score": 0.91,
      "reason": "It is a gate valve."
    }
  ]
}
```"""
        adapter, _ = self._adapter(payload)
        result = adapter.interpret("CS GATE VLV 50MM FLG CL150", "CS GATE VLV 50MM FLG CL150", object(), CATALOG)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].canonical_material_id, "VAL-001")
        self.assertEqual(result[0].score, 0.91)
        self.assertIn("It is a gate valve.", result[0].explanation)
        self.assertIn("Matching: none.", result[0].explanation)
        self.assertIn("Conflicts: none.", result[0].explanation)
        self.assertIn("Missing: none.", result[0].explanation)
        self.assertIn("Technically compatible: unknown.", result[0].explanation)
        self.assertIn("[Gemini Compatibility Score is advisory only; NON-PROBABILISTIC and NON-AUTHORITATIVE.]", result[0].explanation)
        self.assertEqual(result[0].matching_attributes, ())
        self.assertEqual(result[0].conflicting_attributes, ())
        self.assertEqual(result[0].missing_attributes, ())
        self.assertIsNone(result[0].technical_compatible)

    def test_complete_technical_evidence_is_preserved(self):
        payload = '{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.96,"reason":"Technical match","matching_attributes":["valve_type=gate","material=carbon steel"],"conflicting_attributes":[],"missing_attributes":["temperature_class"],"technical_compatible":true}]}'
        adapter, _ = self._adapter(payload)
        result = adapter.interpret("CS GATE VLV 50MM FLG CL150", "CS GATE VLV 50MM FLG CL150", object(), CATALOG)
        self.assertEqual(len(result), 1)
        candidate = result[0]
        self.assertEqual(candidate.score, 0.96)
        self.assertEqual(candidate.matching_attributes, ("valve_type=gate", "material=carbon steel"))
        self.assertEqual(candidate.conflicting_attributes, ())
        self.assertEqual(candidate.missing_attributes, ("temperature_class",))
        self.assertTrue(candidate.technical_compatible)
        self.assertIn("Technically compatible: yes.", candidate.explanation)

    def test_prompt_contains_structured_input_and_candidate_technical_context(self):
        prompt = GeminiLLMAdapter._build_prompt(
            "valve gate carbon steel size 50 mm flanged class 150",
            extract_attributes("CS GATE VLV 50MM FLG CL150").attributes,
            CATALOG,
        )
        self.assertIn("=== INPUT MATERIAL ===", prompt)
        self.assertIn("=== INPUT MATERIAL ATTRIBUTES ===", prompt)
        self.assertIn("=== CANDIDATE CATALOG RECORD ===", prompt)
        self.assertIn("canonical_material_id: VAL-001", prompt)
        self.assertIn("category: Valve", prompt)
        self.assertIn("valve_type: gate", prompt)
        self.assertIn("material: carbon steel", prompt)
        self.assertIn("size_mm: 50", prompt)
        self.assertIn("pressure_class: 150", prompt)
        self.assertIn("connection: flanged", prompt)
        self.assertIn("MATCHING", prompt)
        self.assertIn("CONFLICTING", prompt)
        self.assertIn("MISSING", prompt)

    def test_prompt_uses_actual_pipe_attribute_names_when_present(self):
        catalog = (CatalogRecord("PIPE-001", extract_attributes("PIPE CARBON STEEL OD 100MM THK 5MM SCHEDULE 40 END PLAIN").attributes),)
        attributes = extract_attributes("PIPE CARBON STEEL OD 100MM THK 5MM SCHEDULE 40 END PLAIN").attributes
        prompt = GeminiLLMAdapter._build_prompt("pipe carbon steel od 100 mm thk 5 mm schedule 40 end plain", attributes, catalog)
        self.assertIn("od_mm: 100", prompt)
        self.assertIn("thickness_mm: 5", prompt)
        self.assertIn("schedule: 40", prompt)
        self.assertIn("end: plain", prompt)

    def test_completed_markdown_bullet_list_is_rejected(self):
        adapter, _ = self._adapter("* VAL-001: General category match.\n* VAL-002: Matches gate valve type.")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())

    def test_completed_ordinary_prose_is_rejected(self):
        adapter, _ = self._adapter("VAL-001 looks like the best match because it is a gate valve.")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())

    def test_failed_status_is_rejected_and_unavailable(self):
        adapter, _ = self._adapter('{"candidates":[]}', status="failed")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)
        self.assertIn("status='failed'", adapter.status().detail)

    def test_incomplete_status_is_rejected_and_unavailable(self):
        adapter, _ = self._adapter('{"candidates":[]}', status="incomplete")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)
        self.assertIn("status='incomplete'", adapter.status().detail)

    def test_unknown_catalog_ids_are_rejected(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"NOT-IN-CATALOG","compatibility_score":0.99,"reason":"Nope"},{"canonical_material_id":"VAL-001","compatibility_score":0.87,"reason":"Known"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001"])
        self.assertEqual([x.score for x in result], [0.87])

    def test_exact_catalog_id_is_accepted(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.88,"reason":"Known exact ID"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001"])
        self.assertEqual(result[0].score, 0.88)

    def test_malformed_response_is_rejected_safely(self):
        adapter, _ = self._adapter("not-json")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)

    def test_duplicate_ids_are_removed(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.91,"reason":"first"},{"canonical_material_id":"VAL-001","compatibility_score":0.82,"reason":"duplicate"},{"canonical_material_id":"VAL-002","compatibility_score":0.74,"reason":"second"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001", "VAL-002"])
        self.assertEqual([x.score for x in result], [0.91, 0.74])

    def test_maximum_five_candidates(self):
        catalog = tuple(CatalogRecord(f"VAL-{i:03d}", CATALOG[0].attributes) for i in range(1, 9))
        payload = '{"candidates":[' + ','.join(f'{{"canonical_material_id":"VAL-{i:03d}","compatibility_score":{1.0 - i * 0.05},"reason":"reason {i}"}}' for i in range(1, 9)) + ']}'
        adapter, _ = self._adapter(payload)
        result = adapter.interpret("desc", "desc", object(), catalog)
        self.assertEqual(len(result), 5)
        self.assertEqual([x.score for x in result], [0.95, 0.9, 0.85, 0.8, 0.75])

    def test_api_exception_is_handled_safely(self):
        adapter, _ = self._adapter(None, error=TimeoutError("quota timeout"))
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)

    def test_empty_and_none_description_are_safe(self):
        calls = []
        adapter, interactions = self._adapter('{"candidates": []}', factory_calls=calls)
        self.assertEqual(adapter.interpret("", "", object(), CATALOG), ())
        self.assertEqual(adapter.interpret(None, "", object(), CATALOG), ())
        self.assertEqual(calls, [])
        self.assertEqual(interactions.calls, [])

    def test_gemini_scores_are_preserved_and_not_position_based(self):
        payload = '{"candidates":[{"canonical_material_id":"VAL-002","compatibility_score":0.41,"reason":"second"},{"canonical_material_id":"VAL-001","compatibility_score":0.93,"reason":"first"}]}'
        adapter, _ = self._adapter(payload)
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.score for x in result], [0.41, 0.93])
        self.assertNotEqual(result[0].score, 1.0)
        self.assertNotEqual(result[1].score, 0.8)
        self.assertTrue(all("NON-PROBABILISTIC" in x.explanation for x in result))

    def test_missing_gemini_score_is_rejected_honestly(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"No score supplied"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(result, ())
        self.assertTrue(adapter.status().available)

    def test_invalid_gemini_scores_are_rejected(self):
        for raw_score in ("NaN", "1.5", "-0.1", '"high"', "true"):
            adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":' + raw_score + ',"reason":"invalid"}]}')
            self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())

    def test_mapping_unchanged_when_gemini_succeeds(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-002","compatibility_score":0.88,"reason":"Advisory only"}]}')
        raw = "CS GATE VLV 50MM FLG CL150"
        expected = map_records((LegacyRecord("CHECK", raw),), CATALOG)[0]
        actual = run_hybrid_pipeline(raw, CATALOG, "CHECK", llm_adapter=adapter).mapping_result
        self.assertEqual(actual, expected)
        self.assertEqual(actual.canonical_material_id, "VAL-001")

    def test_mapping_unchanged_when_gemini_fails(self):
        adapter, _ = self._adapter(None, error=RuntimeError("service unavailable"))
        raw = "CS GATE VLV 50MM FLG CL150"
        self.assertEqual(run_hybrid_pipeline(raw, CATALOG, "CHECK", llm_adapter=adapter).mapping_result, map_records((LegacyRecord("CHECK", raw),), CATALOG)[0])

    def test_technical_conflicts_cannot_be_overridden(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.99,"reason":"Gemini suggestion"}]}')
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL300", CATALOG, "CHECK", llm_adapter=adapter)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")
        self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertEqual(result.ai_candidate_suggestions[0].canonical_material_id, "VAL-001")

    def test_no_api_key_value_is_present_in_source(self):
        source = inspect.getsource(GeminiLLMAdapter)
        self.assertNotIn("placeholder", source)
        self.assertIn("GEMINI_API_KEY", source)

    def test_no_authoritative_mapping_result_is_returned(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.9,"reason":"hint"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertIsInstance(result, tuple)
        self.assertTrue(all(isinstance(x, CandidateSuggestion) for x in result))
        self.assertFalse(any(hasattr(x, "decision") for x in result))

    def test_status_uses_environment_key_without_client(self):
        adapter = GeminiLLMAdapter(client_factory=lambda _key: (_ for _ in ()).throw(AssertionError("client should not initialize")))
        self.assertTrue(adapter.status().available)


if __name__ == "__main__":
    unittest.main()
