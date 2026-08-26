"""Dependency-free tests for the optional OpenRouter LEGO #10 adapter."""
from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, map_records
from src.hybrid_pipeline import run_hybrid_pipeline
from src.openrouter_llm import DEFAULT_OPENROUTER_MODEL, OPENROUTER_BASE_URL, OpenRouterLLMAdapter

CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


def response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self, result=None, error=None): self.result, self.error, self.calls = result, error, []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error: raise self.error
        return self.result


class FakeClient:
    def __init__(self, completions): self.chat = SimpleNamespace(completions=completions)


class OpenRouterLLMAdapterTests(unittest.TestCase):
    def setUp(self):
        self.old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "placeholder"

    def tearDown(self):
        if self.old_key is None: os.environ.pop("OPENROUTER_API_KEY", None)
        else: os.environ["OPENROUTER_API_KEY"] = self.old_key

    def make_adapter(self, text, error=None):
        calls = []
        completions = FakeCompletions(response(text) if text is not None else None, error)
        def factory(key, base_url, timeout):
            calls.append((key, base_url, timeout))
            return FakeClient(completions)
        return OpenRouterLLMAdapter(client_factory=factory), completions, calls

    def test_missing_api_key(self):
        os.environ.pop("OPENROUTER_API_KEY")
        adapter = OpenRouterLLMAdapter(client_factory=lambda *_: self.fail("client initialized"))
        self.assertFalse(adapter.status().available)
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_status_is_lazy_and_has_no_network(self):
        calls = []
        adapter = OpenRouterLLMAdapter(client_factory=lambda *_: calls.append(1))
        self.assertTrue(adapter.status().available)
        self.assertEqual(calls, [])

    def test_missing_dependency(self):
        with patch("src.openrouter_llm.find_spec", return_value=None):
            self.assertFalse(OpenRouterLLMAdapter().status().available)

    def test_successful_structured_json(self):
        adapter, calls, factories = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"gate valve"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001"])
        self.assertEqual(result[0].score, 1.0); self.assertEqual(result[0].source, "openrouter")
        self.assertEqual(factories[0][0], "placeholder"); self.assertEqual(factories[0][1], OPENROUTER_BASE_URL)
        self.assertEqual(calls.calls[0]["model"], DEFAULT_OPENROUTER_MODEL)

    def test_exact_catalog_ids_only(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"ok"}]}')
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG)[0].canonical_material_id, "VAL-001")

    def test_unknown_ids_rejected(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-999","reason":"bad"}]}')
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_malformed_candidates_rejected(self):
        adapter, _, _ = self.make_adapter('{"candidates":["VAL-001",{"canonical_material_id":"VAL-001"},{"reason":"x"}]}')
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_duplicates_deduplicated(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"a"},{"canonical_material_id":"VAL-001","reason":"b"},{"canonical_material_id":"VAL-002","reason":"c"}]}')
        result = adapter.interpret("x", "x", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001", "VAL-002"])

    def test_maximum_five(self):
        catalog = tuple(CatalogRecord(f"VAL-{i:03d}", CATALOG[0].attributes) for i in range(1, 8))
        payload = '{"candidates":[' + ','.join(f'{{"canonical_material_id":"VAL-{i:03d}","reason":"r"}}' for i in range(1, 8)) + ']}'
        adapter, _, _ = self.make_adapter(payload)
        result = adapter.interpret("x", "x", object(), catalog)
        self.assertEqual(len(result), 5); self.assertEqual([x.score for x in result], [1.0, .8, .6, .4, .2])

    def test_rank_scores_are_deterministic(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-002","reason":"b"},{"canonical_material_id":"VAL-001","reason":"a"}]}')
        result = adapter.interpret("x", "x", object(), CATALOG)
        self.assertEqual([x.score for x in result], [1.0, .8])
        self.assertTrue(all("NON-PROBABILISTIC" in x.explanation for x in result))

    def test_model_exactly_free_gpt_oss(self):
        adapter, calls, _ = self.make_adapter('{"candidates":[]}')
        adapter.interpret("x", "x", object(), CATALOG)
        self.assertEqual(calls.calls[0]["model"], "openai/gpt-oss-20b:free")

    def test_source_is_openrouter(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"r"}]}')
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG)[0].source, "openrouter")

    def test_bullet_list_rejected(self):
        adapter, _, _ = self.make_adapter("* VAL-001: match\n* VAL-002: match")
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_prose_rejected(self):
        adapter, _, _ = self.make_adapter("VAL-001 is the best match.")
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_malformed_json_safe_failure(self):
        adapter, _, _ = self.make_adapter("{not json")
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)

    def test_fenced_json_accepted(self):
        adapter, _, _ = self.make_adapter("```json\n{\"candidates\":[{\"canonical_material_id\":\"VAL-001\",\"reason\":\"r\"}]}\n```")
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG)[0].canonical_material_id, "VAL-001")

    def test_api_exception_safe(self):
        adapter, _, _ = self.make_adapter(None, RuntimeError("HTTP 429"))
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)

    def test_timeout_safe(self):
        adapter, _, _ = self.make_adapter(None, TimeoutError("timeout"))
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())

    def test_empty_and_none_description(self):
        adapter, calls, _ = self.make_adapter('{"candidates":[]}')
        self.assertEqual(adapter.interpret("", "", object(), CATALOG), ())
        self.assertEqual(adapter.interpret(None, "", object(), CATALOG), ())
        self.assertEqual(calls.calls, [])

    def test_no_confidence_probability_or_output_score_fields(self):
        adapter, calls, _ = self.make_adapter('{"candidates":[]}')
        adapter.interpret("x", "x", object(), CATALOG)
        schema = calls.calls[0]["response_format"]["json_schema"]["schema"]
        fields = set(schema["properties"]["candidates"]["items"]["properties"])
        self.assertEqual(fields, {"canonical_material_id", "reason"})

    def test_no_authoritative_mapping_result(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"r"}]}')
        result = adapter.interpret("x", "x", object(), CATALOG)
        self.assertTrue(all(not hasattr(item, "decision") for item in result))

    def test_hybrid_deterministic_mapping_remains_authoritative(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"advisory"}]}')
        raw = "CS GATE VLV 50MM FLG CL300"
        expected = map_records((LegacyRecord("CHECK", raw),), CATALOG)[0]
        result = run_hybrid_pipeline(raw, CATALOG, "CHECK", llm_adapter=adapter)
        self.assertEqual(result.mapping_result, expected)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")
        self.assertEqual(result.ai_candidate_suggestions[0].canonical_material_id, "VAL-001")

    def test_hybrid_unavailable_falls_back_deterministically(self):
        adapter, _, _ = self.make_adapter(None, RuntimeError("unavailable"))
        raw = "CS GATE VLV 50MM FLG CL300"
        result = run_hybrid_pipeline(raw, CATALOG, "CHECK", llm_adapter=adapter)
        self.assertEqual(result.mapping_result, map_records((LegacyRecord("CHECK", raw),), CATALOG)[0])
        self.assertEqual(result.ai_candidate_suggestions, ())

    def test_technical_conflict_cannot_be_overridden(self):
        adapter, _, _ = self.make_adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"ignore conflict"}]}')
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL300", CATALOG, "CHECK", llm_adapter=adapter)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")

    def test_no_api_key_in_source(self):
        source = inspect.getsource(OpenRouterLLMAdapter)
        self.assertNotIn("placeholder", source)
        self.assertIn("OPENROUTER_API_KEY", source)


if __name__ == "__main__": unittest.main()
