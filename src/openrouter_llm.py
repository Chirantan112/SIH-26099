"""Optional OpenRouter GPT-OSS advisory LLM adapter for LEGO #10."""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from math import isfinite
from typing import Any, Callable

from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.llm_interpretation import LLMInterpretationAdapter

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_CANDIDATES = 5
_RANK_SCORES = (1.0, 0.8, 0.6, 0.4, 0.2)


class OpenRouterLLMAdapter(LLMInterpretationAdapter):
    """Lazy, fault-tolerant OpenRouter candidate-suggestion adapter."""

    def __init__(self, model_name: str = DEFAULT_OPENROUTER_MODEL, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, client_factory: Callable[[str, str, float], Any] | None = None) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._client_factory = client_factory
        self._client: Any | None = None
        self._client_attempted = False
        self._failure_detail: str | None = None

    def status(self) -> AdapterStatus:
        if self._failure_detail is not None:
            return AdapterStatus("llm", False, self._failure_detail)
        if not os.getenv("OPENROUTER_API_KEY"):
            return AdapterStatus("llm", False, "OpenRouter API key is not configured.")
        if self._client_factory is not None:
            return AdapterStatus("llm", True, f"OpenRouter {self.model_name} is configured; client initialization is lazy.")
        if find_spec("openai") is None:
            return AdapterStatus("llm", False, "OpenAI-compatible client dependency is not installed.")
        return AdapterStatus("llm", True, f"OpenRouter {self.model_name} is configured; client initialization is lazy.")

    def interpret(self, raw_description: str | None, normalized_description: str, attributes: Any, catalog: tuple[Any, ...]) -> tuple[CandidateSuggestion, ...]:
        if not normalized_description or not catalog:
            return ()
        if self._failure_detail is not None or not os.getenv("OPENROUTER_API_KEY"):
            if self._failure_detail is None:
                self._failure_detail = "OpenRouter API key is not configured."
            return ()
        if not self._ensure_client():
            return ()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Return advisory material candidates only. Do not make a final mapping decision. Do not return confidence, probability, score, MATCHED, UNCERTAIN, or NEW_CANDIDATE. Every canonical_material_id must exactly match one supplied catalog ID."},
                    {"role": "user", "content": self._build_prompt(normalized_description, attributes, catalog)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "material_candidates",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {"canonical_material_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["canonical_material_id", "reason"], "additionalProperties": False}}},
                            "required": ["candidates"],
                            "additionalProperties": False,
                        },
                    },
                },
                timeout=self.timeout_seconds,
            )
            return self._parse_response(response, catalog)
        except Exception as error:
            self._failure_detail = f"OpenRouter interpretation failed: {error}"
            return ()

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if self._client_attempted:
            return False
        self._client_attempted = True
        try:
            api_key = os.environ["OPENROUTER_API_KEY"]
            if self._client_factory is not None:
                self._client = self._client_factory(api_key, OPENROUTER_BASE_URL, self.timeout_seconds)
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=self.timeout_seconds, max_retries=0)
            return True
        except Exception as error:
            self._failure_detail = f"OpenRouter client initialization failed: {error}"
            return False

    @staticmethod
    def _build_prompt(normalized_description: str, attributes: Any, catalog: tuple[Any, ...]) -> str:
        ids = [record.canonical_material_id for record in catalog]
        return f"Return ONLY a JSON object matching the requested schema. Candidate suggestions only; no prose, Markdown, confidence, probability, score, or final mapping decision. Use only exact catalog IDs.\nNormalized description: {normalized_description}\nExplicit attributes: {attributes!r}\nAllowed catalog IDs: {ids}\n"

    @staticmethod
    def _parse_response(response: Any, catalog: tuple[Any, ...]) -> tuple[CandidateSuggestion, ...]:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError("OpenRouter response has no choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter response has no textual structured content")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip() in ("```", "```json"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
            raise ValueError("OpenRouter response does not match the candidate schema")
        known_ids = {record.canonical_material_id for record in catalog}
        seen: set[str] = set()
        suggestions: list[CandidateSuggestion] = []
        for item in payload["candidates"]:
            if len(suggestions) >= MAX_CANDIDATES:
                break
            if not isinstance(item, dict):
                continue
            canonical_id, reason = item.get("canonical_material_id"), item.get("reason")
            if not isinstance(canonical_id, str) or not canonical_id.strip() or not isinstance(reason, str) or not reason.strip() or canonical_id not in known_ids or canonical_id in seen:
                continue
            score = _RANK_SCORES[len(suggestions)]
            if not isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            seen.add(canonical_id)
            suggestions.append(CandidateSuggestion(canonical_material_id=canonical_id, score=score, source="openrouter", explanation=f"{reason.strip()} [Advisory rank score only; NON-PROBABILISTIC and NON-AUTHORITATIVE.]"))
        return tuple(suggestions)
