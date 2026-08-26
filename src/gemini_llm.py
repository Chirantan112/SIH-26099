"""Optional Gemini LLM interpretation adapter for LEGO #10.

Gemini is advisory only. The deterministic LEGO #2-#5 mapping remains the sole
source of truth for MappingResult. The adapter uses the Google GenAI
Interactions API and structured JSON output, but never asks the model for a
final mapping decision or a confidence/probability value.
"""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from math import isfinite
from typing import Any, Callable

from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.llm_interpretation import LLMInterpretationAdapter

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_CANDIDATES = 5
_RANK_SCORES = (1.0, 0.8, 0.6, 0.4, 0.2)


class GeminiLLMAdapter(LLMInterpretationAdapter):
    """Lazy, fault-tolerant Gemini candidate-suggestion adapter.

    The numeric score is derived only from the model-returned rank. It is a
    bounded compatibility field for ``CandidateSuggestion`` and is explicitly
    NON-PROBABILISTIC and NON-AUTHORITATIVE; it is not Gemini confidence.

    ``client_factory`` is an optional dependency-injection seam for tests. The
    production factory lazily imports ``google-genai`` and constructs a client
    only when ``interpret`` is called.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._client_factory = client_factory
        self._client: Any | None = None
        self._failure_detail: str | None = None
        self._client_attempted = False

    def status(self) -> AdapterStatus:
        """Report readiness without creating a client or making a network request."""
        if self._failure_detail is not None:
            return AdapterStatus("llm", False, self._failure_detail)
        if not os.getenv("GEMINI_API_KEY"):
            return AdapterStatus("llm", False, "Gemini API key is not configured.")
        if self._client_factory is not None:
            return AdapterStatus("llm", True, "Gemini adapter configured; client initialization is lazy.")
        if find_spec("google.genai") is None:
            return AdapterStatus("llm", False, "google-genai is not installed.")
        return AdapterStatus("llm", True, f"Gemini {self.model_name} is configured; client initialization is lazy.")

    def interpret(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: Any,
        catalog: tuple[Any, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        """Ask Gemini for advisory catalog candidates and nothing authoritative."""
        if not normalized_description or not catalog:
            return ()
        if self._failure_detail is not None:
            return ()
        if not os.getenv("GEMINI_API_KEY"):
            self._failure_detail = "Gemini API key is not configured."
            return ()
        if not self._ensure_client():
            return ()

        try:
            response = self._client.interactions.create(
                model=self.model_name,
                input=self._build_prompt(normalized_description, attributes, catalog),
                response_mime_type="application/json",
                response_format=[
                    {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "candidates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "canonical_material_id": {"type": "string"},
                                            "reason": {"type": "string"},
                                        },
                                        "required": ["canonical_material_id", "reason"],
                                    },
                                }
                            },
                            "required": ["candidates"],
                        },
                    }
                ],
            )
            response_status = getattr(response, "status", None)
            if response_status != "completed":
                self._failure_detail = (
                    f"Gemini interaction did not complete successfully: status={response_status!r}"
                )
                return ()
            return self._parse_response(response, catalog)
        except Exception as error:
            self._failure_detail = f"Gemini interpretation failed: {error}"
            return ()

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if self._client_attempted:
            return False
        self._client_attempted = True
        try:
            if self._client_factory is not None:
                self._client = self._client_factory(os.environ["GEMINI_API_KEY"])
            else:
                from google import genai

                self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            return True
        except Exception as error:
            self._failure_detail = f"Gemini SDK/client initialization failed: {error}"
            self._client = None
            return False

    @staticmethod
    def _build_prompt(normalized_description: str, attributes: Any, catalog: tuple[Any, ...]) -> str:
        catalog_ids = [record.canonical_material_id for record in catalog]
        return (
            "You are an advisory material-description interpreter. Return candidate "
            "canonical material IDs only; do not make a final mapping decision, "
            "do not return MATCHED/UNCERTAIN/NEW_CANDIDATE, and do not provide "
            "confidence or probability. Use only IDs from the supplied catalog. "
            "For each candidate give a short reason based on the supplied description "
            "and explicit technical attributes.\n\n"
            f"Normalized description: {normalized_description}\n"
            f"Explicit attributes: {attributes!r}\n"
            f"Allowed catalog IDs: {catalog_ids}\n"
        )

    @staticmethod
    def _parse_response(response: Any, catalog: tuple[Any, ...]) -> tuple[CandidateSuggestion, ...]:
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("Gemini response has no structured output text")
        cleaned = output_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if not lines or lines[0].strip() not in ("```", "```json"):
                raise ValueError("Gemini response contains an unsupported Markdown fence")
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("Gemini response contains an incomplete Markdown fence")
            cleaned = "\n".join(lines[1:-1]).strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
            raise ValueError("Gemini response does not match the candidate schema")

        known_ids = {record.canonical_material_id for record in catalog}
        seen: set[str] = set()
        suggestions: list[CandidateSuggestion] = []
        for item in payload["candidates"]:
            if len(suggestions) >= MAX_CANDIDATES:
                break
            if not isinstance(item, dict):
                continue
            canonical_id = item.get("canonical_material_id")
            reason = item.get("reason")
            if not isinstance(canonical_id, str) or not canonical_id.strip():
                continue
            if not isinstance(reason, str) or not reason.strip():
                continue
            if canonical_id not in known_ids or canonical_id in seen:
                continue
            score = _RANK_SCORES[len(suggestions)]
            if not isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            seen.add(canonical_id)
            suggestions.append(
                CandidateSuggestion(
                    canonical_material_id=canonical_id,
                    score=score,
                    source="gemini",
                    explanation=(
                        f"{reason.strip()} "
                        "[Advisory rank score only; NON-PROBABILISTIC and NON-AUTHORITATIVE.]"
                    ),
                )
            )
        return tuple(suggestions)
