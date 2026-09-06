"""Optional Gemini LLM interpretation adapter for LEGO #10.

Gemini is advisory only. The deterministic LEGO #2-#5 mapping remains the sole
source of truth for MappingResult. The adapter uses structured JSON output and
returns technical evidence, but never asks the model for a final mapping
decision or a confidence/probability value.
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

_FORBIDDEN_OUTPUT_FIELDS = {
    "confidence",
    "probability",
    "score",
    "decision",
    "matched",
    "uncertain",
    "new_candidate",
}


class GeminiLLMAdapter(LLMInterpretationAdapter):
    """Lazy, fault-tolerant Gemini technical candidate-evidence adapter.

    ``compatibility_score`` is supplied by Gemini and preserved as a bounded
    advisory assessment. It is not a calibrated probability or confidence and
    is never authoritative. Technical evidence fields are optional; omitted
    evidence is represented as empty/unknown rather than inferred.
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
        self._response_diagnostic: str | None = None
        self._client_attempted = False

    def status(self) -> AdapterStatus:
        """Report readiness without creating a client or making a network request."""
        if self._failure_detail is not None:
            return AdapterStatus("llm", False, self._failure_detail)
        if not os.getenv("GEMINI_API_KEY"):
            return AdapterStatus("llm", False, "Gemini API key is not configured.")
        if self._client_factory is not None:
            detail = self._response_diagnostic or "Gemini adapter configured; client initialization is lazy."
            return AdapterStatus("llm", True, detail)
        if find_spec("google.genai") is None:
            return AdapterStatus("llm", False, "google-genai is not installed.")
        detail = self._response_diagnostic or f"Gemini {self.model_name} is configured; client initialization is lazy."
        return AdapterStatus("llm", True, detail)

    def interpret(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: Any,
        catalog: tuple[Any, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        """Ask Gemini for advisory candidate evidence and nothing authoritative."""
        if not normalized_description or not catalog:
            self._response_diagnostic = "Gemini request skipped: normalized description or catalog is empty."
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
                input=self._build_prompt(raw_description, normalized_description, attributes, catalog),
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "canonical_material_id": {"type": "string"},
                                "compatibility_score": {"type": "number"},
                                "matching_attributes": {"type": "array", "items": {"type": "string"}},
                                "conflicting_attributes": {"type": "array", "items": {"type": "string"}},
                                "missing_attributes": {"type": "array", "items": {"type": "string"}},
                                "technical_compatible": {"type": "boolean"},
                                "reason": {"type": "string"},
                            },
                            "required": ["canonical_material_id", "compatibility_score"],
                        },
                    },
                },
            )
            response_status = getattr(response, "status", None)
            if response_status != "completed":
                self._failure_detail = (
                    f"Gemini interaction did not complete successfully: status={response_status!r}"
                )
                return ()
            suggestions, diagnostic = self._parse_response(response, catalog)
            self._response_diagnostic = diagnostic
            return suggestions
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
    def _attributes_dict(attributes: Any) -> dict[str, Any]:
        try:
            from dataclasses import fields, is_dataclass
            if is_dataclass(attributes):
                return {
                    field.name: getattr(attributes, field.name)
                    for field in fields(attributes)
                    if getattr(attributes, field.name) is not None
                }
        except Exception:
            pass
        return {"value": repr(attributes)}

    @classmethod
    def _build_prompt(
        cls,
        raw_description: str | None,
        normalized_description: str,
        attributes: Any,
        catalog: tuple[Any, ...],
    ) -> str:
        def format_attributes(values: dict[str, Any], indent: str = "") -> str:
            lines: list[str] = []
            for name, value in values.items():
                if isinstance(value, tuple):
                    rendered = "x".join(str(item) for item in value)
                else:
                    rendered = str(value)
                lines.append(f"{indent}{name}: {rendered}")
            return "\n".join(lines) or f"{indent}<not available>"

        input_values = cls._attributes_dict(attributes)
        sections = [
            "=== INPUT MATERIAL ===",
            f"Raw description: {raw_description or '<not supplied>'}",
            f"Normalized description: {normalized_description}",
            "",
            "=== INPUT MATERIAL ATTRIBUTES ===",
            format_attributes(input_values),
            "",
        ]

        for record in catalog:
            candidate_values = cls._attributes_dict(getattr(record, "attributes", None))
            sections.extend(
                [
                    "=== CANDIDATE CATALOG RECORD ===",
                    f"canonical_material_id: {record.canonical_material_id}",
                    format_attributes(candidate_values),
                    "",
                ]
            )

        sections.extend(
            [
                "TECHNICAL COMPARISON RULES:",
                "1. Matching means both sides contain the attribute and their values are technically compatible.",
                "2. Conflicting means both sides contain the attribute and their values contradict.",
                "3. Missing means one side genuinely lacks the attribute.",
                "4. Never put an input attribute into missing_attributes merely because it is located in the INPUT section.",
                "5. Do not invent attributes or specifications.",
                "6. Compare technical attributes independently.",
                "7. compatibility_score MUST be generated by Gemini from the supplied technical evidence.",
                "8. compatibility_score must be numeric and between 0.0 and 1.0.",
                "9. Do not use positional or rank-based scores.",
                "10. Do not use hard-coded score mappings or positional fallback values.",
                "11. Do not fabricate candidates or scores.",
                "12. Gemini is advisory only.",
                "13. The deterministic MappingResult remains authoritative.",
                "14. Return only supplied canonical material IDs and at most 5 candidates.",
                "15. Return multiple candidates when the input is ambiguous: normally return the 3 strongest distinct plausible candidates. Do not pad the list with weak or fabricated candidates.",
                "16. If one candidate is an overwhelming exact technical match and meaningful alternatives are not plausible, returning only that candidate is acceptable.",
                "17. When multiple candidates are technically or semantically close, include the strongest alternatives even if a technical conflict makes one less suitable; explain the conflict in conflicting_attributes.",
                "18. Order candidates from strongest to weakest according to the Gemini compatibility assessment, not by catalog position.",
                "Return ONLY valid JSON matching the supplied response schema. Do not return confidence, probability, score, or a final decision.",
            ]
        )
        return "\n".join(sections)

    @staticmethod
    def _parse_response(
        response: Any,
        catalog: tuple[Any, ...],
    ) -> tuple[tuple[CandidateSuggestion, ...], str]:
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
        if isinstance(payload, dict):
            candidates = payload.get("candidates")
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = None
        if not isinstance(candidates, list):
            raise ValueError("Gemini response does not match the candidate schema")

        known_ids = {record.canonical_material_id for record in catalog}
        seen: set[str] = set()
        suggestions: list[CandidateSuggestion] = []
        rejected: list[str] = []
        returned_count = len(candidates)
        for item in candidates:
            if len(suggestions) >= MAX_CANDIDATES:
                break
            if not isinstance(item, dict):
                rejected.append("<non-object candidate>: candidate must be an object")
                continue
            canonical_id = item.get("canonical_material_id")
            raw_score = item.get("compatibility_score")
            reason = item.get("reason", "")
            matching = item.get("matching_attributes", [])
            conflicting = item.get("conflicting_attributes", [])
            missing = item.get("missing_attributes", [])
            compatible = item.get("technical_compatible")
            if any(key in item for key in _FORBIDDEN_OUTPUT_FIELDS):
                forbidden = next(key for key in _FORBIDDEN_OUTPUT_FIELDS if key in item)
                rejected.append(f"{canonical_id or '<unknown>'}: forbidden field {forbidden}")
                continue
            if not isinstance(canonical_id, str) or not canonical_id.strip():
                rejected.append("<unknown>: missing canonical_material_id")
                continue
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                rejected.append(f"{canonical_id}: missing or invalid compatibility_score")
                continue
            score = float(raw_score)
            if not isfinite(score) or not 0.0 <= score <= 1.0:
                rejected.append(f"{canonical_id}: compatibility_score outside [0,1]")
                continue
            if reason is not None and not isinstance(reason, str):
                rejected.append(f"{canonical_id}: reason must be a string when supplied")
                continue
            if not all(isinstance(value, list) and all(isinstance(x, str) for x in value) for value in (matching, conflicting, missing)):
                rejected.append(f"{canonical_id}: technical evidence fields must be string arrays")
                continue
            if compatible is not None and not isinstance(compatible, bool):
                rejected.append(f"{canonical_id}: technical_compatible must be boolean or omitted")
                continue
            if canonical_id not in known_ids:
                rejected.append(f"{canonical_id}: unknown catalog ID")
                continue
            if canonical_id in seen:
                rejected.append(f"{canonical_id}: duplicate catalog ID")
                continue
            seen.add(canonical_id)
            compatibility_text = "unknown" if compatible is None else ("yes" if compatible else "no")
            evidence = (
                f"{reason.strip()} Matching: {', '.join(matching) or 'none'}. "
                f"Conflicts: {', '.join(conflicting) or 'none'}. "
                f"Missing: {', '.join(missing) or 'none'}. "
                f"Technically compatible: {compatibility_text}. "
                "[Gemini Compatibility Score is advisory only; NON-PROBABILISTIC and NON-AUTHORITATIVE.]"
            )
            suggestions.append(
                CandidateSuggestion(
                    canonical_material_id=canonical_id,
                    score=score,
                    source="gemini",
                    explanation=evidence,
                    matching_attributes=tuple(matching),
                    conflicting_attributes=tuple(conflicting),
                    missing_attributes=tuple(missing),
                    technical_compatible=compatible,
                )
            )
        if rejected:
            diagnostic = (
                f"Gemini request completed; {returned_count} candidates received; "
                f"{len(suggestions)} accepted; {len(rejected)} rejected: " + "; ".join(rejected)
            )
        elif suggestions:
            diagnostic = f"Gemini request completed; {returned_count} candidates received; {len(suggestions)} accepted."
        else:
            diagnostic = "Gemini returned zero candidates."
        return tuple(suggestions), diagnostic
