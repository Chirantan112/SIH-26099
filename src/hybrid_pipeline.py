"""Fault-tolerant, non-authoritative orchestration for LEGO #9A.

This module adds optional AI suggestions around the existing deterministic
pipeline.  It never changes candidate ranking, linkage, or catalog-mapping
logic: LEGO #4/#5 remains the sole authority for final decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable, TypeVar

from src.ai_retrieval import AdapterStatus, CandidateSuggestion, RetrievalAdapter, UnavailableRetrievalAdapter
from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, MappingResult, map_records
from src.llm_interpretation import LLMInterpretationAdapter, UnavailableLLMAdapter
from src.normalization import normalize_description


MAX_AI_CANDIDATE_SUGGESTIONS = 5
_SuggestionAdapter = TypeVar("_SuggestionAdapter")


@dataclass(frozen=True)
class HybridResult:
    """Deterministic result plus bounded, non-authoritative AI diagnostics."""

    legacy_material_code: str
    normalized_description: str
    normalization_transformations: tuple[str, ...]
    attributes: MaterialAttributes
    mapping_result: MappingResult
    ai_statuses: tuple[AdapterStatus, ...]
    ai_candidate_suggestions: tuple[CandidateSuggestion, ...]
    fallback_used: bool
    fallback_information: str
    explanation: str


def _safe_status(component: str, adapter: object) -> AdapterStatus:
    """Read optional status without allowing an adapter failure to stop mapping."""

    try:
        status = adapter.status()  # type: ignore[attr-defined]
    except Exception as error:  # Adapter boundaries must be fault tolerant.
        return AdapterStatus(component=component, available=False, detail=f"Status unavailable: {error}")
    if not isinstance(status, AdapterStatus) or status.component != component:
        return AdapterStatus(component=component, available=False, detail="Adapter returned an invalid status.")
    return status


def _safe_suggestions(
    component: str,
    adapter: _SuggestionAdapter,
    operation: Callable[[_SuggestionAdapter], tuple[CandidateSuggestion, ...]],
    status: AdapterStatus,
) -> tuple[tuple[CandidateSuggestion, ...], AdapterStatus]:
    """Run one optional adapter and convert failures into a non-fatal status."""

    if not status.available:
        return (), status
    try:
        suggestions = operation(adapter)
    except Exception as error:  # Adapter boundaries must be fault tolerant.
        return (), replace(status, available=False, detail=f"Adapter failed: {error}")
    if not isinstance(suggestions, tuple):
        return (), replace(status, available=False, detail="Adapter returned an invalid suggestion collection.")
    return suggestions, status


def _valid_suggestions(
    suggestions: tuple[CandidateSuggestion, ...],
    catalog: tuple[CatalogRecord, ...],
) -> tuple[CandidateSuggestion, ...]:
    """Accept only bounded, well-formed references to known catalog IDs."""

    known_ids = {record.canonical_material_id for record in catalog}
    valid: list[CandidateSuggestion] = []
    for suggestion in suggestions:
        if not isinstance(suggestion, CandidateSuggestion):
            continue
        if (
            suggestion.canonical_material_id not in known_ids
            or not isinstance(suggestion.score, float)
            or not isfinite(suggestion.score)
            or not isinstance(suggestion.source, str)
            or not suggestion.source.strip()
            or not isinstance(suggestion.explanation, str)
        ):
            continue
        valid.append(suggestion)

    # This deterministic ordering is only presentation/audit metadata.  It does
    # not feed into map_records or alter the final deterministic decision.
    return tuple(sorted(valid, key=lambda item: (-item.score, item.canonical_material_id, item.source)))[:MAX_AI_CANDIDATE_SUGGESTIONS]


def run_hybrid_pipeline(
    raw_description: str | None,
    catalog: tuple[CatalogRecord, ...],
    legacy_material_code: str = "HYBRID-INPUT",
    retrieval_adapter: RetrievalAdapter | None = None,
    llm_adapter: LLMInterpretationAdapter | None = None,
) -> HybridResult:
    """Run AI advice safely around the unchanged deterministic LEGO #2-#5 flow."""

    retrieval = retrieval_adapter or UnavailableRetrievalAdapter()
    llm = llm_adapter or UnavailableLLMAdapter()
    normalization = normalize_description(raw_description)
    extraction = extract_attributes(raw_description)
    mapping_input = raw_description if normalization.normalized_text else ""

    # This is intentionally the same full-catalog deterministic mapping used by
    # LEGO #5; AI suggestions cannot filter, rank, or override it.
    mapping_result = map_records((LegacyRecord(legacy_material_code, mapping_input),), catalog)[0]

    retrieval_status = _safe_status("local_nlp", retrieval)
    llm_status = _safe_status("llm", llm)
    retrieval_suggestions, retrieval_status = _safe_suggestions(
        "local_nlp",
        retrieval,
        lambda adapter: adapter.retrieve(raw_description, normalization.normalized_text, extraction.attributes, catalog),
        retrieval_status,
    )
    llm_suggestions, llm_status = _safe_suggestions(
        "llm",
        llm,
        lambda adapter: adapter.interpret(raw_description, normalization.normalized_text, extraction.attributes, catalog),
        llm_status,
    )
    suggestions = _valid_suggestions(retrieval_suggestions + llm_suggestions, catalog)
    statuses = (retrieval_status, llm_status)
    fallback_used = not all(status.available for status in statuses)
    fallback_information = (
        "Deterministic LEGO #2-#5 mapping remains active; unavailable or failed AI adapters were skipped."
        if fallback_used
        else "Deterministic LEGO #2-#5 mapping verified all advisory AI suggestions."
    )
    return HybridResult(
        legacy_material_code=legacy_material_code,
        normalized_description=normalization.normalized_text,
        normalization_transformations=normalization.transformations,
        attributes=extraction.attributes,
        mapping_result=mapping_result,
        ai_statuses=statuses,
        ai_candidate_suggestions=suggestions,
        fallback_used=fallback_used,
        fallback_information=fallback_information,
        explanation="AI suggestions are advisory only; final mapping is the unchanged deterministic LEGO #4/#5 result.",
    )
