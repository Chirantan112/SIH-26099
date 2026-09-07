"""Fault-tolerant, non-authoritative orchestration for LEGO #9A.

This module adds optional AI suggestions around the existing deterministic
pipeline. It never changes candidate ranking, linkage, or catalog-mapping
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
from src.record_linkage import compare_records


MAX_AI_CANDIDATE_SUGGESTIONS = 5
_SuggestionAdapter = TypeVar("_SuggestionAdapter")


@dataclass(frozen=True)
class AIConsensus:
    """Explicit advisory conclusion derived only from AI evidence."""

    conclusion: str  # MATCHED, UNCERTAIN, NEW_CANDIDATE, or UNAVAILABLE
    canonical_material_id: str | None
    reason: str
    local_candidates: tuple[str, ...]
    gemini_candidates: tuple[str, ...]


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
    ai_consensus: AIConsensus
    fallback_used: bool
    fallback_information: str
    explanation: str


def _safe_status(component: str, adapter: object) -> AdapterStatus:
    """Read optional status without allowing an adapter failure to stop mapping."""
    try:
        status = adapter.status()  # type: ignore[attr-defined]
    except Exception as error:
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
    except Exception as error:
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
            or not isinstance(suggestion.matching_attributes, tuple)
            or not isinstance(suggestion.conflicting_attributes, tuple)
            or not isinstance(suggestion.missing_attributes, tuple)
            or suggestion.technical_compatible not in (True, False, None)
        ):
            continue
        valid.append(suggestion)

    return tuple(sorted(valid, key=lambda item: (-item.score, item.canonical_material_id, item.source)))[:MAX_AI_CANDIDATE_SUGGESTIONS]


def _source_candidates(suggestions: tuple[CandidateSuggestion, ...], source: str) -> tuple[CandidateSuggestion, ...]:
    return tuple(item for item in suggestions if item.source == source)


def _verified_compatibility(
    attributes: MaterialAttributes,
    catalog: tuple[CatalogRecord, ...],
    suggestions: tuple[CandidateSuggestion, ...],
) -> dict[str, bool | None]:
    """Independently verify AI candidates against the existing linkage rules."""
    catalog_by_id = {record.canonical_material_id: record for record in catalog}
    verified: dict[str, bool | None] = {}
    for suggestion in suggestions:
        record = catalog_by_id.get(suggestion.canonical_material_id)
        if record is None:
            continue
        linkage = compare_records(attributes, record.attributes)
        if linkage.decision == "SAME":
            verified[suggestion.canonical_material_id] = True
        elif linkage.decision == "DIFFERENT":
            verified[suggestion.canonical_material_id] = False
        else:
            verified[suggestion.canonical_material_id] = None
    return verified


def _has_explicit_conflict(
    suggestions: tuple[CandidateSuggestion, ...],
    verified: dict[str, bool | None],
) -> bool:
    """Return True only when AI and independent technical evidence rule out every candidate."""
    return bool(suggestions) and all(
        item.technical_compatible is False
        and verified.get(item.canonical_material_id) is False
        for item in suggestions
    )


def _ai_consensus(
    statuses: tuple[AdapterStatus, AdapterStatus],
    suggestions: tuple[CandidateSuggestion, ...],
    verified: dict[str, bool | None],
) -> AIConsensus:
    """Derive conservative consensus from AI candidates plus independent verification."""
    local_status, gemini_status = statuses
    local = _source_candidates(suggestions, "local_embedding")
    gemini = _source_candidates(suggestions, "gemini")
    local_compatible = tuple(item for item in local if verified.get(item.canonical_material_id) is True)
    gemini_compatible = tuple(item for item in gemini if verified.get(item.canonical_material_id) is True)
    local_ids = tuple(item.canonical_material_id for item in local)
    gemini_ids = tuple(item.canonical_material_id for item in gemini)

    if not local_status.available and not gemini_status.available:
        return AIConsensus(
            "UNAVAILABLE",
            None,
            "Both advisory AI components are unavailable; no AI conclusion was produced.",
            local_ids,
            gemini_ids,
        )

    local_evidence_available = local_status.available and bool(local)
    gemini_evidence_available = gemini_status.available and bool(gemini)

    if local_evidence_available and gemini_evidence_available:
        if local_compatible and gemini_compatible:
            if local_compatible[0].canonical_material_id == gemini_compatible[0].canonical_material_id:
                candidate = local_compatible[0].canonical_material_id
                return AIConsensus(
                    "MATCHED",
                    candidate,
                    "Local NLP and Gemini identify the same candidate, and independent deterministic technical validation confirms compatibility. AI remains advisory and does not change the authoritative result.",
                    local_ids,
                    gemini_ids,
                )
            return AIConsensus(
                "UNCERTAIN",
                None,
                "Local NLP and Gemini identify different technically compatible candidates; advisory evidence disagrees.",
                local_ids,
                gemini_ids,
            )

        if local_compatible or gemini_compatible:
            return AIConsensus(
                "UNCERTAIN",
                None,
                "One advisory component supplies a technically compatible candidate while the other does not provide matching verified evidence.",
                local_ids,
                gemini_ids,
            )

        if _has_explicit_conflict(local, verified) and _has_explicit_conflict(gemini, verified):
            return AIConsensus(
                "NEW_CANDIDATE",
                None,
                "Both advisory components supplied only candidates that they explicitly rule out, and independent technical validation also rules them out. No new material identity is fabricated.",
                local_ids,
                gemini_ids,
            )

        return AIConsensus(
            "UNCERTAIN",
            None,
            "AI evidence is incomplete or unresolved; insufficient verified evidence for a new-candidate conclusion.",
            local_ids,
            gemini_ids,
        )

    if local_evidence_available:
        compatible = local_compatible
        if compatible:
            return AIConsensus(
                "UNCERTAIN",
                compatible[0].canonical_material_id,
                "Only Local NLP returned a candidate independently verified as technically compatible; a single advisory source is insufficient for a strong AI consensus match.",
                local_ids,
                gemini_ids,
            )
        if _has_explicit_conflict(local, verified):
            return AIConsensus(
                "NEW_CANDIDATE",
                None,
                "Local NLP supplied only candidates with explicit AI conflicts that are also independently ruled out. No new material identity is fabricated.",
                local_ids,
                gemini_ids,
            )
        return AIConsensus(
            "UNCERTAIN",
            local[0].canonical_material_id,
            "Only Local NLP returned usable candidates, but the available evidence is incomplete or unresolved.",
            local_ids,
            gemini_ids,
        )

    if gemini_evidence_available:
        compatible = gemini_compatible
        if compatible:
            return AIConsensus(
                "UNCERTAIN",
                compatible[0].canonical_material_id,
                "Only Gemini returned a candidate independently verified as technically compatible; a single advisory source is insufficient for a strong AI consensus match.",
                local_ids,
                gemini_ids,
            )
        if _has_explicit_conflict(gemini, verified):
            return AIConsensus(
                "NEW_CANDIDATE",
                None,
                "Gemini supplied only candidates with explicit AI conflicts that are also independently ruled out. No new material identity is fabricated.",
                local_ids,
                gemini_ids,
            )
        return AIConsensus(
            "UNCERTAIN",
            gemini[0].canonical_material_id,
            "Only Gemini returned usable candidates, but the available evidence is incomplete or unresolved.",
            local_ids,
            gemini_ids,
        )

    return AIConsensus(
        "UNCERTAIN",
        None,
        "AI services are available, but neither returned usable candidate evidence. Empty evidence is not treated as a new-candidate finding.",
        local_ids,
        gemini_ids,
    )


def run_hybrid_pipeline(
    raw_description: str | None,
    catalog: tuple[CatalogRecord, ...],
    legacy_material_code: str = "HYBRID-INPUT",
    retrieval_adapter: RetrievalAdapter | None = None,
    llm_adapter: LLMInterpretationAdapter | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> HybridResult:
    """Run the existing deterministic flow with an optional execution-progress hook."""

    def _report(stage: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage)
        except Exception:
            pass

    retrieval = retrieval_adapter or UnavailableRetrievalAdapter()
    llm = llm_adapter or UnavailableLLMAdapter()
    _report("input")
    normalization = normalize_description(raw_description)
    _report("normalize")
    extraction = extract_attributes(raw_description)
    _report("extract")
    mapping_input = raw_description if normalization.normalized_text else ""

    # This is intentionally unchanged deterministic LEGO #2-#5 authority.
    mapping_result = map_records((LegacyRecord(legacy_material_code, mapping_input),), catalog)[0]
    _report("match")

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
    _report("ai_advisory")
    suggestions = _valid_suggestions(retrieval_suggestions + llm_suggestions, catalog)
    statuses = (retrieval_status, llm_status)
    verified = _verified_compatibility(extraction.attributes, catalog, suggestions)
    consensus = _ai_consensus(statuses, suggestions, verified)
    fallback_used = not all(status.available for status in statuses)
    fallback_information = (
        "Deterministic LEGO #2-#5 mapping remains active; unavailable or failed AI adapters were skipped."
        if fallback_used
        else "Deterministic LEGO #2-#5 mapping remains authoritative; AI consensus is advisory evidence only."
    )
    result = HybridResult(
        legacy_material_code=legacy_material_code,
        normalized_description=normalization.normalized_text,
        normalization_transformations=normalization.transformations,
        attributes=extraction.attributes,
        mapping_result=mapping_result,
        ai_statuses=statuses,
        ai_candidate_suggestions=suggestions,
        ai_consensus=consensus,
        fallback_used=fallback_used,
        fallback_information=fallback_information,
        explanation="AI suggestions and consensus are advisory only; final mapping is the unchanged deterministic LEGO #4/#5 result.",
    )
    _report("decide")
    return result
