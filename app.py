"""Judge-ready Streamlit dashboard for CPSE material harmonization."""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.ai_retrieval import CandidateSuggestion, UnavailableRetrievalAdapter
from src.attribute_extraction import MaterialAttributes
from src.demo_pipeline import load_demo_catalog
from src.hybrid_pipeline import HybridResult, run_hybrid_pipeline
from src.llm_interpretation import UnavailableLLMAdapter

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

try:
    from src.gemini_llm import GeminiLLMAdapter
except Exception:  # pragma: no cover
    GeminiLLMAdapter = None  # type: ignore[assignment,misc]

try:
    from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter
except Exception:  # pragma: no cover
    LocalEmbeddingRetrievalAdapter = None  # type: ignore[assignment,misc]

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "demo" / "material_master.csv"
DEMO_EXAMPLES = {
    "MATCHED": "CS GATE VLV 50MM FLG CL150",
    "UNCERTAIN": "GATE VLV CS 50MM CL150",
    "NEW CANDIDATE": "PIPE CS OD 999 MM THK 3 MM SCH-40 PLAIN END",
}
ATTRIBUTE_LABELS = {
    "category": "Category", "valve_type": "Valve type", "material": "Material",
    "size_mm": "Size (mm)", "pressure_class": "Pressure class", "connection": "Connection",
    "bearing_family": "Bearing family", "dimensions": "Dimensions (mm)",
    "dimension_unit_present": "Dimension unit stated", "od_mm": "Outside diameter (mm)",
    "thickness_mm": "Thickness (mm)", "schedule": "Schedule", "end": "End",
}


def _display_value(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, tuple):
        return " × ".join(str(item) for item in value)
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def attribute_rows(attributes: MaterialAttributes) -> list[dict[str, str]]:
    """Return primitive, UI-safe extracted attribute rows."""
    return [
        {"Attribute": ATTRIBUTE_LABELS[field.name], "Extracted value": _display_value(getattr(attributes, field.name))}
        for field in fields(attributes)
    ]


def candidate_rows(candidates: tuple[Any, ...]) -> list[dict[str, str | float]]:
    """Return bounded deterministic evidence in a table-safe representation."""
    return [
        {"Canonical material ID": candidate.canonical_material_id, "Decision": candidate.decision,
         "Score": candidate.score, "Explanation": candidate.explanation}
        for candidate in candidates
    ]


def advisory_rows(suggestions: tuple[CandidateSuggestion, ...]) -> list[dict[str, str | float]]:
    """Return AI suggestions without exposing internal dataclass representations."""
    return [
        {"Canonical material ID": suggestion.canonical_material_id, "Source": suggestion.source,
         "Advisory Rank": suggestion.score, "Reason": suggestion.explanation}
        for suggestion in suggestions
    ]


def _friendly_status(status: Any) -> str:
    if status.available:
        return "Available"
    detail = (status.detail or "").lower()
    if "api key" in detail:
        return "Unavailable — credentials not configured"
    if "not installed" in detail:
        return "Unavailable — optional dependency not installed"
    return "Unavailable — deterministic matching remains active"


def _analysis_adapters(ai_enabled: bool) -> tuple[Any, Any]:
    if not ai_enabled:
        return UnavailableRetrievalAdapter("AI advisory is disabled."), UnavailableLLMAdapter("AI advisory is disabled.")
    retrieval = LocalEmbeddingRetrievalAdapter() if LocalEmbeddingRetrievalAdapter is not None else UnavailableRetrievalAdapter()
    llm = GeminiLLMAdapter() if GeminiLLMAdapter is not None else UnavailableLLMAdapter("Gemini adapter is unavailable.")
    return retrieval, llm


def analyze_material(description: str | None, catalog: tuple[Any, ...], ai_enabled: bool,
                     retrieval_adapter: Any | None = None, llm_adapter: Any | None = None) -> HybridResult:
    """Run the existing hybrid orchestration; never create a second pipeline."""
    if retrieval_adapter is None or llm_adapter is None:
        retrieval_adapter, llm_adapter = _analysis_adapters(ai_enabled)
    return run_hybrid_pipeline(description, catalog, legacy_material_code="DASHBOARD-INPUT",
                               retrieval_adapter=retrieval_adapter, llm_adapter=llm_adapter)


def _inject_styles() -> None:
    st.markdown("""
    <style>
    .hero { padding:1.4rem 1.6rem; border:1px solid rgba(128,128,128,.25); border-radius:18px; background:linear-gradient(135deg,rgba(31,78,121,.12),rgba(255,255,255,.02)); }
    .eyebrow { font-size:.75rem; letter-spacing:.12em; font-weight:700; opacity:.72; text-transform:uppercase; }
    .stage { padding:.9rem; border:1px solid rgba(128,128,128,.25); border-radius:14px; min-height:92px; }
    .stage-num { font-size:.72rem; font-weight:800; opacity:.55; }
    .stage-title { font-weight:750; margin-top:.2rem; }
    .stage-note { font-size:.78rem; opacity:.68; }
    .decision-note { font-size:.82rem; opacity:.7; }
    .advisory-note { padding:.8rem 1rem; border-left:4px solid #d99b27; background:rgba(217,155,39,.08); border-radius:8px; }
    .status-pill { font-weight:650; }
    </style>
    """, unsafe_allow_html=True)


def _render_pipeline() -> None:
    stages = (("01", "NORMALIZE", "Canonical text"), ("02", "EXTRACT", "Technical attributes"),
              ("03", "MATCH", "Deterministic linkage"), ("04", "ADVISE", "Optional AI"), ("05", "DECIDE", "Authoritative result"))
    columns = st.columns(5)
    for column, (number, title, note) in zip(columns, stages):
        with column:
            st.markdown(f'<div class="stage"><div class="stage-num">{number}</div><div class="stage-title">{title}</div><div class="stage-note">{note}</div></div>', unsafe_allow_html=True)


def _render_status(result: HybridResult | None = None) -> None:
    st.markdown("### AI services")
    if result is None:
        retrieval, llm = _analysis_adapters(True)
        statuses = (retrieval.status(), llm.status())
    else:
        statuses = result.ai_statuses
    left, right = st.columns(2)
    for column, status, label in ((left, statuses[0], "Local NLP"), (right, statuses[1], "Gemini")):
        with column:
            symbol = "●" if status.available else "○"
            st.markdown(f'**{label}** &nbsp; <span class="status-pill">{symbol} {_friendly_status(status)}</span>', unsafe_allow_html=True)
            if not status.available:
                st.caption("AI advisory unavailable. The deterministic matching pipeline remains active.")


def _render_decision(result: HybridResult) -> None:
    decision = result.mapping_result.decision
    canonical_id = result.mapping_result.canonical_material_id or "Not assigned"
    score = result.mapping_result.score
    if decision == "MATCHED":
        st.success(f"FINAL DETERMINISTIC DECISION  ·  {decision}")
    elif decision == "UNCERTAIN":
        st.warning(f"FINAL DETERMINISTIC DECISION  ·  {decision}")
    else:
        st.error(f"FINAL DETERMINISTIC DECISION  ·  {decision}")
    first, second, third = st.columns(3)
    first.metric("Canonical Material", canonical_id)
    second.metric("Deterministic Score", f"{score:.3f}")
    third.metric("Decision", decision)
    st.markdown('<div class="decision-note">Authoritative result — produced by the deterministic LEGO #2–#5 engine.</div>', unsafe_allow_html=True)
    st.info(result.mapping_result.explanation)


def _render_analysis(result: HybridResult, raw_description: str | None) -> None:
    st.markdown("## Decision")
    _render_decision(result)
    st.markdown("## Processing trace")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original description**")
        st.code(raw_description or "No description provided", language=None)
        st.markdown("**Normalized description**")
        st.code(result.normalized_description or "No normalized description", language=None)
    with right:
        st.markdown("**Transformations**")
        if result.normalization_transformations:
            for item in result.normalization_transformations:
                st.write(f"• {item}")
        else:
            st.caption("No transformations reported.")
    st.markdown("**Extracted technical attributes**")
    st.dataframe(attribute_rows(result.attributes), hide_index=True, use_container_width=True)
    st.markdown("## Deterministic match evidence")
    st.caption("Deterministic / Authoritative")
    if result.mapping_result.all_candidates:
        st.dataframe(candidate_rows(result.mapping_result.all_candidates), hide_index=True, use_container_width=True)
    else:
        st.caption("No candidate evidence is available.")
    st.markdown("## AI Advisory Suggestions")
    st.caption("Non-authoritative recommendations for analyst review")
    if result.ai_candidate_suggestions:
        st.dataframe(advisory_rows(result.ai_candidate_suggestions), hide_index=True, use_container_width=True)
    else:
        st.caption("No advisory suggestions are available. Deterministic mapping remains fully functional.")
    st.markdown('<div class="advisory-note">AI suggestions are advisory only. The final decision is produced by the deterministic matching engine. Advisory Rank is not a probability.</div>', unsafe_allow_html=True)


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to run this dashboard. Use: streamlit run app.py")
    st.set_page_config(page_title="CPSE Material Harmonization", page_icon="🔗", layout="wide", initial_sidebar_state="expanded")
    _inject_styles()
    with st.sidebar:
        st.markdown("### SYSTEM")
        st.success("Deterministic Engine · Active")
        st.caption("Authoritative mapping engine")
        ai_enabled = st.toggle("AI Advisory", value=False, help="Enable optional local NLP and Gemini suggestions. This never changes the final deterministic decision.")
        st.markdown("### MODE")
        st.info("Hybrid AI Advisory" if ai_enabled else "Deterministic")
        st.markdown("### ARCHITECTURE")
        st.caption("Deterministic engine remains authoritative. AI suggestions are bounded, optional, and non-authoritative.")
    st.markdown('<div class="eyebrow">CPSE MATERIAL HARMONIZATION</div>', unsafe_allow_html=True)
    st.title("Material Harmonization")
    st.caption("AI-Assisted Material Standardization & Cross-Reference")
    st.markdown("Standardize legacy CPSE material descriptions into a common canonical catalog.")
    try:
        catalog = load_demo_catalog(CATALOG_PATH)
    except (OSError, ValueError):
        st.error("The local reference catalog could not be loaded. Deterministic analysis is unavailable until the catalog is restored.")
        return
    _render_pipeline()
    st.caption("AI suggestions never override deterministic decisions.")
    st.divider()
    st.markdown("## Analyze a material")
    example_columns = st.columns(3)
    for column, (outcome, example) in zip(example_columns, DEMO_EXAMPLES.items()):
        column.button(outcome.title(), on_click=lambda value=example: st.session_state.__setitem__("material_description", value), use_container_width=True)
    description = st.text_area("Material Description", key="material_description", height=120, placeholder="Gate Valve Carbon Steel 150 50mm Flanged")
    analyze = st.button("Analyze Material", type="primary", use_container_width=True)
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if analyze:
        try:
            st.session_state["analysis_result"] = analyze_material(description, catalog, ai_enabled)
        except (TypeError, ValueError):
            st.session_state["analysis_result"] = None
            st.error("The description could not be processed safely. Please check the material description and try again.")
    result = st.session_state.get("analysis_result")
    if result is None:
        st.divider()
        st.markdown("### Ready for analysis")
        st.write("Enter a legacy material description to begin.")
        st.caption('Example: "Gate Valve Carbon Steel 150 50mm Flanged"')
        _render_status()
    else:
        st.divider()
        _render_analysis(result, description)
        st.divider()
        _render_status(result)
        if result.fallback_used:
            st.info("AI advisory unavailable or failed. The deterministic matching pipeline remains active.")


if __name__ == "__main__":
    main()
