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
    GeminiLLMAdapter = None

try:
    from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter
except Exception:  # pragma: no cover
    LocalEmbeddingRetrievalAdapter = None

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "demo" / "material_master.csv"
DEMO_EXAMPLES = {
    "MATCHED": "CS GATE VLV 50MM FLG CL150",
    "UNCERTAIN": "GATE VLV CS 50MM CL150",
    "NEW CANDIDATE": "PIPE CS OD 999 MM THK 3 MM SCH-40 PLAIN END",
}

ATTRIBUTE_LABELS = {
    "category": "Category",
    "valve_type": "Valve type",
    "material": "Material",
    "size_mm": "Size (mm)",
    "pressure_class": "Pressure class",
    "connection": "Connection",
    "bearing_family": "Bearing family",
    "dimensions": "Dimensions (mm)",
    "dimension_unit_present": "Dimension unit stated",
    "od_mm": "Outside diameter (mm)",
    "thickness_mm": "Thickness (mm)",
    "schedule": "Schedule",
    "end": "End",
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
    return [
        {"Attribute": ATTRIBUTE_LABELS[field.name], "Extracted value": _display_value(getattr(attributes, field.name))}
        for field in fields(attributes)
    ]


def candidate_rows(candidates: tuple[Any, ...]) -> list[dict[str, str | float]]:
    return [
        {
            "Canonical material ID": candidate.canonical_material_id,
            "Decision": candidate.decision,
            "Score": candidate.score,
            "Explanation": candidate.explanation,
        }
        for candidate in candidates
    ]


def advisory_rows(suggestions: tuple[CandidateSuggestion, ...]) -> list[dict[str, str | float]]:
    return [
        {
            "Canonical material ID": suggestion.canonical_material_id,
            "Source": suggestion.source,
            "Advisory Rank": suggestion.score,
            "Reason": suggestion.explanation,
        }
        for suggestion in suggestions
    ]


def _friendly_status(status: Any) -> str:
    if status.available:
        return "Available"
    detail = (status.detail or "").lower()
    if "api key" in detail or "credentials" in detail:
        return "Unavailable — credentials not configured"
    if "not installed" in detail:
        return "Unavailable — optional dependency not installed"
    return "Unavailable — deterministic matching remains active"


def _analysis_adapters(ai_enabled: bool) -> tuple[Any, Any]:
    if not ai_enabled:
        return (
            UnavailableRetrievalAdapter("AI advisory is disabled."),
            UnavailableLLMAdapter("AI advisory is disabled."),
        )
    retrieval = LocalEmbeddingRetrievalAdapter() if LocalEmbeddingRetrievalAdapter is not None else UnavailableRetrievalAdapter()
    llm = GeminiLLMAdapter() if GeminiLLMAdapter is not None else UnavailableLLMAdapter("Gemini adapter is unavailable.")
    return retrieval, llm


def analyze_material(
    description: str | None,
    catalog: tuple[Any, ...],
    ai_enabled: bool,
    retrieval_adapter: Any | None = None,
    llm_adapter: Any | None = None,
) -> HybridResult:
    """Run the existing hybrid orchestration; never create a second pipeline."""
    if retrieval_adapter is None or llm_adapter is None:
        retrieval_adapter, llm_adapter = _analysis_adapters(ai_enabled)
    return run_hybrid_pipeline(
        description,
        catalog,
        legacy_material_code="DASHBOARD-INPUT",
        retrieval_adapter=retrieval_adapter,
        llm_adapter=llm_adapter,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.35rem 1.55rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(31,78,121,.10), rgba(255,255,255,.02));
            margin-bottom: 1rem;
        }
        .eyebrow {
            font-size: .72rem;
            letter-spacing: .13em;
            font-weight: 750;
            opacity: .68;
            text-transform: uppercase;
        }
        .hero-title { font-size: 2.1rem; font-weight: 800; margin: .15rem 0 .2rem; }
        .hero-subtitle { font-size: 1rem; opacity: .72; }
        .architecture-card {
            padding: .75rem .8rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 12px;
            min-height: 78px;
            background: rgba(255,255,255,.02);
        }
        .architecture-kicker { font-size: .68rem; letter-spacing: .08em; font-weight: 800; opacity: .62; }
        .architecture-main { font-weight: 800; margin-top: .15rem; }
        .architecture-note { font-size: .75rem; opacity: .68; }
        .flow-card {
            padding: .85rem .75rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 14px;
            min-height: 92px;
            background: rgba(255,255,255,.02);
        }
        .flow-num { font-size: .68rem; font-weight: 800; opacity: .52; }
        .flow-title { font-weight: 800; margin-top: .2rem; font-size: .88rem; }
        .flow-note { font-size: .72rem; opacity: .66; }
        .flow-arrow { text-align: center; font-size: 1.2rem; opacity: .45; padding-top: 1.8rem; }
        .authority-card {
            padding: 1.15rem 1.25rem;
            border: 1px solid rgba(48,128,72,.30);
            border-radius: 16px;
            background: rgba(48,128,72,.06);
        }
        .authority-label, .advisory-label {
            font-size: .72rem;
            font-weight: 850;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .authority-label { color: #2f7d4a; }
        .advisory-label { color: #9b6a13; }
        .authority-id { font-size: 1.85rem; font-weight: 850; margin: .25rem 0; }
        .decision-pill {
            display: inline-block;
            padding: .18rem .55rem;
            border-radius: 999px;
            font-size: .74rem;
            font-weight: 800;
            background: rgba(48,128,72,.12);
        }
        .advisory-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(217,155,39,.28);
            border-radius: 16px;
            background: rgba(217,155,39,.06);
        }
        .advisory-warning {
            padding: .85rem 1rem;
            margin-top: .9rem;
            border-left: 4px solid #d99b27;
            border-radius: 8px;
            background: rgba(217,155,39,.08);
            font-weight: 650;
        }
        .status-dot { font-size: 1rem; vertical-align: -1px; }
        .status-title { font-weight: 780; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">CPSE MATERIAL HARMONIZATION</div>'
        '<div class="hero-title">CPSE Material Harmonization</div>'
        '<div class="hero-subtitle">Deterministic material matching with optional local NLP and Gemini advisory intelligence.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cards = (
        ("DETERMINISTIC ENGINE", "AUTHORITATIVE", "Final decision source"),
        ("LOCAL NLP", "ADVISORY", "Technical interpretation"),
        ("GEMINI LLM", "ADVISORY", "Candidate reasoning"),
    )
    for column, (kicker, main, note) in zip(cols, cards):
        with column:
            st.markdown(
                f'<div class="architecture-card"><div class="architecture-kicker">{kicker}</div>'
                f'<div class="architecture-main">{main}</div><div class="architecture-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption("Deterministic matching decides. AI assists.")


def _render_pipeline() -> None:
    stages = (
        ("01", "INPUT", "Legacy material description"),
        ("02", "NORMALIZE", "Canonical text"),
        ("03", "EXTRACT", "NLP / technical attributes"),
        ("04", "MATCH", "Deterministic catalog mapping"),
        ("05", "AI ADVISE", "Local NLP + Gemini"),
        ("06", "DECIDE", "Authoritative result"),
    )
    columns = st.columns([1.7, .25, 1.7, .25, 1.7, .25, 1.7, .25, 1.7, .25, 1.7])
    for index, (number, title, note) in enumerate(stages):
        with columns[index * 2]:
            st.markdown(
                f'<div class="flow-card"><div class="flow-num">{number}</div>'
                f'<div class="flow-title">{title}</div><div class="flow-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
        if index < len(stages) - 1:
            with columns[index * 2 + 1]:
                st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)


def _render_status(result: HybridResult | None = None) -> None:
    st.markdown("### AI Services")
    if result is None:
        retrieval, llm = _analysis_adapters(True)
        statuses = (retrieval.status(), llm.status())
    else:
        statuses = result.ai_statuses
    columns = st.columns(2)
    for column, status, label in ((columns[0], statuses[0], "Local NLP"), (columns[1], statuses[1], "Gemini LLM")):
        symbol = "●" if status.available else "○"
        with column:
            st.markdown(
                f'**{label}** <span class="status-dot">{symbol}</span> '
                f'<span class="status-title">{_friendly_status(status)}</span>',
                unsafe_allow_html=True,
            )
            st.caption("Advisory only — never overrides the deterministic decision.")


def _render_decision(result: HybridResult) -> None:
    decision = result.mapping_result.decision
    canonical_id = result.mapping_result.canonical_material_id or "Not assigned"
    score = result.mapping_result.score
    st.markdown(
        f'<div class="authority-card"><div class="authority-label">Authoritative Decision</div>'
        f'<div class="authority-id">{canonical_id}</div>'
        f'<span class="decision-pill">{decision}</span></div>',
        unsafe_allow_html=True,
    )
    first, second, third = st.columns(3)
    first.metric("Canonical Material", canonical_id)
    second.metric("Deterministic Score", f"{score:.3f}")
    third.metric("Decision", decision)
    st.caption("Produced by the deterministic LEGO #2–#5 engine. This result is authoritative.")
    st.info(result.mapping_result.explanation)


def _render_advisory(result: HybridResult) -> None:
    local = [item for item in result.ai_candidate_suggestions if item.source == "local_embedding"]
    gemini = [item for item in result.ai_candidate_suggestions if item.source == "gemini"]
    st.markdown("## AI Advisory")
    st.markdown(
        '<div class="advisory-warning">AI suggestions are advisory only. They cannot override the deterministic decision.</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for column, label, source_rows in ((cols[0], "Local NLP", local), (cols[1], "Gemini LLM", gemini)):
        with column:
            st.markdown(f'<div class="advisory-card"><div class="advisory-label">{label} · ADVISORY</div>', unsafe_allow_html=True)
            if source_rows:
                st.dataframe(advisory_rows(tuple(source_rows)), hide_index=True, use_container_width=True)
            else:
                st.caption("No suggestions available from this advisory component.")
            st.markdown('</div>', unsafe_allow_html=True)


def _render_analysis(result: HybridResult, raw_description: str | None) -> None:
    st.markdown("## Authoritative Result")
    _render_decision(result)

    st.markdown("## Processing Evidence")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original Description**")
        st.code(raw_description or "No description provided", language=None)
        st.markdown("**Normalized Description**")
        st.code(result.normalized_description or "No normalized description", language=None)
    with right:
        st.markdown("**Normalization Transformations**")
        if result.normalization_transformations:
            for item in result.normalization_transformations:
                st.write(f"• {item}")
        else:
            st.caption("No transformations reported.")

    st.markdown("### NLP / Attribute Extraction")
    st.caption("Local NLP stage: technical interpretation before deterministic catalog matching.")
    st.dataframe(attribute_rows(result.attributes), hide_index=True, use_container_width=True)

    with st.expander("Deterministic Candidate Evidence", expanded=False):
        if result.mapping_result.all_candidates:
            st.dataframe(candidate_rows(result.mapping_result.all_candidates), hide_index=True, use_container_width=True)
        else:
            st.caption("No candidate evidence is available.")

    _render_advisory(result)


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to run this dashboard. Use: streamlit run app.py")
    st.set_page_config(
        page_title="CPSE Material Harmonization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    with st.sidebar:
        st.markdown("### SYSTEM")
        st.success("Deterministic Engine · Active")
        st.caption("AUTHORITATIVE")
        st.divider()
        ai_enabled = st.toggle(
            "AI Advisory",
            value=False,
            help="Enable optional Local NLP and Gemini suggestions. They never change the final decision.",
        )
        st.markdown("### MODE")
        st.info("Hybrid AI Advisory" if ai_enabled else "Deterministic")
        st.markdown("### ARCHITECTURE")
        st.caption("Deterministic matching decides. AI assists.")

    _render_header()
    _render_pipeline()

    try:
        catalog = load_demo_catalog(CATALOG_PATH)
    except (OSError, ValueError):
        st.error("The local reference catalog could not be loaded.")
        return

    st.divider()
    st.markdown("## Analyze Material")
    examples = st.columns(3)
    for column, (outcome, example) in zip(examples, DEMO_EXAMPLES.items()):
        with column:
            if st.button(outcome.title(), use_container_width=True):
                st.session_state["material_description"] = example

    description = st.text_area(
        "Material Description",
        key="material_description",
        height=120,
        placeholder="Gate Valve Carbon Steel 150 50mm Flanged",
    )
    analyze = st.button("Analyze Material", type="primary", use_container_width=True)

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    if analyze:
        try:
            st.session_state["analysis_result"] = analyze_material(description, catalog, ai_enabled)
        except (TypeError, ValueError):
            st.session_state["analysis_result"] = None
            st.error("The description could not be processed safely. Please check the input and try again.")

    result = st.session_state.get("analysis_result")
    if result is None:
        st.divider()
        st.markdown("### Ready for analysis")
        st.write("Enter a legacy material description to begin.")
        st.caption('Example: "Gate Valve Carbon Steel 150 50mm Flanged"')
        _render_status()
        return

    st.divider()
    _render_analysis(result, description)
    st.divider()
    _render_status(result)
    if result.fallback_used:
        st.info("AI advisory unavailable or failed. Deterministic matching remains fully operational.")


if __name__ == "__main__":
    main()
