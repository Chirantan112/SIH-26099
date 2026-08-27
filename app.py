"""Judge-facing Streamlit dashboard for CPSE material harmonization."""

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
        return "Credentials not configured"
    if "not installed" in detail:
        return "Optional dependency not installed"
    return "Advisory unavailable"


def _analysis_adapters(ai_enabled: bool) -> tuple[Any, Any]:
    if not ai_enabled:
        return (
            UnavailableRetrievalAdapter("AI advisory is disabled."),
            UnavailableLLMAdapter("AI advisory is disabled."),
        )
    retrieval = (
        LocalEmbeddingRetrievalAdapter()
        if LocalEmbeddingRetrievalAdapter is not None
        else UnavailableRetrievalAdapter("Local NLP adapter is unavailable.")
    )
    llm = (
        GeminiLLMAdapter()
        if GeminiLLMAdapter is not None
        else UnavailableLLMAdapter("Gemini adapter is unavailable.")
    )
    return retrieval, llm


def analyze_material(
    description: str | None,
    catalog: tuple[Any, ...],
    ai_enabled: bool,
    retrieval_adapter: Any | None = None,
    llm_adapter: Any | None = None,
) -> HybridResult:
    """Run the existing hybrid orchestration; never create a second pipeline."""
    if not ai_enabled:
        retrieval_adapter = UnavailableRetrievalAdapter("AI advisory is disabled.")
        llm_adapter = UnavailableLLMAdapter("AI advisory is disabled.")
    elif retrieval_adapter is None or llm_adapter is None:
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
        :root {
            --cpse-bg:#07111f;
            --cpse-card:#0d1929;
            --cpse-card-2:#101f32;
            --cpse-border:rgba(170,190,215,.16);
            --cpse-muted:#8ea0b7;
            --cpse-text:#eef5ff;
            --cpse-green:#55d68a;
            --cpse-cyan:#58c7e8;
            --cpse-purple:#b48cff;
            --cpse-amber:#e6b35a;
            --cpse-red:#ef6b72;
        }
        .stApp {
            background:radial-gradient(circle at 85% 0%, rgba(44,103,145,.12), transparent 34%), var(--cpse-bg);
            color:var(--cpse-text);
        }
        .block-container {
            max-width:1500px;
            padding-top:1.25rem;
            padding-bottom:2rem;
        }
        [data-testid="stSidebar"] {
            background:#081421;
            border-right:1px solid var(--cpse-border);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top:1.25rem;
        }
        .hero {
            padding:1.6rem 1.7rem;
            border:1px solid var(--cpse-border);
            border-radius:20px;
            background:linear-gradient(135deg,rgba(26,65,95,.34),rgba(9,19,33,.76));
            box-shadow:0 18px 45px rgba(0,0,0,.18);
        }
        .eyebrow,.section-kicker {
            font-size:.72rem;
            letter-spacing:.14em;
            font-weight:800;
            opacity:.68;
            text-transform:uppercase;
        }
        .hero-title {
            margin-top:.3rem;
            font-size:clamp(1.7rem,3.2vw,2.75rem);
            line-height:1.05;
            font-weight:850;
            letter-spacing:-.04em;
        }
        .hero-subtitle {
            max-width:900px;
            margin-top:.7rem;
            color:var(--cpse-muted);
            font-size:clamp(.9rem,1.4vw,1rem);
            line-height:1.55;
        }
        .hero-meta {
            margin-top:1rem;
            display:flex;
            gap:.55rem;
            flex-wrap:wrap;
        }
        .meta-pill {
            display:inline-block;
            padding:.32rem .68rem;
            border:1px solid var(--cpse-border);
            border-radius:999px;
            color:#cbd9e8;
            font-size:.7rem;
            font-weight:750;
        }
        .meta-online {
            color:var(--cpse-green);
            border-color:rgba(85,214,138,.3);
        }
        .architecture-card {
            height:100%;
            min-height:118px;
            padding:1rem 1.05rem;
            margin-top:.8rem;
            border:1px solid var(--cpse-border);
            border-radius:15px;
            background:rgba(13,25,41,.84);
            box-shadow:0 8px 22px rgba(0,0,0,.08);
        }
        .architecture-kicker {
            font-size:.66rem;
            letter-spacing:.1em;
            font-weight:850;
            color:var(--cpse-muted);
        }
        .architecture-main {
            margin-top:.52rem;
            font-size:.98rem;
            font-weight:850;
        }
        .architecture-note {
            margin-top:.35rem;
            color:var(--cpse-muted);
            font-size:.76rem;
            line-height:1.45;
        }
        .authority-card {
            padding:1.35rem 1.4rem;
            border:1px solid rgba(85,214,138,.38);
            border-radius:18px;
            background:linear-gradient(135deg,rgba(25,88,61,.26),rgba(12,27,35,.88));
            box-shadow:0 16px 38px rgba(0,0,0,.16);
        }
        .authority-label {
            color:var(--cpse-green);
            font-size:.69rem;
            font-weight:850;
            letter-spacing:.12em;
            text-transform:uppercase;
        }
        .authority-id {
            display:inline-block;
            margin-top:.55rem;
            margin-right:.7rem;
            font-size:clamp(1.8rem,4vw,2.8rem);
            line-height:1;
            font-weight:900;
            letter-spacing:-.04em;
        }
        .decision-pill {
            display:inline-block;
            vertical-align:middle;
            padding:.38rem .68rem;
            border:1px solid rgba(85,214,138,.4);
            border-radius:999px;
            color:#bff5d3;
            font-size:.69rem;
            font-weight:850;
            letter-spacing:.08em;
        }
        .advisory-warning {
            margin:.65rem 0 1rem;
            padding:.75rem 1rem;
            border-left:3px solid var(--cpse-amber);
            background:rgba(230,179,90,.08);
            border-radius:8px;
            color:#d8c49b;
            font-size:.82rem;
        }
        .advisory-card {
            min-height:170px;
            padding:1rem;
            border:1px solid rgba(180,140,255,.2);
            border-radius:15px;
            background:rgba(17,24,42,.82);
        }
        .advisory-label {
            margin-bottom:.65rem;
            color:var(--cpse-purple);
            font-size:.69rem;
            font-weight:850;
            letter-spacing:.1em;
            text-transform:uppercase;
        }
        .flow-grid {
            display:grid;
            grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr;
            gap:.45rem;
            align-items:stretch;
        }
        .flow-card {
            min-height:106px;
            padding:.8rem .72rem;
            border:1px solid var(--cpse-border);
            border-radius:13px;
            background:var(--cpse-card);
        }
        .flow-num {
            color:var(--cpse-cyan);
            font-size:.67rem;
            font-weight:850;
            letter-spacing:.1em;
        }
        .flow-title {
            margin-top:.32rem;
            font-size:.76rem;
            font-weight:850;
            letter-spacing:.04em;
        }
        .flow-note {
            margin-top:.35rem;
            color:var(--cpse-muted);
            font-size:.67rem;
            line-height:1.38;
        }
        .flow-arrow {
            align-self:center;
            color:#5d738c;
            font-size:1rem;
            text-align:center;
        }
        .mode-banner {
            margin:.75rem 0 1rem;
            padding:.68rem .85rem;
            border:1px solid var(--cpse-border);
            border-radius:10px;
            background:rgba(13,25,41,.7);
            color:var(--cpse-muted);
            font-size:.78rem;
        }
        .mode-banner strong { color:var(--cpse-text); }
        .sidebar-brand {
            font-size:1.02rem;
            font-weight:900;
            line-height:1.05;
            letter-spacing:-.02em;
        }
        .sidebar-sub {
            margin-top:.3rem;
            color:var(--cpse-muted);
            font-size:.7rem;
        }
        .sidebar-label {
            margin-top:1.1rem;
            margin-bottom:.4rem;
            color:var(--cpse-muted);
            font-size:.66rem;
            font-weight:850;
            letter-spacing:.12em;
            text-transform:uppercase;
        }
        .service-row {
            padding:.5rem 0;
            border-bottom:1px solid rgba(170,190,215,.08);
            font-size:.76rem;
        }
        .service-dot { margin-right:.3rem; }
        .online { color:var(--cpse-green); }
        .advisory { color:var(--cpse-purple); }
        .authoritative { color:var(--cpse-green); }
        .mobile-divider { display:none; }
        @media (max-width: 1100px) {
            .flow-grid { grid-template-columns:repeat(3,1fr); }
            .flow-arrow { display:none; }
        }
        @media (max-width: 850px) {
            .block-container { padding:.9rem .75rem 1.4rem; }
            .hero { padding:1.2rem 1rem; }
            .architecture-card { min-height:0; }
        }
        @media (max-width: 720px) {
            .block-container { padding:.75rem .55rem 1.25rem; }
            .flow-grid { grid-template-columns:1fr; }
            .flow-card { min-height:72px; }
            .flow-arrow { display:none; }
            .authority-id { display:block; margin-bottom:.65rem; }
            .hero-title { font-size:1.65rem; }
            .hero-subtitle { font-size:.88rem; }
            .mobile-divider { display:block; height:1px; background:var(--cpse-border); margin:1rem 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(ai_enabled: bool) -> None:
    mode = "HYBRID AI" if ai_enabled else "DETERMINISTIC ONLY"
    st.markdown(
        f'<div class="hero"><div class="eyebrow">CPSE MATERIAL HARMONIZATION</div><div class="hero-title">CPSE Material Harmonization</div><div class="hero-subtitle">AI-assisted standardization of legacy CPSE material descriptions into a common canonical material vocabulary.</div><div class="hero-meta"><span class="meta-pill">SIH 2026 · PS 26099</span><span class="meta-pill meta-online">● SYSTEM ONLINE</span><span class="meta-pill">MODE · {mode}</span></div></div>',
        unsafe_allow_html=True,
    )


def _render_architecture() -> None:
    cols = st.columns(3)
    cards = (
        ("DETERMINISTIC ENGINE", "AUTHORITATIVE", "Final decision source", "authoritative"),
        ("LOCAL NLP", "ADVISORY", "Optional local embedding retrieval", "advisory"),
        ("GEMINI 2.5 FLASH", "LLM · ADVISORY", "Candidate interpretation", "advisory"),
    )
    for column, (kicker, main, note, kind) in zip(cols, cards):
        with column:
            st.markdown(
                f'<div class="architecture-card"><div class="architecture-kicker">{kicker}</div><div class="architecture-main {kind}">{main}</div><div class="architecture-note">{note}</div></div>',
                unsafe_allow_html=True,
            )


def _render_pipeline() -> None:
    stages = (
        ("01", "INPUT", "Legacy description"),
        ("02", "NORMALIZE", "Canonical text"),
        ("03", "EXTRACT", "Technical attributes"),
        ("04", "MATCH", "Deterministic mapping"),
        ("05", "AI ADVISE", "Optional advisory"),
        ("06", "DECIDE", "Authoritative result"),
    )
    st.markdown('<div class="flow-grid">', unsafe_allow_html=True)
    for index, (number, title, note) in enumerate(stages):
        st.markdown(
            f'<div class="flow-card"><div class="flow-num">{number}</div><div class="flow-title">{title}</div><div class="flow-note">{note}</div></div>',
            unsafe_allow_html=True,
        )
        if index < len(stages) - 1:
            st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_status(result: HybridResult | None = None, ai_enabled: bool = True) -> None:
    st.markdown("### Service Status")
    if result is None and not ai_enabled:
        st.markdown('<div class="mode-banner"><strong>AI advisory disabled.</strong> Local NLP and Gemini are not invoked in this mode.</div>', unsafe_allow_html=True)
        return
    if result is None:
        retrieval, llm = _analysis_adapters(True)
        statuses = (retrieval.status(), llm.status())
    else:
        statuses = result.ai_statuses
    columns = st.columns(2)
    for column, status, label in (
        (columns[0], statuses[0], "Local NLP"),
        (columns[1], statuses[1], "Gemini 2.5 Flash"),
    ):
        symbol = "●" if status.available else "○"
        dot_class = "online" if status.available else ""
        with column:
            st.markdown(f'<span class="service-dot {dot_class}">{symbol}</span> <strong>{label}</strong> — {_friendly_status(status)}', unsafe_allow_html=True)
            st.caption("Advisory only — never overrides the deterministic decision.")


def _render_decision(result: HybridResult) -> None:
    decision = result.mapping_result.decision
    canonical_id = result.mapping_result.canonical_material_id or "Not assigned"
    score = result.mapping_result.score
    st.markdown(f'<div class="authority-card"><div class="authority-label">✓ Authoritative Decision</div><div class="authority-id">{canonical_id}</div><span class="decision-pill">{decision}</span></div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    first.metric("Canonical Material", canonical_id)
    second.metric("Deterministic Match Score", f"{score:.3f}")
    third.metric("Decision", decision)
    st.caption("Final decision source: deterministic harmonization engine. AI suggestions cannot change this result.")
    st.info(result.mapping_result.explanation)


def _render_advisory(result: HybridResult) -> None:
    local = [item for item in result.ai_candidate_suggestions if item.source == "local_embedding"]
    gemini = [item for item in result.ai_candidate_suggestions if item.source == "gemini"]
    st.markdown("## AI Advisory")
    st.markdown('<div class="advisory-warning">Advisory intelligence only. Local NLP and Gemini suggestions cannot override the deterministic decision.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for column, label, source_rows in (
        (cols[0], "Local NLP · Embedding Retrieval", local),
        (cols[1], "Gemini 2.5 Flash · LLM", gemini),
    ):
        with column:
            st.markdown(f'<div class="advisory-card"><div class="advisory-label">{label} · ADVISORY</div>', unsafe_allow_html=True)
            if source_rows:
                st.dataframe(advisory_rows(tuple(source_rows)), hide_index=True, use_container_width=True)
            else:
                st.caption("No advisory suggestions available from this component.")
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
                st.write(f"✓ {item}")
        else:
            st.caption("No transformations reported.")
    st.markdown("### Technical Attributes")
    st.dataframe(attribute_rows(result.attributes), hide_index=True, use_container_width=True)
    with st.expander("Deterministic Candidate Evidence", expanded=False):
        if result.mapping_result.all_candidates:
            st.dataframe(candidate_rows(result.mapping_result.all_candidates), hide_index=True, use_container_width=True)
        else:
            st.caption("No candidate evidence is available.")
    _render_advisory(result)


def _set_example(example: str) -> None:
    st.session_state["material_description"] = example
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_mode", None)


def _sidebar_mode_changed() -> None:
    selected = st.session_state.get("analysis_mode_selector")
    new_ai_enabled = selected == "Hybrid AI"
    old_ai_enabled = st.session_state.get("ai_enabled", False)
    if new_ai_enabled != old_ai_enabled:
        st.session_state["ai_enabled"] = new_ai_enabled


def _render_sidebar() -> bool:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">CPSE MATERIAL<br>HARMONIZATION</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">SIH 2026 · Problem Statement 26099</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">System Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="online">●</span> SYSTEM ONLINE</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Analysis Mode</div>', unsafe_allow_html=True)
        current = st.session_state.get("ai_enabled", False)
        st.radio(
            "Analysis Mode",
            options=("Hybrid AI", "Deterministic Only"),
            index=0 if current else 1,
            label_visibility="collapsed",
            key="analysis_mode_selector",
            on_change=_sidebar_mode_changed,
        )
        ai_enabled = st.session_state.get("ai_enabled", False)
        st.markdown('<div class="sidebar-label">Decision Authority</div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="authoritative">●</span> Deterministic Engine</div>', unsafe_allow_html=True)
        st.caption("AUTHORITATIVE · final decision source")
        st.markdown('<div class="sidebar-label">AI Services</div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="advisory">●</span> Local NLP <span style="float:right">ADVISORY</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="advisory">●</span> Gemini 2.5 Flash <span style="float:right">LLM · ADVISORY</span></div>', unsafe_allow_html=True)
        st.caption("AI advisory can be disabled. Disabled mode does not invoke Local NLP or Gemini.")
        return ai_enabled


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to run this dashboard. Use: streamlit run app.py")
    st.set_page_config(page_title="CPSE Material Harmonization", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    _inject_styles()
    if "ai_enabled" not in st.session_state:
        st.session_state["ai_enabled"] = False
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "analysis_mode" not in st.session_state:
        st.session_state["analysis_mode"] = None
    if "material_description" not in st.session_state:
        st.session_state["material_description"] = ""
    ai_enabled = _render_sidebar()
    previous_result = st.session_state.get("analysis_result")
    previous_mode = st.session_state.get("analysis_mode")
    mode_changed = previous_result is not None and previous_mode is not None and previous_mode != ai_enabled
    _render_header(ai_enabled)
    _render_architecture()
    st.markdown('<div class="mode-banner"><strong>Decision authority:</strong> deterministic matching. AI components provide separate advisory intelligence only.</div>', unsafe_allow_html=True)
    _render_pipeline()
    st.divider()
    st.markdown('<div class="section-kicker">MATERIAL ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("### Analyze a legacy material description")
    st.caption("Enter a CPSE material description. The deterministic engine remains authoritative in both modes.")
    examples = st.columns(3)
    for column, (outcome, example) in zip(examples, DEMO_EXAMPLES.items()):
        with column:
            if st.button(outcome.title(), use_container_width=True, key=f"example_{outcome}"):
                _set_example(example)
                st.rerun()
    description = st.text_area(
        "Material Description",
        key="material_description",
        height=125,
        placeholder="Gate Valve Carbon Steel 150 50mm Flanged",
        help="Example: CS GATE VLV 50MM FLG CL150",
    )
    analyze = st.button("Analyze Material →", type="primary", use_container_width=True)
    if analyze:
        if not description or not description.strip():
            st.session_state["analysis_result"] = None
            st.session_state["analysis_mode"] = None
            st.warning("MATERIAL DESCRIPTION REQUIRED — Enter a legacy material description to begin analysis.")
        else:
            with st.spinner("Analyzing material through the selected pipeline…"):
                try:
                    st.session_state["analysis_result"] = analyze_material(description, load_demo_catalog(CATALOG_PATH), ai_enabled)
                    st.session_state["analysis_mode"] = ai_enabled
                except (OSError, TypeError, ValueError):
                    st.session_state["analysis_result"] = None
                    st.session_state["analysis_mode"] = None
                    st.error("The description could not be processed safely. Please check the input and try again.")
    result = st.session_state.get("analysis_result")
    if mode_changed and result is not None:
        old_mode = "Hybrid AI" if previous_mode else "Deterministic Only"
        st.warning(f"ANALYSIS MODE CHANGED — Previous results were generated using {old_mode}. Re-analyze to generate results for the current mode.")
        if st.button("Analyze Again →", type="primary", use_container_width=True):
            with st.spinner("Re-analyzing material through the selected pipeline…"):
                try:
                    st.session_state["analysis_result"] = analyze_material(description, load_demo_catalog(CATALOG_PATH), ai_enabled)
                    st.session_state["analysis_mode"] = ai_enabled
                    st.rerun()
                except (OSError, TypeError, ValueError):
                    st.error("The description could not be processed safely. Please check the input and try again.")
        _render_status(result, ai_enabled=bool(previous_mode))
        return
    if result is None:
        st.divider()
        st.markdown("### Ready for analysis")
        st.write("Enter a legacy material description to begin.")
        st.caption('Example: "CS GATE VLV 50MM FLG CL150"')
        _render_status(ai_enabled=ai_enabled)
        return
    st.divider()
    _render_analysis(result, description)
    st.divider()
    _render_status(result, ai_enabled=ai_enabled)
    if result.fallback_used:
        st.info("One or more optional AI advisory components were unavailable or failed. Deterministic matching remains fully operational.")
    st.divider()
    st.caption("CPSE MATERIAL HARMONIZATION · SIH 2026 · PS 26099 · Deterministic Decision Engine · AI Advisory Layer")


if __name__ == "__main__":
    main()
