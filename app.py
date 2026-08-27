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
    return [{"Attribute": ATTRIBUTE_LABELS[field.name], "Extracted value": _display_value(getattr(attributes, field.name))} for field in fields(attributes)]


def candidate_rows(candidates: tuple[Any, ...]) -> list[dict[str, str | float]]:
    return [{"Canonical material ID": candidate.canonical_material_id, "Decision": candidate.decision, "Score": candidate.score, "Explanation": candidate.explanation} for candidate in candidates]


def advisory_rows(suggestions: tuple[CandidateSuggestion, ...]) -> list[dict[str, str | float]]:
    return [{"Canonical material ID": suggestion.canonical_material_id, "Source": suggestion.source, "Advisory Rank": suggestion.score, "Reason": suggestion.explanation} for suggestion in suggestions]


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
        return UnavailableRetrievalAdapter("AI advisory is disabled."), UnavailableLLMAdapter("AI advisory is disabled.")
    retrieval = LocalEmbeddingRetrievalAdapter() if LocalEmbeddingRetrievalAdapter is not None else UnavailableRetrievalAdapter("Local NLP adapter is unavailable.")
    llm = GeminiLLMAdapter() if GeminiLLMAdapter is not None else UnavailableLLMAdapter("Gemini adapter is unavailable.")
    return retrieval, llm


def analyze_material(description: str | None, catalog: tuple[Any, ...], ai_enabled: bool, retrieval_adapter: Any | None = None, llm_adapter: Any | None = None) -> HybridResult:
    """Run the existing hybrid orchestration; never create a second pipeline."""
    if not ai_enabled:
        retrieval_adapter = UnavailableRetrievalAdapter("AI advisory is disabled.")
        llm_adapter = UnavailableLLMAdapter("AI advisory is disabled.")
    elif retrieval_adapter is None or llm_adapter is None:
        retrieval_adapter, llm_adapter = _analysis_adapters(ai_enabled)
    return run_hybrid_pipeline(description, catalog, legacy_material_code="DASHBOARD-INPUT", retrieval_adapter=retrieval_adapter, llm_adapter=llm_adapter)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:#040a12;
            --panel:#081220;
            --panel-2:#0b1727;
            --panel-3:#0e1c2d;
            --line:rgba(170,195,220,.15);
            --line-strong:rgba(170,195,220,.24);
            --text:#f5f8fc;
            --muted:#8ea3b9;
            --green:#59e39a;
            --cyan:#55d2f2;
            --purple:#bd95ff;
            --amber:#f2bd63;
            --red:#ef7078;
        }
        .stApp {
            background:
                radial-gradient(1000px 520px at 88% -8%, rgba(54,137,185,.13), transparent 58%),
                radial-gradient(780px 400px at 8% 25%, rgba(40,77,124,.07), transparent 60%),
                var(--bg);
        }
        .block-container {
            max-width:1560px;
            padding:1rem clamp(.7rem,2vw,2.2rem) 2.5rem;
        }
        [data-testid="stSidebar"] {
            background:linear-gradient(180deg,#07111d,#06101b);
            border-right:1px solid var(--line);
        }
        [data-testid="stSidebar"] .block-container { padding:1rem .95rem 1.5rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { font-weight:760; }
        .sidebar-brand { font-size:1.02rem; font-weight:950; line-height:1.05; letter-spacing:-.025em; }
        .sidebar-sub { margin-top:.35rem; color:var(--muted); font-size:.68rem; }
        .sidebar-label { margin-top:1.05rem; margin-bottom:.45rem; color:#93a9bf; font-size:.63rem; font-weight:900; letter-spacing:.14em; text-transform:uppercase; }
        .sidebar-note { color:var(--muted); font-size:.69rem; line-height:1.45; }
        .service-row { display:flex; align-items:center; gap:.45rem; padding:.48rem 0; border-bottom:1px solid rgba(170,195,220,.08); font-size:.74rem; }
        .service-row .right { margin-left:auto; color:var(--muted); font-size:.63rem; font-weight:850; letter-spacing:.06em; }
        .green { color:var(--green); }
        .purple { color:var(--purple); }
        .cyan { color:var(--cyan); }
        .amber { color:var(--amber); }
        .hero {
            position:relative;
            overflow:hidden;
            padding:clamp(1.3rem,2.8vw,2rem);
            border:1px solid var(--line);
            border-radius:24px;
            background:
                linear-gradient(135deg,rgba(15,40,61,.86),rgba(6,14,24,.95) 72%),
                var(--panel);
            box-shadow:0 28px 65px rgba(0,0,0,.24);
        }
        .hero::before {
            content:"";
            position:absolute;
            inset:0;
            background:linear-gradient(90deg,transparent,rgba(85,210,242,.035),transparent);
            pointer-events:none;
        }
        .eyebrow,.section-kicker { color:#a8b9cb; font-size:.67rem; letter-spacing:.17em; font-weight:900; text-transform:uppercase; }
        .hero-title { margin-top:.38rem; font-size:clamp(2rem,4.1vw,3.55rem); line-height:.98; font-weight:950; letter-spacing:-.055em; }
        .hero-subtitle { max-width:980px; margin-top:.8rem; color:var(--muted); font-size:clamp(.9rem,1.55vw,1.06rem); line-height:1.6; }
        .hero-meta { margin-top:1.05rem; display:flex; flex-wrap:wrap; gap:.5rem; }
        .meta-pill { padding:.36rem .72rem; border:1px solid var(--line); border-radius:999px; color:#d0dce7; background:rgba(255,255,255,.02); font-size:.68rem; font-weight:820; }
        .meta-online { color:var(--green); border-color:rgba(89,227,154,.32); }
        .architecture-grid { margin-top:1rem; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; }
        .architecture-card {
            min-height:132px;
            padding:1.1rem;
            border:1px solid var(--line);
            border-radius:18px;
            background:linear-gradient(180deg,rgba(13,27,45,.94),rgba(7,16,28,.96));
            box-shadow:0 12px 28px rgba(0,0,0,.12);
        }
        .architecture-kicker { color:#9fb3c7; font-size:.63rem; letter-spacing:.11em; font-weight:900; }
        .architecture-main { margin-top:.62rem; font-size:1rem; font-weight:950; }
        .architecture-main.authoritative { color:var(--green); }
        .architecture-main.advisory { color:var(--purple); }
        .architecture-note { margin-top:.4rem; color:var(--muted); font-size:.74rem; line-height:1.48; }
        .mode-strip { margin:1rem 0 .8rem; padding:.72rem .9rem; border:1px solid var(--line); border-radius:12px; background:rgba(10,21,35,.78); color:var(--muted); font-size:.77rem; }
        .mode-strip strong { color:var(--text); }
        .input-shell { margin-top:1rem; padding:1rem; border:1px solid var(--line); border-radius:19px; background:linear-gradient(180deg,rgba(9,19,32,.82),rgba(6,13,23,.76)); }
        .input-label { font-size:.84rem; font-weight:880; }
        .scenario-caption { margin:.35rem 0 .55rem; color:var(--muted); font-size:.7rem; }
        .scenario-row { display:flex; gap:.5rem; flex-wrap:wrap; }
        .pipeline-shell { margin-top:1rem; padding:1rem; border:1px solid var(--line); border-radius:19px; background:rgba(7,15,26,.78); }
        .pipeline-title { margin-bottom:.8rem; color:#b1c0cf; font-size:.68rem; font-weight:900; letter-spacing:.11em; text-transform:uppercase; }
        .flow-grid { display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr; gap:.42rem; align-items:stretch; }
        .flow-card { position:relative; min-height:110px; padding:.82rem .75rem; border:1px solid var(--line); border-radius:14px; background:linear-gradient(180deg,#0a1625,#07111e); }
        .flow-card.final { border-color:rgba(89,227,154,.24); }
        .flow-num { color:var(--cyan); font-size:.66rem; font-weight:950; letter-spacing:.12em; }
        .flow-title { margin-top:.4rem; font-size:.76rem; font-weight:900; letter-spacing:.045em; }
        .flow-note { margin-top:.36rem; color:var(--muted); font-size:.66rem; line-height:1.4; }
        .flow-arrow { align-self:center; color:#627d97; font-size:1rem; text-align:center; }
        .authority-card { padding:1.45rem; border:1px solid rgba(89,227,154,.42); border-radius:21px; background:radial-gradient(circle at 92% 8%,rgba(89,227,154,.1),transparent 34%),linear-gradient(135deg,rgba(17,65,48,.34),rgba(6,18,25,.97)); box-shadow:0 24px 58px rgba(0,0,0,.22); }
        .authority-label { color:var(--green); font-size:.68rem; font-weight:950; letter-spacing:.14em; text-transform:uppercase; }
        .authority-id { margin-top:.55rem; font-size:clamp(2rem,5vw,3.5rem); line-height:1; font-weight:950; letter-spacing:-.055em; }
        .decision-pill { display:inline-block; margin-top:.62rem; padding:.4rem .75rem; border:1px solid rgba(89,227,154,.4); border-radius:999px; color:#c7f6d9; font-size:.67rem; font-weight:950; letter-spacing:.09em; }
        .decision-pill.uncertain { color:#f4cd80; border-color:rgba(242,189,99,.4); }
        .decision-pill.new { color:#8edaf2; border-color:rgba(85,210,242,.4); }
        .stat-strip { margin-top:.75rem; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; }
        .stat-card { padding:.78rem .85rem; border:1px solid var(--line); border-radius:13px; background:rgba(8,17,28,.74); }
        .stat-label { color:var(--muted); font-size:.63rem; font-weight:780; letter-spacing:.07em; text-transform:uppercase; }
        .stat-value { margin-top:.25rem; font-size:1rem; font-weight:900; }
        .evidence-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
        .evidence-card { padding:.95rem; border:1px solid var(--line); border-radius:16px; background:rgba(9,19,32,.8); }
        .evidence-label { color:#9bb0c4; font-size:.63rem; font-weight:900; letter-spacing:.1em; text-transform:uppercase; }
        .evidence-value { margin-top:.45rem; color:#e5edf5; font-size:.81rem; line-height:1.55; overflow-wrap:anywhere; white-space:pre-wrap; }
        .advisory-zone { margin-top:1rem; padding:1.1rem; border:1px solid rgba(189,149,255,.2); border-radius:20px; background:linear-gradient(135deg,rgba(44,29,70,.18),rgba(8,15,26,.92)); }
        .advisory-title { font-size:1.32rem; font-weight:950; }
        .advisory-subtitle { margin-top:.25rem; color:var(--muted); font-size:.76rem; line-height:1.45; }
        .advisory-grid { margin-top:.9rem; display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
        .advisory-card { min-height:180px; padding:1rem; border:1px solid rgba(189,149,255,.22); border-radius:17px; background:rgba(12,18,32,.86); }
        .advisory-label { color:var(--purple); font-size:.68rem; font-weight:950; letter-spacing:.1em; text-transform:uppercase; }
        .advisory-note { margin-top:.36rem; color:var(--muted); font-size:.72rem; line-height:1.45; }
        .service-grid { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }
        .service-card { padding:.9rem 1rem; border:1px solid var(--line); border-radius:15px; background:rgba(8,17,28,.78); }
        .service-name { font-weight:900; }
        .service-state { margin-top:.32rem; color:var(--muted); font-size:.73rem; }
        .footer-line { padding-top:.9rem; color:#71879d; font-size:.66rem; text-align:center; letter-spacing:.04em; }
        @media (max-width:1100px) {
            .architecture-grid { grid-template-columns:1fr; }
            .flow-grid { grid-template-columns:repeat(3,1fr); }
            .flow-arrow { display:none; }
        }
        @media (max-width:820px) {
            .block-container { padding:.8rem .65rem 1.7rem; }
            .hero-title { font-size:2rem; }
            .evidence-grid,.advisory-grid,.service-grid { grid-template-columns:1fr; }
            .stat-strip { grid-template-columns:1fr; }
        }
        @media (max-width:680px) {
            .block-container { padding:.65rem .5rem 1.4rem; }
            .hero { padding:1rem; border-radius:17px; }
            .hero-title { font-size:1.65rem; }
            .hero-subtitle { font-size:.84rem; }
            .input-shell,.pipeline-shell,.advisory-zone { padding:.78rem; }
            .flow-grid { grid-template-columns:1fr; }
            .flow-card { min-height:70px; }
            .authority-card { padding:1rem; }
            .authority-id { font-size:2.15rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(ai_enabled: bool) -> None:
    mode = "HYBRID AI" if ai_enabled else "DETERMINISTIC ONLY"
    st.markdown(f'<div class="hero"><div class="eyebrow">CPSE MATERIAL HARMONIZATION</div><div class="hero-title">CPSE Material Harmonization</div><div class="hero-subtitle">AI-assisted standardization of legacy CPSE material descriptions into a common canonical material vocabulary.</div><div class="hero-meta"><span class="meta-pill">SIH 2026 · PS 26099</span><span class="meta-pill meta-online">● SYSTEM ONLINE</span><span class="meta-pill">MODE · {mode}</span></div></div>', unsafe_allow_html=True)


def _render_architecture() -> None:
    cols = st.columns(3)
    cards = (("DETERMINISTIC ENGINE", "AUTHORITATIVE", "Final decision source", "authoritative"), ("LOCAL NLP", "ADVISORY", "Optional local embedding retrieval", "advisory"), ("GEMINI 2.5 FLASH", "LLM · ADVISORY", "Candidate reasoning", "advisory"))
    for column, (kicker, main, note, kind) in zip(cols, cards):
        with column:
            st.markdown(f'<div class="architecture-card"><div class="architecture-kicker">{kicker}</div><div class="architecture-main {kind}">{main}</div><div class="architecture-note">{note}</div></div>', unsafe_allow_html=True)


def _render_pipeline() -> None:
    stages = (("01", "INPUT", "Legacy description", ""), ("02", "NORMALIZE", "Canonical text", ""), ("03", "EXTRACT", "Technical attributes", ""), ("04", "MATCH", "Deterministic mapping", ""), ("05", "AI ADVISE", "Local NLP + Gemini", ""), ("06", "DECIDE", "Authoritative result", "final"))
    st.markdown('<div class="pipeline-shell"><div class="pipeline-title">Processing Pipeline</div><div class="flow-grid">', unsafe_allow_html=True)
    for index, (number, title, note, extra_class) in enumerate(stages):
        st.markdown(f'<div class="flow-card {extra_class}"><div class="flow-num">{number}</div><div class="flow-title">{title}</div><div class="flow-note">{note}</div></div>', unsafe_allow_html=True)
        if index < len(stages) - 1:
            st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


def _render_status(result: HybridResult | None = None, ai_enabled: bool = True) -> None:
    st.markdown("### Service Status")
    if result is None and not ai_enabled:
        st.markdown('<div class="mode-strip"><strong>AI advisory disabled.</strong> Local NLP and Gemini are not invoked in this mode.</div>', unsafe_allow_html=True)
        return
    if result is None:
        retrieval, llm = _analysis_adapters(True)
        statuses = (retrieval.status(), llm.status())
    else:
        statuses = result.ai_statuses
    cols = st.columns(2)
    for column, status, label in ((cols[0], statuses[0], "Local NLP"), (cols[1], statuses[1], "Gemini 2.5 Flash")):
        symbol = "●" if status.available else "○"
        dot = "green" if status.available else "amber"
        with column:
            st.markdown(f'<div class="service-card"><div class="service-name"><span class="{dot}">{symbol}</span> {label}</div><div class="service-state">{_friendly_status(status)} · Advisory only</div></div>', unsafe_allow_html=True)


def _render_decision(result: HybridResult) -> None:
    decision = result.mapping_result.decision
    canonical_id = result.mapping_result.canonical_material_id or "Not assigned"
    score = result.mapping_result.score
    pill_class = "uncertain" if decision == "UNCERTAIN" else ("new" if decision == "NEW_CANDIDATE" else "")
    st.markdown(f'<div class="authority-card"><div class="authority-label">✓ Authoritative Decision</div><div class="authority-id">{canonical_id}</div><span class="decision-pill {pill_class}">{decision}</span></div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    first.markdown(f'<div class="stat-card"><div class="stat-label">Canonical material</div><div class="stat-value">{canonical_id}</div></div>', unsafe_allow_html=True)
    second.markdown(f'<div class="stat-card"><div class="stat-label">Deterministic match score</div><div class="stat-value">{score:.3f}</div></div>', unsafe_allow_html=True)
    third.markdown(f'<div class="stat-card"><div class="stat-label">Decision</div><div class="stat-value">{decision}</div></div>', unsafe_allow_html=True)
    st.caption("Final decision source: deterministic harmonization engine. AI suggestions cannot change this result.")
    st.info(result.mapping_result.explanation)


def _render_advisory(result: HybridResult) -> None:
    local = tuple(item for item in result.ai_candidate_suggestions if item.source == "local_embedding")
    gemini = tuple(item for item in result.ai_candidate_suggestions if item.source == "gemini")
    st.markdown('<div class="advisory-zone"><div class="advisory-title">AI Advisory</div><div class="advisory-subtitle">Advisory intelligence only — Local NLP and Gemini suggestions cannot override the deterministic decision.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for column, label, source_rows in ((cols[0], "Local NLP · Embedding Retrieval", local), (cols[1], "Gemini 2.5 Flash · LLM", gemini)):
        with column:
            st.markdown(f'<div class="advisory-card"><div class="advisory-label">{label} · ADVISORY</div><div class="advisory-note">Optional candidate suggestions from this component.</div></div>', unsafe_allow_html=True)
            if source_rows:
                st.dataframe(advisory_rows(source_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No advisory suggestions available from this component.")
    st.markdown('</div>', unsafe_allow_html=True)


def _render_analysis(result: HybridResult, raw_description: str | None) -> None:
    st.markdown("## Authoritative Result")
    _render_decision(result)
    st.markdown("## Processing Evidence")
    left, right = st.columns(2)
    safe_raw = (raw_description or "No description provided").replace("<", "&lt;").replace(">", "&gt;")
    safe_norm = (result.normalized_description or "No normalized description").replace("<", "&lt;").replace(">", "&gt;")
    transform_html = "<br>".join(f"✓ {item}" for item in result.normalization_transformations) if result.normalization_transformations else "No transformations reported."
    with left:
        st.markdown(f'<div class="evidence-card"><div class="evidence-label">Original Description</div><div class="evidence-value">{safe_raw}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="evidence-card"><div class="evidence-label">Normalized Description</div><div class="evidence-value">{safe_norm}</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="evidence-card"><div class="evidence-label">Normalization Transformations</div><div class="evidence-value">{transform_html}</div></div>', unsafe_allow_html=True)
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
    st.session_state["ai_enabled"] = selected == "Hybrid AI"


def _render_sidebar() -> bool:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">CPSE MATERIAL<br>HARMONIZATION</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">SIH 2026 · Problem Statement 26099</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">System Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="green">●</span><span>SYSTEM ONLINE</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Analysis Mode</div>', unsafe_allow_html=True)
        current = st.session_state.get("ai_enabled", False)
        st.radio("Analysis Mode", options=("Hybrid AI", "Deterministic Only"), index=0 if current else 1, label_visibility="collapsed", key="analysis_mode_selector", on_change=_sidebar_mode_changed)
        ai_enabled = st.session_state.get("ai_enabled", False)
        st.markdown('<div class="sidebar-label">Decision Authority</div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="green">●</span><span>Deterministic Engine</span></div>', unsafe_allow_html=True)
        st.caption("AUTHORITATIVE · final decision source")
        st.markdown('<div class="sidebar-label">AI Services</div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="purple">●</span><span>Local NLP</span><span class="right">ADVISORY</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="service-row"><span class="purple">●</span><span>Gemini 2.5 Flash</span><span class="right">LLM · ADVISORY</span></div>', unsafe_allow_html=True)
        st.caption("AI advisory can be disabled. Disabled mode does not invoke Local NLP or Gemini.")
        st.markdown('<div class="sidebar-label">Demo</div>', unsafe_allow_html=True)
        st.caption("Use the scenario buttons in the main workspace to quickly demonstrate MATCHED, UNCERTAIN, and NEW CANDIDATE outcomes.")
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
    st.markdown('<div class="mode-strip"><strong>Decision authority:</strong> deterministic matching is the final decision source. Local NLP and Gemini provide separate advisory intelligence.</div>', unsafe_allow_html=True)
    _render_pipeline()
    st.divider()
    st.markdown('<div class="section-kicker">MATERIAL ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("### Analyze a legacy material description")
    st.caption("Enter a CPSE material description. The deterministic engine remains authoritative in both modes.")
    st.markdown('<div class="input-shell">', unsafe_allow_html=True)
    examples = st.columns(3)
    for column, (outcome, example) in zip(examples, DEMO_EXAMPLES.items()):
        with column:
            if st.button(outcome.title(), use_container_width=True, key=f"example_{outcome}"):
                _set_example(example)
                st.rerun()
    description = st.text_area("Material Description", key="material_description", height=125, placeholder="Gate Valve Carbon Steel 150 50mm Flanged", help="Example: CS GATE VLV 50MM FLG CL150")
    st.markdown('</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="footer-line">CPSE MATERIAL HARMONIZATION · SIH 2026 · PS 26099 · Deterministic Decision Engine · AI Advisory Layer</div>', unsafe_allow_html=True)
        return
    st.divider()
    _render_analysis(result, description)
    st.divider()
    _render_status(result, ai_enabled=ai_enabled)
    if result.fallback_used:
        st.info("One or more optional AI advisory components were unavailable or failed. Deterministic matching remains fully operational.")
    st.markdown('<div class="footer-line">CPSE MATERIAL HARMONIZATION · SIH 2026 · PS 26099 · Deterministic Decision Engine · AI Advisory Layer</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
