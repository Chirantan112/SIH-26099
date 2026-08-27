"""Judge-facing Streamlit dashboard for CPSE material harmonization."""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Any, Callable

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
    "category": "Category", "valve_type": "Valve type", "material": "Material",
    "size_mm": "Size (mm)", "pressure_class": "Pressure class", "connection": "Connection",
    "bearing_family": "Bearing family", "dimensions": "Dimensions (mm)",
    "dimension_unit_present": "Dimension unit stated", "od_mm": "Outside diameter (mm)",
    "thickness_mm": "Thickness (mm)", "schedule": "Schedule", "end": "End",
}

PIPELINE_STAGES = (
    ("01", "INPUT", "Legacy description", "input"),
    ("02", "NORMALIZE", "Canonical text", "normalize"),
    ("03", "EXTRACT", "Technical attributes", "extract"),
    ("04", "MATCH", "Deterministic mapping", "match"),
    ("05", "AI ADVISE", "Advisory intelligence", "ai_advisory"),
    ("06", "DECIDE", "Authoritative result", "decide"),
)


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
    return [{"Canonical material ID": c.canonical_material_id, "Decision": c.decision, "Score": c.score, "Explanation": c.explanation} for c in candidates]


def advisory_rows(suggestions: tuple[CandidateSuggestion, ...]) -> list[dict[str, str | float]]:
    return [{"Canonical material ID": s.canonical_material_id, "Source": s.source, "Advisory Rank": s.score, "Reason": s.explanation} for s in suggestions]


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


def analyze_material(
    description: str | None,
    catalog: tuple[Any, ...],
    ai_enabled: bool,
    retrieval_adapter: Any | None = None,
    llm_adapter: Any | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> HybridResult:
    """Run the existing hybrid orchestration; the deterministic result stays authoritative."""
    if not ai_enabled:
        retrieval_adapter = UnavailableRetrievalAdapter("AI advisory is disabled.")
        llm_adapter = UnavailableLLMAdapter("AI advisory is disabled.")
    elif retrieval_adapter is None or llm_adapter is None:
        retrieval_adapter, llm_adapter = _analysis_adapters(True)
    return run_hybrid_pipeline(
        description,
        catalog,
        legacy_material_code="DASHBOARD-INPUT",
        retrieval_adapter=retrieval_adapter,
        llm_adapter=llm_adapter,
        progress_callback=progress_callback,
    )


def _inject_styles() -> None:
    st.markdown("""
    <style>
    :root{--bg:#050b14;--panel:#091524;--panel2:#0c1a2b;--panel3:#0f2033;--line:#20344a;--text:#f4f8fc;--muted:#91a8be;--green:#55e39a;--cyan:#42c8ee;--purple:#b993ff;--amber:#f1bd63;--danger:#ef6b73}
    .stApp{background:radial-gradient(900px 500px at 80% -5%,rgba(44,121,168,.16),transparent 60%),radial-gradient(700px 420px at 10% 30%,rgba(29,72,118,.08),transparent 62%),var(--bg);color:var(--text)}
    .block-container{max-width:1680px;padding:1rem clamp(.75rem,1.8vw,2rem) 2rem}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#06111e,#071321);border-right:1px solid var(--line)}
    [data-testid="stSidebar"] .block-container{padding:.9rem .85rem 1.2rem}
    [data-testid="stSidebar"] [data-testid="stRadio"] label{font-weight:760}
    .brand{font-size:1rem;font-weight:950;line-height:1.03;letter-spacing:-.02em}.sub{margin-top:.4rem;color:var(--muted);font-size:.68rem}
    .side-kicker{margin-top:1.15rem;margin-bottom:.45rem;color:#8ea6bd;font-size:.62rem;font-weight:900;letter-spacing:.15em;text-transform:uppercase}
    .side-row{display:flex;align-items:center;gap:.45rem;padding:.5rem 0;border-bottom:1px solid rgba(145,168,190,.09);font-size:.73rem}.side-right{margin-left:auto;color:#9db1c4;font-size:.6rem;font-weight:850;letter-spacing:.06em}.side-note{color:var(--muted);font-size:.67rem;line-height:1.45}
    .dot-green{color:var(--green)}.dot-purple{color:var(--purple)}.dot-cyan{color:var(--cyan)}.dot-amber{color:var(--amber)}
    .hero{position:relative;overflow:hidden;padding:clamp(1.25rem,2.5vw,1.9rem);border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(12,37,57,.96),rgba(5,14,24,.98) 74%);box-shadow:0 24px 60px rgba(0,0,0,.22)}
    .hero:after{content:"";position:absolute;right:-10%;top:-65%;width:45%;height:220%;transform:rotate(18deg);background:linear-gradient(90deg,transparent,rgba(66,200,238,.035),transparent);pointer-events:none}
    .eyebrow,.section-kicker{color:#a9bacb;font-size:.63rem;font-weight:900;letter-spacing:.18em;text-transform:uppercase}.hero-title{margin-top:.42rem;font-size:clamp(2.05rem,4.4vw,3.8rem);font-weight:950;line-height:.98;letter-spacing:-.06em}.hero-sub{max-width:1080px;margin-top:.7rem;color:#9db1c4;font-size:clamp(.86rem,1.35vw,1rem);line-height:1.55}.pills{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}.pill{padding:.36rem .7rem;border:1px solid var(--line);border-radius:999px;color:#d2dde7;background:rgba(255,255,255,.025);font-size:.64rem;font-weight:850}.pill.online{color:var(--green);border-color:rgba(85,227,154,.3)}
    .decision-banner{margin-top:.8rem;padding:.65rem .85rem;border:1px solid var(--line);border-radius:12px;background:rgba(8,19,32,.78);color:#9eb3c7;font-size:.72rem}.decision-banner strong{color:#f4f8fc}
    .workspace{margin-top:1rem}.section-title{font-size:.82rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:#b4c3d1;margin-bottom:.65rem}
    .analysis-status{min-height:38px;margin:0 0 .7rem;padding:.55rem .8rem;border:1px solid var(--line);border-radius:12px;background:linear-gradient(90deg,rgba(10,29,46,.96),rgba(8,18,31,.96));color:#afc1d1;font-size:.67rem;font-weight:850;letter-spacing:.04em;display:flex;align-items:center;justify-content:space-between;gap:.65rem}.analysis-status .status-main{display:flex;align-items:center;gap:.45rem;min-width:0}.analysis-status .status-dot{width:8px;height:8px;border-radius:50%;background:#71869a;box-shadow:0 0 0 4px rgba(113,134,154,.08);flex:0 0 auto}.analysis-status.ready .status-dot{background:var(--cyan);box-shadow:0 0 0 4px rgba(66,200,238,.09)}.analysis-status.processing .status-dot{background:var(--cyan);box-shadow:0 0 0 4px rgba(66,200,238,.09);animation:statusPulse 1.15s ease-in-out infinite}.analysis-status.complete .status-dot{background:var(--green);box-shadow:0 0 0 4px rgba(85,227,154,.08)}.analysis-status.warning .status-dot{background:var(--amber);box-shadow:0 0 0 4px rgba(241,189,99,.08)}.analysis-status .status-percent{color:#7e96aa;white-space:nowrap;font-size:.6rem}@keyframes statusPulse{50%{opacity:.42;transform:scale(.78)}}
    .card{padding:1rem;border:1px solid var(--line);border-radius:18px;background:linear-gradient(180deg,rgba(12,27,44,.96),rgba(7,17,29,.96));box-shadow:0 12px 28px rgba(0,0,0,.14)}
    .service-card{min-height:108px}.service-kicker{color:#9fb2c4;font-size:.6rem;font-weight:900;letter-spacing:.11em}.service-main{margin-top:.55rem;font-size:.95rem;font-weight:950}.service-main.green{color:var(--green)}.service-main.purple{color:var(--purple)}.service-note{margin-top:.35rem;color:var(--muted);font-size:.69rem;line-height:1.4}
    .input-card{padding:1rem;border:1px solid var(--line);border-radius:18px;background:rgba(8,18,31,.84)}.input-label{font-size:.78rem;font-weight:900}.scenario-label{margin-top:.7rem;color:#91a7bc;font-size:.63rem;font-weight:800}.scenario-buttons{display:flex;gap:.45rem;flex-wrap:nowrap}.scenario-buttons .stButton{min-width:0}.scenario-buttons .stButton>button{width:100%!important}.scenario-buttons .stButton:nth-child(1)>button{min-width:0}.scenario-buttons .stButton:nth-child(2)>button{min-width:8.1rem}.scenario-buttons .stButton:nth-child(3)>button{min-width:0}
    .pipeline{margin-top:.8rem;padding:.95rem 1rem;border:1px solid var(--line);border-radius:18px;background:rgba(6,15,26,.9)}.pipeline-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem}.pipeline-head span{color:#aebfd0;font-size:.63rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.pipeline-head small{color:#637b92;font-size:.61rem}.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.55rem}.stage{min-width:0;padding:.75rem .65rem;border:1px solid var(--line);border-radius:13px;background:linear-gradient(180deg,#0b1929,#07111e);position:relative;transition:border-color .15s,background .15s,box-shadow .15s;overflow:visible}.stage:not(:last-child):after{content:"→";position:absolute;right:-.57rem;top:50%;transform:translateY(-50%);z-index:2;color:#5d7891;font-size:.78rem}.stage-icon{float:right;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:1px solid #334b60;color:#637b92;font-size:.72rem;font-weight:950}.stage.done{border-color:rgba(85,227,154,.55);background:linear-gradient(180deg,rgba(14,58,42,.72),rgba(7,27,24,.92));box-shadow:0 0 18px rgba(85,227,154,.09)}.stage.done .stage-icon{background:var(--green);border-color:var(--green);color:#062016}.stage.active{border-color:rgba(66,200,238,.65);box-shadow:0 0 20px rgba(66,200,238,.12)}.stage.active .stage-icon{border-color:var(--cyan);color:var(--cyan);animation:pulse 1s infinite}.stage.skipped{opacity:.48}.stage.skipped .stage-icon{color:#5b7084;border-color:#33485b}.stage.unavailable{border-color:rgba(241,189,99,.65);background:linear-gradient(180deg,rgba(68,48,17,.55),rgba(31,23,11,.9));box-shadow:0 0 18px rgba(241,189,99,.08)}.stage.unavailable .stage-icon{background:var(--amber);border-color:var(--amber);color:#2a1c05}.stage-num{display:inline-block;color:var(--cyan);font-size:.6rem;font-weight:950;letter-spacing:.1em}.stage-name{margin-top:.45rem;font-size:.68rem;font-weight:900;line-height:1.2;white-space:normal;overflow-wrap:anywhere;word-break:normal}.stage-note{margin-top:.38rem;color:#8299ad;font-size:.57rem;line-height:1.35;overflow-wrap:anywhere;word-break:normal}@keyframes pulse{50%{box-shadow:0 0 0 6px rgba(66,200,238,0)}}
    .authority{height:100%;min-height:305px;padding:1.2rem;border:1px solid rgba(85,227,154,.4);border-radius:20px;background:radial-gradient(circle at 90% 5%,rgba(85,227,154,.1),transparent 35%),linear-gradient(145deg,rgba(15,62,45,.34),rgba(6,18,26,.98));box-shadow:0 18px 45px rgba(0,0,0,.2)}.authority-label{color:var(--green);font-size:.63rem;font-weight:950;letter-spacing:.13em;text-transform:uppercase}.authority-id{margin-top:.6rem;font-size:clamp(2rem,3.4vw,3rem);font-weight:950;line-height:1;letter-spacing:-.055em}.decision-pill{display:inline-block;margin-top:.55rem;padding:.35rem .62rem;border-radius:999px;border:1px solid rgba(85,227,154,.35);color:#c8f5d9;font-size:.6rem;font-weight:950;letter-spacing:.08em}.decision-pill.uncertain{color:#f5cf82;border-color:rgba(241,189,99,.4)}.decision-pill.new{color:#90dcf2;border-color:rgba(66,200,238,.4)}.authority-score{margin-top:.85rem;color:var(--green);font-size:1.15rem;font-weight:900}.authority-explain{margin-top:.25rem;color:#91a8ba;font-size:.64rem;line-height:1.4}.empty-authority{display:flex;flex-direction:column;justify-content:center;align-items:flex-start;color:#71869a}.empty-authority .authority-id{color:#6f8498;font-size:1.7rem}
    .glance{margin-top:.7rem;padding:.85rem;border:1px solid var(--line);border-radius:14px;background:rgba(7,16,28,.72)}.glance-title{color:#b5c5d4;font-size:.6rem;font-weight:900;letter-spacing:.1em;text-transform:uppercase}.glance-row{display:flex;justify-content:space-between;gap:.5rem;padding:.36rem 0;border-bottom:1px solid rgba(145,168,190,.08);font-size:.62rem}.glance-row:last-child{border-bottom:0}.glance-row span:first-child{color:#8198ac}.glance-row span:last-child{color:#d9e3eb;text-align:right}
    .result-section{margin-top:1rem}.result-head{display:flex;justify-content:space-between;align-items:end;gap:1rem;margin-bottom:.65rem}.result-head h2{margin:0;font-size:1.35rem;letter-spacing:-.03em}.result-head p{margin:0;color:var(--muted);font-size:.67rem;text-align:right}.evidence-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem}.evidence{padding:.8rem;border:1px solid var(--line);border-radius:14px;background:rgba(8,18,31,.78)}.evidence-label{color:#91a8bd;font-size:.59rem;font-weight:900;letter-spacing:.09em;text-transform:uppercase}.evidence-value{margin-top:.4rem;color:#e3ebf2;font-size:.68rem;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
    .advisory{margin-top:.8rem;padding:1rem;border:1px solid rgba(185,147,255,.22);border-radius:18px;background:linear-gradient(145deg,rgba(42,27,65,.16),rgba(7,15,26,.94))}.advisory-head{font-size:.95rem;font-weight:950}.advisory-head span{color:var(--purple)}.advisory-note{margin-top:.25rem;color:var(--muted);font-size:.66rem}.advisory-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:.7rem}.advisory-card{padding:.75rem;border:1px solid rgba(185,147,255,.2);border-radius:14px;background:rgba(10,17,29,.82)}.advisory-label{color:var(--purple);font-size:.6rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase}
    .footer{margin-top:1.4rem;padding-top:.8rem;border-top:1px solid var(--line);color:#62798f;font-size:.58rem;letter-spacing:.04em}
    div[data-testid="stButton"] button{border-radius:10px;font-weight:800;border:1px solid var(--line);min-height:2.35rem;white-space:nowrap;font-size:.72rem;padding:.4rem .42rem}div[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(90deg,#20a7db,#2bc2ec);border:0;color:#04111b;box-shadow:0 8px 22px rgba(40,190,235,.16)}
    textarea{border-radius:12px!important}.stAlert{border-radius:12px}.stExpander{border-color:var(--line);border-radius:13px}
    @media(min-width:1151px){.workspace-columns{grid-template-columns:1.15fr 1.55fr 1fr}}
    @media(max-width:1150px){.flow{grid-template-columns:repeat(3,minmax(0,1fr));row-gap:.65rem}.stage:not(:last-child):after{display:none}.authority{min-height:260px}.hero-title{font-size:clamp(2rem,5vw,3rem)}.scenario-buttons{display:grid;grid-template-columns:1fr 1.15fr 1fr}}
    @media(max-width:760px){.block-container{padding:.65rem .55rem 1.2rem}.hero{border-radius:17px;padding:1rem}.hero-title{font-size:2rem}.architecture-grid,.advisory-grid,.evidence-grid{grid-template-columns:1fr}.flow{grid-template-columns:repeat(2,minmax(0,1fr))}.stage{min-height:92px}.pipeline{padding:.7rem}.authority{min-height:230px}.result-head{display:block}.result-head p{text-align:left;margin-top:.3rem}.side-note{font-size:.65rem}.scenario-buttons{grid-template-columns:1fr 1fr}.scenario-buttons .stButton:nth-child(3){grid-column:1 / -1}.scenario-buttons .stButton:nth-child(2)>button{min-width:0}.analysis-status{font-size:.62rem}}
    @media(max-width:430px){.hero-title{font-size:1.72rem}.hero-sub{font-size:.78rem}.flow{grid-template-columns:1fr}.stage{min-height:70px}.stage:not(:last-child):after{display:block;content:"↓";right:50%;top:auto;bottom:-.7rem;transform:translateX(50%)}.pills{gap:.35rem}.pill{font-size:.57rem}.scenario-buttons{display:grid;grid-template-columns:1fr}.scenario-buttons .stButton:nth-child(3){grid-column:auto}}
    </style>
    """, unsafe_allow_html=True)


def _status_state(result: HybridResult | None, ai_enabled: bool) -> tuple[str, str, str]:
    if result is None:
        return "ready", "READY · Waiting for material", ""
    decision = result.mapping_result.decision
    statuses = result.ai_statuses
    ai_available = any(status.available for status in statuses)
    if decision == "MATCHED":
        suffix = "AI ADVISORY AVAILABLE" if ai_available else "DETERMINISTIC RESULT"
        return "complete", f"✓ ANALYSIS COMPLETE · {suffix}", "100%"
    if decision == "UNCERTAIN":
        suffix = "AI ADVISORY AVAILABLE" if ai_available else "UNCERTAIN"
        return "warning", f"⚠ ANALYSIS COMPLETE · {suffix}", "100%"
    suffix = "AI ADVISORY AVAILABLE" if ai_available else "NEW CANDIDATE"
    return "complete", f"✓ ANALYSIS COMPLETE · {suffix}", "100%"


def _render_analysis_status(result: HybridResult | None, ai_enabled: bool) -> None:
    state, label, percent = _status_state(result, ai_enabled)
    st.markdown(f'<div class="analysis-status {state}"><div class="status-main"><span class="status-dot"></span><span>{escape(label)}</span></div><span class="status-percent">{escape(percent)}</span></div>', unsafe_allow_html=True)


def _render_sidebar() -> bool:
    with st.sidebar:
        st.markdown('<div class="brand">CPSE MATERIAL<br>HARMONIZATION</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">SIH 2026 · Problem Statement 26099</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-kicker">System Status</div><div class="side-row"><span class="dot-green">●</span> SYSTEM ONLINE</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-kicker">Analysis Mode</div>', unsafe_allow_html=True)
        current = st.session_state.get("ai_enabled", False)
        st.radio("Analysis Mode", ("Hybrid AI", "Deterministic Only"), index=0 if current else 1, key="analysis_mode_selector", label_visibility="collapsed", on_change=lambda: st.session_state.__setitem__("ai_enabled", st.session_state["analysis_mode_selector"] == "Hybrid AI"))
        ai_enabled = st.session_state.get("ai_enabled", False)
        st.markdown('<div class="side-kicker">Decision Authority</div><div class="side-row"><span class="dot-green">●</span> Deterministic Engine</div>', unsafe_allow_html=True)
        st.caption("AUTHORITATIVE · final decision source")
        st.markdown('<div class="side-kicker">AI Services</div><div class="side-row"><span class="dot-purple">●</span> 🧠 Local NLP <span class="side-right">ADVISORY</span></div><div class="side-row"><span class="dot-purple">●</span> ✦ Gemini 2.5 Flash <span class="side-right">LLM · ADVISORY</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-note">AI advisory is optional. Disabled mode does not invoke Local NLP or Gemini.</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-kicker">Demo</div><div class="side-note">Use the quick scenarios in the workspace to demonstrate matched, uncertain, and new-candidate outcomes.</div>', unsafe_allow_html=True)
        return ai_enabled


def _render_header(ai_enabled: bool) -> None:
    mode = "HYBRID AI" if ai_enabled else "DETERMINISTIC ONLY"
    st.markdown(f'<div class="hero"><div class="eyebrow">CPSE MATERIAL HARMONIZATION</div><div class="hero-title">CPSE Material Harmonization</div><div class="hero-sub">AI-assisted standardization of legacy CPSE material descriptions into a common canonical material vocabulary.</div><div class="pills"><span class="pill">SIH 2026 · PS 26099</span><span class="pill online">● SYSTEM ONLINE</span><span class="pill">MODE · {mode}</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="decision-banner"><strong>Decision authority:</strong> deterministic matching is always the final decision source. Local NLP and Gemini provide separate advisory intelligence only.</div>', unsafe_allow_html=True)


def _render_architecture() -> None:
    cols = st.columns(3)
    cards = (("DETERMINISTIC ENGINE", "✓ AUTHORITATIVE", "Final decision source", "green"), ("LOCAL NLP", "🧠 ADVISORY", "Optional local embedding retrieval", "purple"), ("GEMINI 2.5 FLASH", "✦ LLM · ADVISORY", "Candidate reasoning", "purple"))
    for col, (kicker, main, note, color) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card service-card"><div class="service-kicker">{kicker}</div><div class="service-main {color}">{main}</div><div class="service-note">{note}</div></div>', unsafe_allow_html=True)


def _render_input(ai_enabled: bool, result: HybridResult | None = None) -> str:
    st.markdown('<div class="section-title">Analyze Material</div><div class="input-card">', unsafe_allow_html=True)
    _render_analysis_status(result, ai_enabled)
    st.markdown('<div class="input-label">Enter a legacy material description</div>', unsafe_allow_html=True)
    st.markdown('<div class="scenario-label">Quick scenarios</div>', unsafe_allow_html=True)
    buttons = st.columns(3)
    for col, (outcome, example) in zip(buttons, DEMO_EXAMPLES.items()):
        with col:
            label = {"MATCHED": "✓ Matched", "UNCERTAIN": "⚠ Uncertain", "NEW CANDIDATE": "✦ New Candidate"}[outcome]
            if st.button(label, use_container_width=True, key=f"example_{outcome}"):
                st.session_state["material_description"] = example
                st.session_state["analysis_result"] = None
                st.session_state["pipeline_completed"] = ()
                st.rerun()
    description = st.text_area("Material Description", key="material_description", height=96, placeholder="Gate Valve Carbon Steel 150 50mm Flanged", help="Example: CS GATE VLV 50MM FLG CL150", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return description


def _render_pipeline(completed: tuple[str, ...] = (), active: str | None = None, container: Any | None = None, ai_enabled: bool = True, result: HybridResult | None = None) -> None:
    target = container or st
    html = '<div class="pipeline"><div class="pipeline-head"><span>Processing Pipeline</span><small>deterministic authority preserved</small></div><div class="flow">'
    completed_set = set(completed)
    statuses = result.ai_statuses if result is not None else ()
    ai_available = any(status.available for status in statuses)
    for num, name, note, key in PIPELINE_STAGES:
        disabled_ai = key == "ai_advisory" and not ai_enabled
        unavailable_ai = key == "ai_advisory" and ai_enabled and result is not None and not ai_available and result.mapping_result.decision in {"MATCHED", "UNCERTAIN", "NEW_CANDIDATE"}
        state = "unavailable" if unavailable_ai else "skipped" if disabled_ai else "done" if key in completed_set else "active" if key == active else ""
        icon = "⚠" if state == "unavailable" else "—" if state == "skipped" else "✓" if state == "done" else "•" if state == "active" else "○"
        display_note = note if not disabled_ai else "Disabled in current mode"
        if unavailable_ai:
            display_note = "No advisory service available"
        html += f'<div class="stage {state}"><span class="stage-icon">{icon}</span><div class="stage-num">{num}</div><div class="stage-name">{name}</div><div class="stage-note">{display_note}</div></div>'
    html += '</div></div>'
    target.markdown(html, unsafe_allow_html=True)


def _decision_class(decision: str) -> str:
    return "uncertain" if decision == "UNCERTAIN" else "new" if decision == "NEW_CANDIDATE" else ""


def _render_authority(result: HybridResult | None) -> None:
    if result is None:
        st.markdown('<div class="authority empty-authority"><div class="authority-label">AUTHORITATIVE DECISION</div><div class="authority-id">Awaiting analysis</div><div class="authority-explain">Run a material description to produce the deterministic result.</div></div>', unsafe_allow_html=True)
        return
    mapping = result.mapping_result
    cid = mapping.canonical_material_id or "Not assigned"
    decision = mapping.decision
    score = mapping.score
    pill = _decision_class(decision)
    explanation = escape(mapping.explanation or "Final decision produced by deterministic harmonization engine.")
    st.markdown(f'<div class="authority"><div class="authority-label">✓ AUTHORITATIVE DECISION</div><div class="authority-id">{escape(str(cid))}</div><span class="decision-pill {pill}">{escape(decision)}</span><div class="authority-score">{score:.3f}</div><div class="authority-explain">Deterministic match score<br>{explanation}</div></div>', unsafe_allow_html=True)
    attrs = result.attributes
    glance = (("Category", _display_value(getattr(attrs, "category", None))), ("Valve Type", _display_value(getattr(attrs, "valve_type", None))), ("Material", _display_value(getattr(attrs, "material", None))), ("Size (mm)", _display_value(getattr(attrs, "size_mm", None))), ("Pressure Class", _display_value(getattr(attrs, "pressure_class", None))), ("Connection", _display_value(getattr(attrs, "connection", None))))
    rows = ''.join(f'<div class="glance-row"><span>{escape(k)}</span><span>{escape(v)}</span></div>' for k, v in glance)
    st.markdown(f'<div class="glance"><div class="glance-title">At a glance attributes</div>{rows}</div>', unsafe_allow_html=True)


def _render_analysis(result: HybridResult) -> None:
    st.markdown('<div class="result-section"><div class="result-head"><h2>Evidence &amp; Details</h2><p>Transparent evidence behind the authoritative decision</p></div>', unsafe_allow_html=True)
    raw = escape(st.session_state.get("material_description", ""))
    norm = escape(result.normalized_description or "Not provided")
    transforms = "\n".join(f"✓ {item}" for item in result.normalization_transformations) if result.normalization_transformations else "No transformations reported."
    st.markdown(f'<div class="evidence-grid"><div class="evidence"><div class="evidence-label">Original Description</div><div class="evidence-value">{raw}</div></div><div class="evidence"><div class="evidence-label">Normalized Description</div><div class="evidence-value">{norm}</div></div><div class="evidence"><div class="evidence-label">Normalization Transformations</div><div class="evidence-value">{escape(transforms)}</div></div><div class="evidence"><div class="evidence-label">Deterministic Explanation</div><div class="evidence-value">{escape(result.mapping_result.explanation or "No explanation reported.")}</div></div></div>', unsafe_allow_html=True)
    with st.expander("Technical Attributes", expanded=False):
        st.dataframe(attribute_rows(result.attributes), hide_index=True, use_container_width=True)
    with st.expander("Deterministic Candidate Evidence", expanded=False):
        candidates = result.mapping_result.all_candidates
        st.dataframe(candidate_rows(candidates), hide_index=True, use_container_width=True) if candidates else st.caption("No candidate evidence is available.")
    _render_advisory(result)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_advisory(result: HybridResult) -> None:
    local = tuple(x for x in result.ai_candidate_suggestions if x.source == "local_embedding")
    gemini = tuple(x for x in result.ai_candidate_suggestions if x.source == "gemini")
    st.markdown('<div class="advisory"><div class="advisory-head"><span>AI Advisory</span> · advisory only</div><div class="advisory-note">AI suggestions are never used as the authoritative decision.</div><div class="advisory-grid"><div class="advisory-card"><div class="advisory-label">🧠 Local NLP · Embedding Retrieval</div></div><div class="advisory-card"><div class="advisory-label">✦ Gemini 2.5 Flash · LLM</div></div></div></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for col, rows in zip(cols, (local, gemini)):
        with col:
            if rows:
                st.dataframe(advisory_rows(rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No advisory suggestions available from this component.")


def _render_status(result: HybridResult | None, ai_enabled: bool) -> None:
    if result is None and not ai_enabled:
        st.caption("AI advisory is disabled; Local NLP and Gemini are not invoked.")
        return
    statuses = result.ai_statuses if result is not None else tuple(adapter.status() for adapter in _analysis_adapters(True))
    cols = st.columns(2)
    for col, status, label in zip(cols, statuses, ("Local NLP", "Gemini 2.5 Flash")):
        with col:
            dot_class = "dot-green" if status.available else "dot-amber"
            st.markdown(f'<div class="side-row"><span class="{dot_class}">●</span> {label}<span class="side-right">{_friendly_status(status).upper()}</span></div>', unsafe_allow_html=True)


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to run this dashboard. Use: streamlit run app.py")
    st.set_page_config(page_title="CPSE Material Harmonization", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    _inject_styles()
    st.session_state.setdefault("ai_enabled", False)
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_mode", None)
    st.session_state.setdefault("material_description", "")
    st.session_state.setdefault("pipeline_completed", ())

    ai_enabled = _render_sidebar()
    previous_result = st.session_state.get("analysis_result")
    previous_mode = st.session_state.get("analysis_mode")
    mode_changed = previous_result is not None and previous_mode is not None and previous_mode != ai_enabled

    _render_header(ai_enabled)
    _render_architecture()

    left, center, right = st.columns([1.18, 1.62, 1.00], gap="medium")
    with left:
        st.markdown('<div class="section-title">Analysis Workspace</div>', unsafe_allow_html=True)
        description = _render_input(ai_enabled, previous_result)
        analyze = st.button("Analyze Material  →", type="primary", use_container_width=True, key="analyze_main")
        if analyze:
            if not description or not description.strip():
                st.session_state["analysis_result"] = None
                st.session_state["pipeline_completed"] = ()
                st.warning("Material description required. Enter a legacy description to begin.")
            else:
                st.session_state["analysis_result"] = None
                st.session_state["pipeline_completed"] = ()
                pipeline_placeholder = center.empty()
                _render_pipeline((), "input", pipeline_placeholder, ai_enabled)
                try:
                    def on_progress(stage: str) -> None:
                        completed = list(st.session_state.get("pipeline_completed", ()))
                        if stage == "ai_advisory" and not ai_enabled:
                            return
                        if stage not in completed:
                            completed.append(stage)
                        st.session_state["pipeline_completed"] = tuple(completed)
                        remaining = next((key for _, _, _, key in PIPELINE_STAGES if key not in completed and (key != "ai_advisory" or ai_enabled)), None)
                        _render_pipeline(tuple(completed), remaining, pipeline_placeholder, ai_enabled)

                    st.session_state["analysis_result"] = analyze_material(
                        description,
                        load_demo_catalog(CATALOG_PATH),
                        ai_enabled,
                        progress_callback=on_progress,
                    )
                    st.session_state["analysis_mode"] = ai_enabled
                    completed_stages = tuple(stage[3] for stage in PIPELINE_STAGES if stage[3] != "ai_advisory" or ai_enabled)
                    st.session_state["pipeline_completed"] = completed_stages
                    _render_pipeline(completed_stages, None, pipeline_placeholder, ai_enabled, st.session_state["analysis_result"])
                    st.rerun()
                except (OSError, TypeError, ValueError):
                    st.session_state["analysis_result"] = None
                    st.session_state["pipeline_completed"] = ()
                    st.error("The description could not be processed safely. Please check the input and try again.")
    with center:
        if not analyze:
            _render_pipeline(st.session_state.get("pipeline_completed", ()), None, None, ai_enabled, previous_result)
        if mode_changed:
            st.warning("Analysis mode changed. Re-analyze to generate evidence for the current mode.")
        if previous_result is None:
            st.markdown('<div class="card" style="margin-top:.8rem;min-height:160px"><div class="service-kicker">WORKFLOW READY</div><div class="service-main">Deterministic engine ready</div><div class="service-note">Input → normalize → extract → match → optional AI advisory → authoritative decision.</div></div>', unsafe_allow_html=True)
    with right:
        _render_authority(previous_result)

    result = st.session_state.get("analysis_result")
    if mode_changed and result is not None and st.session_state.get("analysis_mode") != ai_enabled:
        st.session_state["analysis_result"] = None
        result = None
    if result is not None:
        st.divider()
        _render_analysis(result)
        st.markdown('<div class="section-title" style="margin-top:1rem">Service Status</div>', unsafe_allow_html=True)
        _render_status(result, ai_enabled)
    else:
        st.markdown('<div class="footer">CPSE MATERIAL HARMONIZATION · SIH 2026 · PS 26099 · Deterministic Decision Engine · Optional AI Advisory Layer</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
