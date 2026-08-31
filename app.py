"""Streamlit UI for CPSE material harmonization.

Presentation-only branch: matching, AI, and governance modules remain untouched.
Deterministic mapping remains authoritative; AI output is advisory.
"""
from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Any, Callable

from src.ai_retrieval import CandidateSuggestion, UnavailableRetrievalAdapter
from src.attribute_extraction import MaterialAttributes
from src.governance import AuditTrail
from src.hybrid_pipeline import HybridResult, run_hybrid_pipeline
from src.llm_interpretation import UnavailableLLMAdapter
from src.source_adapters import CSVMaterialCatalogSource

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

try:
    from src.gemini_llm import GeminiLLMAdapter
except Exception:
    GeminiLLMAdapter = None

try:
    from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter
except Exception:
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
KEY_ATTRIBUTES = (
    ("Category", "category"),
    ("Valve type", "valve_type"),
    ("Material", "material"),
    ("Size", "size_mm"),
    ("Pressure", "pressure_class"),
    ("Connection", "connection"),
)
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
        return " × ".join(map(str, value))
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def attribute_rows(attributes: MaterialAttributes) -> list[dict[str, str]]:
    return [
        {
            "Attribute": ATTRIBUTE_LABELS[field.name],
            "Extracted value": _display_value(getattr(attributes, field.name)),
        }
        for field in fields(attributes)
    ]


def candidate_rows(candidates: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "Canonical material ID": candidate.canonical_material_id,
            "Decision": candidate.decision,
            "Score": candidate.score,
            "Explanation": candidate.explanation,
        }
        for candidate in candidates
    ]


def advisory_rows(
    candidates: tuple[CandidateSuggestion, ...], score_label: str
) -> list[dict[str, Any]]:
    return [
        {
            "Canonical material ID": candidate.canonical_material_id,
            "Source": candidate.source,
            score_label: candidate.score,
            "Reason": candidate.explanation,
        }
        for candidate in candidates
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


def _analysis_adapters(enabled: bool) -> tuple[Any, Any]:
    if not enabled:
        return (
            UnavailableRetrievalAdapter("AI advisory is disabled."),
            UnavailableLLMAdapter("AI advisory is disabled."),
        )
    retrieval = (
        LocalEmbeddingRetrievalAdapter()
        if LocalEmbeddingRetrievalAdapter
        else UnavailableRetrievalAdapter("Local NLP adapter is unavailable.")
    )
    llm = (
        GeminiLLMAdapter()
        if GeminiLLMAdapter
        else UnavailableLLMAdapter("Gemini adapter is unavailable.")
    )
    return retrieval, llm


def analyze_material(
    description: str | None,
    catalog: tuple[Any, ...],
    ai_enabled: bool,
    retrieval_adapter: Any | None = None,
    llm_adapter: Any | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> HybridResult:
    if not ai_enabled:
        retrieval_adapter, llm_adapter = _analysis_adapters(False)
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


def _css() -> None:
    st.markdown(
        """
<style>
:root{--bg:#050c15;--panel:#091522;--panel2:#0d1d2d;--line:#203b53;--text:#eef6fb;--muted:#8da6b9;--cyan:#36c6ee;--green:#4ee39a;--purple:#b68cff;--amber:#f2c36e;--red:#ff6f79}
.stApp{background:radial-gradient(900px 500px at 76% -10%,rgba(54,198,238,.12),transparent 62%),radial-gradient(700px 420px at 100% 75%,rgba(78,227,154,.055),transparent 65%),var(--bg);color:var(--text)}
.block-container{max-width:1600px;padding:max(4rem,8vh) clamp(.45rem,1.6vw,1.4rem) 2rem}
[data-testid="stSidebar"]{background:#06111e;border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{padding-top:1rem}
.brand{font-size:1.03rem;font-weight:950;line-height:1.03;letter-spacing:-.025em}.brand-mark{color:var(--cyan);margin-right:.25rem}.sub{color:#7891a5;font-size:.62rem;margin-top:.25rem}.side-kicker{margin:1.15rem 0 .45rem;color:#7f9bb0;font-size:.55rem;font-weight:950;letter-spacing:.16em;text-transform:uppercase}.side-row{display:flex;align-items:center;gap:.42rem;padding:.5rem 0;border-bottom:1px solid rgba(141,166,185,.08);font-size:.68rem}.side-right{margin-left:auto;color:#9db2c1;font-size:.5rem;font-weight:950}.dot-green{color:var(--green)}.dot-cyan{color:var(--cyan)}.dot-purple{color:var(--purple)}.dot-amber{color:var(--amber)}
[data-testid="stSidebar"] [data-testid="stSegmentedControl"]{width:100%}[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button{background:#091827!important;color:#b7c7d3!important;border-color:var(--line)!important;font-weight:850!important}[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:#102e40!important;color:var(--cyan)!important;border-color:var(--cyan)!important}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.15rem 0 .9rem}.top-title{font-size:clamp(1.35rem,2.4vw,2.15rem);font-weight:950;letter-spacing:-.045em}.top-sub{margin-top:.18rem;color:#8da6b9;font-size:.68rem}.top-status{display:flex;align-items:center;gap:.5rem;padding:.45rem .7rem;border:1px solid var(--line);border-radius:999px;background:#091827;color:#c8d8e2;font-size:.57rem;font-weight:900;white-space:nowrap}.top-status b{color:var(--green)}
.hero{position:relative;overflow:hidden;padding:clamp(1.05rem,2.2vw,1.7rem);border:1px solid #25465e;border-radius:22px;background:linear-gradient(135deg,#102d43 0%,#07131f 72%);box-shadow:0 24px 70px #0005}.hero:after{content:"";position:absolute;right:-150px;top:-190px;width:420px;height:420px;border-radius:50%;background:radial-gradient(circle,rgba(54,198,238,.16),transparent 68%)}.eyebrow{position:relative;z-index:1;color:#a9bfce;font-size:.55rem;font-weight:950;letter-spacing:.18em}.hero-title{position:relative;z-index:1;margin-top:.48rem;font-size:clamp(2rem,4.4vw,4.15rem);font-weight:950;line-height:.94;letter-spacing:-.065em}.hero-title span{color:var(--cyan)}.hero-sub{position:relative;z-index:1;max-width:920px;margin-top:.75rem;color:#a7bac8;font-size:clamp(.72rem,1vw,.91rem);line-height:1.55}.hero-pills{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:.38rem;margin-top:1rem}.pill{padding:.33rem .58rem;border:1px solid #315069;border-radius:999px;color:#d0dee7;background:#ffffff05;font-size:.52rem;font-weight:900}.pill.online{color:var(--green);border-color:#4ee39a52}
.trust-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.45rem;margin-top:.55rem}.trust-card{padding:.65rem .72rem;border:1px solid var(--line);border-radius:12px;background:#08131f}.trust-card small{display:block;color:#7994a9;font-size:.48rem;font-weight:950;letter-spacing:.09em;text-transform:uppercase}.trust-card strong{display:block;margin-top:.2rem;font-size:.62rem}.trust-card.primary strong{color:var(--green)}
.workspace{margin-top:.8rem}.section-title{margin-bottom:.5rem;color:#a9bdca;font-size:.61rem;font-weight:950;letter-spacing:.14em;text-transform:uppercase}.card{padding:.9rem;border:1px solid var(--line);border-radius:16px;background:linear-gradient(180deg,#0b1c2c,#07121e);box-shadow:0 12px 32px #0003}.input-head{display:flex;justify-content:space-between;align-items:end;gap:.6rem}.input-title{font-size:.86rem;font-weight:950}.input-note{color:#8199ab;font-size:.55rem}.scenario-label{margin:.62rem 0 .35rem;color:#91a8b8;font-size:.53rem;font-weight:850}.scenario-buttons{display:grid;grid-template-columns:repeat(3,1fr);gap:.35rem}
.pipeline{margin-top:.55rem;padding:.7rem;border:1px solid var(--line);border-radius:15px;background:#050e18e8}.pipeline-head{display:flex;justify-content:space-between;gap:.5rem;margin-bottom:.5rem}.pipeline-head span{color:#aec0cc;font-size:.53rem;font-weight:950;letter-spacing:.1em;text-transform:uppercase}.pipeline-head small{color:#60788b;font-size:.49rem}.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.3rem}.stage{position:relative;min-width:0;padding:.55rem .43rem;border:1px solid var(--line);border-radius:10px;background:#091725}.stage:not(:last-child):after{content:"→";position:absolute;right:-.34rem;top:50%;transform:translateY(-50%);z-index:2;color:#59758a}.stage.done{border-color:#4ee39a70;background:#0b342777}.stage.active{border-color:#36c6ee77}.stage.unavailable{border-color:#f2c36e70;background:#3b2d1370}.stage.skipped{opacity:.48}.stage-icon{float:right;width:19px;height:19px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:1px solid #38556c;color:#7590a3;font-size:.55rem;font-weight:950}.stage.done .stage-icon{background:var(--green);border-color:var(--green);color:#062016}.stage.active .stage-icon{border-color:var(--cyan);color:var(--cyan)}.stage.unavailable .stage-icon{background:var(--amber);border-color:var(--amber);color:#261b05}.stage-num{color:var(--cyan);font-size:.46rem;font-weight:950}.stage-name{margin-top:.27rem;font-size:.53rem;font-weight:900}.stage-note{margin-top:.18rem;color:#7f96a8;font-size:.46rem;line-height:1.2}
.result-layout{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.72fr);gap:.55rem;margin-top:.7rem}.decision-card{position:relative;overflow:hidden;padding:1rem;border:1px solid #4ee39a66;border-radius:17px;background:radial-gradient(circle at 100% 0,#4ee39a18,transparent 34%),linear-gradient(145deg,#0d3027,#07131e 72%);box-shadow:0 16px 44px #0004}.decision-card.uncertain{border-color:#f2c36e66;background:radial-gradient(circle at 100% 0,#f2c36e18,transparent 34%),linear-gradient(145deg,#352a14,#07131e 72%)}.decision-card.new{border-color:#36c6ee66;background:radial-gradient(circle at 100% 0,#36c6ee18,transparent 34%),linear-gradient(145deg,#0c2d3b,#07131e 72%)}.decision-kicker{color:var(--green);font-size:.53rem;font-weight:950;letter-spacing:.13em}.decision-card.uncertain .decision-kicker{color:var(--amber)}.decision-card.new .decision-kicker{color:var(--cyan)}.decision-main{display:flex;align-items:end;justify-content:space-between;gap:1rem;margin-top:.38rem}.decision-state{font-size:clamp(1.55rem,3vw,2.55rem);font-weight:950;letter-spacing:-.055em;line-height:.95}.decision-id{color:#cfe0e9;font-size:.76rem;font-weight:850}.decision-score{color:var(--green);font-size:1rem;font-weight:950;white-space:nowrap}.decision-card.uncertain .decision-score{color:var(--amber)}.decision-card.new .decision-score{color:var(--cyan)}.decision-note{margin-top:.55rem;color:#9db2c0;font-size:14px;line-height:1.4}.reason-box{margin-top:.6rem;padding:.65rem;border:1px solid #36c6ee33;border-radius:11px;background:#36c6ee08}.reason-box b{display:block;color:#a9c9d7;font-size:.5rem;letter-spacing:.09em;text-transform:uppercase}.reason-box span{display:block;margin-top:.24rem;color:#dbe7ed;font-size:14px;line-height:1.45;overflow-wrap:anywhere}.score-meter{margin-top:.6rem}.score-meter-head{display:flex;justify-content:space-between;color:#839aab;font-size:.5rem}.score-track{height:6px;margin-top:.3rem;border-radius:999px;background:#142535;overflow:hidden}.score-fill{height:100%;border-radius:999px;background:var(--green)}.decision-card.uncertain .score-fill{background:var(--amber)}.decision-card.new .score-fill{background:var(--cyan)}
.attribute-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.35rem;margin-top:.55rem}.attribute-card{padding:.5rem;border:1px solid #91a8be18;border-radius:10px;background:#07121ed0}.attribute-card small{display:block;color:#748da1;font-size:.45rem;font-weight:950;text-transform:uppercase}.attribute-card strong{display:block;margin-top:.18rem;color:#e1eaf0;font-size:14px;overflow-wrap:anywhere}.attribute-card.missing strong{color:var(--amber)}
.advisory-panel{padding:.9rem;border:1px solid #b68cff3b;border-radius:17px;background:linear-gradient(145deg,#24163b32,#07111cf5)}.advisory-kicker{color:var(--purple);font-size:.52rem;font-weight:950;letter-spacing:.11em}.advisory-title{margin-top:.22rem;font-size:.84rem;font-weight:950}.advisory-note{margin-top:.25rem;color:#899eae;font-size:14px;line-height:1.35}.advisory-summary{margin-top:.6rem;padding:.55rem;border:1px solid #b68cff2d;border-radius:10px;background:#0a101bd0}.advisory-summary strong{color:#d6c5ff;font-size:.55rem}.advisory-summary > span:not(.status-chip){display:block;margin-top:.2rem;color:#a8b8c4;font-size:14px;line-height:1.4}.status-chip{display:inline-block;margin-top:.4rem;padding:.25rem .42rem;border:1px solid #b68cff45;border-radius:999px;color:#d5c4ff;font-size:.47rem;font-weight:900}
.detail-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(230px,.65fr);gap:.55rem;margin-top:.55rem}.detail-card{padding:.8rem;border:1px solid var(--line);border-radius:15px;background:#07121edb}.detail-title{font-size:.72rem;font-weight:950}.detail-note{margin-top:.18rem;color:#7f96a8;font-size:14px}.verify{display:flex;align-items:center;justify-content:center;min-height:180px;text-align:center}.verify-ring{width:120px;height:120px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;border:9px solid #2d7559;border-top-color:var(--green);border-right-color:var(--green);box-shadow:0 0 28px #4ee39a15}.verify-ring strong{font-size:1.5rem}.verify-ring span{color:#8ca3b2;font-size:.47rem;margin-top:.08rem}.verify-copy{margin-top:.5rem;color:#9eb2bf;font-size:14px;line-height:1.35}
.impact-map{margin-top:.55rem;padding:.8rem;border:1px solid #36c6ee2e;border-radius:15px;background:linear-gradient(145deg,#0b25364a,#07111bf2)}.impact-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:.35rem;margin-top:.5rem}.impact-node{padding:.5rem;border:1px solid #91a8be1b;border-radius:9px;background:#07121ed0;text-align:center}.impact-node strong{display:block;font-size:.55rem}.impact-node span{display:block;margin-top:.12rem;color:#71899c;font-size:14px}.impact-arrow{color:var(--cyan)}
.data-table-wrap{margin-top:.5rem;width:100%;overflow:hidden}.data-table{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#07131f;font-size:.54rem}.data-table th{padding:.48rem .45rem;text-align:left;background:#101d2b;color:#b8c8d4;border-bottom:1px solid var(--line);font-size:.47rem;font-weight:950;text-transform:uppercase;letter-spacing:.05em}.data-table td{padding:.48rem .45rem;color:#dbe6ec;border-bottom:1px solid #91a8be16;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:break-word;line-height:1.35}.data-table tr:last-child td{border-bottom:0}.data-table tr:hover td{background:#36c6ee07}.data-table.candidate th:nth-child(1),.data-table.candidate td:nth-child(1){width:17%}.data-table.candidate th:nth-child(2),.data-table.candidate td:nth-child(2){width:13%}.data-table.candidate th:nth-child(3),.data-table.candidate td:nth-child(3){width:10%}.data-table.candidate th:nth-child(4),.data-table.candidate td:nth-child(4){width:60%}.data-table.advisory th:nth-child(1),.data-table.advisory td:nth-child(1){width:21%}.data-table.advisory th:nth-child(2),.data-table.advisory td:nth-child(2){width:16%}.data-table.advisory th:nth-child(3),.data-table.advisory td:nth-child(3){width:14%}.data-table.advisory th:nth-child(4),.data-table.advisory td:nth-child(4){width:49%}.table-decision{font-weight:950}.table-decision.same{color:var(--green)}.table-decision.uncertain{color:var(--amber)}.table-decision.new{color:var(--cyan)}.score-cell{font-variant-numeric:tabular-nums;color:#b9e9d0;font-weight:850}.explanation-cell{color:#c6d5df!important}.table-empty{padding:.6rem;border:1px dashed var(--line);border-radius:9px;color:#7e95a6;font-size:14px}
.evidence-row{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.55rem}.evidence-box{padding:.6rem;border:1px solid var(--line);border-radius:11px;background:#07121ed0}.evidence-label{color:#7c96a9;font-size:.47rem;font-weight:950;text-transform:uppercase;letter-spacing:.07em}.evidence-value{margin-top:.25rem;color:#d7e4eb;font-size:14px;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere}
.advisory-table{margin-top:.55rem}.review-card,.audit-card{margin-top:.55rem;padding:.8rem;border:1px solid #36c6ee2e;border-radius:15px;background:#07121ed0}.review-kicker{color:var(--cyan);font-size:.49rem;font-weight:950;letter-spacing:.1em}.review-title{margin-top:.2rem;font-size:.7rem;font-weight:950}.review-note{margin-top:.18rem;color:#7f96a8;font-size:14px;line-height:1.35}
.stButton button{min-height:2.15rem!important;border-radius:10px!important;background:#091827!important;border:1px solid var(--line)!important;color:#dce7ee!important;font-size:.61rem!important;font-weight:850!important}.stButton button:hover{border-color:#36c6ee99!important;color:#fff!important}.stButton button[kind="primary"]{background:linear-gradient(90deg,#159fce,#36c6ee)!important;border:0!important;color:#03131a!important}textarea{border-radius:10px!important}.stAlert{border-radius:10px!important}.stSelectbox label,.stTextArea label{font-size:.55rem!important;color:#8ea6b7!important}
.data-table td{font-size:14px}.result-body-note{font-size:14px!important}div[data-testid="stCaptionContainer"] p{font-size:14px!important}
@media(max-width:1100px){.result-layout{grid-template-columns:1fr}.trust-row{grid-template-columns:repeat(2,1fr)}.attribute-grid{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:repeat(3,1fr)}.stage:not(:last-child):after{display:none}}
@media(max-width:760px){.block-container{padding:max(4rem,8vh) .4rem 1.5rem}.topbar{display:block}.top-status{display:inline-flex;margin-top:.5rem}.hero{border-radius:17px}.hero-title{font-size:2rem}.trust-row,.detail-grid,.evidence-row{grid-template-columns:1fr}.scenario-buttons{grid-template-columns:1fr}.flow{grid-template-columns:repeat(2,1fr)}.attribute-grid{grid-template-columns:1fr}.decision-main{display:block}.decision-score{margin-top:.4rem}.impact-flow{grid-template-columns:1fr}.impact-arrow{transform:rotate(90deg)}.data-table,.data-table thead,.data-table tbody,.data-table tr,.data-table th,.data-table td{display:block}.data-table thead{display:none}.data-table{border:0;background:transparent}.data-table tr{margin:.4rem 0;border:1px solid var(--line);border-radius:10px;background:#07131f;overflow:hidden}.data-table td{display:grid;grid-template-columns:minmax(7rem,34%) minmax(0,66%);gap:.45rem;padding:.48rem .52rem;border-bottom:1px solid #91a8be16}.data-table td:last-child{border-bottom:0}.data-table td::before{content:attr(data-label);color:#708a9e;font-size:.45rem;font-weight:950;text-transform:uppercase;letter-spacing:.04em}.data-table.candidate th,.data-table.candidate td,.data-table.advisory th,.data-table.advisory td{width:auto!important}}
@media(max-width:430px){.hero-title{font-size:1.7rem}.flow{grid-template-columns:1fr}.stage{min-height:60px}.stage:not(:last-child):after{display:block;content:"↓";right:50%;top:auto;bottom:-.62rem;transform:translateX(50%)}.data-table td{grid-template-columns:1fr;gap:.16rem}}
</style>
""",
        unsafe_allow_html=True,
    )


def _table_html(rows: list[dict[str, Any]], kind: str = "default") -> str:
    if not rows:
        return '<div class="table-empty">No evidence is available.</div>'
    headers = list(rows[0].keys())
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for header in headers:
            value = row.get(header, "")
            text = f"{value:.3f}" if isinstance(value, float) else _display_value(value)
            classes: list[str] = []
            if header == "Decision":
                decision = str(text).upper()
                classes = [
                    "table-decision",
                    "same" if decision == "SAME" else "uncertain" if decision == "UNCERTAIN" else "new" if decision == "NEW_CANDIDATE" else "",
                ]
            if header == "Score" or "Score" in header:
                classes.append("score-cell")
            if header in {"Explanation", "Reason"}:
                classes.append("explanation-cell")
            class_attr = f' class="{" ".join(c for c in classes if c)}"' if any(classes) else ""
            cells.append(
                f'<td data-label="{escape(str(header))}"{class_attr}>{escape(text)}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    extra = f" {kind}" if kind != "default" else ""
    return (
        '<div class="data-table-wrap"><table class="data-table'
        + extra
        + '"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _sidebar() -> bool:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">◆</span> CPSE MATERIAL<br>HARMONIZATION</div><div class="sub">SIH 2026 · PS 26099</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-kicker">System</div><div class="side-row"><span class="dot-green">●</span>SYSTEM ONLINE</div>',
            unsafe_allow_html=True,
        )
        selected = st.segmented_control(
            "Analysis mode",
            ["Deterministic Only", "Hybrid AI"],
            default="Deterministic Only" if not st.session_state.get("ai_enabled", False) else "Hybrid AI",
            key="analysis_mode_selector",
            label_visibility="collapsed",
            width="stretch",
        )
        ai = selected == "Hybrid AI"
        st.session_state["ai_enabled"] = ai
        st.markdown(
            '<div class="side-kicker">Decision authority</div><div class="side-row"><span class="dot-green">◇</span>Deterministic Engine<span class="side-right">AUTHORITATIVE</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-kicker">Advisory services</div><div class="side-row"><span class="dot-purple">✦</span>Local NLP<span class="side-right">ADVISORY</span></div><div class="side-row"><span class="dot-purple">✦</span>Gemini 2.5 Flash<span class="side-right">ADVISORY</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-kicker">Data source</div><div class="side-row"><span class="dot-green">●</span>Demo material master<span class="side-right">SYNTHETIC</span></div><div class="side-row"><span class="dot-cyan">○</span>SAP / ERP<span class="side-right">BOUNDARY</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-kicker">System status</div><div class="side-row"><span class="dot-green">●</span>Deterministic engine<span class="side-right">READY</span></div><div class="side-row"><span class="dot-purple">●</span>Optional AI layer<span class="side-right">ADVISORY</span></div>',
            unsafe_allow_html=True,
        )
        return ai


def _header(ai: bool) -> None:
    mode = "HYBRID AI" if ai else "DETERMINISTIC ONLY"
    st.markdown(
        f'<div class="topbar"><div><div class="top-title">Material Harmonization Across CPSEs</div><div class="top-sub">Deterministic · Transparent · Trustworthy</div></div><div class="top-status"><b>●</b> System online · {mode}</div></div>'
        f'<div class="hero"><div class="eyebrow">CPSE MATERIAL INTELLIGENCE · SIH 2026</div><div class="hero-title">One material. <span>Many codes.</span><br>One trusted identity.</div><div class="hero-sub">AI-assisted standardization of legacy CPSE material descriptions into a common canonical vocabulary — with deterministic technical verification as the final authority.</div><div class="hero-pills"><span class="pill">PS 26099</span><span class="pill online">● SYSTEM ONLINE</span><span class="pill">MODE · {mode}</span><span class="pill">SYNTHETIC DEMO DATA · CLEARLY LABELLED</span></div></div>'
        '<div class="trust-row"><div class="trust-card primary"><small>Core outcome</small><strong>Legacy descriptions → common material identity</strong></div><div class="trust-card"><small>Decision source</small><strong>Deterministic rules</strong></div><div class="trust-card"><small>AI role</small><strong>Candidate discovery only</strong></div><div class="trust-card"><small>Governance</small><strong>Auditable + reviewable</strong></div></div>',
        unsafe_allow_html=True,
    )


def _pipeline(
    done: tuple[str, ...] = (),
    active: str | None = None,
    container: Any | None = None,
    ai: bool = True,
    ai_state: str | None = None,
) -> None:
    target = container or st
    completed = set(done)
    html = '<div class="pipeline"><div class="pipeline-head"><span>Processing pipeline</span><small>AI advises · deterministic decides</small></div><div class="flow">'
    for number, name, note, key in PIPELINE_STAGES:
        state = (
            "skipped" if key == "ai_advisory" and not ai
            else "unavailable" if key == "ai_advisory" and ai_state == "unavailable"
            else "done" if key in completed
            else "active" if key == active
            else ""
        )
        icon = "—" if state == "skipped" else "⚠" if state == "unavailable" else "✓" if state == "done" else "•" if state == "active" else ""
        stage_note = "Disabled in current mode" if state == "skipped" else "Advisory unavailable" if state == "unavailable" else note
        html += f'<div class="stage {state}"><span class="stage-icon">{icon}</span><div class="stage-num">{number}</div><div class="stage-name">{name}</div><div class="stage-note">{stage_note}</div></div>'
    target.markdown(html + "</div></div>", unsafe_allow_html=True)


def _decision_card(result: HybridResult | None) -> None:
    if result is None:
        st.markdown(
            '<div class="decision-card"><div class="decision-kicker">AUTHORITATIVE DECISION</div><div class="decision-state">Awaiting analysis</div><div class="decision-note">Enter a material description and analyze it to produce the deterministic result.</div></div>',
            unsafe_allow_html=True,
        )
        return
    mapping = result.mapping_result
    decision = str(mapping.decision)
    css_class = "uncertain" if decision == "UNCERTAIN" else "new" if decision == "NEW_CANDIDATE" else ""
    canonical_id = mapping.canonical_material_id or "Not assigned"
    score = float(mapping.score)
    reason = mapping.explanation or "No deterministic explanation was returned."
    score_pct = max(0, min(100, round(score * 100)))
    st.markdown(
        f'<div class="decision-card {css_class}"><div class="decision-kicker">✓ AUTHORITATIVE DECISION</div><div class="decision-main"><div><div class="decision-state">{escape(decision)}</div><div class="decision-id">Canonical material · {escape(str(canonical_id))}</div></div><div class="decision-score">{score:.3f}</div></div><div class="decision-note">The score is the deterministic attribute-comparison score — not an AI probability.</div><div class="score-meter"><div class="score-meter-head"><span>Technical match score</span><span>{score_pct}%</span></div><div class="score-track"><div class="score-fill" style="width:{score_pct}%"></div></div></div><div class="reason-box"><b>Why this decision?</b><span>{escape(reason)}</span></div></div>',
        unsafe_allow_html=True,
    )
    attributes = result.attributes
    items = []
    count = 0
    for label, field_name in KEY_ATTRIBUTES:
        value = getattr(attributes, field_name, None)
        if value is not None:
            count += 1
        cls = "" if value is not None else "missing"
        items.append(f'<div class="attribute-card {cls}"><small>{escape(label)}</small><strong>{escape(_display_value(value))}</strong></div>')
    st.markdown(
        f'<div class="attribute-grid">{"".join(items)}</div><div class="decision-note" style="margin-top:.45rem">{count}/6 key technical attributes extracted from the input.</div>',
        unsafe_allow_html=True,
    )


def _advisory_summary(result: HybridResult) -> None:
    consensus = result.ai_consensus
    local = ", ".join(consensus.local_candidates) if consensus.local_candidates else "No candidate returned"
    gemini = ", ".join(consensus.gemini_candidates) if consensus.gemini_candidates else "No candidate returned"
    st.markdown(
        f'<div class="advisory-panel"><div class="advisory-kicker">ADVISORY INTELLIGENCE · REFERENCE ONLY</div><div class="advisory-title">AI candidate evidence</div><div class="advisory-note">AI output is displayed exactly as advisory evidence. It never changes the deterministic decision.</div><div class="advisory-summary"><strong>Local NLP</strong><span>{escape(local)}</span><span class="status-chip">ADVISORY</span></div><div class="advisory-summary"><strong>Gemini 2.5 Flash</strong><span>{escape(gemini)}</span><span class="status-chip">ADVISORY</span></div></div>',
        unsafe_allow_html=True,
    )


def _details(result: HybridResult) -> None:
    raw = escape(st.session_state.get("material_description", ""))
    normalized = escape(result.normalized_description or "Not provided")
    transformations = result.normalization_transformations
    transformation_text = "\n".join("✓ " + item for item in transformations) if transformations else "No transformations reported."
    st.markdown(
        f'<div class="evidence-row"><div class="evidence-box"><div class="evidence-label">Legacy input</div><div class="evidence-value">{raw}</div></div><div class="evidence-box"><div class="evidence-label">Normalized description</div><div class="evidence-value">{normalized}</div></div><div class="evidence-box"><div class="evidence-label">Normalization steps</div><div class="evidence-value">{escape(transformation_text)}</div></div><div class="evidence-box"><div class="evidence-label">Deterministic explanation</div><div class="evidence-value">{escape(result.mapping_result.explanation or "No explanation reported.")}</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="detail-grid">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="detail-card"><div class="detail-title">Deterministic evidence</div><div class="detail-note">Authoritative candidate comparison from the existing mapping engine.</div>', unsafe_allow_html=True)
        candidates = result.mapping_result.all_candidates
        if candidates:
            st.markdown(_table_html(candidate_rows(candidates), kind="candidate"), unsafe_allow_html=True)
        else:
            st.markdown('<div class="table-empty">No candidate evidence is available.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with st.container():
        extracted = sum(getattr(result.attributes, field, None) is not None for _, field in KEY_ATTRIBUTES)
        st.markdown('<div class="detail-card verify"><div><div class="verify-ring"><strong>' + str(extracted) + '/6</strong><span>attributes extracted</span></div><div class="verify-copy">Only extracted technical values are used in the summary. Missing values remain missing.</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    cid = result.mapping_result.canonical_material_id or "Not assigned"
    st.markdown(
        f'<div class="impact-map"><div class="detail-title">Harmonization path</div><div class="impact-flow"><div class="impact-node"><strong>Legacy description</strong><span>CPSE-specific wording</span></div><div class="impact-arrow">→</div><div class="impact-node"><strong>Technical identity</strong><span>Normalized + verified</span></div><div class="impact-arrow">→</div><div class="impact-node"><strong>{escape(str(cid))}</strong><span>Canonical material ID</span></div></div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("All extracted attributes", expanded=False):
        st.markdown(_table_html(attribute_rows(result.attributes), kind="attributes"), unsafe_allow_html=True)
    with st.expander("AI advisory evidence", expanded=False):
        local = tuple(item for item in result.ai_candidate_suggestions if item.source == "local_embedding")
        gemini = tuple(item for item in result.ai_candidate_suggestions if item.source == "gemini")
        st.markdown('<div class="advisory-table">', unsafe_allow_html=True)
        st.markdown('<div class="detail-title">Local NLP · Embedding Retrieval</div>', unsafe_allow_html=True)
        if local:
            st.markdown(_table_html(advisory_rows(local, "Cosine Similarity"), kind="advisory"), unsafe_allow_html=True)
        else:
            st.caption("No valid Local NLP candidates returned.")
        st.markdown('<div class="detail-title" style="margin-top:.8rem">Gemini 2.5 Flash · LLM</div>', unsafe_allow_html=True)
        if gemini:
            st.markdown(_table_html(advisory_rows(gemini, "Gemini Compatibility Score"), kind="advisory"), unsafe_allow_html=True)
            st.caption("Advisory score only — not a probability or authoritative confidence.")
        else:
            st.caption("No valid Gemini candidates returned.")
        st.markdown('</div>', unsafe_allow_html=True)
    _review(result)
    _audit()


def _review(result: HybridResult) -> None:
    suggestions = result.ai_candidate_suggestions
    ids = [item.canonical_material_id for item in suggestions]
    deterministic_id = result.mapping_result.canonical_material_id
    default = deterministic_id if deterministic_id in ids else (ids[0] if ids else "")
    st.markdown('<div class="review-card"><div class="review-kicker">HUMAN VALIDATION</div><div class="review-title">Review an advisory candidate</div><div class="review-note">Review actions are recorded for governance and never modify the authoritative deterministic result.</div>', unsafe_allow_html=True)
    if not ids:
        st.caption("No advisory candidates are available for human review.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    selected = st.selectbox("Candidate", ids, index=ids.index(default) if default in ids else 0, key="review_candidate")
    note = st.text_area("Reviewer note", key="review_note", height=70, placeholder="Optional review note")
    cols = st.columns(3)
    for col, (label, action, kind) in zip(cols, (("✓ APPROVE", "APPROVE", "primary"), ("✕ REJECT", "REJECT", "secondary"), ("⚠ REVIEW", "REVIEW", "secondary"))):
        with col:
            if st.button(label, type=kind, use_container_width=True, key=f"review_{action.lower()}"):
                event = st.session_state["audit_trail"].record_review(
                    legacy_material_code=result.legacy_material_code,
                    input_description=st.session_state.get("material_description", ""),
                    deterministic_decision=result.mapping_result.decision,
                    deterministic_material_id=result.mapping_result.canonical_material_id,
                    candidate_material_id=selected,
                    action=action,
                    note=note,
                )
                st.session_state["review_action"] = event
                st.rerun()
    previous = st.session_state.get("review_action")
    if previous is not None:
        st.success(f"{previous.reviewer_action}: {previous.selected_material_id or 'no candidate selected'} recorded. Deterministic result remains {result.mapping_result.decision} · {result.mapping_result.canonical_material_id or 'Not assigned'}.")
    st.markdown('</div>', unsafe_allow_html=True)


def _audit() -> None:
    trail = st.session_state.get("audit_trail")
    if not isinstance(trail, AuditTrail) or not trail.events:
        return
    st.markdown('<div class="audit-card"><div class="review-kicker">AUDIT TRAIL</div><div class="review-title">Session review history</div><div class="review-note">Session-local review history; production storage can be replaced by governed storage.</div>', unsafe_allow_html=True)
    st.markdown(_table_html([event.to_dict() for event in reversed(trail.events)], kind="audit"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _status(result: HybridResult | None, ai: bool) -> None:
    if result is None and not ai:
        st.caption("AI advisory is disabled; Local NLP and Gemini are not invoked.")
        return
    statuses = result.ai_statuses if result is not None else tuple(item.status() for item in _analysis_adapters(True))
    cols = st.columns(2)
    for col, status, label in zip(cols, statuses, ("Local NLP", "Gemini 2.5 Flash")):
        with col:
            cls = "dot-green" if status.available else "dot-amber"
            st.markdown(f'<div class="side-row"><span class="{cls}">●</span>{label}<span class="side-right">{_friendly_status(status).upper()}</span></div>', unsafe_allow_html=True)


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required to run this dashboard. Use: streamlit run app.py")
    st.set_page_config(page_title="CPSE Material Intelligence", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    _css()
    defaults = (
        ("ai_enabled", False),
        ("analysis_result", None),
        ("analysis_mode", None),
        ("material_description", ""),
        ("pipeline_completed", ()),
        ("pipeline_ai_status", None),
        ("audit_trail", AuditTrail()),
        ("review_action", None),
        ("review_note", ""),
    )
    for key, value in defaults:
        st.session_state.setdefault(key, value)

    ai = _sidebar()
    previous = st.session_state.get("analysis_result")
    previous_mode = st.session_state.get("analysis_mode")
    mode_changed = previous is not None and previous_mode is not None and previous_mode != ai
    _header(ai)

    st.markdown('<div class="workspace"><div class="section-title">Analyze material</div><div class="card"><div class="input-head"><div><div class="input-title">Analyze a legacy material</div><div class="input-note">Enter a material description or use an example input.</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="scenario-label">Example inputs</div><div class="scenario-buttons">', unsafe_allow_html=True)
    for outcome, example in DEMO_EXAMPLES.items():
        button_label = "New candidate" if outcome == "NEW CANDIDATE" else outcome.title()
        if st.button(button_label, use_container_width=True, key=f"example_{outcome}"):
            st.session_state.update(material_description=example, analysis_result=None, pipeline_completed=(), pipeline_ai_status=None, review_action=None)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    description = st.text_area("Material Description", key="material_description", height=82, placeholder="Gate Valve Carbon Steel 150 50mm Flanged", label_visibility="collapsed")
    analyze = st.button("Analyze material  →", type="primary", use_container_width=True, key="analyze_main")
    st.markdown('</div></div>', unsafe_allow_html=True)

    pipeline_placeholder = st.empty()
    if not analyze:
        _pipeline(st.session_state.get("pipeline_completed", ()), None, pipeline_placeholder, ai, st.session_state.get("pipeline_ai_status"))

    if analyze:
        if not description.strip():
            st.session_state["analysis_result"] = None
            st.warning("Material description required. Enter a legacy description to begin.")
        else:
            st.session_state.update(analysis_result=None, pipeline_completed=(), pipeline_ai_status=None, review_action=None)
            _pipeline((), "input", pipeline_placeholder, ai, None)
            try:
                def progress(stage: str) -> None:
                    done = list(st.session_state.get("pipeline_completed", ()))
                    if stage == "ai_advisory":
                        _pipeline(tuple(done), "ai_advisory", pipeline_placeholder, ai, None)
                        return
                    if stage not in done:
                        done.append(stage)
                    st.session_state["pipeline_completed"] = tuple(done)
                    remaining = next((key for _, _, _, key in PIPELINE_STAGES if key not in done and (key != "ai_advisory" or ai)), None)
                    _pipeline(tuple(done), remaining, pipeline_placeholder, ai, None)

                catalog = CSVMaterialCatalogSource(CATALOG_PATH).load()
                result = analyze_material(description, catalog, ai, progress_callback=progress)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_mode"] = ai
                ai_state = "skipped" if not ai else "complete" if any(status.available for status in result.ai_statuses) else "unavailable"
                completed = [key for _, _, _, key in PIPELINE_STAGES if key != "ai_advisory"]
                if ai_state == "complete":
                    completed.insert(4, "ai_advisory")
                st.session_state.update(pipeline_ai_status=ai_state, pipeline_completed=tuple(completed))
                st.session_state["audit_trail"].record_analysis(
                    legacy_material_code=result.legacy_material_code,
                    input_description=description,
                    deterministic_decision=result.mapping_result.decision,
                    deterministic_material_id=result.mapping_result.canonical_material_id,
                    ai_candidates=tuple(item.canonical_material_id for item in result.ai_candidate_suggestions),
                )
                _pipeline(tuple(completed), None, pipeline_placeholder, ai, ai_state)
                st.rerun()
            except (OSError, TypeError, ValueError):
                st.session_state["analysis_result"] = None
                st.error("The description could not be processed safely. Please check the input and try again.")

    result = st.session_state.get("analysis_result")
    if mode_changed:
        st.warning("Analysis mode changed. Re-analyze to generate evidence for the current mode.")
        result = None
        st.session_state["analysis_result"] = None

    if result is None:
        st.markdown('<div class="card" style="margin-top:.6rem"><div class="section-title" style="margin:0">Workflow</div><div class="input-title">Ready for a governed material decision</div><div class="input-note" style="margin-top:.25rem">Input → normalize → extract → match → optional advisory → authoritative decision.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-layout">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        _decision_card(result)
        _details(result)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        _advisory_summary(result)
        st.markdown('<div class="card" style="margin-top:.55rem"><div class="section-title" style="margin:0">Decision authority</div><div class="input-title" style="margin-top:.25rem">Deterministic engine</div><div class="input-note result-body-note" style="margin-top:.25rem">The authoritative result is calculated from technical attribute comparison. AI evidence is advisory only.</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="workspace"><div class="section-title">Service status</div>', unsafe_allow_html=True)
        _status(result, ai)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;padding:.65rem 0;border-top:1px solid var(--line);color:#60798d;font-size:.5rem">CPSE MATERIAL INTELLIGENCE · SIH 2026 · PS 26099 · Deterministic decision engine · Optional advisory AI layer</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
