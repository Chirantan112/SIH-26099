"""Judge-facing visual demo for SIH PS 26099.

Presentation-only page. It reuses the existing deterministic harmonization
engine and does not modify matching, AI, governance, or dataset logic.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.hybrid_pipeline import run_hybrid_pipeline
from src.llm_interpretation import UnavailableLLMAdapter
from src.ai_retrieval import UnavailableRetrievalAdapter
from src.source_adapters import CSVMaterialCatalogSource

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "demo" / "material_master.csv"
SCENARIOS = {
    "MATCHED": "CS GATE VLV 50MM FLG CL150",
    "UNCERTAIN": "GATE VLV CS 50MM CL150",
    "NEW CANDIDATE": "PIPE CS OD 999 MM THK 3 MM SCH-40 PLAIN END",
}


def css() -> None:
    st.markdown(
        """
<style>
.stApp{background:radial-gradient(900px 520px at 82% -10%,#22c7ee18,transparent 60%),radial-gradient(700px 500px at 10% 90%,#a66cff0b,transparent 65%),#050b12;color:#eef7fb}
.block-container{max-width:1480px;padding:3.6rem clamp(.65rem,2.2vw,2.2rem) 2.5rem}
.hero{position:relative;overflow:hidden;border:1px solid #31516a;border-radius:28px;padding:clamp(1.2rem,3vw,2.2rem);background:linear-gradient(135deg,#102b40,#07131f 72%);box-shadow:0 30px 90px #0007}
.hero:before{content:"";position:absolute;width:380px;height:380px;right:-150px;top:-190px;border-radius:50%;background:radial-gradient(circle,#35c8ef30,transparent 68%)}
.eyebrow{position:relative;color:#79dfff;font-size:.55rem;font-weight:950;letter-spacing:.2em}.title{position:relative;margin-top:.55rem;font-size:clamp(2.3rem,6vw,5.4rem);line-height:.9;font-weight:950;letter-spacing:-.075em}.title span{color:#45d2f3}.subtitle{position:relative;max-width:880px;margin-top:1rem;color:#a9bdca;font-size:.78rem;line-height:1.55}.chips{position:relative;display:flex;flex-wrap:wrap;gap:.4rem;margin-top:1rem}.chip{padding:.35rem .6rem;border:1px solid #31516a;border-radius:999px;background:#ffffff06;color:#d7e7ee;font-size:.5rem;font-weight:900}.chip.green{border-color:#4ee39a55;color:#62efa9}.section{margin-top:1rem}.section-label{margin-bottom:.45rem;color:#7f9bad;font-size:.52rem;font-weight:950;letter-spacing:.16em;text-transform:uppercase}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.45rem}.stat{padding:.8rem;border:1px solid #203b50;border-radius:15px;background:#08131e}.stat small{display:block;color:#7892a5;font-size:.46rem;font-weight:900;text-transform:uppercase;letter-spacing:.08em}.stat strong{display:block;margin-top:.25rem;font-size:.82rem}.stat.accent strong{color:#4ee39a}
.demo{margin-top:1rem;padding:1rem;border:1px solid #203b50;border-radius:20px;background:linear-gradient(180deg,#0a1825,#07111a)}.demo-head{display:flex;align-items:end;justify-content:space-between;gap:1rem}.demo-title{font-size:1rem;font-weight:950}.demo-note{color:#7891a4;font-size:.52rem}.scenario-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;margin-top:.65rem}.scenario-row button{min-height:2.5rem!important;border-radius:12px!important;background:#0b1c2a!important;border:1px solid #29465b!important;font-weight:900!important}.scenario-row button:hover{border-color:#42d0f2!important}
.input-shell{margin-top:.7rem;padding:.8rem;border:1px solid #1f3a4e;border-radius:14px;background:#06101a}.input-label{color:#7892a5;font-size:.48rem;font-weight:950;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.35rem}.input-value{color:#eaf5fa;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.72rem;word-break:break-word}
.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.32rem;margin-top:.7rem}.stage{position:relative;padding:.65rem .5rem;border:1px solid #203b50;border-radius:12px;background:#081521}.stage:not(:last-child):after{content:"→";position:absolute;right:-.34rem;top:50%;transform:translateY(-50%);color:#52758a;z-index:3}.stage b{display:block;color:#44d0f2;font-size:.44rem}.stage strong{display:block;margin-top:.25rem;font-size:.53rem}.stage small{display:block;margin-top:.16rem;color:#7891a4;font-size:.43rem;line-height:1.25}.stage.done{border-color:#4ee39a55;background:#0a2a2166}.stage.done b{color:#4ee39a}
.result{margin-top:.75rem;display:grid;grid-template-columns:minmax(0,1.5fr) minmax(250px,.65fr);gap:.5rem}.decision{padding:1rem;border:1px solid #4ee39a66;border-radius:18px;background:radial-gradient(circle at 100% 0,#4ee39a18,transparent 36%),linear-gradient(145deg,#0b3026,#07121b 70%)}.decision.uncertain{border-color:#f2c36e66;background:radial-gradient(circle at 100% 0,#f2c36e18,transparent 36%),linear-gradient(145deg,#322711,#07121b 70%)}.decision.new{border-color:#42d0f266;background:radial-gradient(circle at 100% 0,#42d0f218,transparent 36%),linear-gradient(145deg,#0a2937,#07121b 70%)}.kicker{color:#4ee39a;font-size:.48rem;font-weight:950;letter-spacing:.14em}.uncertain .kicker{color:#f2c36e}.new .kicker{color:#42d0f2}.decision-main{display:flex;justify-content:space-between;align-items:end;gap:1rem;margin-top:.35rem}.state{font-size:clamp(1.5rem,3vw,2.5rem);font-weight:950;letter-spacing:-.055em}.cid{margin-top:.2rem;color:#c6d8e2;font-size:.65rem;font-weight:850}.score{color:#4ee39a;font-size:1rem;font-weight:950}.uncertain .score{color:#f2c36e}.new .score{color:#42d0f2}.reason{margin-top:.6rem;color:#a8bbc7;font-size:.55rem;line-height:1.45}.meter{height:7px;margin-top:.65rem;border-radius:99px;background:#142635;overflow:hidden}.fill{height:100%;background:#4ee39a;border-radius:99px}.uncertain .fill{background:#f2c36e}.new .fill{background:#42d0f2}
.authority{padding:1rem;border:1px solid #b68cff44;border-radius:18px;background:linear-gradient(145deg,#25153a44,#07111bf5)}.authority .kicker{color:#bd9aff}.authority-title{margin-top:.25rem;font-size:.9rem;font-weight:950}.authority-copy{margin-top:.3rem;color:#899eae;font-size:.53rem;line-height:1.45}.authority-row{display:flex;justify-content:space-between;gap:.5rem;margin-top:.65rem;padding:.55rem;border:1px solid #b68cff28;border-radius:10px;background:#0b101bd0;color:#d7c8f8;font-size:.53rem}.authority-row span{color:#8fa4b4}.evidence{margin-top:.7rem;padding:.8rem;border:1px solid #203b50;border-radius:16px;background:#07121c}.evidence-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;margin-top:.5rem}.evidence-card{padding:.6rem;border:1px solid #1d3547;border-radius:11px;background:#081520}.evidence-card small{display:block;color:#718a9d;font-size:.45rem;font-weight:950;text-transform:uppercase}.evidence-card strong{display:block;margin-top:.2rem;color:#dce9ef;font-size:.57rem;overflow-wrap:anywhere}.footer{margin-top:1.2rem;padding-top:.7rem;border-top:1px solid #1b3040;color:#60798c;font-size:.47rem}
@media(max-width:1000px){.result{grid-template-columns:1fr}.stat-grid{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:repeat(3,1fr)}.stage:not(:last-child):after{display:none}}
@media(max-width:680px){.block-container{padding:3.8rem .45rem 1.5rem}.scenario-row,.evidence-grid,.stat-grid{grid-template-columns:1fr}.flow{grid-template-columns:repeat(2,1fr)}.stage:not(:last-child):after{display:block;content:"↓";right:50%;top:auto;bottom:-.65rem;transform:translateX(50%)}.decision-main{display:block}.score{margin-top:.4rem}}
</style>
""",
        unsafe_allow_html=True,
    )


def run_demo(description: str):
    catalog = CSVMaterialCatalogSource(CATALOG_PATH).load()
    return run_hybrid_pipeline(
        description,
        catalog,
        legacy_material_code="JUDGE-DEMO",
        retrieval_adapter=UnavailableRetrievalAdapter("Judge demo uses deterministic mode."),
        llm_adapter=UnavailableLLMAdapter("Judge demo uses deterministic mode."),
    )


def render() -> None:
    css()
    st.markdown(
        '<div class="hero"><div class="eyebrow">SIH 2026 · PROBLEM STATEMENT 26099</div><div class="title">From many codes<br>to <span>one identity.</span></div><div class="subtitle">A judge-facing walkthrough of CPSE material harmonization. The demo makes the technical journey visible: normalize the legacy description, extract engineering attributes, compare against the catalog, and return an authoritative deterministic decision.</div><div class="chips"><span class="chip">DETERMINISTIC CORE</span><span class="chip green">● AUTHORITATIVE DECISION</span><span class="chip">AI = ADVISORY ONLY</span><span class="chip">SYNTHETIC DATA · CLEARLY LABELLED</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section"><div class="stat-grid"><div class="stat accent"><small>Decision authority</small><strong>Deterministic engine</strong></div><div class="stat"><small>Technical evidence</small><strong>Attribute-aware</strong></div><div class="stat"><small>AI role</small><strong>Candidate discovery</strong></div><div class="stat"><small>Escalation</small><strong>Human review</strong></div></div></div>', unsafe_allow_html=True)

    if "judge_description" not in st.session_state:
        st.session_state["judge_description"] = SCENARIOS["MATCHED"]

    st.markdown('<div class="demo"><div class="demo-head"><div><div class="section-label">Live judge demo</div><div class="demo-title">Choose a scenario</div></div><div class="demo-note">Runs the existing harmonization engine — no mock result.</div></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (label, example) in zip(cols, SCENARIOS.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"judge_{label}"):
                st.session_state["judge_description"] = example
                st.session_state.pop("judge_result", None)
                st.rerun()
    st.markdown(f'<div class="input-shell"><div class="input-label">Selected legacy description</div><div class="input-value">{escape(st.session_state["judge_description"])}</div></div>', unsafe_allow_html=True)
    if st.button("Run live harmonization  →", type="primary", use_container_width=True, key="judge_run"):
        with st.spinner("Running deterministic harmonization…"):
            try:
                st.session_state["judge_result"] = run_demo(st.session_state["judge_description"])
            except (OSError, TypeError, ValueError) as exc:
                st.error(f"Demo could not process this input safely: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

    result: Any = st.session_state.get("judge_result")
    if result is None:
        stages = [("01","INPUT","Legacy wording"),("02","NORMALIZE","Canonical text"),("03","EXTRACT","Engineering attributes"),("04","MATCH","Catalog comparison"),("05","AI ADVISE","Optional evidence"),("06","DECIDE","Authoritative result")]
        stage_html = ''.join(f'<div class="stage"><b>{n}</b><strong>{name}</strong><small>{note}</small></div>' for n,name,note in stages)
        st.markdown(f'<div class="section"><div class="section-label">Decision path</div><div class="flow">{stage_html}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="evidence"><div class="section-label">What the judge should notice</div><div class="evidence-grid"><div class="evidence-card"><small>01 · Trust</small><strong>AI cannot override the authoritative mapping.</strong></div><div class="evidence-card"><small>02 · Safety</small><strong>Technical attributes prevent unsafe near-match merging.</strong></div><div class="evidence-card"><small>03 · Honesty</small><strong>Synthetic data and unavailable enterprise integrations are explicitly labelled.</strong></div></div></div>', unsafe_allow_html=True)
    else:
        decision = str(result.mapping_result.decision)
        css_class = "uncertain" if decision == "UNCERTAIN" else "new" if decision == "NEW_CANDIDATE" else ""
        cid = result.mapping_result.canonical_material_id or "Not assigned"
        score = float(result.mapping_result.score)
        score_pct = max(0, min(100, round(score * 100)))
        reason = result.mapping_result.explanation or "No explanation returned."
        st.markdown('<div class="section"><div class="section-label">Authoritative outcome</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result"><div class="decision {css_class}"><div class="kicker">✓ AUTHORITATIVE DECISION</div><div class="decision-main"><div><div class="state">{escape(decision)}</div><div class="cid">Canonical material · {escape(str(cid))}</div></div><div class="score">{score:.3f}</div></div><div class="reason">{escape(reason)}</div><div class="meter"><div class="fill" style="width:{score_pct}%"></div></div></div><div class="authority"><div class="kicker">DECISION BOUNDARY</div><div class="authority-title">Deterministic engine wins.</div><div class="authority-copy">The judge demo intentionally runs without external AI credentials. This isolates the authoritative path and demonstrates that the system remains useful without an LLM.</div><div class="authority-row"><b>Deterministic</b><span>AUTHORITATIVE</span></div><div class="authority-row"><b>Local NLP</b><span>ADVISORY</span></div><div class="authority-row"><b>Gemini</b><span>ADVISORY</span></div><div class="authority-row"><b>Human review</b><span>ESCALATION</span></div></div></div>', unsafe_allow_html=True)
        attrs = result.attributes
        values = [("Category", attrs.category),("Material", attrs.material),("Size", attrs.size_mm),("Pressure", attrs.pressure_class),("Connection", attrs.connection),("Valve type", attrs.valve_type)]
        cards = ''.join(f'<div class="evidence-card"><small>{escape(label)}</small><strong>{escape(str(value if value is not None else "Not provided"))}</strong></div>' for label,value in values)
        st.markdown(f'<div class="evidence"><div class="section-label">Extracted technical identity</div><div class="evidence-grid">{cards}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="evidence"><div class="section-label">Harmonization path</div><div class="flow"><div class="stage done"><b>01</b><strong>INPUT</strong><small>{escape(st.session_state["judge_description"])}</small></div><div class="stage done"><b>02</b><strong>NORMALIZE</strong><small>{escape(result.normalized_description or "Not provided")}</small></div><div class="stage done"><b>03</b><strong>EXTRACT</strong><small>Technical attributes identified</small></div><div class="stage done"><b>04</b><strong>MATCH</strong><small>Deterministic catalog comparison</small></div><div class="stage done"><b>05</b><strong>AI ADVISE</strong><small>Optional — not invoked here</small></div><div class="stage done"><b>06</b><strong>DECIDE</strong><small>Authoritative result locked to deterministic engine</small></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">CPSE MATERIAL INTELLIGENCE · SIH 2026 · PS 26099 · Judge Demo is presentation-only and reuses the existing decision engine.</div>', unsafe_allow_html=True)


render()
