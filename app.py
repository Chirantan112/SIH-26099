"""Judge-facing Streamlit dashboard for CPSE material harmonization."""
from __future__ import annotations
import time
from dataclasses import fields
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Any
from src.ai_retrieval import CandidateSuggestion, UnavailableRetrievalAdapter
from src.attribute_extraction import MaterialAttributes
from src.demo_pipeline import load_demo_catalog
from src.hybrid_pipeline import HybridResult, run_hybrid_pipeline
from src.llm_interpretation import UnavailableLLMAdapter
try:
    import streamlit as st
except ModuleNotFoundError: st = None
try:
    from src.gemini_llm import GeminiLLMAdapter
except Exception: GeminiLLMAdapter = None
try:
    from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter
except Exception: LocalEmbeddingRetrievalAdapter = None
ROOT=Path(__file__).resolve().parent
CATALOG_PATH=ROOT/"data"/"demo"/"material_master.csv"
DEMO_EXAMPLES={"MATCHED":"CS GATE VLV 50MM FLG CL150","UNCERTAIN":"GATE VLV CS 50MM CL150","NEW CANDIDATE":"PIPE CS OD 999 MM THK 3 MM SCH-40 PLAIN END"}
ATTRIBUTE_LABELS={"category":"Category","valve_type":"Valve type","material":"Material","size_mm":"Size (mm)","pressure_class":"Pressure class","connection":"Connection","bearing_family":"Bearing family","dimensions":"Dimensions (mm)","dimension_unit_present":"Dimension unit stated","od_mm":"Outside diameter (mm)","thickness_mm":"Thickness (mm)","schedule":"Schedule","end":"End"}
def _display_value(v:Any)->str:
    if v is None:return "Not provided"
    if isinstance(v,bool):return "Yes" if v else "No"
    if isinstance(v,tuple):return " × ".join(map(str,v))
    if isinstance(v,Decimal):return str(v)
    return str(v)
def attribute_rows(a:MaterialAttributes)->list[dict[str,str]]: return [{"Attribute":ATTRIBUTE_LABELS[f.name],"Extracted value":_display_value(getattr(a,f.name))} for f in fields(a)]
def candidate_rows(cs:tuple[Any,...])->list[dict[str,str|float]]: return [{"Canonical material ID":c.canonical_material_id,"Decision":c.decision,"Score":c.score,"Explanation":c.explanation} for c in cs]
def advisory_rows(ss:tuple[CandidateSuggestion,...])->list[dict[str,str|float]]: return [{"Canonical material ID":s.canonical_material_id,"Source":s.source,"Advisory Rank":s.score,"Reason":s.explanation} for s in ss]
def _friendly_status(s:Any)->str:
    if s.available:return "Available"
    d=(s.detail or "").lower()
    if "api key" in d or "credentials" in d:return "Credentials not configured"
    if "not installed" in d:return "Optional dependency not installed"
    return "Advisory unavailable"
def _analysis_adapters(ai:bool)->tuple[Any,Any]:
    if not ai:return UnavailableRetrievalAdapter("AI advisory is disabled."),UnavailableLLMAdapter("AI advisory is disabled.")
    r=LocalEmbeddingRetrievalAdapter() if LocalEmbeddingRetrievalAdapter else UnavailableRetrievalAdapter("Local NLP adapter is unavailable.")
    l=GeminiLLMAdapter() if GeminiLLMAdapter else UnavailableLLMAdapter("Gemini adapter is unavailable.")
    return r,l
def analyze_material(description:str|None,catalog:tuple[Any,...],ai_enabled:bool,retrieval_adapter:Any|None=None,llm_adapter:Any|None=None)->HybridResult:
    if not ai_enabled:r,l=UnavailableRetrievalAdapter("AI advisory is disabled."),UnavailableLLMAdapter("AI advisory is disabled.")
    elif retrieval_adapter is None or llm_adapter is None:r,l=_analysis_adapters(True)
    else:r,l=retrieval_adapter,llm_adapter
    return run_hybrid_pipeline(description,catalog,legacy_material_code="DASHBOARD-INPUT",retrieval_adapter=r,llm_adapter=l)
def _inject_styles()->None:
    st.markdown("""<style>
:root{--bg:#050b14;--line:#20344a;--text:#f4f8fc;--muted:#91a8be;--green:#55e39a;--cyan:#42c8ee;--purple:#b993ff;--amber:#f1bd63}.stApp{background:radial-gradient(900px 500px at 80% -5%,rgba(44,121,168,.16),transparent 60%),var(--bg);color:var(--text)}.block-container{max-width:1680px;padding:1rem clamp(.75rem,1.8vw,2rem) 2rem}[data-testid="stSidebar"]{background:linear-gradient(180deg,#06111e,#071321);border-right:1px solid var(--line)}[data-testid="stSidebar"] .block-container{padding:.9rem .85rem 1.2rem}.brand{font-size:1rem;font-weight:950;line-height:1.03}.sub{margin-top:.4rem;color:var(--muted);font-size:.68rem}.side-kicker{margin-top:1.15rem;margin-bottom:.45rem;color:#8ea6bd;font-size:.62rem;font-weight:900;letter-spacing:.15em;text-transform:uppercase}.side-row{display:flex;align-items:center;gap:.45rem;padding:.5rem 0;border-bottom:1px solid rgba(145,168,190,.09);font-size:.73rem}.side-right{margin-left:auto;color:#9db1c4;font-size:.6rem;font-weight:850;letter-spacing:.06em}.side-note{color:var(--muted);font-size:.67rem;line-height:1.45}.dot-green{color:var(--green)}.dot-purple{color:var(--purple)}
.hero{padding:clamp(1.25rem,2.5vw,1.9rem);border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(12,37,57,.96),rgba(5,14,24,.98) 74%);box-shadow:0 24px 60px rgba(0,0,0,.22)}.eyebrow{color:#a9bacb;font-size:.63rem;font-weight:900;letter-spacing:.18em}.hero-title{margin-top:.42rem;font-size:clamp(2.05rem,4.4vw,3.8rem);font-weight:950;line-height:.98;letter-spacing:-.06em}.hero-sub{margin-top:.7rem;color:#9db1c4;font-size:clamp(.86rem,1.35vw,1rem);line-height:1.55}.pills{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}.pill{padding:.36rem .7rem;border:1px solid var(--line);border-radius:999px;color:#d2dde7;background:rgba(255,255,255,.025);font-size:.64rem;font-weight:850}.pill.online{color:var(--green);border-color:rgba(85,227,154,.3)}.decision-banner{margin-top:.8rem;padding:.65rem .85rem;border:1px solid var(--line);border-radius:12px;background:rgba(8,19,32,.78);color:#9eb3c7;font-size:.72rem}.decision-banner strong{color:#f4f8fc}.section-title{font-size:.82rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:#b4c3d1;margin-bottom:.65rem}.card,.input-card,.pipeline{border:1px solid var(--line);border-radius:18px;background:linear-gradient(180deg,rgba(12,27,44,.96),rgba(7,17,29,.96));box-shadow:0 12px 28px rgba(0,0,0,.14)}.card{padding:1rem}.service-card{min-height:108px}.service-kicker{color:#9fb2c4;font-size:.6rem;font-weight:900;letter-spacing:.11em}.service-main{margin-top:.55rem;font-size:.95rem;font-weight:950}.green{color:var(--green)}.purple{color:var(--purple)}.service-note{margin-top:.35rem;color:var(--muted);font-size:.69rem;line-height:1.4}.input-card{padding:1rem;background:rgba(8,18,31,.84)}.input-label{font-size:.78rem;font-weight:900}.scenario-label{margin-top:.7rem;color:#91a7bc;font-size:.63rem;font-weight:800}
.pipeline{margin-top:.8rem;padding:.9rem;background:rgba(6,15,26,.9)}.pipeline-head{display:flex;justify-content:space-between;margin-bottom:.75rem}.pipeline-head span{color:#aebfd0;font-size:.63rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.pipeline-head small{color:#637b92;font-size:.61rem}.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.4rem}.stage{min-width:0;padding:.72rem .55rem;border:1px solid var(--line);border-radius:13px;background:linear-gradient(180deg,#0b1929,#07111e);position:relative;transition:.25s}.stage:not(:last-child):after{content:"→";position:absolute;right:-.42rem;top:50%;color:#5d7891;font-size:.8rem}.stage-icon{float:right;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:1px solid #334b60;color:#637b92;font-size:.75rem;font-weight:950}.stage.done{border-color:rgba(85,227,154,.55);background:linear-gradient(180deg,rgba(14,58,42,.72),rgba(7,27,24,.92));box-shadow:0 0 18px rgba(85,227,154,.09)}.stage.done .stage-icon{background:var(--green);border-color:var(--green);color:#062016}.stage.active{border-color:rgba(66,200,238,.65);box-shadow:0 0 20px rgba(66,200,238,.12)}.stage.active .stage-icon{border-color:var(--cyan);color:var(--cyan);animation:pulse 1s infinite}.stage.skipped{opacity:.5}.stage-num{color:var(--cyan);font-size:.6rem;font-weight:950}.stage-name{margin-top:.35rem;font-size:.65rem;font-weight:900;white-space:nowrap}.stage-note{margin-top:.3rem;color:#8299ad;font-size:.57rem;line-height:1.3}@keyframes pulse{50%{box-shadow:0 0 0 6px rgba(66,200,238,0)}}
.authority{height:100%;min-height:305px;padding:1.2rem;border:1px solid rgba(85,227,154,.4);border-radius:20px;background:radial-gradient(circle at 90% 5%,rgba(85,227,154,.1),transparent 35%),linear-gradient(145deg,rgba(15,62,45,.34),rgba(6,18,26,.98))}.authority-label{color:var(--green);font-size:.63rem;font-weight:950;letter-spacing:.13em}.authority-id{margin-top:.6rem;font-size:clamp(2rem,3.4vw,3rem);font-weight:950;line-height:1}.decision-pill{display:inline-block;margin-top:.55rem;padding:.35rem .62rem;border-radius:999px;border:1px solid rgba(85,227,154,.35);color:#c8f5d9;font-size:.6rem;font-weight:950}.decision-pill.uncertain{color:#f5cf82;border-color:rgba(241,189,99,.4)}.decision-pill.new{color:#90dcf2;border-color:rgba(66,200,238,.4)}.authority-score{margin-top:.85rem;color:var(--green);font-size:1.15rem;font-weight:900}.authority-explain{margin-top:.25rem;color:#91a8ba;font-size:.64rem;line-height:1.4}.glance{margin-top:.7rem;padding:.85rem;border:1px solid var(--line);border-radius:14px;background:rgba(7,16,28,.72)}.glance-title{color:#b5c5d4;font-size:.6rem;font-weight:900;letter-spacing:.1em;text-transform:uppercase}.glance-row{display:flex;justify-content:space-between;padding:.36rem 0;border-bottom:1px solid rgba(145,168,190,.08);font-size:.62rem}.glance-row span:first-child{color:#8198ac}.glance-row span:last-child{color:#d9e3eb}.result-section{margin-top:1rem}.result-head{display:flex;justify-content:space-between;margin-bottom:.65rem}.result-head h2{margin:0;font-size:1.35rem}.result-head p{margin:0;color:var(--muted);font-size:.67rem}.evidence-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem}.evidence{padding:.8rem;border:1px solid var(--line);border-radius:14px;background:rgba(8,18,31,.78)}.evidence-label{color:#91a8bd;font-size:.59rem;font-weight:900;letter-spacing:.09em}.evidence-value{margin-top:.4rem;color:#e3ebf2;font-size:.68rem;line-height:1.5;white-space:pre-wrap}.advisory{margin-top:.8rem;padding:1rem;border:1px solid rgba(185,147,255,.22);border-radius:18px;background:linear-gradient(145deg,rgba(42,27,65,.16),rgba(7,15,26,.94))}.advisory-head{font-size:.95rem;font-weight:950}.advisory-head span{color:var(--purple)}.advisory-note{margin-top:.25rem;color:var(--muted);font-size:.66rem}.advisory-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:.7rem}.advisory-card{padding:.75rem;border:1px solid rgba(185,147,255,.2);border-radius:14px}.advisory-label{color:var(--purple);font-size:.6rem;font-weight:900}.footer{margin-top:1.4rem;padding-top:.8rem;border-top:1px solid var(--line);color:#62798f;font-size:.58rem}@media(max-width:1150px){.flow{grid-template-columns:repeat(3,minmax(0,1fr))}.stage:not(:last-child):after{display:none}}@media(max-width:760px){.architecture-grid,.advisory-grid,.evidence-grid{grid-template-columns:1fr}.flow{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:430px){.flow{grid-template-columns:1fr}.stage:not(:last-child):after{display:block;content:"↓";right:50%;top:auto;bottom:-.7rem;transform:translateX(50%)}}
    </style>""",unsafe_allow_html=True)
def _render_sidebar()->bool:
    with st.sidebar:
        st.markdown('<div class="brand">CPSE MATERIAL<br>HARMONIZATION</div><div class="sub">SIH 2026 · Problem Statement 26099</div>',unsafe_allow_html=True)
        st.markdown('<div class="side-kicker">System Status</div><div class="side-row"><span class="dot-green">●</span> SYSTEM ONLINE</div><div class="side-kicker">Analysis Mode</div>',unsafe_allow_html=True)
        current=st.session_state.get("ai_enabled",False);st.radio("Analysis Mode",("Hybrid AI","Deterministic Only"),index=0 if current else 1,key="analysis_mode_selector",label_visibility="collapsed",on_change=lambda:st.session_state.__setitem__("ai_enabled",st.session_state["analysis_mode_selector"]=="Hybrid AI"))
        ai=st.session_state.get("ai_enabled",False)
        st.markdown('<div class="side-kicker">Decision Authority</div><div class="side-row"><span class="dot-green">●</span> Deterministic Engine</div>',unsafe_allow_html=True);st.caption("AUTHORITATIVE · final decision source")
        st.markdown('<div class="side-kicker">AI Services</div><div class="side-row"><span class="dot-purple">●</span> Local NLP <span class="side-right">ADVISORY</span></div><div class="side-row"><span class="dot-purple">●</span> Gemini 2.5 Flash <span class="side-right">LLM · ADVISORY</span></div><div class="side-note">AI advisory can be disabled. Disabled mode does not invoke Local NLP or Gemini.</div>',unsafe_allow_html=True)
        return ai
def _header(ai:bool)->None:
    mode="HYBRID AI" if ai else "DETERMINISTIC ONLY";st.markdown(f'<div class="hero"><div class="eyebrow">CPSE MATERIAL HARMONIZATION</div><div class="hero-title">CPSE Material Harmonization</div><div class="hero-sub">AI-assisted standardization of legacy CPSE material descriptions into a common canonical material vocabulary.</div><div class="pills"><span class="pill">SIH 2026 · PS 26099</span><span class="pill online">● SYSTEM ONLINE</span><span class="pill">MODE · {mode}</span></div></div><div class="decision-banner"><strong>Decision authority:</strong> deterministic matching is always the final decision source. Local NLP and Gemini provide separate advisory intelligence.</div>',unsafe_allow_html=True)
def _architecture()->None:
    for col,(k,m,n,c) in zip(st.columns(3),(("DETERMINISTIC ENGINE","AUTHORITATIVE","Final decision source","green"),("LOCAL NLP","ADVISORY","Optional local embedding retrieval","purple"),("GEMINI 2.5 FLASH","LLM · ADVISORY","Candidate reasoning","purple"))):
        with col:st.markdown(f'<div class="card service-card"><div class="service-kicker">{k}</div><div class="service-main {c}">{m}</div><div class="service-note">{n}</div></div>',unsafe_allow_html=True)
def _input()->str:
    st.markdown('<div class="section-title">Analyze Material</div><div class="input-card"><div class="input-label">Enter a legacy material description</div><div class="scenario-label">Quick scenarios</div>',unsafe_allow_html=True)
    for col,(outcome,example) in zip(st.columns(3),DEMO_EXAMPLES.items()):
        with col:
            if st.button(outcome.title(),use_container_width=True,key="example_"+outcome):st.session_state["material_description"]=example;st.session_state["analysis_result"]=None;st.session_state["pipeline_state"]=None;st.rerun()
    d=st.text_area("Material Description",key="material_description",height=96,placeholder="Gate Valve Carbon Steel 150 50mm Flanged",label_visibility="collapsed");st.markdown('</div>',unsafe_allow_html=True);return d
def _pipeline_html(states:list[str],status:str)->str:
    stages=(("01","INPUT","Legacy description"),("02","NORMALIZE","Canonical text"),("03","EXTRACT","Technical attributes"),("04","MATCH","Deterministic mapping"),("05","AI ADVISE","Local NLP + Gemini"),("06","DECIDE","Authoritative result"));h=f'<div class="pipeline"><div class="pipeline-head"><span>Processing Pipeline</span><small>{escape(status)}</small></div><div class="flow">'
    for i,(num,name,note) in enumerate(stages):
        s=states[i];icon="✓" if s=="done" else "•" if s=="active" else "—" if s=="skipped" else "○";h+=f'<div class="stage {s}"><div class="stage-icon">{icon}</div><div class="stage-num">{num}</div><div class="stage-name">{name}</div><div class="stage-note">{"Skipped · deterministic mode" if s=="skipped" else note}</div></div>'
    return h+'</div></div>'
def _render_pipeline(slot:Any,states:list[str],status:str)->None:slot.markdown(_pipeline_html(states,status),unsafe_allow_html=True)
def _animated(slot:Any,d:str,ai:bool,catalog:tuple[Any,...])->HybridResult:
    states=["pending"]*6
    for i,label in enumerate(("Reading legacy description…","Normalizing technical text…","Extracting material attributes…","Matching deterministic catalog…")):
        states[i]="active";_render_pipeline(slot,states,label);time.sleep(.28);states[i]="done";_render_pipeline(slot,states,label.replace("…"," complete"))
    if ai:
        states[4]="active";_render_pipeline(slot,states,"AI advisory running…");result=analyze_material(d,catalog,True);states[4]="done"
    else:states[4]="skipped";_render_pipeline(slot,states,"AI advisory disabled");result=analyze_material(d,catalog,False)
    states[5]="active";_render_pipeline(slot,states,"Finalizing authoritative decision…");time.sleep(.28);states[5]="done";_render_pipeline(slot,states,"analysis complete");st.session_state["pipeline_state"]=states;return result
def _authority(slot:Any,r:HybridResult|None)->None:
    if r is None:slot.markdown('<div class="authority"><div class="authority-label">AUTHORITATIVE DECISION</div><div class="authority-id">Awaiting analysis</div><div class="authority-explain">Run a material description to produce the deterministic result.</div></div>',unsafe_allow_html=True);return
    m=r.mapping_result;dec=m.decision;cid=m.canonical_material_id or "Not assigned";cls="uncertain" if dec=="UNCERTAIN" else "new" if dec=="NEW_CANDIDATE" else "";slot.markdown(f'<div class="authority"><div class="authority-label">✓ AUTHORITATIVE DECISION</div><div class="authority-id">{escape(str(cid))}</div><span class="decision-pill {cls}">{escape(dec)}</span><div class="authority-score">{m.score:.3f}</div><div class="authority-explain">Deterministic match score<br>{escape(m.explanation or "Final decision produced by deterministic harmonization engine.")}</div></div>',unsafe_allow_html=True)
    a=r.attributes;rows=''.join(f'<div class="glance-row"><span>{k}</span><span>{escape(_display_value(v))}</span></div>' for k,v in (("Category",getattr(a,"category",None)),("Valve Type",getattr(a,"valve_type",None)),("Material",getattr(a,"material",None)),("Size (mm)",getattr(a,"size_mm",None)),("Pressure Class",getattr(a,"pressure_class",None)),("Connection",getattr(a,"connection",None))));slot.markdown(f'<div class="glance"><div class="glance-title">At a glance attributes</div>{rows}</div>',unsafe_allow_html=True)
def _analysis(r:HybridResult)->None:
    st.markdown('<div class="result-section"><div class="result-head"><h2>Evidence &amp; Details</h2><p>Transparent evidence behind the authoritative decision</p></div>',unsafe_allow_html=True);raw=escape(st.session_state.get("material_description",""));norm=escape(r.normalized_description or "Not provided");trans="\n".join("✓ "+x for x in r.normalization_transformations) if r.normalization_transformations else "No transformations reported.";st.markdown(f'<div class="evidence-grid"><div class="evidence"><div class="evidence-label">Original Description</div><div class="evidence-value">{raw}</div></div><div class="evidence"><div class="evidence-label">Normalized Description</div><div class="evidence-value">{norm}</div></div><div class="evidence"><div class="evidence-label">Normalization Transformations</div><div class="evidence-value">{escape(trans)}</div></div><div class="evidence"><div class="evidence-label">Deterministic Explanation</div><div class="evidence-value">{escape(r.mapping_result.explanation or "No explanation reported.")}</div></div></div>',unsafe_allow_html=True)
    with st.expander("Technical Attributes",expanded=False):st.dataframe(attribute_rows(r.attributes),hide_index=True,use_container_width=True)
    with st.expander("Deterministic Candidate Evidence",expanded=False):
        cs=r.mapping_result.all_candidates;st.dataframe(candidate_rows(cs),hide_index=True,use_container_width=True) if cs else st.caption("No candidate evidence is available.")
    local=tuple(x for x in r.ai_candidate_suggestions if x.source=="local_embedding");gem=tuple(x for x in r.ai_candidate_suggestions if x.source=="gemini");st.markdown('<div class="advisory"><div class="advisory-head"><span>AI Advisory</span> · advisory only</div><div class="advisory-note">AI suggestions are never used as the authoritative decision.</div></div>',unsafe_allow_html=True)
    for col,rows in zip(st.columns(2),(local,gem)):
        with col:st.dataframe(advisory_rows(rows),hide_index=True,use_container_width=True) if rows else st.caption("No advisory suggestions available.")

def main()->None:
    if st is None:raise RuntimeError("Streamlit is required to run this dashboard.")
    st.set_page_config(page_title="CPSE Material Harmonization",page_icon="◆",layout="wide",initial_sidebar_state="expanded");_inject_styles();st.session_state.setdefault("ai_enabled",False);st.session_state.setdefault("analysis_result",None);st.session_state.setdefault("analysis_mode",None);st.session_state.setdefault("material_description","");st.session_state.setdefault("pipeline_state",None)
    ai=_render_sidebar();prev=st.session_state.get("analysis_result");oldmode=st.session_state.get("analysis_mode");changed=prev is not None and oldmode is not None and oldmode!=ai;_header(ai);_architecture();left,center,right=st.columns([1.05,1.55,1.05],gap="medium");pipe= center.empty();auth=right.empty()
    with left:
        d=_input();clicked=st.button("Analyze Material  →",type="primary",use_container_width=True,key="analyze_main")
        if clicked:
            if not d or not d.strip():st.session_state["analysis_result"]=None;st.warning("Material description required. Enter a legacy description to begin.")
            else:
                try:st.session_state["analysis_result"]=_animated(pipe,d,ai,load_demo_catalog(CATALOG_PATH));st.session_state["analysis_mode"]=ai
                except (OSError,TypeError,ValueError):st.session_state["analysis_result"]=None;st.error("The description could not be processed safely. Please check the input and try again.")
    result=st.session_state.get("analysis_result");_render_pipeline(pipe,st.session_state.get("pipeline_state") or ["pending"]*6,"analysis complete" if result else "deterministic authority preserved");_authority(auth,result)
    if changed:st.warning("Analysis mode changed. Re-analyze to generate evidence for the current mode.")
    if result:st.divider();_analysis(result)
    else:st.markdown('<div class="footer">CPSE MATERIAL HARMONIZATION · SIH 2026 · PS 26099 · Deterministic Decision Engine · Optional AI Advisory Layer</div>',unsafe_allow_html=True)
if __name__=="__main__":main()
