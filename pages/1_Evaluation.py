"""Judge-facing evaluation dashboard for SIH-26099.

Presentation-only page: it measures the existing harmonization engine and does not
change normalization, extraction, linkage, mapping, AI, or governance behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robust_evaluation import (  # noqa: E402
    EvaluationPair,
    build_evaluation_pairs,
    build_retrieval_cases,
    evaluate_pairs,
    load_legacy_descriptions,
    load_legacy_records,
    retrieval_recall_at_k,
)
from src.catalog_mapping import CatalogRecord  # noqa: E402
from src.demo_pipeline import load_demo_catalog  # noqa: E402
from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter  # noqa: E402
from src.record_linkage import compare_records  # noqa: E402

DATA_PATH = ROOT / "data" / "demo" / "material_master.csv"

st.set_page_config(
    page_title="SIH 26099 · Evaluation",
    page_icon="◆",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp { background: #050c15; }
.block-container { max-width: 1500px; padding-top: 1.4rem; }
.eval-hero { padding: 1.35rem 1.5rem; border: 1px solid #25465e; border-radius: 20px;
  background: linear-gradient(135deg,#102d43 0%,#07131f 72%); }
.eyebrow { color:#36c6ee; font-size:.62rem; font-weight:900; letter-spacing:.16em; }
.hero-title { margin-top:.35rem; font-size:clamp(1.8rem,3.5vw,3rem); font-weight:950; letter-spacing:-.05em; }
.hero-copy { max-width:900px; margin-top:.55rem; color:#9db2c0; font-size:.78rem; line-height:1.55; }
.badge { display:inline-block; margin-top:.8rem; margin-right:.35rem; padding:.32rem .55rem;
  border:1px solid #315069; border-radius:999px; color:#cfe0e9; font-size:.52rem; font-weight:900; }
.badge.safe { color:#4ee39a; border-color:#4ee39a55; }
.section { margin:1rem 0 .55rem; color:#9eb5c4; font-size:.58rem; font-weight:950; letter-spacing:.14em; text-transform:uppercase; }
.metric-card { min-height:100px; padding:.8rem .9rem; border:1px solid #203b53; border-radius:14px; background:#091725; }
.metric-label { color:#7994a8; font-size:.52rem; font-weight:900; letter-spacing:.09em; text-transform:uppercase; }
.metric-value { margin-top:.22rem; font-size:1.45rem; font-weight:950; }
.metric-note { margin-top:.18rem; color:#6f8799; font-size:.48rem; }
.callout { padding:.8rem .9rem; border:1px solid #36c6ee40; border-radius:14px; background:#36c6ee08; color:#a9c3d0; font-size:.64rem; line-height:1.5; }
.callout strong { color:#eef6fb; }
.safe-card { padding:.85rem .9rem; border:1px solid #4ee39a40; border-radius:14px; background:#4ee39a08; }
.danger-card { padding:.85rem .9rem; border:1px solid #ff6f7940; border-radius:14px; background:#ff6f7908; }
.case-card { padding:.75rem; border:1px solid #203b53; border-radius:12px; background:#08131f; margin-bottom:.45rem; }
.case-label { color:#7892a5; font-size:.48rem; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }
.case-value { margin-top:.25rem; color:#dce8ee; font-size:.63rem; line-height:1.45; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="eval-hero">
  <div class="eyebrow">SIH 2026 · PS 26099 · EVIDENCE CENTER</div>
  <div class="hero-title">Harmonization Evaluation</div>
  <div class="hero-copy">
    Judge-facing evidence for the existing material harmonization engine. The page
    stress-tests difficult technical pairs, reports unresolved cases separately,
    and optionally measures Local NLP candidate retrieval.
  </div>
  <span class="badge safe">DETERMINISTIC ENGINE · AUTHORITATIVE</span>
  <span class="badge">AI / NLP · ADVISORY ONLY</span>
  <span class="badge">DATA · SYNTHETIC DEMO, CLEARLY LABELLED</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section">Benchmark controls</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1, 1, 1.2])
with col_a:
    max_hard_negatives = st.number_input("Hard negatives", min_value=1, max_value=500, value=100, step=25)
with col_b:
    run_retrieval = st.checkbox("Run Local NLP Recall@K", value=False)
with col_c:
    st.caption("The benchmark reads repository-owned synthetic material data. No production CPSE data is claimed.")

@st.cache_data(show_spinner=False)
def run_deterministic_evaluation(limit: int) -> tuple[object, tuple[EvaluationPair, ...], tuple[tuple[str, object, str], ...]]:
    records = load_legacy_records(DATA_PATH)
    pairs = build_evaluation_pairs(records, max_hard_negatives=limit)
    result = evaluate_pairs(pairs, records)
    return result, pairs, records

if st.button("Run evaluation", type="primary", use_container_width=True):
    st.session_state["judge_eval_run"] = True

if not st.session_state.get("judge_eval_run", False):
    st.info("Run the benchmark to populate the judge-facing evidence.")
    st.stop()

with st.spinner("Running deterministic robustness evaluation…"):
    result, pairs, records = run_deterministic_evaluation(int(max_hard_negatives))

st.markdown('<div class="section">Core decision performance</div>', unsafe_allow_html=True)
metric_values = (
    ("F1 score", f"{result.f1:.1%}", "SAME/Different resolved decisions"),
    ("Precision", f"{result.precision:.1%}", "Resolved SAME precision"),
    ("Recall", f"{result.recall:.1%}", "Resolved SAME recall"),
    ("Resolved accuracy", f"{result.resolved_accuracy:.1%}", "Excludes UNCERTAIN from denominator"),
    ("UNCERTAIN rate", f"{result.uncertain_rate:.1%}", "Surfaced for human review"),
    ("Hard-negative rejection", f"{result.hard_negative_rejections / result.hard_negative_count:.1%}" if result.hard_negative_count else "—", "Technically similar pairs rejected as DIFFERENT"),
)
metric_cols = st.columns(3)
for index, (label, value, note) in enumerate(metric_values):
    if index and index % 3 == 0:
        metric_cols = st.columns(3)
    with metric_cols[index % 3]:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Safety boundary</div>', unsafe_allow_html=True)
safe_col, review_col = st.columns(2)
with safe_col:
    st.markdown(
        f'<div class="safe-card"><strong>✓ Hard negatives correctly rejected</strong><br><span>{result.hard_negative_rejections} of {result.hard_negative_count} selected hard negatives were classified DIFFERENT.</span></div>',
        unsafe_allow_html=True,
    )
with review_col:
    st.markdown(
        f'<div class="callout"><strong>UNCERTAIN is not silently forced into a match.</strong><br>{result.uncertain} cases were unresolved and remain visible as a human-review workload.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section">Decision mix</div>', unsafe_allow_html=True)
mix_cols = st.columns(4)
for col, label, value in zip(mix_cols, ("Evaluation pairs", "Expected SAME", "Expected DIFFERENT", "UNCERTAIN"), (result.pairs, result.same_expected, result.different_expected, result.uncertain)):
    with col:
        st.metric(label, value)

if run_retrieval:
    st.markdown('<div class="section">Local NLP retrieval</div>', unsafe_allow_html=True)
    with st.spinner("Running Local NLP retrieval evaluation…"):
        descriptions = load_legacy_descriptions(DATA_PATH)
        catalog = load_demo_catalog(DATA_PATH)
        cases = build_retrieval_cases(records, descriptions)
        adapter = LocalEmbeddingRetrievalAdapter()
        status = adapter.status()
    if not status.available:
        st.warning(f"Local NLP is unavailable: {status.detail}")
    else:
        with st.spinner("Calculating Recall@1 / @3 / @5…"):
            recall = {k: retrieval_recall_at_k(cases, catalog, adapter, k) for k in (1, 3, 5)}
        recall_cols = st.columns(3)
        for col, k in zip(recall_cols, (1, 3, 5)):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Recall@{k}</div><div class="metric-value">{recall[k]:.1%}</div><div class="metric-note">Correct canonical ID appears in the top {k} candidates.</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="callout"><strong>Local NLP is optional.</strong><br>Enable the checkbox above when the embedding dependency/model is available. Retrieval metrics are advisory and do not alter the authoritative decision.</div>', unsafe_allow_html=True)

st.markdown('<div class="section">Difficult-case explorer</div>', unsafe_allow_html=True)
by_code = {code: (attributes, ground_truth) for code, attributes, ground_truth in records}
hard_cases = [pair for pair in pairs if pair.hardness == "hard-negative"]
for index, pair in enumerate(hard_cases[:12], start=1):
    left_attr, left_gt = by_code[pair.left_code]
    right_attr, right_gt = by_code[pair.right_code]
    comparison = compare_records(left_attr, right_attr)
    with st.expander(f"Case {index:02d} · {pair.category} · expected DIFFERENT"):
        left, right = st.columns(2)
        with left:
            st.markdown(f'<div class="case-card"><div class="case-label">{pair.left_code} · canonical {left_gt}</div><div class="case-value">{left_attr}</div></div>', unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="case-card"><div class="case-label">{pair.right_code} · canonical {right_gt}</div><div class="case-value">{right_attr}</div></div>', unsafe_allow_html=True)
        if comparison.decision == "DIFFERENT":
            st.success("Correctly rejected as DIFFERENT.")
        elif comparison.decision == "UNCERTAIN":
            st.warning("UNCERTAIN — safe escalation rather than a forced match.")
        else:
            st.error("Potential false SAME — requires investigation.")
        if comparison.conflicting_fields:
            st.caption("Conflicting attributes: " + ", ".join(comparison.conflicting_fields))

st.markdown('<div class="section">What this benchmark proves</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="callout"><strong>Evidence boundary:</strong> these metrics measure the repository-owned synthetic CPSE-style benchmark. They demonstrate deterministic behavior and retrieval coverage, but they are not claims of production CPSE accuracy. Production validation still requires representative CPSE material-master data.</div>',
    unsafe_allow_html=True,
)
