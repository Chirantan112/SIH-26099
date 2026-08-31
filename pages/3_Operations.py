"""Operational SIH-26099 demo page.

This page is additive: the existing matcher and LLM adapters are not modified.
Batch mode is deterministic-only; AI is opt-in for single-record evidence review.
"""
from __future__ import annotations

import csv
import io
import sys
from dataclasses import fields
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_pipeline import load_demo_catalog, run_demo
from src.governance import create_audit_trail
from src.hybrid_pipeline import run_hybrid_pipeline
from src.operations import process_batch, review_item

CATALOG_PATH = ROOT / "data" / "demo" / "material_master.csv"


@st.cache_resource
def get_catalog():
    return load_demo_catalog(CATALOG_PATH)


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=("legacy_material_code", "description", "decision", "material_id", "score", "explanation"))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _result_row(result) -> dict[str, object]:
    mapping = result.mapping_result
    return {
        "legacy_material_code": result.legacy_material_code,
        "description": result.original_raw_description or "",
        "decision": mapping.decision,
        "material_id": mapping.canonical_material_id or "",
        "score": mapping.score,
        "explanation": mapping.explanation,
    }


def _record_audit(result, raw_description: str | None) -> None:
    trail = create_audit_trail(st.session_state.get("audit_trail"))
    st.session_state.audit_trail = trail
    mapping = result.mapping_result
    trail.record_analysis(
        legacy_material_code=result.legacy_material_code,
        input_description=raw_description or "",
        deterministic_decision=mapping.decision,
        deterministic_material_id=mapping.canonical_material_id,
        ai_candidates=tuple(
            candidate.canonical_material_id for candidate in getattr(result, "ai_candidate_suggestions", ())
        ),
    )


st.set_page_config(page_title="Operations | CPSE Harmonization", page_icon="⚙️", layout="wide")
st.title("Operations & Evidence")
st.caption("Batch operations, review queue, audit trail, and judge-facing evidence. Existing decision and LLM modules remain unchanged.")

st.markdown(
    """
<style>
[data-testid="stCaptionContainer"] p,[data-testid="stWidgetLabel"] p,[data-testid="stMarkdownContainer"] p{font-size:14px!important;line-height:1.5}
[data-testid="stAlert"] p{font-size:14px!important;line-height:1.5}
/* Readability weight update: medium body text for dashboard and pages. */
.sub,.top-sub,.hero-sub,.side-row,.input-note,.scenario-label,.stage-note,.decision-id,.decision-note,.reason-box span,.score-meter-head,.attribute-card strong,.advisory-note,.advisory-summary > span:not(.status-chip),.detail-note,.verify-copy,.impact-node span,.table-empty,.evidence-value,.review-note,.pipeline-head small{font-weight:600!important}
.stage-name{font-weight:700!important}
.data-table td{font-weight:600!important}
</style>
""",
    unsafe_allow_html=True,
)

catalog = get_catalog()

single_tab, batch_tab, review_tab, benchmark_tab = st.tabs(["Single Evidence", "Batch Processing", "Review Queue", "Benchmark"])

with single_tab:
    st.subheader("Explainable decision")
    col1, col2 = st.columns([3, 1])
    with col1:
        description = st.text_input("Legacy material description", value="CS GATE VLV 50MM FLG CL150")
    with col2:
        ai_enabled = st.toggle("AI advisory", value=False, help="Opt-in only. Uses the existing Local NLP and Gemini adapters; AI never changes the deterministic final decision.")

    if st.button("Analyze", type="primary", use_container_width=True):
        if ai_enabled:
            retrieval = None
            llm = None
            try:
                from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter
                retrieval = LocalEmbeddingRetrievalAdapter()
            except Exception:
                pass
            try:
                from src.gemini_llm import GeminiLLMAdapter
                llm = GeminiLLMAdapter()
            except Exception:
                pass
            result = run_hybrid_pipeline(description, catalog, legacy_material_code="OPS-INPUT", retrieval_adapter=retrieval, llm_adapter=llm)
        else:
            result = run_demo(description, catalog, legacy_material_code="OPS-INPUT")
        st.session_state.last_result = result
        _record_audit(result, description)

    result = st.session_state.get("last_result")
    if result:
        mapping = result.mapping_result
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown(f"### **{mapping.decision}**")
            st.metric("Authoritative material", mapping.canonical_material_id or "Human review required")
            st.write(mapping.explanation)
            st.markdown("#### Why this decision?")
            attrs = result.attributes
            for field in fields(attrs):
                value = getattr(attrs, field.name)
                if value is not None:
                    st.write(f"**{field.name.replace('_', ' ').title()}:** {value}")
        with right:
            st.markdown("#### Deterministic evidence")
            for candidate in mapping.all_candidates[:5]:
                st.write(f"`{candidate.canonical_material_id}` · {candidate.decision} · {candidate.score:.3f}")
                st.caption(candidate.explanation)
            if hasattr(result, "ai_statuses"):
                st.markdown("#### AI advisory — never authoritative")
                for status in result.ai_statuses:
                    st.write(f"**{status.component}:** {'Available' if status.available else 'Unavailable'}")
                for candidate in result.ai_candidate_suggestions:
                    st.write(f"`{candidate.canonical_material_id}` · {candidate.source} · {candidate.score:.3f}")
                    st.caption(candidate.explanation)
        st.info("Decision boundary: deterministic mapping is authoritative. AI output is advisory only and cannot override it.")

with batch_tab:
    st.subheader("Batch processing")
    st.caption("Batch mode is deliberately deterministic-only: it never constructs or calls Local NLP or Gemini, avoiding unexpected API/model traffic.")
    uploaded = st.file_uploader("Upload CSV with a description column", type=["csv"])
    pasted = st.text_area("Or paste one description per line", height=120)
    descriptions: list[str] = []
    if uploaded:
        text = uploaded.getvalue().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        fields_available = reader.fieldnames or []
        description_field = st.selectbox("Description column", fields_available, index=0 if fields_available else None)
        if description_field:
            descriptions = [row.get(description_field, "").strip() for row in reader if row.get(description_field, "").strip()]
    elif pasted.strip():
        descriptions = [line.strip() for line in pasted.splitlines() if line.strip()]

    if descriptions:
        st.write(f"Ready: **{len(descriptions):,}** descriptions")
    if st.button("Process batch", type="primary", disabled=not descriptions):
        batch = process_batch(descriptions, catalog)
        st.session_state.batch_result = batch
        st.session_state.review_items = [review_item(row) for row in batch.rows if row.mapping_result.decision != "MATCHED"]

    batch = st.session_state.get("batch_result")
    if batch:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", batch.total)
        m2.metric("Matched", batch.matched)
        m3.metric("Uncertain", batch.uncertain)
        m4.metric("New candidate", batch.new_candidate)
        st.caption(f"{batch.elapsed_seconds:.3f}s · {batch.throughput_per_second:.1f} descriptions/s · {batch.review_required} require review")
        rows = [_result_row(row) for row in batch.rows]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("Export results", _csv_bytes(rows), "harmonization_results.csv", "text/csv")

with review_tab:
    st.subheader("Human review queue")
    items = st.session_state.get("review_items", [])
    if not items:
        st.success("No review items in this session. Run a batch containing UNCERTAIN or NEW_CANDIDATE cases.")
    else:
        st.write(f"**{len(items)}** items awaiting review")
        index = st.number_input("Review item", min_value=1, max_value=len(items), value=1, step=1) - 1
        item = items[index]
        st.code(item.input_description)
        st.warning(f"{item.decision}: {item.reason}")
        candidate = st.text_input("Selected canonical material ID", value=item.material_id or "")
        action = st.selectbox("Reviewer action", ["APPROVE", "REJECT", "REVIEW"])
        note = st.text_input("Reviewer note")
        if st.button("Record review", type="primary"):
            trail = create_audit_trail(st.session_state.get("audit_trail"))
            st.session_state.audit_trail = trail
            trail.record_review(
                legacy_material_code=f"REVIEW-{index + 1:04d}",
                input_description=item.input_description,
                deterministic_decision=item.decision,
                deterministic_material_id=item.material_id,
                candidate_material_id=candidate or None,
                action=action,
                note=note,
            )
            st.success("Review action recorded in the session audit trail.")

    trail = st.session_state.get("audit_trail")
    if trail and trail.events:
        st.markdown("#### Audit trail")
        st.dataframe([event.to_dict() for event in trail.events], use_container_width=True, hide_index=True)

with benchmark_tab:
    st.subheader("Reproducible benchmark")
    st.caption("Synthetic CPSE-style data only. This benchmark does not call Gemini or Local NLP and is not a claim of production CPSE accuracy.")
    variants = st.number_input("Variants per material", min_value=1, max_value=200, value=40, step=1)
    if st.button("Run benchmark", type="primary"):
        from scripts.benchmark import build_cases, evaluate_mapping, evaluate_pairs
        cases = build_cases(catalog, int(variants))
        mapping = evaluate_mapping(cases, catalog)
        pairs = evaluate_pairs(catalog)
        st.session_state.benchmark_result = (mapping, pairs)

    benchmark = st.session_state.get("benchmark_result")
    if benchmark:
        mapping, pairs = benchmark
        a, b, c, d = st.columns(4)
        a.metric("Exact mapping", f"{mapping['exact_mapping_rate']:.2%}")
        b.metric("Uncertain", f"{mapping['uncertain_rate']:.2%}")
        c.metric("Wrong", f"{mapping['wrong_rate']:.2%}")
        d.metric("Throughput", f"{mapping['throughput_per_second']:.1f}/s")
        st.write({"Generated descriptions": mapping["cases"], "Runtime (s)": round(mapping["seconds"], 3), "Distinct pairs": pairs["pairs"], "Pair false-positive rate": pairs["false_positive_rate"]})
        st.info("Use these measurements as regression/stress evidence for the repository's synthetic dataset. Replace with approved CPSE data before making production-accuracy claims.")
