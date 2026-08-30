"""Judge-facing synthetic cross-CPSE harmonization overview.

Presentation-only page: it consumes the repository-owned demo CSV and the
existing deterministic catalog model. It does not change matching behavior.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "demo" / "material_master.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_overview(rows: list[dict[str, str]]) -> dict[str, Any]:
    canonical_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        canonical_groups[row["ground_truth_material_id"]].append(row)
    return {
        "records": len(rows),
        "canonical_materials": len(canonical_groups),
        "cpse_counts": dict(Counter(row["cpse"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "groups": canonical_groups,
    }


st.set_page_config(page_title="SIH 26099 · Harmonization", page_icon="◆", layout="wide")

st.markdown(
    """
<style>
.stApp{background:#050c15;color:#eef6fb}
.block-container{max-width:1500px;padding:2.5rem 1.2rem 2rem}
.hero{padding:1.35rem 1.5rem;border:1px solid #25465e;border-radius:20px;background:linear-gradient(135deg,#102d43,#07131f 72%)}
.eyebrow{color:#36c6ee;font-size:.62rem;font-weight:900;letter-spacing:.16em}.title{margin-top:.35rem;font-size:clamp(1.8rem,3.8vw,3.1rem);font-weight:950;letter-spacing:-.055em}.copy{max-width:900px;margin-top:.55rem;color:#9db2c0;font-size:.76rem;line-height:1.55}.pill{display:inline-block;margin-top:.75rem;margin-right:.35rem;padding:.3rem .52rem;border:1px solid #315069;border-radius:999px;color:#cfe0e9;font-size:.5rem;font-weight:900}.safe{color:#4ee39a;border-color:#4ee39a55}.section{margin:1rem 0 .55rem;color:#9eb5c4;font-size:.58rem;font-weight:950;letter-spacing:.14em;text-transform:uppercase}.card{padding:.9rem;border:1px solid #203b53;border-radius:15px;background:#091725}.metric{padding:.8rem;border:1px solid #203b53;border-radius:13px;background:#08131f}.label{color:#7994a8;font-size:.48rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.value{margin-top:.2rem;font-size:1.5rem;font-weight:950}.node{padding:.65rem;border:1px solid #315069;border-radius:12px;background:#07131f;text-align:center}.node strong{display:block;font-size:.62rem}.node span{display:block;margin-top:.18rem;color:#7892a5;font-size:.48rem}.arrow{text-align:center;color:#36c6ee;font-size:1.2rem;font-weight:900}.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:.4rem}.canonical{padding:1rem;border:1px solid #4ee39a55;border-radius:15px;background:linear-gradient(145deg,#0d3027,#07131e 72%)}.canonical-id{color:#4ee39a;font-size:.52rem;font-weight:950;letter-spacing:.12em}.canonical-title{margin-top:.25rem;font-size:1.1rem;font-weight:950}.attrs{margin-top:.45rem;color:#b6c7d2;font-size:.57rem;line-height:1.6}.record{padding:.65rem .75rem;margin-top:.4rem;border:1px solid #203b53;border-radius:10px;background:#07131f}.record-top{display:flex;justify-content:space-between;gap:.5rem}.record-code{font-size:.55rem;font-weight:900}.record-cpse{color:#36c6ee;font-size:.5rem;font-weight:900}.record-desc{margin-top:.2rem;color:#9db2c0;font-size:.54rem;line-height:1.4}
</style>
""",
    unsafe_allow_html=True,
)

try:
    rows = load_rows(DATA_PATH)
except (OSError, csv.Error) as error:
    st.error(f"Demo dataset could not be loaded: {error}")
    st.stop()

summary = build_overview(rows)

st.markdown(
    '<div class="hero"><div class="eyebrow">SIH 2026 · PS 26099 · HARMONIZATION VIEW</div><div class="title">One material. Many legacy codes.</div><div class="copy">A judge-facing view of how the repository-owned synthetic records from different CPSE-style sources converge on a shared canonical material identity. This page visualizes the demo dataset only; it does not claim production CPSE data.</div><span class="pill safe">DETERMINISTIC ENGINE · AUTHORITATIVE</span><span class="pill">SYNTHETIC DEMO DATA</span><span class="pill">NO LIVE CPSE CONNECTOR</span></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section">Demo dataset at a glance</div>', unsafe_allow_html=True)
metric_cols = st.columns(4)
for col, label, value in zip(
    metric_cols,
    ("Legacy records", "Canonical materials", "CPSE-style sources", "Categories"),
    (summary["records"], summary["canonical_materials"], len(summary["cpse_counts"]), len(summary["category_counts"])),
):
    with col:
        st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Cross-CPSE harmonization concept</div>', unsafe_allow_html=True)
flow_cols = st.columns([1, .12, 1, .12, 1])
for col, content in zip(
    flow_cols,
    (
        '<div class="node"><strong>CPCL</strong><span>Legacy material descriptions</span></div>',
        '<div class="arrow">→</div>',
        '<div class="node"><strong>Canonical identity</strong><span>Normalized technical attributes</span></div>',
        '<div class="arrow">→</div>',
        '<div class="node"><strong>IOCL / NTPC</strong><span>Equivalent legacy records</span></div>',
    ),
):
    with col:
        st.markdown(content, unsafe_allow_html=True)

st.markdown('<div class="section">Canonical material explorer</div>', unsafe_allow_html=True)
canonical_ids = sorted(summary["groups"])
selected = st.selectbox("Canonical material", canonical_ids, label_visibility="collapsed")
group = summary["groups"][selected]
first = group[0]

st.markdown(
    f'<div class="canonical"><div class="canonical-id">CANONICAL MATERIAL</div><div class="canonical-title">{selected}</div><div class="attrs"><b>{first.get("category", "—")}</b> · {first.get("valve_type") or first.get("bearing_family") or "Material specification"}<br>Material: {first.get("material") or first.get("pipe_material") or "—"} · Size / OD: {first.get("size_mm") or first.get("pipe_od_mm") or "—"} mm<br>Pressure / Schedule: {first.get("pressure_class") or first.get("pipe_schedule") or "—"} · Connection / End: {first.get("end_connection") or first.get("pipe_end") or "—"}</div></div>',
    unsafe_allow_html=True,
)

st.markdown(f'<div class="section">Equivalent legacy records · {len(group)}</div>', unsafe_allow_html=True)
for row in group:
    st.markdown(
        f'<div class="record"><div class="record-top"><span class="record-code">{row["legacy_material_code"]}</span><span class="record-cpse">{row["cpse"]}</span></div><div class="record-desc">{row["raw_description"]}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section">Source distribution</div>', unsafe_allow_html=True)
source_cols = st.columns(max(1, len(summary["cpse_counts"])))
for col, (cpse, count) in zip(source_cols, sorted(summary["cpse_counts"].items())):
    with col:
        st.metric(cpse, count)

st.caption("Evidence boundary: this page visualizes the repository-owned synthetic development dataset. It is intended to demonstrate the harmonization workflow, not to represent real CPSE procurement statistics.")
