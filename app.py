"""Judge-facing Streamlit presentation for the deterministic SIH demo.

Run locally with: ``streamlit run app.py``.
All material processing is delegated to ``src.demo_pipeline``.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.attribute_extraction import MaterialAttributes
from src.demo_pipeline import CandidateEvidence, load_demo_catalog, run_demo

try:  # Keep presentation helpers importable when Streamlit is not installed.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - depends on local optional UI package
    st = None


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
    """Format existing attribute values without inventing missing information."""

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
    """Return clean, primitive display rows for extracted technical attributes."""

    return [
        {
            "Attribute": ATTRIBUTE_LABELS[field.name],
            "Extracted value": _display_value(getattr(attributes, field.name)),
        }
        for field in fields(attributes)
    ]


def candidate_rows(candidates: tuple[CandidateEvidence, ...]) -> list[dict[str, str | float]]:
    """Return the already bounded LEGO #7 evidence in a table-ready form."""

    return [
        {
            "Canonical material ID": candidate.canonical_material_id,
            "Decision": candidate.decision,
            "Score": candidate.score,
            "Explanation": candidate.explanation,
        }
        for candidate in candidates
    ]


if st is not None:

    @st.cache_resource(show_spinner=False)
    def load_startup_catalog():
        """Load the immutable demo catalog once per Streamlit process."""

        return load_demo_catalog(CATALOG_PATH)


def _select_example(example: str) -> None:
    """Populate the main input with a selected judge-facing example."""

    if st is not None:
        st.session_state["material_description"] = example


def _render_outcome(decision: str, canonical_material_id: str | None, score: float) -> None:
    """Render the three existing mapping outcomes without changing their meaning."""

    message = f"Decision: {decision}"
    if decision == "MATCHED":
        st.success(message)
    elif decision == "UNCERTAIN":
        st.warning(message)
    else:
        st.error(message)

    first, second = st.columns(2)
    first.metric("Similarity score", f"{score:.3f}")
    second.metric("Canonical material ID", canonical_material_id or "Not assigned")


def main() -> None:
    """Render the thin Streamlit presentation layer."""

    if st is None:  # pragma: no cover - only reached when launching without Streamlit
        raise RuntimeError("Streamlit is required to run this demo. Install Streamlit, then run: streamlit run app.py")

    st.set_page_config(page_title="CPSE Material Harmonization", page_icon="🔗", layout="wide")
    st.title("AI-Driven CPSE Material Code Harmonization")
    st.caption("Offline, deterministic material standardization for the SIH demonstration.")

    try:
        catalog = load_startup_catalog()
    except (OSError, ValueError) as error:
        st.error("Demo startup error: the local reference catalog could not be loaded.")
        st.caption(str(error))
        st.stop()

    st.subheader("Decision pipeline")
    st.markdown("Raw Description  \\n+↓  \\n+Normalization  \\n+↓  \\n+Attribute Extraction  \\n+↓  \\n+Catalog Matching  \\n+↓  \\n+**MATCHED / UNCERTAIN / NEW CANDIDATE**")

    st.subheader("Try a material description")
    example_columns = st.columns(3)
    for column, (outcome, example) in zip(example_columns, DEMO_EXAMPLES.items()):
        column.button(outcome.title(), on_click=_select_example, args=(example,), use_container_width=True)

    description = st.text_area(
        "Raw material description",
        key="material_description",
        height=150,
        placeholder="Example: CS GATE VLV 50MM FLG CL150",
        help="Enter a legacy CPSE-style material description, or select a demo example above.",
    )
    submitted = st.button("Analyze description", type="primary")

    if submitted:
        try:
            result = run_demo(description, catalog)
        except (TypeError, ValueError) as error:
            st.error("The description could not be processed safely.")
            st.caption(str(error))
            st.stop()

        st.divider()
        st.subheader("Analysis result")
        _render_outcome(
            result.mapping_result.decision,
            result.mapping_result.canonical_material_id,
            result.mapping_result.score,
        )
        st.info(result.mapping_result.explanation)

        left, right = st.columns(2)
        with left:
            st.markdown("**Raw description**")
            st.code(result.original_raw_description or "No description provided", language=None)
            st.markdown("**Normalized description**")
            st.code(result.normalized_description or "No normalized description", language=None)
        with right:
            st.markdown("**Transformations applied**")
            for transformation in result.normalization_transformations:
                st.write(f"• {transformation}")

        st.markdown("**Extracted technical attributes**")
        st.dataframe(attribute_rows(result.attributes), hide_index=True, use_container_width=True)

        st.markdown("**Candidate evidence**")
        if result.candidate_evidence:
            st.dataframe(candidate_rows(result.candidate_evidence), hide_index=True, use_container_width=True)
            with st.expander("View candidate audit notes"):
                for candidate in result.candidate_evidence:
                    st.markdown(f"**{candidate.canonical_material_id} — {candidate.decision} ({candidate.score:.3f})**")
                    st.write(candidate.explanation)
        else:
            st.caption("No candidate evidence is available for an empty or missing description.")

    st.divider()
    st.subheader("How the system decides")
    st.markdown(
        "- It compares explicit technical attributes, such as material, size, pressure class, dimensions, and connection.\n"
        "- Missing information stays missing; the system does not invent technical values.\n"
        "- Conflicting technical attributes prevent an unsafe match.\n"
        "- Incomplete or ambiguous matches are marked **UNCERTAIN** for review."
    )
    st.subheader("System characteristics")
    st.markdown("Deterministic • Offline • Explainable • No LLM required • No external API required")


if __name__ == "__main__":
    main()
