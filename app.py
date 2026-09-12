import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from textwrap import dedent

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from model_service import (
    analyze_patient,
    load_model_artifacts,
)

from pdf_parser import (
    parse_patient_pdf,
)

from evidence_pipeline import (
    build_evidence_from_model_result,
)

from claude_client import (
    build_patient_explanation,
)

@st.cache_data
def load_cached_demo_report():

    demo_path = (
        PROJECT_ROOT
        / "outputs"
        / "patient_2561_explanation.json"
    )

    if not demo_path.exists():
        raise FileNotFoundError(
            f"Demo JSON not found: {demo_path}"
        )

    with open(
        demo_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)
# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroDecel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .hero {
        padding: 3rem 2rem;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(98, 79, 255, 0.12),
                rgba(40, 180, 170, 0.08)
            );
        margin-bottom: 2rem;
        text-align: center;
    }

    .hero h1 {
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.15rem;
        opacity: 0.8;
        max-width: 760px;
        margin: auto;
    }

    .section-title {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .age-card {
        padding: 1.5rem;
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 18px;
        text-align: center;
    }

    .big-age {
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }

    .subtle {
        opacity: 0.65;
        font-size: 0.92rem;
    }

    .younger {
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(50, 180, 120, 0.12);
        display: inline-block;
    }

    .older {
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(230, 140, 60, 0.12);
        display: inline-block;
    }

    .disclaimer {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(120,120,120,0.08);
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Dataset helpers for demo
# --------------------------------------------------

META_COLS = [
    "AIRC_ID",
    "ConstructName",
    "Wave",
    "HasData",
    "NumScores",
]

FLAG_COLS = [
    "Global",
    "Thickness",
    "Volume",
    "Area",
    "SubVolumes",
]


@st.cache_data
def load_demo_patient(
    patient_id=2561
):

    data_path = (
        PROJECT_ROOT
        / "dataset"
        / "BIOMARKERDATA.xlsx"
    )


    def load_sheet(sheet_name):

        df = pd.read_excel(
            data_path,
            sheet_name=sheet_name
        )

        drop_cols = [
            col
            for col in META_COLS + FLAG_COLS
            if col in df.columns
        ]

        return df.drop(
            columns=drop_cols
        )


    age_df = pd.read_excel(
        data_path,
        sheet_name="Age-W1,W2,W3"
    )

    age_df = age_df[
        ["S#", "AgeMRI_W1"]
    ].dropna(
        subset=["AgeMRI_W1"]
    )


    sheets = [
        "GlobalVariables-W1",
        "CorticalThickness-W1",
        "GMVolume-W1",
        "SurfaceArea-W1",
        "SubcorticalVolume-W1",
    ]


    merged = age_df.copy()


    for sheet in sheets:

        merged = merged.merge(
            load_sheet(sheet),
            on="S#",
            how="inner"
        )


    patient = merged[
        pd.to_numeric(
            merged["S#"],
            errors="coerce"
        ) == patient_id
    ]


    if patient.empty:

        raise ValueError(
            f"Demo patient "
            f"{patient_id} not found."
        )


    patient = patient.iloc[0]


    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]


    feature_values = {
        feature: float(
            patient[feature]
        )
        for feature in feature_cols
    }


    return {
        "patient_id":
            patient_id,

        "actual_age":
            float(
                patient["AgeMRI_W1"]
            ),

        "features":
            feature_values,
    }


# --------------------------------------------------
# Run complete pipeline
# --------------------------------------------------

def run_complete_pipeline(
    features,
    actual_age,
    patient_id=None,
):
    """
    MRI features
        ↓
    XGBoost
        ↓
    SHAP
        ↓
    Amass
        ↓
    Claude
    """

    with st.status(
        "Analyzing brain-age profile...",
        expanded=True,
    ) as status:

        st.write(
            "🧠 Running brain-age model..."
        )

        model_result = analyze_patient(
            feature_values=features,
            actual_age=actual_age,
            patient_id=patient_id,
            top_n=5,
        )


        st.write(
            "🔍 Identifying the five "
            "strongest contributing features..."
        )


        st.write(
            "📚 Searching scientific evidence "
            "with Amass..."
        )

        evidence_result = (
            build_evidence_from_model_result(
                model_result
            )
        )


        st.write(
            "✨ Translating the evidence "
            "into plain language with Claude..."
        )

        final_report = (
            build_patient_explanation(
                evidence_result
            )
        )


        status.update(
            label="Analysis complete",
            state="complete",
            expanded=False,
        )


    return final_report


# --------------------------------------------------
# Render report
# --------------------------------------------------

def render_report(report):

    patient = report["patient"]
    summary = report["summary"]


    st.divider()


    st.markdown(
        "## Your Brain-Age Report"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Chronological age",
            f'{patient["actual_age"]:.0f} years'
        )


    with col2:

        st.metric(
            "Estimated brain age",
            f'{patient["predicted_age"]:.1f} years'
        )


    with col3:

        gap = patient[
            "brain_age_gap"
        ]

        st.metric(
            "Brain-age gap",
            (
                f"{abs(gap):.1f} years "
                f"{'younger' if gap < 0 else 'older'}"
            )
        )


    st.markdown(
        f"### {summary['headline']}"
    )

    st.write(
        summary["overview"]
    )


    st.info(
        "💡 " +
        summary["key_takeaway"]
    )


    # --------------------------------------------------
    # Factors
    # --------------------------------------------------

    st.markdown(
        "## What influenced the prediction?"
    )

    st.caption(
        "These are the five MRI-derived measurements "
        "that had the strongest influence on this "
        "individual prediction."
    )


    for index, factor in enumerate(
        report["factors"],
        start=1
    ):

        direction = factor[
            "direction"
        ]

        arrow = (
            "↓"
            if direction == "younger"
            else "↑"
        )

        label = (
            "Younger-associated"
            if direction == "younger"
            else "Older-associated"
        )


        with st.expander(
            f"{index}. {arrow} "
            f"{factor['feature']} — "
            f"{label}"
        ):

            st.markdown(
                "**How it affected the model**"
            )

            st.write(
                factor[
                    "model_explanation"
                ]
            )


            st.markdown(
                "**What this measurement is**"
            )

            st.write(
                factor[
                    "what_it_is"
                ]
            )


            st.markdown(
                "**What research says**"
            )

            st.write(
                factor[
                    "research_summary"
                ]
            )


            st.markdown(
                "**What it means here**"
            )

            st.write(
                factor[
                    "personal_interpretation"
                ]
            )


            strength = factor[
                "evidence_strength"
            ].capitalize()

            st.markdown(
                f"**Evidence strength:** "
                f"{strength}"
            )

            st.caption(
                factor[
                    "evidence_note"
                ]
            )


            sources = factor.get(
                "sources",
                []
            )

            if sources:

                st.markdown(
                    "**Scientific sources**"
                )

                for source in sources:

                    title = source[
                        "title"
                    ]

                    doi = source.get(
                        "doi"
                    )

                    if doi:

                        st.markdown(
                            f"- [{title}]"
                            f"(https://doi.org/{doi})"
                        )

                    else:

                        st.markdown(
                            f"- {title}"
                        )


            st.warning(
                factor[
                    "important_caution"
                ]
            )


    # --------------------------------------------------
    # Evidence summary
    # --------------------------------------------------

    st.markdown(
        "## How strong is the evidence?"
    )

    st.write(
        summary[
            "evidence_overview"
        ]
    )


    st.markdown(
        f"""
        <div class="disclaimer">
        <strong>Research prototype</strong><br>
        {summary["important_caution"]}
        NeuroDecel is designed to explain model predictions
        and supporting research. It is not a diagnostic tool.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
<div class="hero">
<h1>🧠 NeuroDecel</h1>
<h3>Your brain age is more than a number.</h3>
<p>
NeuroDecel estimates brain age from MRI-derived measurements,
identifies which brain structures influenced the prediction,
and connects those findings with published scientific evidence.
</p>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
    ### Most brain-age tools stop at a prediction.

    **NeuroDecel explains why.**

    Instead of simply telling you that a brain appears
    older or younger, NeuroDecel combines interpretable
    machine learning with scientific literature to show
    which structural measurements mattered and what
    current research says about them.
    """
)


# --------------------------------------------------
# Product value
# --------------------------------------------------

c1, c2, c3 = st.columns(3)


with c1:

    st.markdown("### 🎯 Predict")

    st.write(
        "Estimate brain age from hundreds of "
        "MRI-derived structural measurements."
    )


with c2:

    st.markdown("### 🔎 Explain")

    st.write(
        "Use SHAP to identify the measurements "
        "that most influenced each individual result."
    )


with c3:

    st.markdown("### 📚 Ground")

    st.write(
        "Connect model explanations to published "
        "research through Amass and translate the "
        "evidence into understandable language."
    )


st.divider()


# --------------------------------------------------
# Input choices
# --------------------------------------------------

st.markdown(
    "## Try NeuroDecel"
)


upload_tab, demo_tab = st.tabs(
    [
        "📄 Upload MRI report",
        "⚡ Try a demo",
    ]
)


# --------------------------------------------------
# PDF tab
# --------------------------------------------------

with upload_tab:

    st.markdown(
        "### Upload an MRI-derived brain report"
    )

    st.write(
        """
        Upload a structured PDF containing the patient's
        chronological age and MRI-derived structural
        measurements.

        NeuroDecel will extract the measurements,
        estimate brain age, identify the most influential
        features, search scientific literature, and create
        an evidence-grounded explanation.
        """
    )


    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="patient_pdf",
    )


    if uploaded_file:

        st.success(
            f"Loaded: {uploaded_file.name}"
        )


        # ------------------------------------------
        # First parse + validate PDF
        # ------------------------------------------

        try:

            with st.spinner(
                "Checking MRI report..."
            ):

                parsed = parse_patient_pdf(
                    uploaded_file
                )


            st.markdown(
                "#### Report successfully parsed"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Chronological age",
                    f"{parsed['actual_age']:.0f}"
                )


            with col2:

                st.metric(
                    "MRI features detected",
                    f"{parsed['feature_count']} / 270"
                )


            st.success(
                "All required MRI measurements "
                "were found. Ready for analysis."
            )


            # ------------------------------------------
            # Full live analysis
            # ------------------------------------------

            if st.button(
                "🧠 Analyze brain age",
                type="primary",
                use_container_width=True,
            ):

                report = run_complete_pipeline(

                    features=
                        parsed["features"],

                    actual_age=
                        parsed["actual_age"],

                    patient_id=
                        parsed["patient_id"],
                )


                st.session_state[
                    "report"
                ] = report


        except Exception as exc:

            st.error(
                "This PDF could not be used "
                "for brain-age analysis."
            )

            st.warning(
                str(exc)
            )

            print(
                "PDF parsing error:",
                repr(exc)
            )


# --------------------------------------------------
# Demo tab
# --------------------------------------------------

with demo_tab:

    st.write(
        "Explore NeuroDecel using an anonymized "
        "sample from our research dataset."
    )

    st.caption(
        "Demo patient: anonymized participant #2561"
    )

    if st.button(
        "🚀 Run demo",
        type="primary",
        use_container_width=True,
    ):

        try:

            report = load_cached_demo_report()

            st.session_state[
                "report"
            ] = report

            st.success(
                "Demo loaded successfully."
            )

        except Exception as exc:

            st.error(
                "Could not load the demo patient."
            )

            print(
                "Demo loading error:",
                repr(exc)
            )

# --------------------------------------------------
# Render persisted result
# --------------------------------------------------

if "report" in st.session_state:

    render_report(
        st.session_state[
            "report"
        ]
    )


# --------------------------------------------------
# Future vision
# --------------------------------------------------

st.divider()

st.markdown(
    "## From MRI scan to understandable insight"
)

st.write(
    """
    Today's prototype works with structural measurements
    already extracted from MRI scans.

    The next step is direct MRI ingestion: upload a scan,
    automatically extract structural brain measurements,
    estimate brain age, identify the most influential
    regions, and generate an evidence-grounded explanation
    — all in one workflow.
    """
)

st.caption(
    "NeuroDecel is a research prototype and is not "
    "intended for clinical diagnosis or medical decision-making."
)