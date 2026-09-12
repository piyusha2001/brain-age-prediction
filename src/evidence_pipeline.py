from pathlib import Path
import json
import pandas as pd

from feature_mapping import (
    clean_feature_name,
    is_amass_searchable,
)

from amass_client import (
    fetch_biomed_evidence,
    filter_biomed_evidence,
)



# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPLANATIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "all_test_patients_explained.csv"
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------
# Load SHAP explanations
# --------------------------------------------------

def load_explanations():
    """
    Load patient-level SHAP explanations produced
    by the brain-age model.
    """

    return pd.read_csv(EXPLANATIONS_PATH)


# --------------------------------------------------
# Select top N searchable features
# --------------------------------------------------
def build_evidence_from_model_result(
    model_result
):
    """
    Take the output of model_service.analyze_patient()
    and attach scientific evidence from Amass.
    """

    result = {
        "patient": model_result["patient"],
        "factors": []
    }

    total = len(
        model_result["factors"]
    )

    for index, factor in enumerate(
        model_result["factors"],
        start=1
    ):

        medical_name = factor[
            "medical_name"
        ]

        print(
            f"[{index}/{total}] "
            f"Searching Amass for: "
            f"{medical_name}"
        )

        raw_papers = fetch_biomed_evidence(
            medical_name
        )

        papers = filter_biomed_evidence(
            medical_name,
            raw_papers,
            top_k=3
        )

        factor_with_evidence = (
            factor.copy()
        )

        factor_with_evidence[
            "amass"
        ] = {
            "papers": papers
        }

        result["factors"].append(
            factor_with_evidence
        )

    return result
    
def get_top_patient_factors(patient_id, top_n=5):
    """
    Get the strongest SHAP factors for one patient.

    Non-biological / QC features are skipped before
    selecting the final top N features.
    """

    df = load_explanations()

    patient = df[
        df["S#"] == patient_id
    ].copy()

    if patient.empty:
        raise ValueError(
            f"Patient {patient_id} was not found."
        )

    # Absolute SHAP strength
    patient["abs_shap"] = (
        patient["shap_impact"].abs()
    )

    # Strongest contribution first
    patient = patient.sort_values(
        "abs_shap",
        ascending=False
    )

    # Remove features we should not query in Amass
    patient = patient[
        patient["feature"].apply(
            is_amass_searchable
        )
    ]

    # Final top N searchable features
    return patient.head(top_n)


# --------------------------------------------------
# Retrieve Amass evidence
# --------------------------------------------------

def build_patient_evidence(patient_id, top_n=5):
    """
    Retrieve Amass scientific evidence for the
    patient's top SHAP brain-age factors.
    """

    patient = get_top_patient_factors(
        patient_id,
        top_n
    )

    first_row = patient.iloc[0]

    result = {
        "patient": {
            "id": int(patient_id),

            "actual_age": float(
                first_row["actual_age"]
            ),

            "predicted_age": float(
                first_row["predicted_age"]
            ),

            "brain_age_gap": float(
                first_row["brain_age_gap"]
            ),
        },

        "factors": []
    }


    # --------------------------------------------------
    # Query Amass for every selected factor
    # --------------------------------------------------

    for number, (_, row) in enumerate(
        patient.iterrows(),
        start=1
    ):

        raw_feature = str(
            row["feature"]
        )

        medical_name = clean_feature_name(
            raw_feature
        )

        print()
        print("=" * 70)
        print(
            f"FACTOR {number}/{len(patient)}"
        )
        print("=" * 70)

        print(
            "Raw feature:",
            raw_feature
        )

        print(
            "Medical term:",
            medical_name
        )

        print(
            "SHAP impact:",
            row["shap_impact"]
        )

        print(
            "Direction:",
            row["direction"]
        )


        # ----------------------------------------------
        # AMASS
        # ----------------------------------------------

        raw_papers = fetch_biomed_evidence(
             medical_name
        )

        papers = filter_biomed_evidence(
            medical_name,
            raw_papers,
            top_k=3
        )

        factor_result = {

            "rank": number,

            "raw_feature":
                raw_feature,

            "medical_name":
                medical_name,

            "shap_impact": float(
                row["shap_impact"]
            ),

            "direction": str(
                row["direction"]
            ),

            "patient_level": str(
                row["patient_level"]
            ),

            "z_score": float(
                row["z_score"]
            ),

            "amass": {
                "papers": papers
            }
        }


        result["factors"].append(
            factor_result
        )

        print(
            f"Amass papers found: "
            f"{len(papers)}"
        )


    return result


# --------------------------------------------------
# Save complete evidence object
# --------------------------------------------------

def save_patient_evidence(result):
    """
    Save the combined ML + SHAP + Amass output
    as JSON for the next stage of the pipeline.
    """

    patient_id = result["patient"]["id"]

    output_path = (
        OUTPUTS_DIR
        / f"patient_{patient_id}_amass_evidence.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    return output_path


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    PATIENT_ID = 2561

    print("\nBuilding evidence for patient:")
    print(PATIENT_ID)

    result = build_patient_evidence(
        patient_id=PATIENT_ID,
        top_n=5
    )

    output_path = save_patient_evidence(
        result
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "\nPatient:",
        result["patient"]
    )

    print(
        "\nFactors processed:",
        len(result["factors"])
    )

    for factor in result["factors"]:

        print()
        print(
            factor["rank"],
            factor["medical_name"]
        )

        print(
            "  SHAP:",
            factor["shap_impact"]
        )

        print(
            "  Direction:",
            factor["direction"]
        )

        print(
            "  Papers:",
            len(
                factor["amass"]["papers"]
            )
        )

    print(
        "\nSaved to:",
        output_path
    )