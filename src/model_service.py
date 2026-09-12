from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from feature_mapping import (
    clean_feature_name,
    is_amass_searchable,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODELS_DIR / "brainage_xgb_model.json"
)

FEATURE_COLS_PATH = (
    MODELS_DIR / "feature_cols.pkl"
)

BIAS_MODEL_PATH = (
    MODELS_DIR / "bias_correction_model.pkl"
)

REFERENCE_STATS_PATH = (
    MODELS_DIR / "training_reference_stats.pkl"
)


# --------------------------------------------------
# Load saved model artifacts
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_model_artifacts():
    """
    Load everything required to make a prediction
    for a new patient.

    Cached so Streamlit does not reload the model
    every time the page reruns.
    """

    # XGBoost model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # Exact 270 feature names / ordering
    feature_cols = joblib.load(
        FEATURE_COLS_PATH
    )

    # Bias-correction model
    bias_model = joblib.load(
        BIAS_MODEL_PATH
    )

    # Training-set reference mean/std
    reference_stats = joblib.load(
        REFERENCE_STATS_PATH
    )

    # SHAP explainer
    explainer = shap.TreeExplainer(
        model
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "bias_model": bias_model,
        "reference_stats": reference_stats,
        "explainer": explainer,
    }


# --------------------------------------------------
# Validate incoming patient
# --------------------------------------------------

def validate_patient_features(
    feature_values: dict
):
    """
    Make sure a new patient contains every feature
    required by the trained model.
    """

    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]

    missing_features = [
        feature
        for feature in feature_cols
        if feature not in feature_values
    ]

    if missing_features:

        preview = missing_features[:10]

        raise ValueError(
            f"Patient is missing "
            f"{len(missing_features)} required features. "
            f"First missing features: {preview}"
        )

    return True


# --------------------------------------------------
# Convert patient dictionary into model input
# --------------------------------------------------

def prepare_patient_input(
    feature_values: dict
):
    """
    Put patient features into the exact same
    270-feature order used during model training.
    """

    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]

    validate_patient_features(
        feature_values
    )

    ordered_values = [
        float(feature_values[feature])
        for feature in feature_cols
    ]

    # One patient, 270 features
    X_patient = np.array(
        [ordered_values],
        dtype=float
    )

    return X_patient


# --------------------------------------------------
# Predict brain age
# --------------------------------------------------

def predict_brain_age(
    feature_values: dict
):
    """
    Generate raw and bias-corrected brain-age
    predictions for one patient.
    """

    artifacts = load_model_artifacts()

    model = artifacts["model"]
    bias_model = artifacts["bias_model"]

    X_patient = prepare_patient_input(
        feature_values
    )

    # Raw XGBoost prediction
    raw_prediction = float(
        model.predict(X_patient)[0]
    )

    # Same correction used in notebook:
    #
    # bias_model learned:
    # y_train - train_prediction
    #
    # so:
    # corrected = raw + predicted correction

    correction = float(
        bias_model.predict(
            np.array(
                [[raw_prediction]]
            )
        )[0]
    )

    corrected_prediction = (
        raw_prediction + correction
    )

    return {
        "raw_prediction":
            raw_prediction,

        "bias_correction":
            correction,

        "predicted_age":
            corrected_prediction,
    }


# --------------------------------------------------
# SHAP explanation
# --------------------------------------------------

def explain_patient_features(
    feature_values: dict,
    top_n: int = 5
):
    """
    Find the top N searchable features driving
    this patient's prediction.
    """

    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]

    reference_stats = artifacts[
        "reference_stats"
    ]

    explainer = artifacts[
        "explainer"
    ]

    X_patient = prepare_patient_input(
        feature_values
    )

    # SHAP values
    shap_values = explainer.shap_values(
        X_patient
    )

    # One patient
    patient_shap = np.asarray(
        shap_values
    )[0]

    patient_values = X_patient[0]

    train_means = np.asarray(
        reference_stats["mean"]
    )

    train_stds = np.asarray(
        reference_stats["std"]
    )


    # Strongest absolute SHAP contributions first
    ranked_indices = np.argsort(
        np.abs(patient_shap)
    )[::-1]


    factors = []


    for index in ranked_indices:

        raw_feature = feature_cols[index]

        # Skip processing/QC features
        if not is_amass_searchable(
            raw_feature
        ):
            continue


        shap_impact = float(
            patient_shap[index]
        )

        raw_value = float(
            patient_values[index]
        )

        cohort_mean = float(
            train_means[index]
        )

        cohort_std = float(
            train_stds[index]
        )


        # Calculate z-score
        if cohort_std > 0:

            z_score = (
                raw_value - cohort_mean
            ) / cohort_std

        else:

            z_score = 0.0


        # Same thresholds as your notebook
        if z_score > 0.5:

            patient_level = "HIGH"

        elif z_score < -0.5:

            patient_level = "LOW"

        else:

            patient_level = "about average"


        direction = (
            "older"
            if shap_impact > 0
            else "younger"
        )


        factors.append({

            "rank":
                len(factors) + 1,

            "raw_feature":
                raw_feature,

            "medical_name":
                clean_feature_name(
                    raw_feature
                ),

            "raw_value":
                raw_value,

            "reference_mean":
                cohort_mean,

            "reference_std":
                cohort_std,

            "z_score":
                float(z_score),

            "patient_level":
                patient_level,

            "shap_impact":
                shap_impact,

            "direction":
                direction,
        })


        if len(factors) == top_n:
            break


    return factors


# --------------------------------------------------
# Complete model analysis
# --------------------------------------------------

def analyze_patient(
    feature_values: dict,
    actual_age: float,
    patient_id=None,
    top_n: int = 5
):
    """
    Run the complete ML portion of the pipeline.

    Input:
        270 MRI-derived feature values
        chronological age

    Output:
        predicted brain age
        brain-age gap
        top SHAP factors
    """

    prediction = predict_brain_age(
        feature_values
    )

    predicted_age = prediction[
        "predicted_age"
    ]

    brain_age_gap = (
        predicted_age - float(actual_age)
    )

    factors = explain_patient_features(
        feature_values,
        top_n=top_n
    )


    result = {

        "patient": {
            "id":
                patient_id,

            "actual_age":
                float(actual_age),

            "predicted_age":
                float(predicted_age),

            "brain_age_gap":
                float(brain_age_gap),
        },

        "model_debug": {
            "raw_prediction":
                prediction[
                    "raw_prediction"
                ],

            "bias_correction":
                prediction[
                    "bias_correction"
                ],
        },

        "factors":
            factors,
    }


    return result

if __name__ == "__main__":

    # --------------------------------------------------
    # Regression test using known patient 2561
    # --------------------------------------------------

    PATIENT_ID = 2561

    DATA_PATH = (
        PROJECT_ROOT
        / "dataset"
        / "BIOMARKERDATA.xlsx"
    )

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


    def load_sheet(sheet_name):

        df = pd.read_excel(
            DATA_PATH,
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


    # ----------------------------------------------
    # Load age
    # ----------------------------------------------

    age_df = pd.read_excel(
        DATA_PATH,
        sheet_name="Age-W1,W2,W3"
    )

    age_w1 = age_df[
        ["S#", "AgeMRI_W1"]
    ].dropna(
        subset=["AgeMRI_W1"]
    )


    # ----------------------------------------------
    # Load MRI feature sheets
    # ----------------------------------------------

    sheets_w1 = [
        "GlobalVariables-W1",
        "CorticalThickness-W1",
        "GMVolume-W1",
        "SurfaceArea-W1",
        "SubcorticalVolume-W1",
    ]


    merged = age_w1.copy()

    for sheet in sheets_w1:

        merged = merged.merge(
            load_sheet(sheet),
            on="S#",
            how="inner"
        )


    # ----------------------------------------------
    # Find known patient
    # ----------------------------------------------

    patient_row = merged[
        pd.to_numeric(
            merged["S#"],
            errors="coerce"
        ) == PATIENT_ID
    ]

    if patient_row.empty:
        raise ValueError(
            f"Patient {PATIENT_ID} not found."
        )

    patient_row = patient_row.iloc[0]


    # ----------------------------------------------
    # Get exact model features
    # ----------------------------------------------

    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]


    patient_features = {
        feature: float(
            patient_row[feature]
        )
        for feature in feature_cols
    }


    actual_age = float(
        patient_row["AgeMRI_W1"]
    )


    # ----------------------------------------------
    # Run the SAVED model
    # ----------------------------------------------

    result = analyze_patient(
        feature_values=patient_features,
        actual_age=actual_age,
        patient_id=PATIENT_ID,
        top_n=5,
    )


    # ----------------------------------------------
    # Compare with notebook
    # ----------------------------------------------

    EXPECTED_PREDICTION = 47.34774
    EXPECTED_GAP = -6.652259826660156


    print("\n" + "=" * 70)
    print("SAVED MODEL REGRESSION TEST")
    print("=" * 70)

    print(
        "\nPatient:",
        PATIENT_ID
    )

    print(
        "Actual age:",
        result["patient"]["actual_age"]
    )

    print(
        "Predicted age:",
        result["patient"]["predicted_age"]
    )

    print(
        "Expected prediction:",
        EXPECTED_PREDICTION
    )

    print(
        "Prediction difference:",
        abs(
            result["patient"]["predicted_age"]
            - EXPECTED_PREDICTION
        )
    )

    print(
        "\nBrain-age gap:",
        result["patient"]["brain_age_gap"]
    )

    print(
        "Expected gap:",
        EXPECTED_GAP
    )


    print("\nTOP 5 FACTORS")
    print("-" * 70)

    for factor in result["factors"]:

        print()
        print(
            factor["rank"],
            factor["raw_feature"]
        )

        print(
            "  Medical name:",
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
            "  Z-score:",
            factor["z_score"]
        )