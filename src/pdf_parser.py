import io
import re

from pypdf import PdfReader

from model_service import (
    load_model_artifacts
)


# --------------------------------------------------
# Extract PDF text
# --------------------------------------------------

def extract_pdf_text(
    uploaded_file
):
    """
    Extract text from a text-based PDF.
    """

    uploaded_file.seek(0)

    pdf_bytes = uploaded_file.read()

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    full_text = "\n".join(pages)

    if not full_text.strip():

        raise ValueError(
            "No readable text was found in the PDF. "
            "The current prototype requires a "
            "text-based MRI measurement report."
        )

    return full_text


# --------------------------------------------------
# Patient age
# --------------------------------------------------

def extract_age(
    text
):

    patterns = [

        r"AgeMRI_W1\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",

        r"Chronological\s+Age\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",

        r"Patient\s+Age\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",

        r"\bAge\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            age = float(
                match.group(1)
            )

            if age <= 0 or age > 120:

                raise ValueError(
                    f"Invalid patient age: {age}"
                )

            return age


    raise ValueError(
        "Patient age could not be found in the PDF. "
        "Include a line such as 'Patient Age: 54'."
    )


# --------------------------------------------------
# Optional patient ID
# --------------------------------------------------

def extract_patient_id(
    text
):

    patterns = [

        r"Patient\s+ID\s*[:=]?\s*([A-Za-z0-9_-]+)",

        r"Participant\s+ID\s*[:=]?\s*([A-Za-z0-9_-]+)",

        r"\bS#\s*[:=]?\s*([A-Za-z0-9_-]+)",
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            return match.group(1)


    return None


# --------------------------------------------------
# Extract model features
# --------------------------------------------------

def extract_features(
    text
):

    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]


    feature_values = {}

    missing_features = []


    # Normalise some PDF extraction quirks
    normalized_text = (
        text
        .replace("\u00a0", " ")
        .replace(",", "")
    )


    for feature in feature_cols:

        escaped_feature = re.escape(
            str(feature)
        )

        patterns = [

            # ThirdVentVol: 123.45
            rf"{escaped_feature}\s*[:=]\s*"
            rf"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",

            # ThirdVentVol    123.45
            rf"{escaped_feature}\s+"
            rf"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        ]


        value = None


        for pattern in patterns:

            match = re.search(
                pattern,
                normalized_text,
                flags=re.IGNORECASE
            )

            if match:

                value = float(
                    match.group(1)
                )

                break


        if value is None:

            missing_features.append(
                feature
            )

        else:

            feature_values[
                feature
            ] = value


    return {
        "features":
            feature_values,

        "missing_features":
            missing_features,

        "total_required":
            len(feature_cols),

        "total_found":
            len(feature_values),
    }


# --------------------------------------------------
# Complete PDF parser
# --------------------------------------------------

def parse_patient_pdf(
    uploaded_file
):

    text = extract_pdf_text(
        uploaded_file
    )

    patient_id = extract_patient_id(
        text
    )

    actual_age = extract_age(
        text
    )

    feature_result = extract_features(
        text
    )


    if feature_result[
        "missing_features"
    ]:

        missing = feature_result[
            "missing_features"
        ]

        raise ValueError(
            f"Only "
            f"{feature_result['total_found']} / "
            f"{feature_result['total_required']} "
            f"required MRI features were found. "
            f"Missing {len(missing)} features. "
            f"First missing features: "
            f"{missing[:10]}"
        )


    return {

        "patient_id":
            patient_id,

        "actual_age":
            actual_age,

        "features":
            feature_result[
                "features"
            ],

        "feature_count":
            feature_result[
                "total_found"
            ],
    }