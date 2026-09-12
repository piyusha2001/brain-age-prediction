from pathlib import Path
import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv


# --------------------------------------------------
# Paths / environment
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-5"
)


if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY was not found in .env"
    )


client = Anthropic(
    api_key=ANTHROPIC_API_KEY
)


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You explain brain-age model results to a general audience using
scientific evidence supplied to you.

Your job is to translate technical neuroimaging and brain-aging
research into medically responsible, easy-to-understand language.

STRICT RULES:

1. Use only the scientific evidence supplied in the request.
   Do not introduce medical facts from memory.

2. Clearly separate:
   - what the machine-learning model found
   - what published research reports

3. A SHAP value describes how a feature influenced this model's
   prediction. It does NOT prove that the feature biologically caused
   brain aging.

4. Do not diagnose any disease or neurological condition.

5. Do not claim that a patient has Alzheimer's disease, dementia,
   vascular disease, neurodegeneration, or another disorder simply
   because a paper discusses that condition.

6. Prefer association language:
   "associated with", "linked with", "research has observed".
   Do not claim causation unless the supplied evidence establishes it,
   and in this application causal wording should generally be avoided.

7. Explain anatomical terms in plain language.

8. "HIGH", "LOW", or a z-score refers to the reference distribution
   used in this analysis. Do not call a measurement clinically
   abnormal unless clinical evidence explicitly establishes that.

9. Do not give medication, treatment, or medical-action advice.

10. If the evidence is indirect, inconsistent, disease-specific,
    based on a different population, or only loosely related to the
    patient's feature, explicitly say that.

11. The audience is a normal person with no neuroscience background.
    Prefer short sentences and simple words.

12. Never exaggerate the meaning of the brain-age prediction.
    Brain age is a model estimate, not the person's literal biological
    age and not a diagnosis.

13. The machine-learning model and the scientific evidence are
    completely separate sources of information.

    The model was trained on the provided brain-imaging dataset.
    The scientific papers were retrieved AFTER the prediction to
    provide external context.

    NEVER imply that the model was trained on, learned from, or used
    the supplied research papers.

14. When both healthy-aging studies and disease-specific studies are
    supplied, prioritize healthy-aging evidence.

    Do not mention disease-specific studies in the user-facing
    explanation unless they are necessary to explain an important
    limitation or there is insufficient healthy-aging evidence.

15. Do not make disease-specific findings sound applicable to the
    patient.

16. A z-score describes where the measurement lies relative to the
    reference distribution. Describe it numerically when useful, but
    do not equate it with clinical abnormality.

17. When describing SHAP direction:
    - "older" means the feature pushed the model toward a higher
      predicted brain age.
    - "younger" means the feature pushed the model toward a lower
      predicted brain age.
    Never say that a single factor pushed the prediction above or
    below the person's chronological age unless the supplied model
    result explicitly establishes that.

18. When discussing z-scores, ONLY use wording such as:

    "above the reference average"
    "below the reference average"
    "close to the reference average"

    Never use:
    "normal range"
    "typical range"
    "healthy range"
    "abnormal"

    unless a supplied clinical threshold explicitly establishes it.

19. Preserve the specificity of the evidence.
    If research discusses ventricles generally, do not claim that it
    specifically studied the third ventricle.
    If research discusses a broad brain region, do not convert that
    into evidence for a more specific subregion unless the paper
    explicitly does so.

20. Keep the research summary consistent with the evidence limitations.
    If the evidence discusses ventricles generally, say "ventricles"
    rather than implying the study specifically analyzed the third
    ventricle.

21. If the relevant evidence is primarily disease-specific and does
    not support an interpretation in healthy aging, do not name the
    diseases in the user-facing research_summary or
    personal_interpretation.

    Instead say that the available evidence comes from specific
    clinical populations and is not sufficient to interpret the
    feature in healthy aging.

    Disease-specific sources may remain in backend provenance data.

Return valid JSON only.
Do not use Markdown.
Do not wrap the JSON in code fences.
"""

PATIENT_SUMMARY_SYSTEM_PROMPT = """
You summarize a brain-age prediction for a general audience.

You receive:
- the person's chronological age
- the model's predicted brain age
- the brain-age gap
- five already-generated factor explanations

Your job is to create a short, medically cautious overall explanation.

STRICT RULES:

1. Brain age is a machine-learning estimate, not literal biological age
   and not a medical diagnosis.

2. Do not diagnose disease.

3. Do not introduce any medical facts that are not already present in
   the supplied factor explanations.

4. Do not imply causation.

5. Clearly explain that some features can push the prediction older
   while the overall prediction is still younger, or vice versa.

6. Do not describe the person as healthy, unhealthy, normal, abnormal,
   diseased, or disease-free.

7. Use simple language suitable for someone without medical training.

8. Do not repeat every factor explanation in full.

9. If evidence is limited or insufficient for some factors, say so
   briefly.

10. Avoid alarmist wording.

Return valid JSON only.
Do not use Markdown.
Do not wrap the JSON in code fences.
"""

# --------------------------------------------------
# Load patient evidence
# --------------------------------------------------

def load_patient_evidence(patient_id: int):

    path = (
        OUTPUTS_DIR
        / f"patient_{patient_id}_amass_evidence.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find evidence file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Find one feature
# --------------------------------------------------

def get_factor(
    patient_data,
    raw_feature: str
):

    for factor in patient_data["factors"]:

        if factor["raw_feature"] == raw_feature:
            return factor

    raise ValueError(
        f"Feature {raw_feature} was not found "
        f"in this patient's evidence."
    )


# --------------------------------------------------
# Keep only useful paper information
# --------------------------------------------------

def prepare_papers(factor):

    papers = []

    for index, paper in enumerate(
        factor["amass"]["papers"],
        start=1
    ):

        papers.append({
            "source_number": index,

            "title":
                paper.get("title"),

            "abstract":
                paper.get("abstract"),

            "journal":
                paper.get("journal"),

            "publication_date":
                paper.get("publicationDate"),

            "doi":
                paper.get("doi"),

            "pmid":
                paper.get("pmid"),

            "citation_count":
                paper.get("citationCount"),

            "journal_quality":
                paper.get("journalQualityJufo"),

            "relevance_score":
                paper.get("relevanceScore"),
        })

    return papers


# --------------------------------------------------
# Claude explanation
# --------------------------------------------------

def explain_factor(
    patient_data,
    factor,
    max_retries=2
):

    patient = patient_data["patient"]

    papers = prepare_papers(factor)

    input_data = {

        "patient_model_result": {
            "actual_age":
                patient["actual_age"],

            "predicted_brain_age":
                patient["predicted_age"],

            "brain_age_gap":
                patient["brain_age_gap"],
        },

        "factor": {
            "raw_feature":
                factor["raw_feature"],

            "medical_name":
                factor["medical_name"],

            "shap_impact":
                factor["shap_impact"],

            "direction":
                factor["direction"],

            "patient_level":
                factor["patient_level"],

            "z_score":
                factor["z_score"],
        },

        "scientific_evidence":
            papers,
    }


    prompt = f"""
Here is the patient-specific brain-age model result and the scientific
evidence retrieved for one contributing feature.

<evidence>
{json.dumps(input_data, indent=2)}
</evidence>

Explain this factor for a non-medical user.

First determine what the MODEL tells us:
- whether this factor pushed the brain-age prediction older or younger
- how the patient's measurement compares with the reference distribution
- do not interpret the SHAP value as a number of biological years

Then determine what the RESEARCH tells us:
- explain what this brain structure or measurement is
- summarize only findings directly relevant to aging and this feature
- prioritize healthy/normal aging evidence over disease-specific evidence
- if healthy-aging evidence is sufficient, omit disease-specific studies
  from the user-facing explanation
- disease-specific studies may be used only to describe limitations,
  never to suggest the patient has or is at risk for that disease
- remember that these papers were retrieved after prediction and played
  no role in training or generating the model result
- note important limitations

Return exactly this JSON structure:

{{
  "feature": "short human-readable feature name",

  "direction": "older or younger",

  "model_explanation":
    "1-2 simple sentences explaining how this feature affected the model",

  "what_it_is":
    "1-2 simple sentences describing this brain structure or measurement",

  "research_summary":
    "2-4 simple sentences summarizing what the supplied research says about this feature and aging",

  "personal_interpretation":
    "2-3 cautious sentences connecting the model result with the research without diagnosing or claiming causation",

  "evidence_strength":
    "strong, moderate, limited, or insufficient",

  "evidence_note":
    "one sentence explaining why that evidence-strength label was chosen",

  "important_caution":
    "one short sentence explaining what the result does NOT mean",

  "sources": [
    {{
      "source_number": 1,
      "title": "paper title",
      "doi": "paper DOI or null"
    }}
  ]
}}

Only include sources that actually support the explanation.
"""


    for attempt in range(
        max_retries + 1
    ):

        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=3500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )


        text_blocks = [
            block.text
            for block in message.content
            if block.type == "text"
        ]


        if not text_blocks:

            if attempt < max_retries:
                continue

            raise ValueError(
                "Claude returned no text response."
            )


        response_text = "\n".join(
            text_blocks
        )


        try:

            return parse_claude_json(
                response_text
            )

        except ValueError:

            if attempt < max_retries:

                print(
                    f"Invalid Claude JSON. "
                    f"Retrying "
                    f"({attempt + 1}/{max_retries})..."
                )

                continue

            raise
    
# --------------------------------------------------
# Test on ThirdVentVol
# --------------------------------------------------

def explain_all_factors(patient_data):
    """
    Generate a Claude explanation for every
    Amass-backed factor in the patient result.
    """

    explanations = []

    total_factors = len(
        patient_data["factors"]
    )

    for index, factor in enumerate(
        patient_data["factors"],
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"EXPLAINING FACTOR "
            f"{index}/{total_factors}"
        )
        print("=" * 70)

        print(
            "Feature:",
            factor["medical_name"]
        )

        explanation = explain_factor(
            patient_data,
            factor
        )

        explanations.append(
            explanation
        )

    return explanations

def synthesize_patient_report(
    patient_data,
    factor_explanations
):
    """
    Generate one overall, user-friendly interpretation
    from the five factor explanations.
    """

    patient = patient_data["patient"]

    # Keep directions deterministic rather than
    # asking Claude to infer them again.
    younger_factors = [
        factor["feature"]
        for factor in factor_explanations
        if factor["direction"] == "younger"
    ]

    older_factors = [
        factor["feature"]
        for factor in factor_explanations
        if factor["direction"] == "older"
    ]


    synthesis_input = {

        "patient": {
            "actual_age":
                patient["actual_age"],

            "predicted_brain_age":
                patient["predicted_age"],

            "brain_age_gap":
                patient["brain_age_gap"],
        },

        "younger_direction_factors":
            younger_factors,

        "older_direction_factors":
            older_factors,

        "factor_explanations":
            factor_explanations,
    }


    prompt = f"""
Create the overall explanation for this brain-age result.

<data>
{json.dumps(
    synthesis_input,
    indent=2,
    ensure_ascii=False
)}
</data>

Return exactly this JSON structure:

{{
  "headline":
    "one short sentence describing the overall brain-age result",

  "overview":
    "2-4 plain-language sentences explaining the result as a whole",

  "key_takeaway":
    "one concise sentence with the most useful interpretation",

  "evidence_overview":
    "1-3 sentences explaining how strong or limited the supporting research is across the factors",

  "important_caution":
    "one short sentence explaining that brain age is a model estimate and not a diagnosis"
}}

The headline should describe the overall result, for example:
"Estimated brain age is about 6.7 years younger than chronological age."

Do not say the person's brain literally is that many years younger.
"""


    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=PATIENT_SUMMARY_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    text_blocks = [
        block.text
        for block in message.content
        if block.type == "text"
    ]

    if not text_blocks:
        raise ValueError(
            "Claude returned no text for patient synthesis."
        )

    response_text = "\n".join(
        text_blocks
    )

    return json.loads(
        response_text
    )

def build_patient_explanation(patient_data):
    """
    Build the complete patient result for the UI.
    """

    patient = patient_data["patient"]

    # Individual explanations
    explanations = explain_all_factors(
        patient_data
    )

    # Overall Claude synthesis
    overall = synthesize_patient_report(
        patient_data,
        explanations
    )


    # Exact factual values calculated by Python
    gap = float(
        patient["brain_age_gap"]
    )

    age_direction = (
        "younger"
        if gap < 0
        else "older"
        if gap > 0
        else "same"
    )


    younger_factors = [
        factor["feature"]
        for factor in explanations
        if factor["direction"] == "younger"
    ]

    older_factors = [
        factor["feature"]
        for factor in explanations
        if factor["direction"] == "older"
    ]


    result = {

        "patient": {
            "id":
                patient["id"],

            "actual_age":
                patient["actual_age"],

            "predicted_age":
                patient["predicted_age"],

            "brain_age_gap":
                gap,
        },


        "summary": {

            "age_direction":
                age_direction,

            "gap_years":
                abs(gap),

            "headline":
                overall["headline"],

            "overview":
                overall["overview"],

            "key_takeaway":
                overall["key_takeaway"],

            "evidence_overview":
                overall["evidence_overview"],

            "important_caution":
                overall["important_caution"],
        },


        "factor_groups": {

            "younger":
                younger_factors,

            "older":
                older_factors,
        },


        "factors":
            explanations,
    }

    return result

def parse_claude_json(response_text):
    """
    Parse Claude's JSON response with clearer errors.
    """

    response_text = response_text.strip()

    # Remove accidental Markdown fences
    if response_text.startswith("```json"):
        response_text = response_text[7:]

    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:
        return json.loads(response_text)

    except json.JSONDecodeError as exc:

        print("\nINVALID CLAUDE JSON")
        print("=" * 70)
        print(response_text)
        print("=" * 70)

        raise ValueError(
            f"Claude returned invalid JSON: {exc}"
        )

def save_patient_explanation(
    result
):
    """
    Save final Claude explanations for the UI.
    """

    patient_id = result["patient"]["id"]

    output_path = (
        OUTPUTS_DIR
        / f"patient_{patient_id}_explanation.json"
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
            ensure_ascii=False
        )

    return output_path

if __name__ == "__main__":

    PATIENT_ID = 2561

    print(
        f"\nLoading patient {PATIENT_ID}..."
    )

    patient_data = load_patient_evidence(
        PATIENT_ID
    )

    result = build_patient_explanation(
        patient_data
    )

    output_path = save_patient_explanation(
        result
    )

    print()
    print("=" * 70)
    print("ALL FACTORS COMPLETE")
    print("=" * 70)

    print(
        "\nActual age:",
        result["patient"]["actual_age"]
    )

    print(
        "Predicted brain age:",
        result["patient"]["predicted_age"]
    )

    print(
        "Brain-age gap:",
        result["patient"]["brain_age_gap"]
    )

    print(
        "\nFactors explained:",
        len(result["factors"])
    )

    for index, factor in enumerate(
        result["factors"],
        start=1
    ):

        print()
        print(
            index,
            factor["feature"]
        )

        print(
            "  Direction:",
            factor["direction"]
        )

        print(
            "  Evidence:",
            factor["evidence_strength"]
        )

    print(
        "\nSaved to:",
        output_path
    )