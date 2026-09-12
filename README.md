# 🧠 NeuroDecel

> **Your brain age is more than a number.**

NeuroDecel is an explainable brain-age prediction system that estimates a person's brain age from MRI-derived structural measurements, identifies the brain regions that most influenced the prediction, retrieves relevant scientific evidence, and converts the result into a clear, evidence-grounded explanation.

Most brain-age models stop at:

> "Your predicted brain age is 47."

NeuroDecel goes further and asks:

> **Why did the model make that prediction, and what does scientific research say about the contributing brain measurements?**

---

## 🚀 What NeuroDecel Does

NeuroDecel combines machine learning, explainability, scientific literature retrieval, and LLM-based synthesis into one pipeline:

```text
MRI-derived brain measurements
            ↓
        XGBoost
            ↓
  Predicted brain age
            ↓
     Brain-age gap
            ↓
          SHAP
            ↓
Top patient-specific features
            ↓
       Feature mapping
            ↓
          Amass
            ↓
Scientific literature retrieval
            ↓
      Claude Sonnet
            ↓
Evidence-grounded explanation
            ↓
       Streamlit UI
```

The scientific papers are **not used to make the prediction**.

The trained machine-learning model first predicts brain age. Scientific evidence is retrieved afterwards to explain and contextualize the model's most influential features.

---

## ✨ Key Features

### 🧠 Brain-Age Prediction

A trained XGBoost regression model takes approximately **270 MRI-derived structural features** and predicts brain age.

The system then calculates:

```text
Brain-age gap = predicted brain age - chronological age
```

For example:

```text
Chronological age: 54
Predicted brain age: 47.7
Brain-age gap: -6.3 years
```

A negative gap means the model estimated a younger brain age, while a positive gap means the model estimated an older brain age.

---

### 🔎 Patient-Specific Explainability

NeuroDecel uses **SHAP** to explain each individual prediction.

Instead of only showing global feature importance, the system identifies which measurements had the strongest influence on that specific person's prediction and whether they pushed the model toward:

- an **older predicted brain age**
- a **younger predicted brain age**

The strongest searchable features are then passed to the scientific evidence retrieval stage.

---

### 📚 Scientific Evidence with Amass

For the most influential MRI-derived features, NeuroDecel queries **Amass BiomedCore** for relevant biomedical literature.

Retrieved studies are filtered and ranked using factors such as:

- relevance to the brain measurement
- relevance to aging
- journal quality
- citation count
- retraction status

Healthy and normal-aging evidence is prioritized whenever possible.

---

### 🤖 Evidence-Grounded Explanations with Claude

Claude receives:

- the patient's model result
- SHAP direction
- reference-distribution information
- scientific evidence retrieved through Amass

It then generates a medically cautious, plain-language explanation containing:

- how the feature affected the model
- what the brain measurement represents
- what scientific research says about the feature and aging
- how strong the evidence is
- important limitations
- supporting scientific sources

The system explicitly separates:

```text
What the MODEL tells us
            +
What the RESEARCH tells us
```

SHAP values are treated as model influences, not biological causes.

---

## 📄 MRI Report Upload

The Streamlit interface supports structured PDF reports containing:

```text
Patient ID: MRI-DEMO-001
Patient Age: 54

RhSuperiortemporalThick: 2.7615
ThirdVentVol: 1084.32
WMHypointensitiesVol: 742.11
...
```

The uploaded report contains MRI-derived structural measurements expected by the trained model.

The full pipeline is then executed:

```text
PDF
 ↓
Feature extraction
 ↓
XGBoost
 ↓
Brain-age prediction
 ↓
SHAP
 ↓
Top contributing features
 ↓
Amass
 ↓
Scientific evidence
 ↓
Claude
 ↓
Plain-language report
```

The current prototype works with measurements that have already been extracted from MRI scans.

A future version could integrate directly with MRI-processing pipelines so that users can upload the scan itself.

---

## ⚡ Demo Mode

NeuroDecel also includes an instant demo using a precomputed anonymized patient example.

This allows users to explore the complete interface without waiting for:

- model execution
- Amass retrieval
- Claude generation

This is especially useful during live presentations where many users may open the application simultaneously.

---

## 🏗️ Project Structure

```text
brain-age-prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── dataset/
│   └── BIOMARKERDATA.xlsx
│
├── models/
│   ├── brainage_xgb_model.json
│   ├── bias_correction_model.pkl
│   ├── feature_cols.pkl
│   └── training_reference_stats.pkl
│
├── notebooks/
│   └── Braincodetest.ipynb
│
├── outputs/
│   ├── all_test_patients_explained.csv
│   ├── test_set_brain_age_gap.csv
│   ├── predicted_vs_actual_test.png
│   ├── patient_2561_amass_evidence.json
│   └── patient_2561_explanation.json
│
└── src/
    ├── __init__.py
    ├── model_service.py
    ├── pdf_parser.py
    ├── feature_mapping.py
    ├── amass_client.py
    ├── evidence_pipeline.py
    └── claude_client.py
```

---

## ⚙️ System Components

### `model_service.py`

Handles inference for new patients.

Responsibilities include:

- loading the saved XGBoost model
- validating the required MRI features
- generating the raw brain-age prediction
- applying bias correction
- calculating brain-age gap
- computing SHAP values
- ranking patient-specific contributors
- calculating reference-distribution z-scores

---

### `feature_mapping.py`

Converts internal MRI feature names into medically understandable search terms.

For example:

```text
RhSuperiortemporalThick
```

becomes:

```text
right superior temporal cortex cortical thickness
```

This allows literature retrieval to use meaningful biomedical terminology.

---

### `amass_client.py`

Handles communication with the Amass API.

Current functionality includes:

- authentication
- biomedical literature search
- retries
- rate-limit handling
- evidence filtering
- relevance scoring

The current pipeline primarily uses:

```text
Amass BiomedCore
```

---

### `evidence_pipeline.py`

Takes the top patient-specific SHAP factors and retrieves relevant scientific evidence.

Example structure:

```json
{
  "raw_feature": "ThirdVentVol",
  "medical_name": "cerebral third ventricle volume",
  "direction": "older",
  "z_score": 1.42,
  "amass": {
    "papers": []
  }
}
```

---

### `claude_client.py`

Uses Claude to convert model output and scientific evidence into understandable explanations.

Claude is instructed to:

- separate model findings from scientific evidence
- prioritize healthy-aging research
- avoid diagnosis
- avoid causal claims
- avoid interpreting z-scores as clinical abnormalities
- acknowledge weak or indirect evidence
- distinguish statistical associations from biological conclusions

---

### `pdf_parser.py`

Extracts patient information and MRI-derived measurements from structured PDFs.

The parser currently expects feature names matching those used by the trained model.

Example:

```text
Patient ID: MRI-DEMO-001
Patient Age: 54

ThirdVentVol: 1084.32
RhSuperiortemporalThick: 2.7615
WMHypointensitiesVol: 742.11
...
```

---

## 🧠 Model Artifacts

The trained model is saved so the Streamlit app does not need to retrain it.

```python
model.save_model(
    MODELS_DIR / "brainage_xgb_model.json"
)

joblib.dump(
    feature_cols,
    MODELS_DIR / "feature_cols.pkl"
)

joblib.dump(
    bias_model,
    MODELS_DIR / "bias_correction_model.pkl"
)
```

Training-distribution statistics are also saved:

```python
training_reference = {
    "mean": X_train.mean(axis=0),
    "std": X_train.std(axis=0),
}

joblib.dump(
    training_reference,
    MODELS_DIR / "training_reference_stats.pkl"
)
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
AMASS_API_KEY=your_amass_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-sonnet-5
```

Do not commit `.env`.

Add it to `.gitignore`:

```text
.env
__pycache__/
.DS_Store
```

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd brain-age-prediction
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Example dependencies:

```text
streamlit
pandas
numpy
openpyxl
xgboost
scikit-learn
shap
joblib
requests
python-dotenv
anthropic
pypdf
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
python3 -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📄 MRI Report Format

The prototype expects a text-based PDF containing:

1. patient age
2. optional patient ID
3. all required MRI-derived measurements

Example:

```text
NEUROIMAGING BIOMARKER REPORT

Patient ID: MRI-DEMO-001
Patient Age: 54

MRI STRUCTURAL MEASUREMENTS

LhBanksstsThick: 2.4123
LhCaudalanteriorcingulateThick: 2.6741
RhSuperiortemporalThick: 2.7615
ThirdVentVol: 1084.32
WMHypointensitiesVol: 742.11
...
```

All feature names must currently match the feature names stored in:

```text
models/feature_cols.pkl
```

---

## 🔬 Example Output

A NeuroDecel result may look like:

```text
Chronological age
54

Estimated brain age
47.7

Brain-age gap
6.3 years younger
```

The system then identifies features that pushed the prediction in different directions.

Example:

```text
Younger-associated

- Right superior temporal cortex thickness
- Non-white matter hypointensity volume

Older-associated

- Third ventricle volume
- White matter hypointensity volume
```

Each feature can include:

- model explanation
- anatomical explanation
- research summary
- patient-specific interpretation
- evidence strength
- supporting scientific sources
- important caution

---

## 💡 Why NeuroDecel?

Brain-age prediction has growing applications in neuroscience research, but a single predicted age is difficult to interpret.

A result such as:

> "Your predicted brain age is 7 years older than your chronological age."

does not explain:

- which measurements influenced the prediction
- whether one or many brain regions drove the result
- whether scientific research supports those relationships
- how strong the evidence is
- what limitations exist

NeuroDecel bridges that gap by combining:

```text
Prediction
+
Explainability
+
Scientific evidence retrieval
+
Grounded explanation
```

The goal is not just to produce another brain-age number.

The goal is to make brain-age prediction **more transparent, evidence-aware, and understandable**.

---

## 🌍 Future Work

Future extensions include:

- direct MRI or DICOM upload
- automated brain segmentation
- automatic MRI feature extraction
- FreeSurfer integration
- longitudinal brain-age tracking
- prediction uncertainty estimates
- broader biomedical evidence retrieval
- demographic fairness analysis
- improved evidence-ranking models
- secure clinical/research deployment
- automated downloadable reports

The future pipeline could become:

```text
Raw MRI
   ↓
Automatic segmentation
   ↓
Structural feature extraction
   ↓
Brain-age prediction
   ↓
SHAP
   ↓
Scientific evidence retrieval
   ↓
Grounded explanation
   ↓
Interactive report
```

---

## ⚠️ Disclaimer

NeuroDecel is a **research prototype**.

Brain age is a statistical estimate produced by a machine-learning model. It is not a direct measurement of biological brain age and should not be interpreted as a medical diagnosis.

SHAP values describe how features influenced the machine-learning model's prediction. They do not establish biological causation, disease, disease risk, or clinical abnormality.

Scientific literature retrieved by the system provides external context and is not used to generate the original brain-age prediction.

NeuroDecel is not intended for clinical diagnosis or medical decision-making.

---

## 🛠️ Built With

- Python
- XGBoost
- SHAP
- Pandas
- scikit-learn
- Amass
- Anthropic Claude
- Streamlit
- PyPDF

---

## 👩‍💻 Team

Built as a hackathon prototype exploring explainable and evidence-grounded AI for brain aging.

---

## One-line Summary

**NeuroDecel turns brain-age prediction from a single number into an interpretable, evidence-grounded explanation.**