# Synapse AI

**Explainable brain-age estimation from MRI biomarkers**

Synapse AI estimates a person's **brain age** from structural MRI biomarkers, compares it with their chronological age, and explains which brain features are contributing most to the prediction.

The goal is not to diagnose disease. It is to turn a difficult-to-interpret result into something a person can understand and discuss with a clinician.

---

## Problem and Intended User

Medical reports are often full of jargon that is hardest to parse at the exact moment it matters most. This anxiety around scans and results is sometimes described as **scanxiety**.

Synapse AI is built for the gap between receiving a result and being able to discuss it with a clinician.

Instead of producing another unexplained medical score, it focuses on helping users understand:

- their predicted brain age
- their brain-age gap
- which MRI biomarkers influenced the result
- which features pushed the prediction older or younger
- what the result may be worth discussing with a clinician

Another key goal is **supporting earlier awareness of brain health**. Synapse AI can help people understand potential signs of accelerated brain aging earlier, giving them useful context about their brain health before meeting with a doctor. This can help users arrive at their appointment better informed and more prepared to discuss their results.

The intended user is someone who has undergone structural brain imaging and wants a clearer explanation of their result.

Synapse AI is an educational and decision-support prototype, not a diagnostic system. It is not intended to diagnose disease or replace clinical advice.

---

## What We Built and Why

Synapse AI takes structural MRI-derived biomarkers and predicts a person's **brain age** using **270 named structural MRI features**.

We calculate:

**Brain Age Gap = Predicted Brain Age - Chronological Age**

For example:

- Chronological age: 42
- Predicted brain age: 49
- Brain age gap: +7 years

Rather than stopping at the number, Synapse AI uses model explainability to identify which biomarkers contributed most strongly to the prediction.

Those contributions are then translated into plain-language explanations so the result is understandable rather than a black box.

---

## Technical Architecture and Tools Used

### Modeling

- **XGBoost** for brain-age regression
- **270 structural MRI biomarkers** as input features
- **pandas** for data handling
- **scikit-learn** for preprocessing and evaluation
- **joblib** for model persistence

### Calibration

Brain-age models can show age-related prediction bias, where younger participants are predicted older and older participants are predicted younger.

We fit a **bias correction on a held-out calibration set** and apply it to a **separate untouched test set**.

This reduces prediction bias while avoiding information leakage from the test data.

### Explainability

We use **SHAP** for per-prediction attribution.

For each prediction, SHAP identifies:

- which MRI biomarkers influenced the prediction most
- whether each biomarker pushed the predicted age higher or lower
- the relative contribution of each feature

### AI and Product Layer

- **Streamlit** — interactive web application
- **Amass** — evidence-based reasoning
- **Anthropic Claude** — plain-language explanation generation
- **ElevenLabs** — optional voice read-out

The LLM does not generate the brain-age prediction itself. It explains the output of the trained ML model and its SHAP attributions.

### Architecture

### Architecture

```text
Structural MRI biomarkers
          ↓
      XGBoost
          ↓
 Predicted brain age
          ↓
   Bias correction
          ↓
   Brain age gap
          ↓
        SHAP
          ↓
Top contributing biomarkers
          ↓
        Amass
Scientific evidence retrieval
          ↓
       Claude
Plain-language explanation
          ↓
     Streamlit
          ↓
   ElevenLabs
Optional voice output
```

**Flow:** XGBoost predicts → SHAP explains the prediction → Amass retrieves relevant scientific evidence → Claude turns the model output and evidence into a plain-language explanation → Streamlit presents the result → ElevenLabs optionally reads it aloud.

---

## Data Sources, Licences and Evidence

### Data Source

**OpenNeuro ds004856**

The dataset contains structural MRI and cognitive assessment data used for the brain-age modeling pipeline.

OpenNeuro datasets are distributed under the **CC0 public-domain dedication**.

### Evidence Layer

The explanation pipeline is designed to keep prediction and language generation separate:

```text
Model prediction
      ↓
SHAP attribution
      ↓
Relevant evidence
      ↓
Plain-language explanation
```

This reduces the risk of the language model inventing reasons that are unrelated to the underlying prediction.

---

## Working Demo

### Live Website

https://synapse-ai-lab.streamlit.app/

### Recorded Demo

A recorded 2–3 minute website demo is also included with the submission.

**Video link:** 

https://github.com/user-attachments/assets/e91acba4-f46d-4254-8967-fbaa09c142da

The demo shows:

1. entering or loading participant biomarker data
2. generating a brain-age prediction
3. calculating the brain-age gap
4. displaying the strongest SHAP contributors
5. generating a plain-language explanation
6. optional voice read-out

---

## Results and Success Metrics

Current test-set performance:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | **7.11 years** |
| Pearson correlation (r) | **0.875** |
| Held-out test set | **n = 93** |
| Validation | **5-fold cross-validation** |

The model's performance was also consistent under **5-fold cross-validation**.

### Why These Metrics?

**Mean Absolute Error (MAE)** measures the average difference between predicted brain age and chronological age.

An MAE of **7.11 years** means the model differs from chronological age by about seven years on average.

**Pearson correlation (r)** measures how strongly predicted brain age tracks chronological age across participants.

Our result of **r = 0.875** indicates a strong relationship between predicted and chronological age.

---

## Limitations, Risks and Safety Considerations

### Not Diagnostic

Synapse AI is not a diagnostic system.

It does not currently produce:

- disease risk scores
- neurological diagnoses
- treatment recommendations
- medication recommendations

Brain age should not be interpreted as a diagnosis.

### Small Held-Out Test Set

The untouched test set contains **93 participants**, which is relatively small for a medical machine-learning system.

Larger external datasets would be needed before making claims about real-world generalization.

### No Demographic Fairness Check Yet

The current dataset does not provide enough demographic information to support a robust fairness analysis.

A production system would need evaluation across factors such as:

- age groups
- sex
- ethnicity
- scanner manufacturers
- imaging sites
- geographic populations

### No Raw MRI Processing Yet

The current system works with **pre-extracted structural MRI biomarkers**.

It does not yet take raw MRI scans and automatically perform segmentation and biomarker extraction.

### Language Model Risk

Claude is used only to explain structured outputs from the brain-age model, SHAP analysis, and evidence layer.

The system follows a simple principle:

> **The ML model makes the prediction. The LLM explains the prediction.**

AI-generated explanations can still contain inaccuracies and should not replace consultation with a healthcare professional.

---

## Team Members

- **Piyusha Patil** — Computer Science Engineer
- **Shakeel J** — Medical Engineer
- **Haripriya Sampath** — Medical Engineer
- **Soundharya Y** — Medical Engineer

---

## Next Steps

### 1. Longitudinal Brain-Age Trajectories

Add support for repeated scans over time so users can track brain-age changes rather than relying on a single prediction.

```text
MRI at T1 → Brain age
MRI at T2 → Brain age
MRI at T3 → Brain age
       ↓
Brain-age trajectory
```

### 2. Next-Step Guidance Layer

Build an evidence-grounded guidance layer that helps users understand which questions or follow-up topics may be worth discussing with a clinician.

This would remain educational rather than diagnostic.

### 3. Direct MRI Ingestion

Extend the pipeline from:

```text
Extracted biomarkers → Brain age
```

to:

```text
Raw MRI
   ↓
Segmentation
   ↓
Biomarker extraction
   ↓
Brain-age model
   ↓
SHAP explanation
   ↓
Evidence-grounded explanation
```

### 4. External Validation

Evaluate Synapse AI on independent neuroimaging datasets collected at different institutions and using different MRI scanners.

### 5. Fairness Evaluation

Once sufficiently diverse datasets are available, evaluate model performance across demographic and clinical subgroups.

---

## Disclaimer

Synapse AI is a research and educational prototype.

It is **not a medical device and does not provide medical diagnosis or treatment recommendations**.

Predictions and explanations should not be used as a substitute for professional medical advice.
