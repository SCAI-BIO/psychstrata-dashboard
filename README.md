# Treatment Resistance Classifier Demo

Interactive dashboard for predicting treatment resistance in depression using machine learning with uncertainty quantification.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- Random Forest classifier with ROC-AUC evaluation
- Conformal prediction for calibrated uncertainty estimates
- SHAP-based feature contribution analysis
- Plain-language explanation with literature citations
- Role-based navigation from a main page (Patient vs Clinician views)
- Password-protected dashboard access
- REST API for predictions and SHAP values
- t-SNE population visualization

## Project Structure

```
├── app.py              # Dash application entry point
├── auth.py             # Session-based login protection for the dashboard
├── api.py              # REST API endpoints for prediction and SHAP access
├── callbacks.py        # Dash callback registration
├── components.py       # Dash UI components
├── config.py           # UI configuration and feature definitions
├── data_synth.py       # Synthetic data generation
├── llm_summary.py      # Prompt construction and LLM summary generation
├── model.py            # ML model training and inference
├── visualization.py    # Plotly figure generation
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Locally

Quickstart:

```bash
export APP_PASSWORD=change-me
export OPENAI_API_KEY=your-api-key
python app.py
```

Open `http://localhost:8050`, sign in with `APP_PASSWORD`, and choose a role from the main page:

- **Patient view**: treatment-resistance prediction + plain-language explanation
- **Clinician view**: full dashboard including SHAP and t-SNE visualizations

Development server:

```bash
export APP_PASSWORD=change-me
export OPENAI_API_KEY=your-api-key
python app.py
```

Production-like (Gunicorn):

```bash
export APP_PASSWORD=change-me
export OPENAI_API_KEY=your-api-key
gunicorn app:server -b 0.0.0.0:8050
```

Docker:

```bash
docker build -t treatment-classifier .
docker run \
  -e APP_PASSWORD=change-me \
  -e APP_SESSION_SECRET=change-me-too \
  -e OPENAI_API_KEY=your-api-key \
  -p 8050:8050 \
  treatment-classifier
```

Access at `http://localhost:8050`

Set `APP_PASSWORD` in the container environment to protect the dashboard with a simple login page. For stable login sessions across restarts, you can also set `APP_SESSION_SECRET`. Set `OPENAI_API_KEY` to enable the explanation component. In Kubernetes, these values should be provided through secrets. The browser dashboard is gated by the login page, while `/api/*` remains available for service-to-service calls.

## REST API

The app exposes a small JSON API on the same server:

- `GET /api/health` - basic health check
- `GET /api/features` - feature schema, defaults, ranges, and categorical options
- `POST /api/predict` - prediction and SHAP values for a feature selection

Example request:

```bash
curl -X POST http://localhost:8050/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 40,
      "sex_female": 0,
      "phq9": 18,
      "duration_months": 6,
      "previous_failures": 1,
      "adherence_pct": 80,
      "sertraline_mg": 100,
      "quetiapine_mg": 0,
      "lithium_mg": 0,
      "early_improvement": 0,
      "sleep_severity": 1,
      "substance_use": 0,
      "comorbid_anxiety": 0,
      "side_effects": 1
    }
  }'
```

`POST /api/predict` requires all model features and returns `400` for missing, unknown, or out-of-range values.

## Literature Evidence on Predictors of Treatment Resistance

Generation of synthetic data ([data_synth.py](data_synth.py)) is based on general literature evidence of treatment resistance factors:

| Predictor                     | Association                                                                                  | References                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Baseline severity (PHQ-9)     | Higher severity → higher TR risk                                                             | Souery 2007 (PMID: 17685743)                                             |
| Episode duration / chronicity | Longer current episode duration is part of higher TR staging / burden                        | Fekadu 2009 (PMID: 19192471)                                             |
| Prior treatment failures      | Multiple adequate prior antidepressant failures define or increase TR staging                | Berlim & Turecki 2007 (PMID: 17444078), Fekadu 2009 (PMID: 19192471)     |
| Poor adherence                | Nonadherence can create apparent resistance *(pseudo-resistance rather than a direct predictor)* | Sackeim 2001 (PMID: 11480879), Steegen 2021 (PMID: 33779973)         |
| SSRI dose (sertraline)        | Lower licensed SSRI dose ranges tend to balance efficacy and tolerability best *(treatment, not predictor)* | Furukawa 2019 (PMID: 31178367), Cipriani 2018 (PMID: 29477251) |
| Quetiapine augmentation       | Quetiapine augmentation improves response/remission in difficult-to-treat depression *(treatment, not predictor)* | Nuñez 2022 (PMID: 34986373), Yan 2022 (PMID: 35993319)        |
| Lithium augmentation          | Lithium augmentation is evidence-supported in inadequate antidepressant response *(treatment, not predictor)* | Bschor 2014 (PMID: 24825489), Nuñez 2022 (PMID: 34986373)      |
| Early improvement (week 2)    | No early improvement predicts poorer later response/remission *(not TRD-specific)*           | Szegedi 2009 (PMID: 19254516)                                            |
| Sleep disturbance / insomnia  | Sleep disturbance is common in depression and may complicate treatment response *(weak / indirect evidence for TR)* | Wichniak 2012 (PMID: 22681161), Wichniak 2017 (PMID: 28791566) |
| Substance use                 | Comorbid substance use complicates depression treatment; direct evidence for TR prediction is limited | Nunes et al. 2004 (PMID: 15100209)                                 |
| Comorbid anxiety              | Anxiety → higher TR risk *(supported in TRD cohorts)*                                        | Souery 2007 (PMID: 17685743)                                             |
| Side effect burden            | Side effects can impair tolerability/adherence, but are not well established as a direct TR predictor | Furukawa 2019 (PMID: 31178367), Steegen 2021 (PMID: 33779973)  |
| Sex (female vs male)          | Sex differences in antidepressant response exist, but are not large enough to guide care alone | Khan 2005 (PMID: 16012273)                                            |
