# Treatment Resistance Classifier Demo

Interactive dashboard for predicting treatment resistance in depression using machine learning with uncertainty quantification.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- Random Forest classifier with ROC-AUC evaluation
- Conformal prediction for calibrated uncertainty estimates
- SHAP-based feature contribution analysis
- LLM-generated plain-language explanation grounded in SHAP and README evidence snippets
- t-SNE population visualization

## Project Structure

```
├── app.py              # Dash application entry point
├── config.py           # UI configuration and feature definitions
├── model.py            # ML model training and inference
├── visualization.py    # Plotly figure generation
├── components.py       # Dash UI components
├── callbacks.py        # Dash callback registration
├── data_synth.py       # Synthetic data generation
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Locally

Development server:

```bash
python app.py
```

Production-like (Gunicorn):

```bash
gunicorn app:server -b 0.0.0.0:8050
```

Docker:

```bash
docker build -t treatment-classifier .
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" -p 8050:8050 treatment-classifier
```

Access at `http://localhost:8050`

The LLM explanation reads the OpenAI API key from the `OPENAI_API_KEY` environment variable. The app constrains the prompt to live SHAP results plus the literature snippets and PMIDs listed below so the generated summary stays as grounded as possible.

## Literature Evidence on Predictors of Treatment Resistance

Generation of synthetic data ([data_synth.py](data_synth.py)) is based on general literature evidence of treatment resistance factors:

| Predictor                     | Association                                                                                  | References                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Baseline severity (PHQ-9)     | Higher severity → higher TR risk                                                             | Souery 2007 (PMID: 17685743) *(Papakostas 2015 removed – not relevant)*  |
| Episode duration / chronicity | Longer duration → higher TR risk                                                             | Souery 2007 (PMID: 17685743), Fekadu 2009 (PMID: 19193338)               |
| Prior treatment failures      | More failures → higher TR risk                                                               | Berlim & Turecki 2007 (PMID: 17388795), Fava 2003 (PMID: 12716236)       |
| Poor adherence                | Nonadherence → higher TR risk *(conceptual; confounder of pseudo-resistance)*                | Souery 2006 (PMID: 16727297)                                             |
| SSRI dose (sertraline)        | 50–200 mg effective; limited benefit above 100 mg *(not a predictor of TR)*                  | Hieronymus 2016 (PMID: 27052632), Cipriani 2018 (PMID: 29477251)         |
| Quetiapine augmentation       | 150–300 mg/day; modest effect, tolerability concerns *(treatment, not predictor)*            | Komossa 2010 (PMID: 20091549), Nelson & Papakostas 2009 (PMID: 19289445) |
| Lithium augmentation          | 600–900 mg/day; strong evidence for TRD *(treatment, not predictor)*                         | Nelson 2014 (PMID: 25016772), Bschor 2014 (PMID: 24792557)               |
| Early improvement (week 2)    | No early improvement → higher TR risk *(predicts non-response rather than TRD specifically)* | Stassen 2007 (PMID: 17388796), Szegedi 2009 (PMID: 19607757)             |
| Sleep disturbance / insomnia  | Insomnia → higher TR risk *(weak / indirect evidence)*                                       | Wichniak 2017 (PMID: 28427964), Dew 1997 (PMID: 9255847)                 |
| Substance use                 | Regular use → higher TR risk *(evidence exists but not directly supported by cited paper)*   | Nunes & Levin 2004 (PMID: 15033227), Fava 2003 (PMID: 12716236)          |
| Comorbid anxiety              | Anxiety → higher TR risk *(supported in TRD cohorts; Papakostas 2015 removed)*               | Souery 2007 (PMID: 17685743)                                             |
| Side effect burden            | Side effects → dose reduction → higher TR risk *(not well established as predictor)*         | Souery 2006 (PMID: 16727297)                                             |
| Sex (female vs male)          | Mixed/inconsistent evidence                                                                  | Khan 2005 (PMID: 15794786)                                               |
