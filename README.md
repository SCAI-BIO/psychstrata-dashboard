# Treatment Resistance Classifier Demo

Interactive dashboard for predicting treatment resistance in depression using machine learning with uncertainty quantification.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- Random Forest classifier with ROC-AUC evaluation
- Conformal prediction for calibrated uncertainty estimates
- SHAP-based feature contribution analysis
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
docker run -p 8050:8050 treatment-classifier
```

Access at `http://localhost:8050`

## Literature Evidence on Predictors of Treatment Resistance

Generation of synthetic data ([data_synth.py](data_synth.py)) is based on general literature evidence of treatment resistance factors:

| Predictor | Association | References |
|-----------|-------------|------------|
| Baseline severity (PHQ-9) | Higher severity → higher TR risk | [Papakostas 2015](https://pubmed.ncbi.nlm.nih.gov/26016766/), [Souery 2006](https://pubmed.ncbi.nlm.nih.gov/16727297/) |
| Episode duration / chronicity | Longer duration → higher TR risk | [Souery 2006](https://pubmed.ncbi.nlm.nih.gov/16727297/), [Fekadu 2009](https://pubmed.ncbi.nlm.nih.gov/19193338/) |
| Prior treatment failures | More failures → higher TR risk | [Berlim & Turecki 2007](https://pubmed.ncbi.nlm.nih.gov/17388795/), [Fava 2003](https://pubmed.ncbi.nlm.nih.gov/12716236/) |
| Poor adherence | Nonadherence → higher TR risk | [Souery 2006](https://pubmed.ncbi.nlm.nih.gov/16727297/) |
| SSRI dose (sertraline) | 50–200 mg effective; limited benefit above 100 mg | [Hieronymus 2016](https://pubmed.ncbi.nlm.nih.gov/27052632/), [Cipriani 2018](https://pubmed.ncbi.nlm.nih.gov/29477251/) |
| Quetiapine augmentation | 150–300 mg/day; modest effect, tolerability concerns | [Komossa 2010](https://pubmed.ncbi.nlm.nih.gov/20091549/), [Nelson & Papakostas 2009](https://pubmed.ncbi.nlm.nih.gov/19289445/) |
| Lithium augmentation | 600–900 mg/day; strong evidence for TRD | [Nelson 2014](https://pubmed.ncbi.nlm.nih.gov/25016772/), [Bschor 2014](https://pubmed.ncbi.nlm.nih.gov/24792557/) |
| Early improvement (week 2) | No early improvement → higher TR risk | [Stassen 2007](https://pubmed.ncbi.nlm.nih.gov/17388796/), [Szegedi 2009](https://pubmed.ncbi.nlm.nih.gov/19607757/) |
| Sleep disturbance / insomnia | Insomnia → higher TR risk | [Wichniak 2017](https://pubmed.ncbi.nlm.nih.gov/28427964/), [Dew 1997](https://pubmed.ncbi.nlm.nih.gov/9255847/) |
| Substance use | Regular use → higher TR risk | [Nunes & Levin 2004](https://pubmed.ncbi.nlm.nih.gov/15033227/), [Fava 2003](https://pubmed.ncbi.nlm.nih.gov/12716236/) |
| Comorbid anxiety | Anxiety → higher TR risk | [Papakostas 2015](https://pubmed.ncbi.nlm.nih.gov/26016766/) |
| Side effect burden | Side effects → dose reduction → higher TR risk | [Souery 2006](https://pubmed.ncbi.nlm.nih.gov/16727297/) |
| Sex (female vs male) | Mixed/inconsistent evidence | [Khan 2005](https://pubmed.ncbi.nlm.nih.gov/15794786/), [Papakostas 2015](https://pubmed.ncbi.nlm.nih.gov/26016766/) |
