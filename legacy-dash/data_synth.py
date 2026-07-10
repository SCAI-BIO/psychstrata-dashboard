from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

@dataclass
class FeatureConfig:
    id: str
    label: str
    kind: str  # "numeric" or "categorical"
    default: Any
    params: Dict[str, Any]  # For numeric: min, max, step; For categorical: options (list of dicts)

def generate_synthetic_dataset(n: int = 2500, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)

    # Distributions for features
    age = rng.integers(18, 81, size=n)
    sex = rng.choice([0, 1], p=[0.5, 0.5], size=n)  # 0=Male, 1=Female
    phq9 = rng.integers(0, 28, size=n)  # depression severity
    duration_months = rng.integers(0, 61, size=n)  # current episode length
    previous_failures = rng.integers(0, 6, size=n)
    adherence_pct = np.clip(rng.normal(80, 20, size=n), 0, 100).astype(int)

    # Medication doses (mg)
    sertraline_mg = np.clip(np.round(rng.normal(100, 40, size=n)), 0, 200).astype(int)
    quetiapine_mg = np.clip((rng.choice([0, 50, 100, 150, 200, 300], size=n, p=[0.5, 0.15, 0.12, 0.1, 0.08, 0.05])), 0, 300)
    lithium_mg = np.clip((rng.choice([0, 300, 600, 900, 1200], size=n, p=[0.7, 0.08, 0.1, 0.07, 0.05])), 0, 1200)

    early_improvement = rng.choice([0, 1], size=n, p=[0.6, 0.4])  # improvement by week 2
    sleep_severity = rng.choice([0, 1, 2], size=n, p=[0.3, 0.45, 0.25])  # None/Mild/Severe
    substance_use = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1])  # None/Occasional/Regular
    comorbid_anxiety = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    side_effects = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.4, 0.2, 0.05])

    # Latent risk model (log-odds) for treatment resistance (higher -> more resistant)
    intercept = -0.9  # tuned for ~30-35% resistance

    age_eff = 0.02 * (age - 45) / 10.0
    phq_eff = 0.12 * phq9
    duration_eff = 0.06 * (duration_months / 6.0)  # per half-year
    prevfail_eff = 0.45 * previous_failures
    adherence_eff = -0.01 * adherence_pct

    # Dose-response with diminishing returns
    ssri_eff = -0.01 * np.minimum(sertraline_mg, 100) - 0.002 * np.maximum(sertraline_mg - 100, 0)
    quet_eff = -0.005 * np.minimum(quetiapine_mg, 150)
    li_eff = -0.0008 * np.minimum(lithium_mg, 1200)

    early_imp_eff = -1.4 * early_improvement
    sleep_eff = 0.55 * sleep_severity
    substance_eff = 0.65 * substance_use
    anxiety_eff = 0.45 * comorbid_anxiety
    sidefx_eff = 0.25 * side_effects
    sex_eff = 0.05 * (sex == 1).astype(int)  # tiny effect

    logit = (
        intercept + age_eff + phq_eff + duration_eff + prevfail_eff + adherence_eff
        + ssri_eff + quet_eff + li_eff + early_imp_eff + sleep_eff + substance_eff
        + anxiety_eff + sidefx_eff + sex_eff
    )
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob, size=n)

    data = pd.DataFrame({
        "age": age,
        "sex_female": sex,
        "phq9": phq9,
        "duration_months": duration_months,
        "previous_failures": previous_failures,
        "adherence_pct": adherence_pct,
        "sertraline_mg": sertraline_mg,
        "quetiapine_mg": quetiapine_mg,
        "lithium_mg": lithium_mg,
        "early_improvement": early_improvement,
        "sleep_severity": sleep_severity,
        "substance_use": substance_use,
        "comorbid_anxiety": comorbid_anxiety,
        "side_effects": side_effects,
    })
    return data, pd.Series(y, name="treatment_resistant")