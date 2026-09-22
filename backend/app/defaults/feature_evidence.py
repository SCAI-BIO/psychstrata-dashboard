from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEvidence:
    association: str
    pmids: list[str]
    note: str = ""


FEATURE_EVIDENCE: dict[str, FeatureEvidence] = {
    "phq9": FeatureEvidence(
        association="Higher baseline depression severity is associated with higher treatment resistance risk.",
        pmids=["17685743"],
    ),
    "duration_months": FeatureEvidence(
        association="Longer current episode duration is used in staging higher treatment resistance burden.",
        pmids=["19192471"],
    ),
    "previous_failures": FeatureEvidence(
        association="Multiple adequate prior antidepressant failures define or increase treatment resistance staging.",
        pmids=["17444078", "19192471"],
    ),
    "adherence_pct": FeatureEvidence(
        association="Poor adherence can create apparent treatment resistance and may reflect pseudo-resistance.",
        pmids=["11480879", "33779973"],
        note="This is conceptual evidence about pseudo-resistance rather than a direct treatment-resistance predictor.",
    ),
    "sertraline_mg": FeatureEvidence(
        association="Lower licensed SSRI dose ranges tend to balance efficacy and tolerability best in acute depression treatment, but dose is not a treatment-resistance predictor.",
        pmids=["31178367", "29477251"],
        note="Treatment dosing evidence rather than predictor evidence.",
    ),
    "quetiapine_mg": FeatureEvidence(
        association="Quetiapine augmentation improves response or remission in difficult-to-treat depression, but it is a treatment variable rather than a predictor.",
        pmids=["34986373", "35993319"],
        note="Treatment evidence rather than predictor evidence.",
    ),
    "lithium_mg": FeatureEvidence(
        association="Lithium augmentation is evidence-supported in inadequate antidepressant response, but it is a treatment variable rather than a predictor.",
        pmids=["24825489", "34986373"],
        note="Treatment evidence rather than predictor evidence.",
    ),
    "early_improvement": FeatureEvidence(
        association="Lack of early improvement is associated with higher later non-response risk.",
        pmids=["19254516"],
        note="This is stronger for later non-response than for treatment resistance specifically.",
    ),
    "sleep_severity": FeatureEvidence(
        association="Sleep disturbance is common in depression and can complicate treatment response, but evidence as a treatment-resistance predictor is weak and indirect.",
        pmids=["22681161", "28791566"],
        note="Evidence is weaker and more indirect than for core resistance predictors.",
    ),
    "substance_use": FeatureEvidence(
        association="Comorbid substance use can complicate depression treatment, but direct evidence for treatment-resistance prediction is limited.",
        pmids=["15100209"],
        note="This is indirect evidence from co-occurring depression and substance-use treatment outcomes.",
    ),
    "comorbid_anxiety": FeatureEvidence(
        association="Comorbid anxiety is associated with higher treatment resistance risk.",
        pmids=["17685743"],
    ),
    "side_effects": FeatureEvidence(
        association="Side effect burden may contribute to dose reduction or poor adherence, but it is not well established as a direct predictor of treatment resistance.",
        pmids=["31178367", "33779973"],
        note="Indirect evidence rather than a well-established predictor.",
    ),
    "sex_at_birth": FeatureEvidence(
        association="Sex differences in antidepressant response exist, but they are not large enough to guide care alone and do not establish a strong standalone treatment-resistance predictor.",
        pmids=["16012273"],
        note="Use cautious wording because this is about antidepressant response differences, not a direct treatment-resistance predictor.",
    ),
}
