import json

import numpy as np

from app.services.llm_summary_service import build_llm_prompt


def _values() -> dict:
    return {"phq9": 20, "duration_months": 24, "previous_failures": 3}


def test_prompt_instructs_inline_pmid_citation_format() -> None:
    shap_vals = np.array([0.5, -0.3, 0.2])

    prompt = build_llm_prompt(_values(), 0.7, shap_vals, list(_values()))

    rules = json.loads(prompt)["rules"]
    assert any("PMID: <id>" in rule for rule in rules)
