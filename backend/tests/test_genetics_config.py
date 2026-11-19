import json
from pathlib import Path

import pytest

from app.io.genetics_config import load_genetics_config


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "genetics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _genes() -> list[dict]:
    return [
        {"id": "CYP2C19", "label": "CYP2C19", "common_alleles": ["*1", "*2"]},
        {"id": "CYP2D6", "label": "CYP2D6", "common_alleles": ["*1", "*4"]},
    ]


def _drugs() -> list[dict]:
    return [
        {"id": "sertraline", "label": "Sertraline"},
        {"id": "amitriptyline", "label": "Amitriptyline"},
    ]


def test_load_genetics_config_returns_definitions(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"genes": _genes(), "drugs": _drugs()})

    genes, drugs = load_genetics_config(path)

    assert [gene.id for gene in genes] == ["CYP2C19", "CYP2D6"]
    assert genes[0].label == "CYP2C19"
    assert genes[1].common_alleles == ("*1", "*4")
    assert [drug.id for drug in drugs] == ["sertraline", "amitriptyline"]
    assert drugs[0].label == "Sertraline"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {"genes": [_genes()[0], _genes()[0]], "drugs": _drugs()},
            "duplicate gene ids",
        ),
        (
            {"genes": _genes(), "drugs": [_drugs()[0], _drugs()[0]]},
            "duplicate drug ids",
        ),
        (
            {
                "genes": [{"id": "CYP2C19", "label": "CYP2C19", "common_alleles": ["*1", "*1"]}],
                "drugs": _drugs(),
            },
            "common_alleles must not contain duplicates",
        ),
        (
            {"genes": [], "drugs": _drugs()},
            "Invalid genetics configuration",
        ),
        (
            {"genes": _genes(), "drugs": []},
            "Invalid genetics configuration",
        ),
        (
            {
                "genes": [{"id": "CYP2C19", "label": "CYP2C19", "common_alleles": ["*1"], "extra": 1}],
                "drugs": _drugs(),
            },
            "Invalid genetics configuration",
        ),
        (
            {
                "genes": [{"id": "CYP2C19", "label": "  ", "common_alleles": ["*1"]}],
                "drugs": _drugs(),
            },
            "Invalid genetics configuration",
        ),
    ],
)
def test_load_genetics_config_rejects_invalid_payloads(tmp_path: Path, payload: dict, match: str) -> None:
    path = _write_config(tmp_path, payload)

    with pytest.raises(RuntimeError, match=match):
        load_genetics_config(path)
