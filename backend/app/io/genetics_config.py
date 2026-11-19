import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..domain.genetics import DrugDefinition, GeneDefinition


class _GeneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    common_alleles: list[str] = Field(min_length=1)

    @field_validator("id", "label")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty.")
        return stripped

    @field_validator("common_alleles")
    @classmethod
    def _validate_alleles(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("common_alleles must not contain empty values.")
        if len(stripped) != len(set(stripped)):
            raise ValueError("common_alleles must not contain duplicates.")
        return stripped


class _DrugInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)

    @field_validator("id", "label")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty.")
        return stripped


class _GeneticsConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genes: list[_GeneInput] = Field(min_length=1)
    drugs: list[_DrugInput] = Field(min_length=1)


def load_genetics_config(path: Path) -> tuple[tuple[GeneDefinition, ...], tuple[DrugDefinition, ...]]:
    with path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    try:
        payload = _GeneticsConfigInput.model_validate(raw_payload)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid genetics configuration in {path}: {exc}") from exc

    gene_ids = [gene.id for gene in payload.genes]
    if len(gene_ids) != len(set(gene_ids)):
        raise RuntimeError("Genetics config contains duplicate gene ids.")
    drug_ids = [drug.id for drug in payload.drugs]
    if len(drug_ids) != len(set(drug_ids)):
        raise RuntimeError("Genetics config contains duplicate drug ids.")

    genes = tuple(
        GeneDefinition(id=gene.id, label=gene.label, common_alleles=tuple(gene.common_alleles))
        for gene in payload.genes
    )
    drugs = tuple(DrugDefinition(id=drug.id, label=drug.label) for drug in payload.drugs)
    return genes, drugs
