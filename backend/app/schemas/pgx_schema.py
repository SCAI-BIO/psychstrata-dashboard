from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DiplotypeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allele1: str = Field(min_length=1)
    allele2: str = Field(min_length=1)


class PgxInterpretRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cyp2c19: DiplotypeInput | None = None
    cyp2d6: DiplotypeInput | None = None
    drugs: list[str] | None = None


class GeneInterpretation(BaseModel):
    gene: str
    allele1: str
    allele2: str
    diplotype: str
    phenotype: str
    activity_score: float | None
    function1: str | None
    function2: str | None
    priority: str | None
    is_indeterminate: bool


class DrugRecommendation(BaseModel):
    drug: str
    required_genes: list[str]
    provided_genes: list[str]
    status: Literal["available", "unavailable"]
    recommendation: str | None
    unavailable_reason: str | None
    guideline: str | None
    cpic_level: str | None
    cpic_level_status: str | None
    pmids: list[str]


class PgxInterpretResponse(BaseModel):
    genes: list[GeneInterpretation]
    drug_recommendations: list[DrugRecommendation]
    disclaimer: str


class GeneOptions(BaseModel):
    gene: str
    alleles: list[str]
    common_alleles: list[str]
    phenotypes: list[str]


class DrugOption(BaseModel):
    drug: str
    label: str
    genes: list[str]


class PgxOptionsResponse(BaseModel):
    genes: list[GeneOptions]
    drugs: list[DrugOption]
