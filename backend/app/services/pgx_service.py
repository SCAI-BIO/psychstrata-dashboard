"""Pharmacogenetic interpretation built on pypgx and CPIC guidance."""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Any

import pypgx

from ..domain.genetics import DrugDefinition, GeneDefinition
from ..io.genetics_loader import (
    get_configured_drug_ids,
    get_configured_drugs,
    get_configured_gene_ids,
    get_configured_genes,
)
from .errors import InvalidGeneticsDataError

DISCLAIMER = (
    "Pharmacogenetic interpretation based on CPIC guidance. Synthetic demo data only; "
    "not a medical device and not a substitute for clinical judgement."
)

_CPIC_LEVEL_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


@lru_cache(maxsize=None)
def _allele_set(gene: str) -> frozenset[str]:
    return frozenset(pypgx.list_alleles(gene))


def _is_valid_allele(gene: str, allele: str) -> bool:
    if allele in _allele_set(gene):
        return True
    if gene != "CYP2D6":
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = float(pypgx.predict_score(gene, allele))
    except Exception:
        return False
    return not math.isnan(score)


@lru_cache(maxsize=None)
def _drug_genes(drug: str) -> tuple[str, ...]:
    table = pypgx.load_recommendation_table()
    rows = table[table.Drug == drug]
    if rows.empty:
        raise InvalidGeneticsDataError(f"Unsupported drug: {drug!r}")
    genes: list[str] = []
    for gene in [*rows.Gene1.unique().tolist(), *rows.Gene2.unique().tolist()]:
        if gene != "None" and gene not in genes:
            genes.append(gene)
    return tuple(genes)


@lru_cache(maxsize=None)
def _cpic_metadata(drug: str, genes: tuple[str, ...]) -> dict[str, Any]:
    table = pypgx.load_cpic_table()
    rows = table[(table.Drug == drug) & (table.Gene.isin(genes))]
    if rows.empty:
        return {"guideline": None, "cpic_level": None, "cpic_level_status": None, "pmids": []}

    guidelines = [url for url in rows.Guideline.unique().tolist() if isinstance(url, str) and url]
    levels = [level for level in rows.CPICLevel.unique().tolist() if isinstance(level, str) and level]
    statuses = [status for status in rows.CPICLevelStatus.unique().tolist() if isinstance(status, str) and status]
    pmids: list[str] = []
    for value in rows.PMID.tolist():
        if not isinstance(value, str):
            continue
        for pmid in value.replace(";", ",").split(","):
            pmid = pmid.strip()
            if pmid and pmid not in pmids:
                pmids.append(pmid)

    return {
        "guideline": guidelines[0] if guidelines else None,
        "cpic_level": sorted(levels, key=lambda level: _CPIC_LEVEL_RANK.get(level, 9))[0] if levels else None,
        "cpic_level_status": statuses[0] if statuses else None,
        "pmids": pmids,
    }


def _optional_function(gene: str, allele: str) -> str | None:
    if allele not in _allele_set(gene):
        return None
    try:
        return str(pypgx.get_function(gene, allele))
    except Exception:
        return None


@lru_cache(maxsize=512)
def interpret_gene(gene: str, allele1: str, allele2: str) -> dict[str, Any]:
    if gene not in get_configured_gene_ids():
        raise InvalidGeneticsDataError(f"Unsupported gene: {gene!r}")
    for allele in (allele1, allele2):
        if not _is_valid_allele(gene, allele):
            raise InvalidGeneticsDataError(f"Unknown {gene} allele: {allele!r}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phenotype = str(pypgx.predict_phenotype(gene, allele1, allele2))
        score = float(pypgx.predict_score(gene, allele1)) + float(pypgx.predict_score(gene, allele2))
        priority = pypgx.get_priority(gene, phenotype)

    return {
        "gene": gene,
        "allele1": allele1,
        "allele2": allele2,
        "diplotype": f"{allele1}/{allele2}",
        "phenotype": phenotype,
        "activity_score": None if math.isnan(score) else score,
        "function1": _optional_function(gene, allele1),
        "function2": _optional_function(gene, allele2),
        "priority": priority if isinstance(priority, str) and priority else None,
        "is_indeterminate": phenotype == "Indeterminate",
    }


@lru_cache(maxsize=512)
def _recommend(
    drug: str,
    gene1: str,
    phenotype1: str,
    gene2: str | None = None,
    phenotype2: str | None = None,
) -> str | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = pypgx.get_recommendation(drug, gene1, phenotype1, gene2, phenotype2)
    except (KeyError, IndexError, ValueError) as exc:
        raise InvalidGeneticsDataError(f"No recommendation for {drug}: {exc}") from exc
    text = str(text).strip()
    return None if not text or text == "None" else text


def _interpret_drug(drug: str, genes: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    required = _drug_genes(drug)
    provided = [gene for gene in required if gene in genes]
    if not provided:
        return None

    recommendation: str | None
    unavailable_reason: str | None = None
    if len(provided) < len(required):
        missing = [gene for gene in required if gene not in genes]
        recommendation = None
        unavailable_reason = f"Missing {' and '.join(missing)} genotype"
    else:
        if len(required) > 1:
            recommendation = _recommend(
                drug,
                required[0],
                genes[required[0]]["phenotype"],
                required[1],
                genes[required[1]]["phenotype"],
            )
        else:
            recommendation = _recommend(drug, required[0], genes[required[0]]["phenotype"])
        if recommendation is None:
            unavailable_reason = "No CPIC guidance for this phenotype combination"

    metadata = _cpic_metadata(drug, required)
    return {
        "drug": _drug_label(drug),
        "required_genes": list(required),
        "provided_genes": provided,
        "status": "available" if recommendation else "unavailable",
        "recommendation": recommendation,
        "unavailable_reason": unavailable_reason,
        "guideline": metadata["guideline"],
        "cpic_level": metadata["cpic_level"],
        "cpic_level_status": metadata["cpic_level_status"],
        "pmids": metadata["pmids"],
    }


def _drug_label(drug: str) -> str:
    for definition in get_configured_drugs():
        if definition.id == drug:
            return definition.label
    return drug.title()


def interpret(
    diplotypes: dict[str, tuple[str, str]],
    drugs: list[str] | None = None,
) -> dict[str, Any]:
    if not diplotypes:
        raise InvalidGeneticsDataError("Provide at least one gene diplotype.")
    configured_genes = get_configured_gene_ids()
    unknown = set(diplotypes) - set(configured_genes)
    if unknown:
        raise InvalidGeneticsDataError(f"Unsupported genes: {', '.join(sorted(unknown))}")

    gene_results = [interpret_gene(gene, *diplotypes[gene]) for gene in configured_genes if gene in diplotypes]
    genes_by_name = {result["gene"]: result for result in gene_results}

    configured_drugs = get_configured_drug_ids()
    selected = list(configured_drugs)
    if drugs is not None:
        selected = []
        for drug in drugs:
            key = drug.strip().lower()
            if not key:
                continue
            if key not in configured_drugs:
                raise InvalidGeneticsDataError(f"Unsupported drug: {drug!r}")
            if key not in selected:
                selected.append(key)

    recommendations = [
        result
        for drug in selected
        if (result := _interpret_drug(drug, genes_by_name)) is not None
    ]

    return {
        "genes": gene_results,
        "drug_recommendations": recommendations,
        "disclaimer": DISCLAIMER,
    }


@lru_cache(maxsize=None)
def _sorted_alleles(gene: str) -> tuple[str, ...]:
    return tuple(str(allele) for allele in pypgx.sort_alleles(list(pypgx.list_alleles(gene)), by="name"))


_ALLOWED_GENETICS_FIELDS = {"available", "genes"}
_ALLOWED_DIPLOTYPE_FIELDS = {"allele1", "allele2"}


def validate_genetics(values: dict[str, Any]) -> dict[str, Any]:
    """Validate a patient genetics payload against the configured genes and alleles."""
    if not isinstance(values, dict):
        raise InvalidGeneticsDataError("Genetics data must be an object.")
    unknown_fields = sorted(set(values) - _ALLOWED_GENETICS_FIELDS)
    if unknown_fields:
        raise InvalidGeneticsDataError(f"Unknown genetics fields: {', '.join(unknown_fields)}")

    available = values.get("available", False)
    if not isinstance(available, bool):
        raise InvalidGeneticsDataError("Genetics availability must be a boolean.")

    raw_genes = values.get("genes", {})
    if not isinstance(raw_genes, dict):
        raise InvalidGeneticsDataError("Genetics genes must be an object.")
    unknown_genes = sorted(set(raw_genes) - set(get_configured_gene_ids()))
    if unknown_genes:
        raise InvalidGeneticsDataError(f"Unsupported genes: {', '.join(unknown_genes)}")

    genes: dict[str, dict[str, str]] = {}
    for gene, raw_diplotype in raw_genes.items():
        if not isinstance(raw_diplotype, dict):
            raise InvalidGeneticsDataError(f"{gene} diplotype must be an object.")
        unknown_diplotype_fields = sorted(set(raw_diplotype) - _ALLOWED_DIPLOTYPE_FIELDS)
        if unknown_diplotype_fields:
            raise InvalidGeneticsDataError(
                f"Unknown {gene} diplotype fields: {', '.join(unknown_diplotype_fields)}"
            )
        diplotype: dict[str, str] = {}
        for field in ("allele1", "allele2"):
            allele = raw_diplotype.get(field)
            if not isinstance(allele, str) or not allele.strip():
                raise InvalidGeneticsDataError(f"{gene} {field} must be a non-empty string.")
            allele = allele.strip()
            if not _is_valid_allele(gene, allele):
                raise InvalidGeneticsDataError(f"Unknown {gene} allele: {allele!r}")
            diplotype[field] = allele
        genes[gene] = diplotype

    return {"available": available, "genes": genes}


def _gene_options(definition: GeneDefinition) -> dict[str, Any]:
    try:
        alleles = list(_sorted_alleles(definition.id))
        phenotypes = [str(phenotype) for phenotype in pypgx.list_phenotypes(definition.id)]
    except Exception as exc:
        raise RuntimeError(f"Genetics config gene {definition.id!r} is not supported by pypgx.") from exc
    unknown = [allele for allele in definition.common_alleles if not _is_valid_allele(definition.id, allele)]
    if unknown:
        raise RuntimeError(
            f"Genetics config gene {definition.id!r} has unknown alleles: {', '.join(unknown)}."
        )
    return {
        "gene": definition.id,
        "label": definition.label,
        "alleles": alleles,
        "common_alleles": list(definition.common_alleles),
        "phenotypes": phenotypes,
    }


def _drug_options(definition: DrugDefinition) -> dict[str, Any]:
    try:
        genes = list(_drug_genes(definition.id))
    except InvalidGeneticsDataError as exc:
        raise RuntimeError(
            f"Genetics config drug {definition.id!r} is not supported by pypgx."
        ) from exc
    return {"drug": definition.id, "label": definition.label, "genes": genes}


@lru_cache(maxsize=1)
def build_options() -> dict[str, Any]:
    return {
        "genes": [_gene_options(definition) for definition in get_configured_genes()],
        "drugs": [_drug_options(definition) for definition in get_configured_drugs()],
    }
