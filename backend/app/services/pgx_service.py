"""Pharmacogenetic interpretation built on pypgx and CPIC guidance."""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Any

import pypgx

from .errors import InvalidGeneticsDataError

SUPPORTED_GENES = ("CYP2C19", "CYP2D6")

COMMON_ALLELES: dict[str, tuple[str, ...]] = {
    "CYP2C19": ("*1", "*2", "*3", "*4", "*5", "*6", "*7", "*8", "*9", "*10", "*17"),
    "CYP2D6": (
        "*1", "*2", "*3", "*4", "*5", "*6", "*7", "*8", "*9", "*10", "*11",
        "*14", "*15", "*17", "*19", "*20", "*29", "*35", "*36", "*39", "*40",
        "*41", "*43", "*45", "*46", "*56", "*61", "*63", "*64", "*68", "*69",
        "*71", "*82", "*100", "*101", "*114",
        "*1x2", "*2x2", "*4x2", "*10x2", "*36+*10",
    ),
}

DRUG_CATALOGUE: tuple[str, ...] = (
    "escitalopram",
    "citalopram",
    "sertraline",
    "fluvoxamine",
    "paroxetine",
    "amitriptyline",
    "clomipramine",
    "doxepin",
    "imipramine",
    "trimipramine",
    "desipramine",
    "nortriptyline",
    "atomoxetine",
    "codeine",
    "tramadol",
    "ondansetron",
)

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
    if gene not in SUPPORTED_GENES:
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
        "drug": drug.title(),
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


def interpret(
    diplotypes: dict[str, tuple[str, str]],
    drugs: list[str] | None = None,
) -> dict[str, Any]:
    if not diplotypes:
        raise InvalidGeneticsDataError("Provide at least one gene diplotype.")
    unknown = set(diplotypes) - set(SUPPORTED_GENES)
    if unknown:
        raise InvalidGeneticsDataError(f"Unsupported genes: {', '.join(sorted(unknown))}")

    gene_results = [interpret_gene(gene, *diplotypes[gene]) for gene in SUPPORTED_GENES if gene in diplotypes]
    genes_by_name = {result["gene"]: result for result in gene_results}

    selected = list(DRUG_CATALOGUE)
    if drugs is not None:
        selected = []
        for drug in drugs:
            key = drug.strip().lower()
            if not key:
                continue
            if key not in DRUG_CATALOGUE:
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


@lru_cache(maxsize=1)
def build_options() -> dict[str, Any]:
    genes = [
        {
            "gene": gene,
            "alleles": list(_sorted_alleles(gene)),
            "common_alleles": list(COMMON_ALLELES[gene]),
            "phenotypes": [str(phenotype) for phenotype in pypgx.list_phenotypes(gene)],
        }
        for gene in SUPPORTED_GENES
    ]
    drugs = [
        {"drug": drug, "label": drug.title(), "genes": list(_drug_genes(drug))}
        for drug in DRUG_CATALOGUE
    ]
    return {"genes": genes, "drugs": drugs}
