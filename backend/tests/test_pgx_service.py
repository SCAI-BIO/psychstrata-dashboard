import pytest

from app.services import pgx_service
from app.services.errors import InvalidGeneticsDataError


def test_interpret_gene_normalises_diplotype_shape() -> None:
    result = pgx_service.interpret_gene("CYP2C19", "*17", "*17")

    assert result["diplotype"] == "*17/*17"
    assert result["phenotype"] == "Ultrarapid Metabolizer"
    assert result["function1"] == result["function2"] == "Increased Function"
    assert result["activity_score"] is None


def test_interpret_gene_computes_cyp2d6_activity_score() -> None:
    result = pgx_service.interpret_gene("CYP2D6", "*36+*10", "*5")

    assert result["activity_score"] == 0.25
    assert result["phenotype"] == "Intermediate Metabolizer"
    assert result["function1"] is None


def test_interpret_gene_rejects_unsupported_gene() -> None:
    with pytest.raises(InvalidGeneticsDataError):
        pgx_service.interpret_gene("CYP3A4", "*1", "*1")


def test_interpret_rejects_unlisted_gene_key() -> None:
    with pytest.raises(InvalidGeneticsDataError, match="Unsupported genes"):
        pgx_service.interpret({"SLC6A4": ("*1", "*1")})


def test_interpret_keeps_request_order_and_deduplicates_drugs() -> None:
    result = pgx_service.interpret(
        {"CYP2D6": ("*1", "*1")},
        ["ondansetron", "codeine", "ondansetron"],
    )

    drugs = [item["drug"] for item in result["drug_recommendations"]]
    assert drugs == ["Ondansetron", "Codeine"]


def test_interpret_returns_every_configured_drug_and_flags_missing_genotypes() -> None:
    result = pgx_service.interpret({"CYP2C19": ("*1", "*1")})

    drugs = result["drug_recommendations"]
    assert len(drugs) == len(pgx_service.build_options()["drugs"])
    assert len({item["drug_id"] for item in drugs}) == len(drugs)
    for item in drugs:
        if "CYP2D6" in item["required_genes"]:
            assert item["status"] == "unavailable"
            assert item["unavailable_reason"] == "CYP2D6 genotype not entered"
        else:
            assert item["status"] == "available"
            assert item["unavailable_reason"] is None


def test_build_options_is_cached_and_consistent() -> None:
    first = pgx_service.build_options()
    second = pgx_service.build_options()

    assert first is second
    cyp2d6 = next(gene for gene in first["genes"] if gene["gene"] == "CYP2D6")
    assert cyp2d6["alleles"][:3] == ["*1", "*2", "*3"]
    assert all(drug["genes"] for drug in first["drugs"])


def test_interpret_returns_verbatim_cpic_text_and_metadata() -> None:
    result = pgx_service.interpret({"CYP2C19": ("*1", "*1")}, ["escitalopram"])

    assert result["recommendation_source"] == "CPIC"
    assert result["recommendation_source_url"] == "https://api.cpicpgx.org/v1/recommendation"
    assert result["recommendation_retrieved_at"]
    recommendation = result["drug_recommendations"][0]
    assert recommendation["status"] == "available"
    assert recommendation["recommendation"] == "Initiate therapy with recommended starting dose"
    assert recommendation["recommendation_source"] == "CPIC"
    assert recommendation["cpic_classification"] == "Strong"
    assert isinstance(recommendation["cpic_recommendation_id"], int)


def test_interpret_treats_no_recommendation_rows_as_unavailable() -> None:
    result = pgx_service.interpret({"CYP2D6": ("*1", "*5")}, ["ondansetron"])

    recommendation = result["drug_recommendations"][0]
    assert recommendation["status"] == "unavailable"
    assert recommendation["recommendation"] is None
    assert recommendation["recommendation_source"] is None
    assert recommendation["unavailable_reason"] == "No CPIC guidance for this phenotype combination"
    assert recommendation["cpic_classification"] == "No Recommendation"


def test_interpret_uses_cyp2b6_no_result_rows_for_sertraline() -> None:
    result = pgx_service.interpret({"CYP2C19": ("*1", "*1")}, ["sertraline"])

    recommendation = result["drug_recommendations"][0]
    assert recommendation["required_genes"] == ["CYP2C19"]
    assert recommendation["status"] == "available"
    assert recommendation["recommendation"] == "Initiate therapy with recommended starting dose"


def test_interpret_selects_rows_by_cyp2d6_activity_score() -> None:
    lower = pgx_service.interpret({"CYP2D6": ("*10", "*10")}, ["nortriptyline"])
    higher = pgx_service.interpret({"CYP2D6": ("*10", "*10x2")}, ["nortriptyline"])

    lower_recommendation = lower["drug_recommendations"][0]
    higher_recommendation = higher["drug_recommendations"][0]
    assert lower_recommendation["cpic_classification"] == "Moderate"
    assert higher_recommendation["cpic_classification"] == "Optional"
    assert lower_recommendation["cpic_recommendation_id"] != higher_recommendation["cpic_recommendation_id"]


def test_interpret_matches_indeterminate_rows_with_na_lookup() -> None:
    result = pgx_service.interpret(
        {"CYP2D6": ("*1", "*43"), "CYP2C19": ("*1", "*1")}
    )

    recommendation = next(
        item for item in result["drug_recommendations"] if item["drug_id"] == "amitriptyline"
    )
    assert recommendation["cpic_recommendation_id"] == 8608614
    assert recommendation["cpic_classification"] == "No Recommendation"
    assert recommendation["status"] == "unavailable"
    assert recommendation["recommendation"] is None
