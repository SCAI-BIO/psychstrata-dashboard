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


def test_interpret_returns_only_drugs_with_a_provided_gene() -> None:
    result = pgx_service.interpret({"CYP2C19": ("*1", "*1")})

    assert {item["drug"] for item in result["drug_recommendations"]} == {
        "Escitalopram",
        "Citalopram",
        "Sertraline",
        "Amitriptyline",
        "Clomipramine",
        "Doxepin",
        "Imipramine",
        "Trimipramine",
    }
    for item in result["drug_recommendations"]:
        if item["required_genes"] == ["CYP2C19"]:
            assert item["status"] == "available"
        else:
            assert item["status"] == "unavailable"
            assert item["unavailable_reason"] == "Missing CYP2D6 genotype"


def test_build_options_is_cached_and_consistent() -> None:
    first = pgx_service.build_options()
    second = pgx_service.build_options()

    assert first is second
    cyp2d6 = next(gene for gene in first["genes"] if gene["gene"] == "CYP2D6")
    assert cyp2d6["alleles"][:3] == ["*1", "*2", "*3"]
    assert all(drug["genes"] for drug in first["drugs"])
