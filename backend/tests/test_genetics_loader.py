import json
from pathlib import Path

import pytest

from app.io import genetics_loader
from app.io.genetics_loader import (
    _reset_genetics_loader_for_tests,
    genetics_source,
    get_configured_drug_ids,
    get_configured_drugs,
    get_configured_gene_ids,
    get_configured_genes,
)
from app.services import pgx_service
from app.services.errors import InvalidGeneticsDataError
from app.settings import _reset_backend_settings_for_tests


@pytest.fixture(autouse=True)
def _reset_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GENETICS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("BACKEND_CONFIG_FILE", raising=False)
    _reset_backend_settings_for_tests()
    _reset_genetics_loader_for_tests()
    pgx_service.build_options.cache_clear()
    pgx_service.interpret_gene.cache_clear()
    yield
    _reset_backend_settings_for_tests()
    _reset_genetics_loader_for_tests()
    pgx_service.build_options.cache_clear()
    pgx_service.interpret_gene.cache_clear()


def _write_config(
    tmp_path: Path,
    genes: list[dict] | None = None,
    drugs: list[dict] | None = None,
) -> Path:
    path = tmp_path / "genetics.json"
    path.write_text(
        json.dumps(
            {
                "genes": genes
                or [{"id": "CYP2C19", "label": "CYP2C19 custom", "common_alleles": ["*1", "*2"]}],
                "drugs": drugs or [{"id": "sertraline", "label": "Sertraline custom"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _use_config(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("GENETICS_CONFIG_PATH", str(path))
    _reset_backend_settings_for_tests()
    _reset_genetics_loader_for_tests()
    pgx_service.build_options.cache_clear()
    pgx_service.interpret_gene.cache_clear()


def test_builtin_defaults_are_used_without_override() -> None:
    assert genetics_source() == "default"
    assert get_configured_gene_ids() == ("CYP2C19", "CYP2D6")
    assert "sertraline" in get_configured_drug_ids()


def test_configured_file_overrides_builtin_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, _write_config(tmp_path))

    assert genetics_source() == "file"
    assert get_configured_gene_ids() == ("CYP2C19",)
    assert get_configured_drug_ids() == ("sertraline",)
    assert get_configured_genes()[0].label == "CYP2C19 custom"
    assert get_configured_drugs()[0].label == "Sertraline custom"

    options = pgx_service.build_options()
    assert [gene["gene"] for gene in options["genes"]] == ["CYP2C19"]
    assert options["genes"][0]["label"] == "CYP2C19 custom"
    assert options["genes"][0]["common_alleles"] == ["*1", "*2"]
    assert [drug["label"] for drug in options["drugs"]] == ["Sertraline custom"]


def test_missing_configured_file_fails_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, tmp_path / "missing.json")

    with pytest.raises(RuntimeError, match="GENETICS_CONFIG_PATH is set but file does not exist"):
        get_configured_genes()


def test_directory_config_path_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="GENETICS_CONFIG_PATH must point to a file"):
        get_configured_genes()


def test_build_options_rejects_gene_unknown_to_pypgx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(
        monkeypatch,
        _write_config(tmp_path, genes=[{"id": "CYP9Z9", "label": "CYP9Z9", "common_alleles": ["*1"]}]),
    )

    with pytest.raises(RuntimeError, match="CYP9Z9.*not supported by pypgx"):
        pgx_service.build_options()


def test_build_options_rejects_unknown_common_allele(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(
        monkeypatch,
        _write_config(tmp_path, genes=[{"id": "CYP2C19", "label": "CYP2C19", "common_alleles": ["*999"]}]),
    )

    with pytest.raises(RuntimeError, match=r"CYP2C19.*unknown alleles: \*999"):
        pgx_service.build_options()


def test_build_options_rejects_drug_unknown_to_pypgx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, _write_config(tmp_path, drugs=[{"id": "lithium", "label": "Lithium"}]))

    with pytest.raises(RuntimeError, match="lithium.*not supported by pypgx"):
        pgx_service.build_options()


def test_interpret_uses_only_configured_genes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, _write_config(tmp_path))

    result = pgx_service.interpret({"CYP2C19": ("*1", "*2")})
    assert [gene["gene"] for gene in result["genes"]] == ["CYP2C19"]
    assert [drug["drug"] for drug in result["drug_recommendations"]] == ["Sertraline custom"]

    with pytest.raises(InvalidGeneticsDataError, match="Unsupported genes: CYP2D6"):
        pgx_service.interpret({"CYP2D6": ("*1", "*1")})


def test_validate_genetics_enforces_configured_genes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_config(monkeypatch, _write_config(tmp_path))

    assert pgx_service.validate_genetics(
        {"available": True, "genes": {"CYP2C19": {"allele1": "*1", "allele2": "*2"}}}
    ) == {"available": True, "genes": {"CYP2C19": {"allele1": "*1", "allele2": "*2"}}}

    with pytest.raises(InvalidGeneticsDataError, match="Unsupported genes: CYP2D6"):
        pgx_service.validate_genetics(
            {"available": True, "genes": {"CYP2D6": {"allele1": "*1", "allele2": "*1"}}}
        )
    with pytest.raises(InvalidGeneticsDataError, match="Unknown CYP2C19 allele"):
        pgx_service.validate_genetics(
            {"available": True, "genes": {"CYP2C19": {"allele1": "*1", "allele2": "*999"}}}
        )
    with pytest.raises(InvalidGeneticsDataError, match="availability must be a boolean"):
        pgx_service.validate_genetics({"available": "yes"})
