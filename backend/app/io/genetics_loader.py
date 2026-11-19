import threading
from pathlib import Path

from ..domain.genetics import DrugDefinition, GeneDefinition
from ..settings import get_backend_settings
from .genetics_config import load_genetics_config

_BUILTIN_GENETICS_PATH = Path(__file__).resolve().parents[1] / "defaults" / "genetics_definitions.json"

_genes: tuple[GeneDefinition, ...] | None = None
_drugs: tuple[DrugDefinition, ...] | None = None
_source: str | None = None
_load_lock = threading.Lock()


def _configured_genetics_path() -> str | None:
    return get_backend_settings().genetics_config_path


def _ensure_loaded() -> None:
    global _genes, _drugs, _source
    if _genes is not None:
        return
    with _load_lock:
        if _genes is not None:
            return
        configured_path = _configured_genetics_path()
        genetics_path = _BUILTIN_GENETICS_PATH if configured_path is None else Path(configured_path)
        if not genetics_path.exists():
            if configured_path is None:
                raise RuntimeError(f"Built-in genetics definitions file does not exist: {genetics_path}.")
            raise RuntimeError(f"GENETICS_CONFIG_PATH is set but file does not exist: {configured_path}.")
        if not genetics_path.is_file():
            if configured_path is None:
                raise RuntimeError(f"Built-in genetics definitions path must be a file: {genetics_path}.")
            raise RuntimeError(f"GENETICS_CONFIG_PATH must point to a file: {configured_path}.")

        genes, drugs = load_genetics_config(genetics_path)
        _genes = genes
        _drugs = drugs
        _source = "default" if configured_path is None else "file"


def get_configured_genes() -> list[GeneDefinition]:
    _ensure_loaded()
    return list(_genes or ())


def get_configured_gene_ids() -> tuple[str, ...]:
    _ensure_loaded()
    return tuple(gene.id for gene in (_genes or ()))


def get_configured_drugs() -> list[DrugDefinition]:
    _ensure_loaded()
    return list(_drugs or ())


def get_configured_drug_ids() -> tuple[str, ...]:
    _ensure_loaded()
    return tuple(drug.id for drug in (_drugs or ()))


def genetics_source() -> str:
    _ensure_loaded()
    return _source or "default"


def _reset_genetics_loader_for_tests() -> None:
    global _genes, _drugs, _source
    _genes = None
    _drugs = None
    _source = None
    from ..services import pgx_service

    pgx_service.clear_caches()
