from dataclasses import dataclass


@dataclass(frozen=True)
class GeneDefinition:
    """A pharmacogene exposed for intake and CPIC interpretation."""

    id: str
    label: str
    common_alleles: tuple[str, ...]


@dataclass(frozen=True)
class DrugDefinition:
    """A drug whose CPIC recommendations are offered by the interpretation API."""

    id: str
    label: str
