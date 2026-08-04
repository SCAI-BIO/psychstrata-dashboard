from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Feature:
    id: str
    label: str
    kind: str
    default: Any
    params: dict[str, Any]

