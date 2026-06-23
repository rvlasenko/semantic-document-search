from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    source: str
