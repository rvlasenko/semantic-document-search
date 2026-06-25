from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    score: float
