from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
