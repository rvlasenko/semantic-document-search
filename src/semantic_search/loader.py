from pathlib import Path

from semantic_search.document import Document
from semantic_search.exceptions import DataDirectoryNotFoundError


class DocumentLoader:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(self) -> list[Document]:
        if not self._data_dir.exists():
            raise DataDirectoryNotFoundError(
                f"Data directory not found: {self._data_dir}"
            )

        documents = []

        for path in sorted(self._data_dir.glob("*.txt")):
            text = self._read_file(path)
            documents.append(
                Document(
                    text=text,
                    source=str(path),
                )
            )

        return documents

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")
