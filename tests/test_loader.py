from pathlib import Path

import pytest

from semantic_search.document import Document
from semantic_search.exceptions import DataDirectoryNotFoundError
from semantic_search.loader import DocumentLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_empty_directory(tmp_path: Path) -> None:
    loader = DocumentLoader(tmp_path)
    assert loader.load() == []


def test_txt_file_is_loaded() -> None:
    loader = DocumentLoader(FIXTURES_DIR)
    documents = loader.load()
    sources = [doc.source for doc in documents]
    assert any("refund_policy.txt" in s for s in sources)


def test_document_text_is_not_empty() -> None:
    loader = DocumentLoader(FIXTURES_DIR)
    documents = loader.load()
    assert all(len(doc.text) > 0 for doc in documents)


def test_non_txt_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("markdown file")
    (tmp_path / "data.pdf").write_text("pdf file")
    loader = DocumentLoader(tmp_path)
    assert loader.load() == []


def test_missing_directory_raises_error() -> None:
    loader = DocumentLoader(Path("/non-existent"))
    with pytest.raises(DataDirectoryNotFoundError):
        loader.load()


def test_returns_document_instances() -> None:
    loader = DocumentLoader(FIXTURES_DIR)
    documents = loader.load()
    assert all(isinstance(doc, Document) for doc in documents)
