class SemanticSearchError(Exception):
    """Base exception for semantic-document-search."""


class DataDirectoryNotFoundError(SemanticSearchError):
    """Raised when the data directory does not exist."""


class IndexNotFoundError(SemanticSearchError):
    """Raised when the index directory does not exist."""


class IndexCorruptedError(SemanticSearchError):
    """Raised when the index files are missing or cannot be loaded."""
