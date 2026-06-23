class SemanticSearchError(Exception):
    """Base exception for semantic-document-search."""


class DataDirectoryNotFoundError(SemanticSearchError):
    """Raised when the data directory does not exist."""
