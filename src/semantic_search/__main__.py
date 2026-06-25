import sys
from pathlib import Path

from semantic_search.chunker import TextChunker
from semantic_search.embedding_model import EmbeddingModel
from semantic_search.loader import DocumentLoader
from semantic_search.search_index import SearchIndex

DATA_DIR = Path("data")
TOP_K = 3


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "search":
        print('Usage: python -m semantic_search search "your query"')
        sys.exit(1)

    query = sys.argv[2]

    loader = DocumentLoader(DATA_DIR)
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    model = EmbeddingModel()
    index = SearchIndex(chunker, model)

    documents = loader.load()

    if not documents:
        print(f"No documents found in {DATA_DIR}/")
        sys.exit(1)

    index.add_documents(documents)
    results = index.search(query=query, top_k=TOP_K)

    print(f"\nQuery: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"#{i} score={result.score:.3f} source={result.source}")
        print(result.text)
        print()


if __name__ == "__main__":
    main()
