import sys
from pathlib import Path

from semantic_search.exceptions import IndexNotFoundError
from semantic_search.indexing.chunker import TextChunker
from semantic_search.indexing.embedding_model import EmbeddingModel
from semantic_search.indexing.loader import DocumentLoader
from semantic_search.search.search_index import SearchIndex

DATA_DIR = Path("data")
INDEX_DIR = Path(".index")
TOP_K = 3


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m semantic_search index")
        print('  python -m semantic_search search "your query"')
        sys.exit(1)

    command = sys.argv[1]
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    model = EmbeddingModel()

    if command == "index":
        documents = DocumentLoader(DATA_DIR).load()

        if not documents:
            print(f"No documents found in {DATA_DIR}/")
            sys.exit(1)

        index = SearchIndex(chunker, model)
        index.add_documents(documents)
        index.save(INDEX_DIR)
        print(f"Indexed {len(documents)} documents → {INDEX_DIR}/")

    elif command == "search":
        if len(sys.argv) < 3:
            print('Usage: python -m semantic_search search "your query"')
            sys.exit(1)

        query = sys.argv[2]

        try:
            index = SearchIndex.load(INDEX_DIR, chunker, model)
        except IndexNotFoundError:
            print("Index not found. Run first: python -m semantic_search index")
            sys.exit(1)

        results = index.search(query, top_k=TOP_K)

        print(f"\nQuery: {query}\n")
        for i, result in enumerate(results, 1):
            print(f"#{i} score={result.score:.3f} source={result.source}")
            print(result.text)
            print()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
