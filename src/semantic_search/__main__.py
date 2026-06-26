import argparse
import sys
from pathlib import Path

from semantic_search.exceptions import IndexNotFoundError
from semantic_search.indexing.chunker import TextChunker
from semantic_search.indexing.embedding_model import EmbeddingModel
from semantic_search.indexing.loader import DocumentLoader
from semantic_search.search.search_index import SearchIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic_search",
        description="Semantic search engine for plain text documents.",
    )

    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser(
        "index", help="Index documents from data directory."
    )
    index_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        metavar="PATH",
        help="Directory with .txt documents (default: data/).",
    )
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(".index"),
        metavar="PATH",
        help="Where to save the index (default: .index/).",
    )

    search_parser = subparsers.add_parser("search", help="Search indexed documents.")
    search_parser.add_argument("query", type=str, help="Search query.")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="N",
        help="Number of results to return (default: 3).",
    )
    search_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(".index"),
        metavar="PATH",
        help="Index directory to load (default: .index/).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    model = EmbeddingModel()

    if args.command == "index":
        documents = DocumentLoader(args.data_dir).load()

        if not documents:
            print(f"No documents found in {args.data_dir}/")
            sys.exit(1)

        index = SearchIndex(chunker, model)
        index.add_documents(documents)
        index.save(args.index_dir)
        print(f"Indexed {len(documents)} documents → {args.index_dir}/")

    elif args.command == "search":
        try:
            index = SearchIndex.load(args.index_dir, chunker, model)
        except IndexNotFoundError:
            print("Index not found. Run first: python -m semantic_search index")
            sys.exit(1)

        results = index.search(args.query, top_k=args.top_k)

        print(f"\nQuery: {args.query}\n")
        for i, result in enumerate(results, 1):
            print(f"#{i} score={result.score:.3f} source={result.source}")
            print(result.text)
            print()


if __name__ == "__main__":
    main()
