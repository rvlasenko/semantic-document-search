from pathlib import Path

from semantic_search.document import Document
from semantic_search.loader import DocumentLoader
from semantic_search.chunker import TextChunker
from semantic_search.embedding_model import EmbeddingModel
from semantic_search.chunk import TextChunk
from semantic_search.search import search

loader = DocumentLoader(Path("data"))
chunker = TextChunker(chunk_size=50, chunk_overlap=20)
model = EmbeddingModel()

documents = loader.load()

chunks = []

for document in documents:
    chunks.extend(chunker.split(document))

chunk_embeddings = [model.encode(chunk) for chunk in chunks]

query = "How long can I return an item?"
query_embedding = model.encode(TextChunk(text=query, source="query"))

results = search(query_embedding, chunk_embeddings, chunks, top_k=3)

for i, result in enumerate(results, 1):
    print(f"# {i}, score={result.score:.3f}")
    print(result.text)
