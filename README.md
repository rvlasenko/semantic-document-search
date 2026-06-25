# Semantic Document Search

A semantic search engine for plain text documents built with Python and Hugging Face.

Unlike keyword search, this project finds documents by meaning — not exact words.

## How it works

```
.txt files → chunks → embeddings → cosine similarity → results
```

1. Documents are split into overlapping chunks
2. Each chunk is encoded into a 384-dimensional vector using a sentence transformer model
3. At query time, the query is encoded the same way
4. Cosine similarity finds the closest chunks

## Stack

- [`sentence-transformers`](https://www.sbert.net/) — text embeddings
- [`scikit-learn`](https://scikit-learn.org/) — cosine similarity
- [`numpy`](https://numpy.org/) — vector operations

## Setup

```bash
git clone https://github.com/your-username/semantic-document-search
cd semantic-document-search
uv sync
```

## Usage

Add `.txt` files to `data/` and run:

```bash
python -m semantic_search search "How does the refund policy work?"
```

Example output:

```
Query: How does the refund policy work?

#1 score=0.588 source=data/refund_policy.txt
Our refund policy allows returns within 30 days of purchase...

#2 score=0.331 source=data/refund_policy.txt
Original shipping costs are non-refundable...

#3 score=0.244 source=data/faq.txt
How do I contact customer support?...
```

## Project structure

```
src/semantic_search/
├── document.py        # Document dataclass
├── loader.py          # Loads .txt files from data/
├── chunk.py           # TextChunk dataclass
├── chunker.py         # Splits documents into overlapping chunks
├── embedding_model.py # Encodes chunks via sentence-transformers
├── search.py          # Cosine similarity search function
├── search_index.py    # Stateful index — add documents, run queries
└── __main__.py        # CLI entry point
```

## Limitations

- English text only (`all-MiniLM-L6-v2` is not multilingual)
- `.txt` files only — no PDF, no Word
- No persistent index — rebuilt on every run
- Brute-force similarity search — not suitable for large document sets
