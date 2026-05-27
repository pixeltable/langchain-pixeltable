# langchain-pixeltable

LangChain VectorStore integration backed by [Pixeltable](https://github.com/pixeltable/pixeltable) -- multimodal data infrastructure with built-in embedding indexes, metadata filtering, computed column lineage, and incremental computation.

## Installation

```bash
pip install langchain-pixeltable
```

## Quick Start

Works with any LangChain `Embeddings` model -- cloud or local:

```python
from langchain_pixeltable import PixeltableVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # no API key needed

vs = PixeltableVectorStore.from_texts(
    texts=[
        "Pixeltable handles multimodal data",
        "LangChain builds LLM applications",
        "Vector databases store embeddings",
    ],
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    metadatas=[
        {"category": "infra"},
        {"category": "framework"},
        {"category": "infra"},
    ],
    table_name="mydir.docs",
)

# Similarity search
results = vs.similarity_search("multimodal data management", k=2)
for doc in results:
    print(doc.page_content)
```

## Filtered Similarity Search

The `filter` parameter maps to Pixeltable's `.where()` clause -- predicates are evaluated **before** ranking, so only matching rows participate in the similarity sort:

```python
# Only search within "infra" documents
results = vs.similarity_search(
    "data storage", k=5, filter={"category": "infra"},
)

# With scores
results = vs.similarity_search_with_score(
    "embeddings", k=3, filter={"category": "infra"},
)
for doc, score in results:
    print(f"[{score:.3f}] {doc.page_content}")
```

## Access the Underlying Pixeltable Table

The `.table` property gives direct access to the Pixeltable table for operations beyond the VectorStore interface -- computed columns, lineage, version history, and arbitrary predicates:

```python
import pixeltable as pxt

t = vs.table

# Inspect all data
t.select(t.text, t.metadata, t.embedding).collect()

# Add a computed column -- auto-backfills all existing rows
t.add_computed_column(word_count=my_word_counter(t.text))

# New inserts via the wrapper auto-compute lineage columns
vs.add_texts(["New document"], metadatas=[{"category": "infra"}])

# WHERE on computed columns + similarity
import numpy as np
sim = t.embedding.similarity(vector=np.array(query_vec, dtype=np.float32))
results = (
    t.where(t.word_count > 5)
    .order_by(sim, asc=False)
    .limit(3)
    .select(t.text, t.word_count, sim=sim)
    .collect()
)
```

## Connect to an Existing Pixeltable Table

Connect to any existing Pixeltable table -- including tables with multimodal columns like images or video:

```python
vs = PixeltableVectorStore.from_existing_table(
    table_name="mydir.existing_docs",
    embedding=OpenAIEmbeddings(),
    text_column="content",
    embedding_column="content_embedding",
)
results = vs.similarity_search("search query", filter={"source": "arxiv"})
```

## Use as a LangChain Retriever

```python
retriever = vs.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("What is Pixeltable?")
```

## Why Pixeltable as a Vector Backend?

- **Metadata filtering via `.where()`**: Filter on metadata fields *before* ranking, not post-hoc
- **Computed column lineage**: Add derived columns that auto-backfill and auto-compute on new inserts
- **Persistent and versioned**: Data survives restarts; every change is tracked
- **Incremental**: Only new/changed rows get re-embedded
- **Multimodal native**: Images, video, audio, and documents alongside text
- **Any embedding model**: Works with OpenAI, Hugging Face, or any local model
- **No external services**: Embedded PostgreSQL, no Docker required

## Links

- [Pixeltable Docs](https://docs.pixeltable.com/)
- [LangChain Integration Docs](https://docs.pixeltable.com/libraries/langchain)
- [GitHub](https://github.com/pixeltable/pixeltable)
- [Discord](https://discord.gg/QPyqFYx2UN)
