"""Tests that exercise real Pixeltable operations without API keys.

Uses a deterministic fake embedding so these run in CI with zero config.
"""

import hashlib
from typing import Any

import numpy as np
import pixeltable as pxt
import pytest
from langchain_core.embeddings import Embeddings

from langchain_pixeltable import PixeltableVectorStore

EMBED_DIM = 32
TABLE_DIR = "test_lc_ci"


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for CI — hashes text into a fixed-dim vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8)[:EMBED_DIM].astype(np.float32)
        return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()


@pytest.fixture()
def embedding() -> Embeddings:
    return FakeEmbeddings()


@pytest.fixture()
def table_name() -> str:
    return f"{TABLE_DIR}.docs"


@pytest.fixture(autouse=True)
def clean_table(table_name: str) -> Any:
    try:
        pxt.drop_dir(TABLE_DIR, force=True)
    except Exception:
        pass
    yield
    try:
        pxt.drop_dir(TABLE_DIR, force=True)
    except Exception:
        pass


# ---- CRUD ----


class TestCRUD:
    def test_add_texts(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embedding=embedding)
        ids = vs.add_texts(["alpha", "beta", "gamma"], metadatas=[{"k": 1}, {"k": 2}, {"k": 3}])
        assert len(ids) == 3
        assert vs.table.count() == 3

    def test_add_texts_with_custom_ids(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embedding=embedding)
        ids = vs.add_texts(["one", "two"], ids=["id-1", "id-2"])
        assert ids == ["id-1", "id-2"]

    def test_delete(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embedding=embedding)
        ids = vs.add_texts(["to-keep", "to-delete"])
        assert vs.table.count() == 2
        vs.delete(ids=[ids[1]])
        assert vs.table.count() == 1

    def test_delete_none_is_noop(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embedding=embedding)
        assert vs.delete(ids=None) is False

    def test_from_texts(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore.from_texts(
            texts=["hello", "world"],
            embedding=embedding,
            table_name=table_name,
        )
        assert vs.table.count() == 2

    def test_from_existing_table(self, embedding: Embeddings, table_name: str) -> None:
        vs1 = PixeltableVectorStore.from_texts(
            texts=["doc1", "doc2"],
            embedding=embedding,
            table_name=table_name,
        )
        assert vs1.table.count() == 2

        vs2 = PixeltableVectorStore.from_existing_table(
            table_name=table_name,
            embedding=embedding,
        )
        assert vs2.table.count() == 2


# ---- Search ----


class TestSearch:
    def test_similarity_search(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore.from_texts(
            texts=["Pixeltable is great", "Weather is nice", "AI rocks"],
            embedding=embedding,
            table_name=table_name,
        )
        results = vs.similarity_search("Pixeltable", k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "page_content") for r in results)

    def test_similarity_search_with_score(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore.from_texts(
            texts=["hello world", "goodbye world"],
            embedding=embedding,
            table_name=table_name,
        )
        results = vs.similarity_search_with_score("hello", k=2)
        assert len(results) == 2
        for _doc, score in results:
            assert isinstance(score, float)

    def test_similarity_search_by_vector(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore.from_texts(
            texts=["alpha", "beta"],
            embedding=embedding,
            table_name=table_name,
        )
        vec = embedding.embed_query("alpha")
        results = vs.similarity_search_by_vector(vec, k=1)
        assert len(results) == 1


# ---- Filter ----


class TestFilter:
    def _make_store(self, embedding: Embeddings, table_name: str) -> PixeltableVectorStore:
        return PixeltableVectorStore.from_texts(
            texts=["doc-a", "doc-b", "doc-c"],
            embedding=embedding,
            metadatas=[
                {"category": "science", "priority": 3},
                {"category": "art", "priority": 1},
                {"category": "science", "priority": 5},
            ],
            table_name=table_name,
        )

    def test_filter_eq(self, embedding: Embeddings, table_name: str) -> None:
        vs = self._make_store(embedding, table_name)
        results = vs.similarity_search("doc", k=10, filter={"category": "science"})
        assert len(results) == 2
        for r in results:
            assert r.metadata["category"] == "science"

    def test_filter_multiple_keys(self, embedding: Embeddings, table_name: str) -> None:
        vs = self._make_store(embedding, table_name)
        results = vs.similarity_search("doc", k=10, filter={"category": "science", "priority": 5})
        assert len(results) == 1

    def test_filter_no_match(self, embedding: Embeddings, table_name: str) -> None:
        vs = self._make_store(embedding, table_name)
        results = vs.similarity_search("doc", k=10, filter={"category": "nonexistent"})
        assert len(results) == 0

    def test_filter_with_score(self, embedding: Embeddings, table_name: str) -> None:
        vs = self._make_store(embedding, table_name)
        results = vs.similarity_search_with_score("doc", k=10, filter={"category": "art"})
        assert len(results) == 1
        assert results[0][0].metadata["category"] == "art"

    def test_filter_by_vector(self, embedding: Embeddings, table_name: str) -> None:
        vs = self._make_store(embedding, table_name)
        vec = embedding.embed_query("doc")
        results = vs.similarity_search_by_vector(vec, k=10, filter={"category": "science"})
        assert len(results) == 2


# ---- .table Escape Hatch ----


class TestTableProperty:
    def test_table_returns_pxt_table(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embedding=embedding)
        vs.add_texts(["test"])
        t = vs.table
        assert t is not None
        assert t.count() == 1

    def test_table_data_accessible(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore.from_texts(
            texts=["hello", "world"],
            embedding=embedding,
            metadatas=[{"tag": "a"}, {"tag": "b"}],
            table_name=table_name,
        )
        t = vs.table
        rows = t.select(t.text, t.metadata).collect()
        assert len(rows) == 2
        texts = {r["text"] for r in rows}
        assert texts == {"hello", "world"}

    def test_where_on_table(self, embedding: Embeddings, table_name: str) -> None:
        vs = PixeltableVectorStore.from_texts(
            texts=["doc-1", "doc-2", "doc-3"],
            embedding=embedding,
            metadatas=[{"n": 10}, {"n": 20}, {"n": 30}],
            table_name=table_name,
        )
        t = vs.table
        rows = t.where(t.metadata["n"] > 15).select(t.text).collect()
        assert len(rows) == 2
