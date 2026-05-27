"""Unit tests for PixeltableVectorStore.

These tests verify the class interface without requiring a running
Pixeltable instance or API keys.
"""

from langchain_pixeltable import PixeltableVectorStore


def test_import():
    """Verify the package exports PixeltableVectorStore."""
    assert hasattr(PixeltableVectorStore, "similarity_search")
    assert hasattr(PixeltableVectorStore, "similarity_search_with_score")
    assert hasattr(PixeltableVectorStore, "similarity_search_by_vector")
    assert hasattr(PixeltableVectorStore, "add_texts")
    assert hasattr(PixeltableVectorStore, "delete")
    assert hasattr(PixeltableVectorStore, "from_texts")
    assert hasattr(PixeltableVectorStore, "from_existing_table")
    assert hasattr(PixeltableVectorStore, "as_retriever")


def test_version():
    """Verify version is set."""
    import langchain_pixeltable

    assert langchain_pixeltable.__version__ == "0.1.1"
