"""Integration tests for PixeltableVectorStore.

Requires:
- A running Pixeltable instance
- OPENAI_API_KEY environment variable
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set',
)


@pytest.fixture
def embedding():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model='text-embedding-3-small')


@pytest.fixture
def clean_table():
    """Ensure a clean table for each test, and clean up after."""
    import pixeltable as pxt
    table_name = 'test_lc_integration.docs'
    try:
        pxt.drop_table(table_name, force=True)
    except Exception:
        pass
    try:
        pxt.drop_dir('test_lc_integration', force=True)
    except Exception:
        pass
    yield table_name
    try:
        pxt.drop_table(table_name, force=True)
    except Exception:
        pass
    try:
        pxt.drop_dir('test_lc_integration', force=True)
    except Exception:
        pass


class TestPixeltableVectorStore:
    def test_from_texts_and_search(self, embedding, clean_table):
        from langchain_pixeltable import PixeltableVectorStore

        vs = PixeltableVectorStore.from_texts(
            texts=[
                'Pixeltable is multimodal data infrastructure',
                'LangChain builds LLM applications',
                'Vector databases store embeddings',
            ],
            embedding=embedding,
            table_name=clean_table,
        )

        results = vs.similarity_search('multimodal data', k=2)
        assert len(results) == 2
        assert 'Pixeltable' in results[0].page_content

    def test_similarity_search_with_score(self, embedding, clean_table):
        from langchain_pixeltable import PixeltableVectorStore

        vs = PixeltableVectorStore.from_texts(
            texts=['hello world', 'goodbye world'],
            embedding=embedding,
            table_name=clean_table,
        )

        results = vs.similarity_search_with_score('hello', k=2)
        assert len(results) == 2
        doc, score = results[0]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_as_retriever(self, embedding, clean_table):
        from langchain_pixeltable import PixeltableVectorStore

        vs = PixeltableVectorStore.from_texts(
            texts=['retriever test document'],
            embedding=embedding,
            table_name=clean_table,
        )

        retriever = vs.as_retriever(search_kwargs={'k': 1})
        docs = retriever.invoke('retriever test')
        assert len(docs) == 1

    def test_delete(self, embedding, clean_table):
        import pixeltable as pxt
        from langchain_pixeltable import PixeltableVectorStore

        vs = PixeltableVectorStore.from_texts(
            texts=['doc to keep', 'doc to delete'],
            embedding=embedding,
            table_name=clean_table,
        )

        t = pxt.get_table(clean_table)
        assert t.count() == 2

        docs = vs.similarity_search('delete', k=1)
        vs.delete(ids=[docs[0].id])
        assert t.count() == 1

    def test_similarity_search_by_vector(self, embedding, clean_table):
        from langchain_pixeltable import PixeltableVectorStore

        vs = PixeltableVectorStore.from_texts(
            texts=['vector search test'],
            embedding=embedding,
            table_name=clean_table,
        )

        vec = embedding.embed_query('vector search')
        results = vs.similarity_search_by_vector(vec, k=1)
        assert len(results) == 1

    def test_from_existing_table(self, embedding, clean_table):
        from langchain_pixeltable import PixeltableVectorStore

        PixeltableVectorStore.from_texts(
            texts=['existing table test'],
            embedding=embedding,
            table_name=clean_table,
        )

        vs2 = PixeltableVectorStore.from_existing_table(
            table_name=clean_table,
            embedding=embedding,
        )
        results = vs2.similarity_search('existing', k=1)
        assert len(results) == 1
