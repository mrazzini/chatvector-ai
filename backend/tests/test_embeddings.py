import pytest
from services.embedding_service import get_embedding, get_embeddings
import services.providers as providers_mod

pytestmark = pytest.mark.asyncio


class _FakeEmbeddingProvider:
    """Minimal stand-in that satisfies the EmbeddingProvider contract."""

    def __init__(self, dim: int = 3072, value: float = 0.1):
        self._dim = dim
        self._value = value

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[self._value] * self._dim for _ in texts]


@pytest.fixture(autouse=True)
def reset_provider():
    """Inject a fake provider and reset after each test."""
    providers_mod._embedding_provider = _FakeEmbeddingProvider()
    yield
    providers_mod._embedding_provider = None


async def test_get_embedding_success():
    text = "Hello world"
    embedding = await get_embedding(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 3072
    assert all(isinstance(v, float) for v in embedding)


async def test_get_embeddings_batch_success():
    texts = ["Hello world", "Another sentence"]
    embeddings = await get_embeddings(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2

    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == 3072
        assert all(isinstance(v, float) for v in emb)


async def test_embedding_dimension_consistency():
    texts = ["short text", "longer text with more content"]
    embeddings = await get_embeddings(texts)

    dims = {len(e) for e in embeddings}
    assert dims == {3072}


async def test_get_embeddings_raises_on_failure():
    """Embedding failures must propagate; the zero-vector fallback is gone."""

    class _FailingProvider:
        async def embed(self, texts):
            raise RuntimeError("forced failure")

    providers_mod._embedding_provider = _FailingProvider()

    with pytest.raises(RuntimeError, match="forced failure"):
        await get_embeddings(["this should raise"])
