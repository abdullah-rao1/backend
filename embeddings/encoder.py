from __future__ import annotations

import logging
import os
from functools import lru_cache

# stray tensorflow install in the venv breaks transformers' TF auto-detection
# (Keras 3 incompatibility); force the torch-only path since that's all we use.
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    model = SentenceTransformer(settings.embedding_model)
    dim = model.get_sentence_embedding_dimension()
    if dim != settings.embedding_dimension:
        raise RuntimeError(
            f"Embedding model {settings.embedding_model!r} produces {dim}-dim vectors, "
            f"but EMBEDDING_DIMENSION={settings.embedding_dimension}. "
            "Use a 384-dim model (all-MiniLM-L6-v2) or clear storage/chroma after switching."
        )
    logger.info("Loaded embedding model %s (%d dims)", settings.embedding_model, dim)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_encoder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
