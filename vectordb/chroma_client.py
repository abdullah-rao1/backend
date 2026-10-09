from __future__ import annotations

from functools import lru_cache

import chromadb

from core.config import settings


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def _collection_name(session_id: str) -> str:
    return f"{settings.chroma_collection}_{session_id}"


def get_session_collection(session_id: str):
    return get_client().get_or_create_collection(
        name=_collection_name(session_id), metadata={"hnsw:space": "cosine"}
    )


def upsert_chunks(
    session_id: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    if not ids:
        return
    collection = get_session_collection(session_id)
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_top_k(session_id: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    collection = get_session_collection(session_id)
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches: list[dict] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        page_number = metadata.get("page_number", -1)
        matches.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata.get("document_id"),
                "text": text,
                "document_name": metadata.get("document_name"),
                "page_number": None if page_number == -1 else page_number,
                "chunk_index": metadata.get("chunk_index"),
                "score": 1 - distance,
            }
        )
    return matches


def clear_session(session_id: str) -> None:
    try:
        get_client().delete_collection(name=_collection_name(session_id))
    except (ValueError, chromadb.errors.NotFoundError):
        pass


def delete_document_chunks(session_id: str, document_id: str) -> None:
    collection = get_session_collection(session_id)
    if collection.count() == 0:
        return
    collection.delete(where={"document_id": document_id})


def get_session_texts(session_id: str, limit: int = 200) -> list[str]:
    collection = get_session_collection(session_id)
    if collection.count() == 0:
        return []
    result = collection.get(limit=limit, include=["documents"])
    return result.get("documents", []) or []
