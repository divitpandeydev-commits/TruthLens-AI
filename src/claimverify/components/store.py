"""
Storage component — PostgreSQL/pgvector and Local In-Memory/File Store
======================================================================

Supports two backends transparently:
1. PostgreSQL with pgvector extension (when DB_URL points to postgresql://...)
2. Local file/memory store (when DB_URL is 'local' or None or PostgreSQL is offline)

All higher-level answering and indexing modules interact through this API
without needing to know which storage backend is active.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np

from claimverify.components.embedding import EMBEDDING_DIM
from claimverify.contracts import Chunk, Passage

TOP_K_PER_DOC = 2
LOCAL_STORE_PATH = Path("data/local_store.json")


class LocalStore:
    """In-memory and file-backed chunk repository with vector similarity search."""

    def __init__(self, path: Path = LOCAL_STORE_PATH):
        self.path = path
        self.chunks: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []
        else:
            self.chunks = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def replace(self, chunks: list[Chunk]):
        self.chunks = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "filename": c.filename,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "embedding": c.embedding,
                "source_url": getattr(c, "source_url", None),
                "title": getattr(c, "title", None),
            }
            for c in chunks
        ]
        self.save()

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def nearest(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Return top-k Chunk objects ordered by cosine similarity to query_embedding."""
        scored = []
        for c in self.chunks:
            emb = c.get("embedding")
            if emb:
                sim = self.cosine_similarity(query_embedding, emb)
                scored.append((sim, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                filename=c["filename"],
                source_type=c["source_type"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                embedding=c.get("embedding"),
                source_url=c.get("source_url"),
                title=c.get("title"),
            )
            for _, c in scored[:top_k]
        ]


_GLOBAL_LOCAL_STORE = LocalStore()


def _is_local(conn: Any) -> bool:
    return conn is None or conn == "local" or isinstance(conn, (str, LocalStore))


# ---------------------------------------------------------------------------
# Database Setup & Ingestion
# ---------------------------------------------------------------------------

def setup_db(conn):
    if _is_local(conn):
        LOCAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return

    from psycopg2.extras import execute_values  # lazy import
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                source_type TEXT NOT NULL,
                chunk_index INT NOT NULL,
                text TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM})
            );
        """)
        cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx;")
    conn.commit()


def replace_chunks(conn, chunks: list[Chunk]):
    if not chunks:
        raise ValueError("No chunk to store: check --corpus_dir. The store is not modified.")

    missing = [c.chunk_id for c in chunks if c.embedding is None]
    if missing:
        raise ValueError(
            f"{len(missing)} chunk(s) without an embedding (e.g. {missing[0]}): "
            "run embed_chunks before replace_chunks."
        )

    if _is_local(conn):
        _GLOBAL_LOCAL_STORE.replace(chunks)
        return

    from psycopg2.extras import execute_values
    rows = [
        (c.chunk_id, c.doc_id, c.filename, c.source_type, c.chunk_index, c.text, c.embedding)
        for c in chunks
    ]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks;")
        execute_values(
            cur,
            """
            INSERT INTO chunks (chunk_id, doc_id, filename, source_type, chunk_index, text, embedding)
            VALUES %s;
            """,
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Searching (answering pipeline)
# ---------------------------------------------------------------------------

def search_global(conn, query_embedding: list[float], top_k: int = 5) -> list[Passage]:
    if _is_local(conn):
        scored = []
        for c in _GLOBAL_LOCAL_STORE.chunks:
            if c.get("embedding"):
                sim = _GLOBAL_LOCAL_STORE.cosine_similarity(query_embedding, c["embedding"])
                scored.append(
                    Passage(
                        filename=c["filename"],
                        source_type=c["source_type"],
                        chunk_index=c["chunk_index"],
                        text=c["text"],
                        score=round(sim, 4),
                        source_url=c.get("source_url"),
                        title=c.get("title"),
                    )
                )
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:top_k]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT filename, source_type, chunk_index, text,
                   1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k),
        )
        return [_to_passage(row) for row in cur.fetchall()]


def search_per_document(conn, query_embedding: list[float],
                        top_k_per_doc: int = TOP_K_PER_DOC) -> list[Passage]:
    if _is_local(conn):
        by_doc: dict[str, list[Passage]] = {}
        for c in _GLOBAL_LOCAL_STORE.chunks:
            if c.get("embedding"):
                sim = _GLOBAL_LOCAL_STORE.cosine_similarity(query_embedding, c["embedding"])
                p = Passage(
                    filename=c["filename"],
                    source_type=c["source_type"],
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    score=round(sim, 4),
                    source_url=c.get("source_url"),
                    title=c.get("title"),
                )
                by_doc.setdefault(c["filename"], []).append(p)

        all_results = []
        for doc_passages in by_doc.values():
            doc_passages.sort(key=lambda p: p.score, reverse=True)
            all_results.extend(doc_passages[:top_k_per_doc])

        all_results.sort(key=lambda p: p.score, reverse=True)
        return all_results

    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT filename FROM chunks;")
        filenames = [row[0] for row in cur.fetchall()]

        all_results = []
        for filename in filenames:
            cur.execute(
                """
                SELECT filename, source_type, chunk_index, text,
                       1 - (embedding <=> %s::vector) AS cosine_similarity
                FROM chunks
                WHERE filename = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, filename, query_embedding, top_k_per_doc),
            )
            all_results.extend(_to_passage(row) for row in cur.fetchall())

        all_results.sort(key=lambda p: p.score, reverse=True)
        return all_results


def rank_all_per_document(conn, query_embedding: list[float]) -> dict[str, list[Passage]]:
    if _is_local(conn):
        ranked: dict[str, list[Passage]] = {}
        for c in _GLOBAL_LOCAL_STORE.chunks:
            if c.get("embedding"):
                sim = _GLOBAL_LOCAL_STORE.cosine_similarity(query_embedding, c["embedding"])
                p = Passage(
                    filename=c["filename"],
                    source_type=c["source_type"],
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    score=round(sim, 4),
                    source_url=c.get("source_url"),
                    title=c.get("title"),
                )
                ranked.setdefault(c["filename"], []).append(p)
        for doc_name in ranked:
            ranked[doc_name].sort(key=lambda p: p.score, reverse=True)
        return ranked

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT filename, source_type, chunk_index, text,
                   1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM chunks
            ORDER BY filename, embedding <=> %s::vector;
            """,
            (query_embedding, query_embedding),
        )
        ranked: dict[str, list[Passage]] = {}
        for row in cur.fetchall():
            passage = _to_passage(row)
            ranked.setdefault(passage.filename, []).append(passage)
        return ranked


def load_chunk_texts(conn) -> list[tuple[str, int, str]]:
    if _is_local(conn):
        return [(c["filename"], c["chunk_index"], c["text"]) for c in _GLOBAL_LOCAL_STORE.chunks]

    with conn.cursor() as cur:
        cur.execute("SELECT filename, chunk_index, text FROM chunks ORDER BY filename, chunk_index;")
        return cur.fetchall()


def _to_passage(row) -> Passage:
    filename, source_type, chunk_index, text, score = row
    return Passage(
        filename=filename,
        source_type=source_type,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )
