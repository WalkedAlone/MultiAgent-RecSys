"""Prepare real review chunks and build the persistent Chroma review index."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from src.core.logger import get_logger
from src.rag.embedding import BGEEmbedding
from src.rag.vector_store import ChromaCollection

logger = get_logger(__name__)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def prepare_top_reviews(
    *,
    reviews_path: str | Path,
    output_path: str | Path,
    max_per_item: int = 3,
    memory_limit: str = "3GB",
) -> dict[str, Any]:
    """Select up to ``max_per_item`` real reviews per parent_asin with DuckDB.

    Selection is deterministic: verified purchase, helpful votes, newest first,
    then review_id.  Only text-bearing reviews are eligible.
    """
    import duckdb

    reviews_path = Path(reviews_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
        src = _sql_path(reviews_path)
        ids_path = output_path.with_name(
            f".review_top{max_per_item}_ids.parquet"
        )
        con.execute(
            f"""
            COPY (
              SELECT review_id, parent_asin, verified_purchase, helpful_vote,
                     timestamp_ms, item_idx, rn
              FROM (
                SELECT review_id, parent_asin, verified_purchase, helpful_vote,
                       timestamp_ms, item_idx,
                       row_number() OVER (
                         PARTITION BY parent_asin
                         ORDER BY verified_purchase DESC, helpful_vote DESC,
                                  timestamp_ms DESC, review_id
                       ) AS rn
                FROM read_parquet('{src}')
                WHERE has_text
              ) ranked
              WHERE rn <= {int(max_per_item)}
            ) TO '{_sql_path(ids_path)}' (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT i.rn, r.review_id, r.parent_asin, r.user_id, r.rating,
                     r.timestamp_ms, r.title, r.text_clean, r.verified_purchase,
                     r.helpful_vote, r.item_idx
              FROM read_parquet('{_sql_path(ids_path)}') i
              JOIN read_parquet('{src}') r ON i.review_id = r.review_id
            ) TO '{_sql_path(output_path)}' (FORMAT PARQUET)
            """
        )
        count, items = con.execute(
            f"SELECT count(*), count(distinct parent_asin) "
            f"FROM read_parquet('{_sql_path(output_path)}')"
        ).fetchone()
    finally:
        con.close()
    stats = {
        "reviews_source": str(reviews_path),
        "output": str(output_path),
        "max_per_item": int(max_per_item),
        "reviews_selected": int(count),
        "items_with_reviews": int(items),
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    logger.info("review_selection_completed %s", stats)
    return stats


def review_document(row: dict[str, Any], *, max_chars: int = 700) -> str:
    """Build a retrieval document from one real review only."""
    text = str(row.get("text_clean") or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    title = str(row.get("title") or "").strip()
    if title:
        return f"{title}. {text}"
    return text


def review_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Metadata suitable for Chroma; values are primitive."""
    return {
        "review_id": str(row.get("review_id") or ""),
        "parent_asin": str(row.get("parent_asin") or ""),
        "rating": float(row.get("rating") or 0.0),
        "timestamp_ms": int(row.get("timestamp_ms") or 0),
        "timestamp_iso": _iso_from_ms(row.get("timestamp_ms")),
        "helpful_vote": int(row.get("helpful_vote") or 0),
        "verified_purchase": bool(row.get("verified_purchase") or False),
        "title": str(row.get("title") or "")[:300],
        "item_idx": int(row.get("item_idx") or 0),
    }


def _iso_from_ms(value: Any) -> str:
    try:
        return datetime.fromtimestamp(
            int(value) / 1000.0, tz=timezone.utc
        ).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


class ReviewIndexBuilder:
    """Embed real review chunks in batches and persist them in Chroma."""

    def __init__(
        self,
        *,
        config: Any,
        embedder: BGEEmbedding,
        collection: ChromaCollection | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.collection = collection or ChromaCollection(
            path=config.resolve_path(config.agent.rag.review_index_dir),
            collection_name=config.agent.rag.collection_name,
        )

    def build(
        self,
        *,
        source_path: str | Path,
        max_per_item: int = 3,
        limit_items: int | None = None,
        allowed_item_ids: set[str] | None = None,
        batch_size: int = 64,
        upsert_batch_size: int = 500,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(str(source_path))
        # ``rebuild`` is handled by the CLI before constructing the persistent
        # client; deleting a live HNSW collection in-process can corrupt the
        # local index.
        _ = rebuild
        started = time.perf_counter()
        frame = pl.read_parquet(
            source_path,
            columns=[
                "review_id",
                "parent_asin",
                "user_id",
                "rating",
                "timestamp_ms",
                "title",
                "text_clean",
                "verified_purchase",
                "helpful_vote",
                "item_idx",
            ],
        )
        if allowed_item_ids is not None:
            frame = frame.filter(pl.col("parent_asin").is_in(list(allowed_item_ids)))
        if limit_items is not None and limit_items > 0:
            keep = (
                frame.select("parent_asin")
                .unique()
                .sort("parent_asin")
                .head(int(limit_items))
                .get_column("parent_asin")
                .to_list()
            )
            frame = frame.filter(pl.col("parent_asin").is_in(keep))

        total = frame.height
        indexed = 0
        skipped = 0
        valid_rows: list[dict[str, Any]] = []
        for row in frame.iter_rows(named=True):
            document = review_document(row)
            if not document:
                skipped += 1
                continue
            valid_rows.append({"row": row, "document": document})
        logger.info(
            "review_index_build_start docs=%d skipped=%d limit_items=%s",
            len(valid_rows),
            skipped,
            limit_items,
        )
        pending_ids: list[str] = []
        pending_documents: list[str] = []
        pending_metadatas: list[dict[str, Any]] = []
        pending_embeddings: list[np.ndarray] = []

        def flush() -> None:
            nonlocal pending_ids, pending_documents, pending_metadatas, pending_embeddings
            if not pending_ids:
                return
            self.collection.upsert(
                ids=pending_ids,
                documents=pending_documents,
                metadatas=pending_metadatas,
                embeddings=np.asarray(pending_embeddings, dtype=np.float32),
                batch_size=upsert_batch_size,
            )
            pending_ids = []
            pending_documents = []
            pending_metadatas = []
            pending_embeddings = []

        for start in range(0, len(valid_rows), batch_size):
            batch = valid_rows[start : start + batch_size]
            documents = [item["document"] for item in batch]
            embeddings = self.embedder.encode_documents(documents)
            pending_ids.extend(
                str(item["row"]["review_id"]) for item in batch
            )
            pending_documents.extend(documents)
            pending_metadatas.extend(
                review_metadata(item["row"]) for item in batch
            )
            pending_embeddings.extend(np.asarray(embeddings, dtype=np.float32))
            indexed += len(batch)
            if len(pending_ids) >= upsert_batch_size:
                flush()
            if indexed % 5000 < batch_size:
                logger.info(
                    "review_index_progress indexed=%d/%d", indexed, len(valid_rows)
                )
        flush()
        self.collection.persist()
        stats = {
            "collection": self.collection.collection_name,
            "collection_count": self.collection.count,
            "source_rows": int(total),
            "documents_indexed": int(indexed),
            "skipped_empty_text": int(skipped),
            "max_per_item": int(max_per_item),
            "limit_items": limit_items,
            "embedding_dimension": int(self.embedder.dimension),
            "runtime_seconds": round(time.perf_counter() - started, 1),
        }
        logger.info("review_index_build_completed %s", stats)
        return stats


class ShardedReviewIndexBuilder:
    """Build multiple bounded Chroma collections (safe local HNSW size).

    A single 144k-document persistent HNSW collection was not reloadable on
    this Windows/Chroma combination.  Sharding by a deterministic parent_asin
    hash keeps each local HNSW under 10k vectors and still gives an exact
    product-scoped lookup (one shard per item).
    """

    def __init__(
        self,
        *,
        config: Any,
        embedder: BGEEmbedding,
        base_dir: str | Path | None = None,
        collection_name: str | None = None,
        num_shards: int = 16,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.base_dir = Path(
            base_dir
            or config.resolve_path(config.agent.rag.review_index_dir)
        )
        self.collection_name = collection_name or config.agent.rag.collection_name
        self.num_shards = max(1, int(num_shards))
        # Keep every shard client/collection alive until interpreter teardown.
        # Dropping a local Chroma collection immediately after persist can
        # prevent its HNSW header from being flushed on Windows.
        self._collections: list[ChromaCollection] = []

    @staticmethod
    def shard_index(item_id: str, num_shards: int) -> int:
        import hashlib

        digest = hashlib.sha1(str(item_id).encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % int(num_shards)

    def build(
        self,
        *,
        source_path: str | Path,
        max_per_item: int = 2,
        limit_items: int | None = None,
        allowed_item_ids: set[str] | None = None,
        batch_size: int = 256,
        upsert_batch_size: int = 2000,
    ) -> dict[str, Any]:
        source_path = Path(source_path)
        frame = pl.read_parquet(
            source_path,
            columns=[
                "review_id",
                "parent_asin",
                "user_id",
                "rating",
                "timestamp_ms",
                "title",
                "text_clean",
                "verified_purchase",
                "helpful_vote",
                "item_idx",
            ],
        )
        if allowed_item_ids is not None:
            frame = frame.filter(pl.col("parent_asin").is_in(list(allowed_item_ids)))
        if limit_items is not None and limit_items > 0:
            keep = (
                frame.select("parent_asin")
                .unique()
                .sort("parent_asin")
                .head(int(limit_items))
                .get_column("parent_asin")
                .to_list()
            )
            frame = frame.filter(pl.col("parent_asin").is_in(keep))
        shard_column = frame.get_column("parent_asin").map_elements(
            lambda value: self.shard_index(str(value), self.num_shards),
            return_dtype=pl.Int64,
        )
        frame = frame.with_columns(shard_column.alias("_shard"))
        shards = {
            shard: frame.filter(pl.col("_shard") == shard)
            for shard in range(self.num_shards)
        }
        started = time.perf_counter()
        stats_shards: list[dict[str, Any]] = []
        total_indexed = 0
        for shard, shard_frame in shards.items():
            if shard_frame.height == 0:
                continue
            rows = []
            for row in shard_frame.iter_rows(named=True):
                document = review_document(row)
                if document:
                    rows.append({"row": row, "document": document})
            shard_path = self.base_dir / f"shard_{shard:02d}"
            shard_path.mkdir(parents=True, exist_ok=True)
            collection = ChromaCollection(
                path=shard_path,
                collection_name=self.collection_name,
            )
            self._collections.append(collection)
            pending_ids: list[str] = []
            pending_docs: list[str] = []
            pending_meta: list[dict[str, Any]] = []
            pending_emb: list[np.ndarray] = []

            def flush() -> None:
                nonlocal pending_ids, pending_docs, pending_meta, pending_emb
                if not pending_ids:
                    return
                collection.upsert(
                    ids=pending_ids,
                    documents=pending_docs,
                    metadatas=pending_meta,
                    embeddings=np.asarray(pending_emb, dtype=np.float32),
                    batch_size=upsert_batch_size,
                )
                pending_ids = []
                pending_docs = []
                pending_meta = []
                pending_emb = []

            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                docs = [item["document"] for item in batch]
                vectors = self.embedder.encode_documents(docs)
                pending_ids.extend(str(item["row"]["review_id"]) for item in batch)
                pending_docs.extend(docs)
                pending_meta.extend(review_metadata(item["row"]) for item in batch)
                pending_emb.extend(
                    np.asarray(vectors, dtype=np.float32)
                )
                if len(pending_ids) >= upsert_batch_size:
                    flush()
                total_indexed += len(batch)
            flush()
            collection.persist()
            shard_count = collection.count
            stats_shards.append(
                {
                    "shard": shard,
                    "path": str(shard_path),
                    "documents": int(shard_count),
                }
            )
            logger.info(
                "review_shard_built shard=%d documents=%d", shard, shard_count
            )
        stats = {
            "mode": "sharded",
            "base_dir": str(self.base_dir),
            "collection": self.collection_name,
            "num_shards": self.num_shards,
            "collection_count": int(sum(s["documents"] for s in stats_shards)),
            "documents_indexed": int(total_indexed),
            "max_per_item": int(max_per_item),
            "limit_items": limit_items,
            "embedding_dimension": int(self.embedder.dimension),
            "shards": stats_shards,
            "runtime_seconds": round(time.perf_counter() - started, 1),
        }
        logger.info("review_sharded_build_completed %s", stats)
        return stats
