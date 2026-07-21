from __future__ import annotations
from config.settings import settings

import json
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from config.path_config import *


LOGGER = logging.getLogger("qdrant_ingestion")

load_dotenv()

class QdrantIngestionPipeline:
    """
    Embed JSONL text chunks and upload them to Qdrant.

    Expected JSONL fields:
    - chunk_id
    - document_id
    - source_file
    - chunk_index
    - page_start
    - page_end
    - text
    - character_count
    - metadata
    """

    REQUIRED_FIELDS = {
        "chunk_id",
        "document_id",
        "chunk_index",
        "page_start",
        "page_end",
        "text",
    }
    def __init__(
            self,
            chunks_path: str | Path,
            qdrant_url: str = "http://localhost:6333",
            collection_name: str = "pdf_chunks",
            embedding_model: str = "BAAI/bge-small-en-v1.5",
            api_key: str | None = None,
            batch_size: int = 64,
            recreate_collection: bool = False,
            verbose_logging: bool = False,
        ) -> None:
            self.chunks_path = Path(chunks_path)
            self.qdrant_url = qdrant_url
            self.collection_name = collection_name
            self.embedding_model = embedding_model
            self.api_key = api_key
            self.batch_size = batch_size
            self.recreate_collection = recreate_collection
            self.verbose_logging = verbose_logging

            self.client: QdrantClient | None = None

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=(
                logging.DEBUG
                if self.verbose_logging
                else logging.INFO
            ),
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def connect(self) -> QdrantClient:
        """Create and verify the Qdrant connection."""

        LOGGER.info("Connecting to Qdrant at %s", self.qdrant_url)

        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.api_key,
            timeout=120,
        )

        self.client.get_collections()

        LOGGER.info("Connected to Qdrant")

        return self.client

    def get_client(self) -> QdrantClient:
        """Return the connected client or create one."""

        if self.client is None:
            return self.connect()

        return self.client

    def read_chunks(
        self,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Read and validate JSONL chunks one record at a time."""

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file does not exist: {self.chunks_path}"
            )

        if not self.chunks_path.is_file():
            raise ValueError(
                f"Chunks path is not a file: {self.chunks_path}"
            )

        with self.chunks_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            for line_number, raw_line in enumerate(
                input_file,
                start=1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} must contain a JSON object"
                    )

                missing_fields = self.REQUIRED_FIELDS.difference(
                    record
                )

                if missing_fields:
                    raise ValueError(
                        f"Line {line_number} is missing fields: "
                        f"{sorted(missing_fields)}"
                    )

                text = record.get("text")

                if not isinstance(text, str) or not text.strip():
                    LOGGER.warning(
                        "Skipping line %s because text is empty",
                        line_number,
                    )
                    continue

                yield line_number, record

    def batched(
        self,
        records: Iterator[tuple[int, dict[str, Any]]],
    ) -> Iterator[list[tuple[int, dict[str, Any]]]]:
        """Yield records in batches."""

        batch: list[tuple[int, dict[str, Any]]] = []

        for record in records:
            batch.append(record)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    @staticmethod
    def make_point_id(chunk_id: str) -> str:
        """
        Convert the parser chunk ID into a stable Qdrant UUID.

        Re-ingesting the same chunk produces the same point ID.
        """

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"omnibrain-chunk:{chunk_id}",
            )
        )

    @staticmethod
    def clean_payload_value(value: Any) -> Any:
        """Convert values into Qdrant-compatible JSON values."""

        if value is None:
            return None

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): QdrantIngestionPipeline.clean_payload_value(
                    item
                )
                for key, item in value.items()
                if item is not None
            }

        if isinstance(value, (list, tuple, set)):
            return [
                QdrantIngestionPipeline.clean_payload_value(item)
                for item in value
                if item is not None
            ]

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    def build_payload(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert one chunk record into a Qdrant payload."""

        metadata = record.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        payload = {
            "chunk_id": str(record["chunk_id"]),
            "document_id": str(record["document_id"]),
            "source_file": record.get("source_file"),
            "chunk_index": int(record["chunk_index"]),
            "page_start": int(record["page_start"]),
            "page_end": int(record["page_end"]),
            "text": str(record["text"]).strip(),
            "character_count": record.get("character_count"),

            "filename": metadata.get("filename"),
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "keywords": metadata.get("keywords"),
            "creator": metadata.get("creator"),
            "producer": metadata.get("producer"),
            "file_sha256": metadata.get("file_sha256"),

            "metadata": metadata,
        }

        return {
            key: self.clean_payload_value(value)
            for key, value in payload.items()
            if value is not None
        }

    def create_payload_indexes(self) -> None:
        """Create indexes for commonly filtered payload fields."""

        client = self.get_client()

        indexes = (
            (
                "chunk_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "document_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "filename",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "file_sha256",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "chunk_index",
                models.PayloadSchemaType.INTEGER,
            ),
            (
                "page_start",
                models.PayloadSchemaType.INTEGER,
            ),
            (
                "page_end",
                models.PayloadSchemaType.INTEGER,
            ),
        )

        for field_name, field_schema in indexes:
            try:
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )

                LOGGER.info(
                    "Created payload index: %s",
                    field_name,
                )

            except Exception as exc:
                LOGGER.debug(
                    "Could not create index '%s': %s",
                    field_name,
                    exc,
                )

    def ensure_collection(self) -> None:
        """Create or recreate the Qdrant collection."""

        client = self.get_client()

        collection_exists = client.collection_exists(
            self.collection_name
        )

        if collection_exists and self.recreate_collection:
            LOGGER.warning(
                "Deleting collection: %s",
                self.collection_name,
            )

            client.delete_collection(
                collection_name=self.collection_name
            )

            collection_exists = False

        if collection_exists:
            LOGGER.info(
                "Using existing collection: %s",
                self.collection_name,
            )
            return

        vector_size = client.get_embedding_size(
            self.embedding_model
        )

        LOGGER.info(
            "Creating collection '%s' with vector size %s",
            self.collection_name,
            vector_size,
        )

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
            on_disk_payload=True,
        )

        self.create_payload_indexes()

    def upload_batch(
        self,
        batch: list[tuple[int, dict[str, Any]]],
    ) -> int:
        """Embed and upload one batch."""

        client = self.get_client()

        ids: list[str] = []
        documents: list[models.Document] = []
        payloads: list[dict[str, Any]] = []

        for line_number, record in batch:
            try:
                chunk_id = str(record["chunk_id"])
                text = str(record["text"]).strip()

                ids.append(self.make_point_id(chunk_id))

                documents.append(
                    models.Document(
                        text=text,
                        model=self.embedding_model,
                    )
                )

                payloads.append(
                    self.build_payload(record)
                )

            except Exception as exc:
                raise ValueError(
                    f"Could not prepare line {line_number}: {exc}"
                ) from exc

        client.upload_collection(
            collection_name=self.collection_name,
            ids=ids,
            vectors=documents,
            payload=payloads,
            batch_size=self.batch_size,
            parallel=1,
            max_retries=3,
            wait=True,
        )

        return len(ids)

    def count_source_records(self) -> int:
        """Count valid, non-empty JSONL chunk records."""

        return sum(1 for _ in self.read_chunks())

    def count_qdrant_points(self) -> int:
        """Return the exact number of points in the collection."""

        client = self.get_client()

        result = client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def verify(self, test_query: str | None = None) -> None:
        """Verify collection status, point count, and semantic search."""

        client = self.get_client()

        if not client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Collection does not exist: "
                f"{self.collection_name}"
            )

        collection_info = client.get_collection(
            self.collection_name
        )

        qdrant_count = self.count_qdrant_points()

        print()
        print("Qdrant verification")
        print("=" * 70)
        print("Collection:", self.collection_name)
        print("Status:", collection_info.status)
        print("Exact point count:", qdrant_count)

        if qdrant_count == 0:
            raise RuntimeError(
                "Collection exists but contains no points"
            )

        sample_points, _ = client.scroll(
            collection_name=self.collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )

        print()
        print("Stored point samples")
        print("=" * 70)

        for index, point in enumerate(sample_points, start=1):
            payload = point.payload or {}

            print(f"Sample {index}")
            print("Qdrant ID:", point.id)
            print("Chunk ID:", payload.get("chunk_id"))
            print("Document ID:", payload.get("document_id"))
            print("Filename:", payload.get("filename"))
            print(
                "Pages:",
                payload.get("page_start"),
                "-",
                payload.get("page_end"),
            )
            print(
                "Text:",
                str(payload.get("text", ""))[:300],
            )
            print("-" * 70)

        if test_query:
            self.search(
                query=test_query,
                limit=5,
            )

    def search(
        self,
        query: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[models.ScoredPoint]:
        """Run semantic search against the collection."""

        client = self.get_client()

        query_filter = None

        if document_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )

        response = client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query,
                model=self.embedding_model,
            ),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        print()
        print("Semantic search")
        print("=" * 70)
        print("Query:", query)

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            payload = point.payload or {}

            print()
            print(f"Result {rank}")
            print("Score:", round(float(point.score), 4))
            print("Filename:", payload.get("filename"))
            print(
                "Pages:",
                payload.get("page_start"),
                "-",
                payload.get("page_end"),
            )
            print(
                "Text:",
                str(payload.get("text", ""))[:500],
            )

        return response.points

    def run(
        self,
        limit: int | None = None,
        verify_after_ingestion: bool = True,
        test_query: str | None = None,
    ) -> int:
        """
        Run the complete ingestion pipeline.

        Steps:
        1. configure logging;
        2. connect to Qdrant;
        3. create or reuse the collection;
        4. read and embed chunks;
        5. upload points;
        6. optionally verify the result.
        """

        self.configure_logging()
        self.connect()
        self.ensure_collection()

        uploaded_total = 0
        batch_number = 0

        records = self.read_chunks()

        for batch in self.batched(records):
            if limit is not None:
                remaining = limit - uploaded_total

                if remaining <= 0:
                    break

                batch = batch[:remaining]

            if not batch:
                break

            batch_number += 1

            uploaded_count = self.upload_batch(batch)
            uploaded_total += uploaded_count

            LOGGER.info(
                "Batch %s uploaded %s points. Total=%s",
                batch_number,
                uploaded_count,
                uploaded_total,
            )

            if limit is not None and uploaded_total >= limit:
                break

        LOGGER.info(
            "Ingestion finished. Uploaded this run=%s",
            uploaded_total,
        )

        LOGGER.info(
            "Exact Qdrant point count=%s",
            self.count_qdrant_points(),
        )

        if verify_after_ingestion:
            self.verify(test_query=test_query)

        return uploaded_total


if __name__ == "__main__":
    pipeline = QdrantIngestionPipeline(CHUNKS,
        qdrant_url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        recreate_collection=False,
    )

    pipeline.run(
        limit=100,
        test_query=(
            "What methods are used for biomedical "
            "signal processing?"
        )
    )