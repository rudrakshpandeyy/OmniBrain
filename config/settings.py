from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    embedding_model: str
    embedding_batch_size: int
    openai_api_key: str | None


def load_settings() -> Settings:
    return Settings(
        qdrant_url=os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        qdrant_api_key=(
            os.getenv("QDRANT_API_KEY") or None
        ),
        qdrant_collection=os.getenv(
            "QDRANT_COLLECTION",
            "pdf_chunks",
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5",
        ),
        embedding_batch_size=int(
            os.getenv("EMBEDDING_BATCH_SIZE", "64")
        ),
        openai_api_key=(
            os.getenv("OPENAI_API_KEY") or None
        ),
    )


settings = load_settings()