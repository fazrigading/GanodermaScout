from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Protocol

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class MemoryStore(Protocol):
    def load_history(
        self,
        *,
        palm_id: str | None = None,
        palm_code: str | None = None,
        block_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]: ...


def _sync_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to load inspection history")
    return create_engine(_sync_database_url(database_url))


class SQLAlchemyMemoryStore:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine

    def load_history(
        self,
        *,
        palm_id: str | None = None,
        palm_code: str | None = None,
        block_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        palm_history = []
        block_history = []
        with (self.engine or get_database_engine()).connect() as connection:
            if palm_id:
                palm_history = self._query(connection, "p.id = CAST(:palm_id AS UUID)", {"palm_id": palm_id})
            elif palm_code:
                palm_history = self._query(connection, "p.palm_code = :palm_code", {"palm_code": palm_code})
            if block_id:
                block_history = self._query(connection, "b.id = CAST(:block_id AS UUID)", {"block_id": block_id})
        return {"palm_history": palm_history, "block_history": block_history}

    @staticmethod
    def _query(connection: Any, condition: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        statement = text(
            "SELECT i.id::text AS inspection_id, p.id::text AS palm_id, p.palm_code, "
            "b.id::text AS block_id, b.name AS block_name, i.status, i.created_at, "
            "COALESCE(json_agg(json_build_object("
            "'class_name', d.class_name, 'confidence', d.confidence, "
            "'bbox', json_build_array(d.bbox_x_min, d.bbox_y_min, d.bbox_x_max, d.bbox_y_max)) "
            "ORDER BY d.class_name) FILTER (WHERE d.id IS NOT NULL), '[]'::json) AS detections "
            "FROM inspections i "
            "JOIN palms p ON p.id = i.palm_id "
            "JOIN blocks b ON b.id = p.block_id "
            "LEFT JOIN detections d ON d.inspection_id = i.id "
            f"WHERE {condition} "
            "GROUP BY i.id, p.id, b.id ORDER BY i.created_at DESC LIMIT 50"
        )
        records = []
        for row in connection.execute(statement, parameters).mappings():
            record = dict(row)
            if record["created_at"] is not None:
                record["created_at"] = record["created_at"].isoformat()
            records.append(record)
        return records
