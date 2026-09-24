"""Persistent cache for deterministic reuse of audio observations."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def normalize_query(query: str) -> str:
    return " ".join((query or "").split())


def make_cache_key(*, audio_id: str, query: str, omni_model: str, prompt_version: str) -> str:
    payload = "\0".join((str(audio_id), normalize_query(query), str(omni_model), str(prompt_version)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AudioObservationCache:
    """SQLite-backed cache safe for independent reward-worker processes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS omni_observations (
                    cache_key TEXT PRIMARY KEY,
                    audio_id TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    omni_model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=1)
        connection.execute("PRAGMA busy_timeout=1000")
        return connection

    def get(self, *, audio_id: str, query: str, omni_model: str, prompt_version: str) -> str | None:
        key = make_cache_key(
            audio_id=audio_id,
            query=query,
            omni_model=omni_model,
            prompt_version=prompt_version,
        )
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT observation FROM omni_observations WHERE cache_key = ?", (key,)
            ).fetchone()
        finally:
            connection.close()
        return row[0] if row else None

    def put_if_absent(self, *, audio_id: str, query: str, omni_model: str, prompt_version: str, observation: str) -> str:
        normalized_query = normalize_query(query)
        key = make_cache_key(
            audio_id=audio_id,
            query=normalized_query,
            omni_model=omni_model,
            prompt_version=prompt_version,
        )
        values = (
            key,
            str(audio_id),
            normalized_query,
            str(omni_model),
            str(prompt_version),
            observation,
            datetime.now(UTC).isoformat(),
        )
        with self._write_lock:
            delay = 0.05
            for attempt in range(6):
                connection = None
                try:
                    connection = self._connect()
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO omni_observations (
                            cache_key, audio_id, normalized_query, omni_model,
                            prompt_version, observation, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
                    connection.commit()
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    if attempt == 5:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 1.0)
                finally:
                    if connection is not None:
                        connection.close()
        return key


# Compatibility alias for the pinned September SLIME integration.
OmniCache = AudioObservationCache
