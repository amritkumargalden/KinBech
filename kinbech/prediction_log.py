"""Durable, local audit logging for successful KinBech predictions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PredictionLogError(RuntimeError):
    """Raised when a completed prediction cannot be recorded."""


def log_prediction(
    database_path: str | Path,
    *,
    request_data: dict[str, Any],
    result: dict[str, Any],
    source: str,
    model_version: str | None,
) -> int:
    """Persist only validated valuation inputs and the result, never user identity."""
    path = Path(database_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5)
        try:
            with connection:
                connection.execute("PRAGMA busy_timeout = 5000")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        source TEXT NOT NULL,
                        model_version TEXT,
                        request_json TEXT NOT NULL,
                        result_json TEXT NOT NULL
                    )
                    """
                )
                cursor = connection.execute(
                    """
                    INSERT INTO prediction_events (
                        created_at, source, model_version, request_json, result_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        source,
                        model_version,
                        json.dumps(request_data, sort_keys=True, allow_nan=False),
                        json.dumps(result, sort_keys=True, allow_nan=False),
                    ),
                )
                event_id = int(cursor.lastrowid)
            return event_id
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        raise PredictionLogError("Could not persist prediction event") from error
