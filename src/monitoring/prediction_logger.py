"""
SQLite-backed logger for prediction requests.
Logs every /predict call and exposes get_recent() for the /history endpoint.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.parent
_default_data_dir = _project_root / "data"

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logs prediction calls to a local SQLite database.

    One instance is created at API startup and reused for the lifetime
    of the process. Each public method opens and closes its own connection
    to remain safe under FastAPI's default thread-pool executor.
    """

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(os.environ.get("DATA_DIR", str(_default_data_dir)))
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "predictions.db"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT    NOT NULL,
                    inputs_json       TEXT    NOT NULL,
                    prediction        INTEGER NOT NULL,
                    churn_probability REAL    NOT NULL,
                    risk_level        TEXT    NOT NULL,
                    model_name        TEXT    NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def log(
        self,
        inputs: dict[str, Any],
        prediction: int,
        churn_probability: float,
        risk_level: str,
        model_name: str,
    ) -> None:
        """Insert one prediction record. Never raises — DB errors are logged only."""
        timestamp = datetime.now(timezone.utc).isoformat()
        inputs_json = json.dumps(inputs)
        conn = None
        try:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO prediction_log
                    (timestamp, inputs_json, prediction, churn_probability, risk_level, model_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, inputs_json, prediction, churn_probability, risk_level, model_name),
            )
            conn.commit()
        except Exception as exc:
            logger.error("Failed to log prediction: %s", exc)
        finally:
            if conn is not None:
                conn.close()

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the N most recent prediction records, newest first."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, timestamp, inputs_json, prediction,
                       churn_probability, risk_level, model_name
                FROM prediction_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (n,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        records = []
        for row in rows:
            record = dict(row)
            record["inputs"] = json.loads(record.pop("inputs_json"))
            records.append(record)
        return records

    def get_distribution_stats(
        self,
        baseline_churn_rate: float,
        window: int = 50,
        drift_threshold: float = 0.10,
    ) -> dict[str, Any]:
        """
        Compare recent prediction distribution against a training baseline.

        Args:
            baseline_churn_rate: Mean churn probability from training data (0-1).
            window: Number of recent predictions to evaluate.
            drift_threshold: Fractional drift that triggers a warning (default 10%).

        Returns:
            Dict with keys: recent_mean, baseline_mean, drift, drift_detected,
            sample_size, risk_counts.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT churn_probability, risk_level
                FROM prediction_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (window,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "recent_mean": None,
                "baseline_mean": baseline_churn_rate,
                "drift": None,
                "drift_detected": False,
                "sample_size": 0,
                "risk_counts": {"high": 0, "medium": 0, "low": 0},
            }

        probabilities = [row["churn_probability"] for row in rows]
        recent_mean = sum(probabilities) / len(probabilities)
        drift = abs(recent_mean - baseline_churn_rate)

        risk_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for row in rows:
            level = row["risk_level"]
            if level in risk_counts:
                risk_counts[level] += 1

        return {
            "recent_mean": round(recent_mean, 4),
            "baseline_mean": round(baseline_churn_rate, 4),
            "drift": round(drift, 4),
            "drift_detected": drift > drift_threshold,
            "sample_size": len(rows),
            "risk_counts": risk_counts,
        }
