import json
import logging
import sqlite3
from datetime import datetime

from agent_utilities.core.paths import memory_view_dir

from .base import TimeSeriesBackend, TimeSeriesDataPoint

logger = logging.getLogger(__name__)


class SQLiteTimeSeriesBackend(TimeSeriesBackend):
    """Local embedded SQLite backend for high-frequency time-series data."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(memory_view_dir() / "timeseries.db")
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Initialize the database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeseries_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metrics JSON NOT NULL,
                tags JSON,
                UNIQUE(symbol, timestamp)
            )
        """)
        # Create indexes for fast querying
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_time ON timeseries_data (symbol, timestamp)"
        )
        self.conn.commit()
        logger.debug(f"SQLiteTimeSeriesBackend initialized at {self.db_path}")

    def insert(self, points: list[TimeSeriesDataPoint]) -> None:
        """Insert a batch of time-series points."""
        if not self.conn:
            self.initialize()
        assert self.conn is not None

        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO timeseries_data (symbol, timestamp, metrics, tags)
                VALUES (?, ?, ?, ?)
            """,
                [
                    (
                        p.symbol,
                        p.timestamp.isoformat(),
                        json.dumps(p.metrics),
                        json.dumps(p.tags) if p.tags else None,
                    )
                    for p in points
                ],
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to insert time-series batch: {e}")
            raise

    def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        tags: dict[str, str] | None = None,
    ) -> list[TimeSeriesDataPoint]:
        """Query time-series points for a specific symbol within a time range."""
        if not self.conn:
            self.initialize()
        assert self.conn is not None

        cursor = self.conn.cursor()

        query = "SELECT symbol, timestamp, metrics, tags FROM timeseries_data WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?"
        params = [symbol, start_time.isoformat(), end_time.isoformat()]

        # Note: SQLite JSON querying is possible, but for simplicity we filter tags in memory
        # since this is an embedded abstraction.
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            row_tags = json.loads(row["tags"]) if row["tags"] else None

            # Filter by tags if provided
            if tags and row_tags:
                if not all(row_tags.get(k) == v for k, v in tags.items()):
                    continue

            results.append(
                TimeSeriesDataPoint(
                    symbol=row["symbol"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metrics=json.loads(row["metrics"]),
                    tags=row_tags,
                )
            )

        return results

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
