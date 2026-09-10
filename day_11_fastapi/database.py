import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "observability.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL,
            start_time TEXT NOT NULL,
            total_latency_ms REAL,
            model_version TEXT,
            prompt_version TEXT,
            outcome TEXT NOT NULL,
            error_category TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieved_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            score REAL,
            FOREIGN KEY (request_id)
                REFERENCES requests(request_id)
                ON DELETE CASCADE
        )
        """
    )

    connection.commit()
    connection.close()


def log_request(
    request_id,
    endpoint,
    start_time,
    total_latency_ms,
    model_version,
    prompt_version,
    outcome,
    error_category=None
):
    connection = get_connection()

    connection.execute(
        """
        INSERT OR REPLACE INTO requests (
            request_id,
            endpoint,
            start_time,
            total_latency_ms,
            model_version,
            prompt_version,
            outcome,
            error_category
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            endpoint,
            start_time,
            total_latency_ms,
            model_version,
            prompt_version,
            outcome,
            error_category
        )
    )

    connection.commit()
    connection.close()


def log_retrieved_sources(
    request_id,
    sources,
    scores
):
    connection = get_connection()

    for source_id, score in zip(sources, scores):
        connection.execute(
            """
            INSERT INTO retrieved_sources (
                request_id,
                source_id,
                score
            )
            VALUES (?, ?, ?)
            """,
            (
                request_id,
                source_id,
                score
            )
        )

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully.")