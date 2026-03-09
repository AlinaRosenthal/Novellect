from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "novellect.db"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


@contextmanager
def get_connection():
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    file_hash TEXT UNIQUE,
    title TEXT NOT NULL,
    author TEXT,
    format TEXT,
    file_path TEXT,
    text_path TEXT,
    added_at TEXT,
    last_opened_at TEXT,
    open_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    token_count INTEGER DEFAULT 0,
    FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    intent TEXT,
    executed_at TEXT,
    result_count INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    used_llm INTEGER DEFAULT 0,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_books_last_opened ON books(last_opened_at);
"""


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def insert_book(book: Dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO books (
                id, file_hash, title, author, format, file_path, text_path,
                added_at, last_opened_at, open_count, char_count, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book["id"],
                book.get("file_hash"),
                book.get("title") or "Без названия",
                book.get("author"),
                book.get("format"),
                book.get("file_path"),
                book.get("text_path"),
                book.get("added_at"),
                book.get("last_opened_at"),
                int(book.get("open_count", 0)),
                int(book.get("char_count", 0)),
                int(book.get("word_count", 0)),
            ),
        )


def replace_chunks(book_id: str, chunks: Sequence[Dict[str, Any]]) -> List[int]:
    inserted_ids: List[int] = []
    with get_connection() as conn:
        conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
        for chunk in chunks:
            cursor = conn.execute(
                """
                INSERT INTO chunks (book_id, chunk_index, text, char_start, char_end, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    book_id,
                    int(chunk.get("chunk_index", 0)),
                    chunk.get("text", ""),
                    chunk.get("char_start"),
                    chunk.get("char_end"),
                    int(chunk.get("token_count", 0)),
                ),
            )
            inserted_ids.append(int(cursor.lastrowid))
    return inserted_ids


def list_books() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT b.*, COUNT(c.id) AS chunks_count
            FROM books b
            LEFT JOIN chunks c ON c.book_id = b.id
            GROUP BY b.id
            ORDER BY b.added_at DESC, b.title COLLATE NOCASE ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_book(book_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
    return dict(row) if row else None


def get_book_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM books WHERE file_hash = ?", (file_hash,)).fetchone()
    return dict(row) if row else None


def get_all_chunks() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.book_id,
                c.chunk_index,
                c.text,
                c.char_start,
                c.char_end,
                c.token_count,
                b.title,
                b.author,
                b.format,
                b.added_at,
                b.last_opened_at,
                b.open_count
            FROM chunks c
            JOIN books b ON b.id = c.book_id
            ORDER BY b.title COLLATE NOCASE ASC, c.chunk_index ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chunks_for_book(book_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.book_id, c.chunk_index, c.text, c.char_start, c.char_end,
                   c.token_count, b.title, b.author, b.format, b.added_at, b.last_opened_at, b.open_count
            FROM chunks c
            JOIN books b ON b.id = c.book_id
            WHERE c.book_id = ?
            ORDER BY c.chunk_index ASC
            """,
            (book_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        books_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        opened_count = conn.execute(
            "SELECT COUNT(*) FROM books WHERE COALESCE(open_count, 0) > 0"
        ).fetchone()[0]
        total_chars = conn.execute("SELECT COALESCE(SUM(char_count), 0) FROM books").fetchone()[0]
    return {
        "books_count": int(books_count),
        "chunks_count": int(chunks_count),
        "opened_books_count": int(opened_count),
        "total_chars": int(total_chars or 0),
    }


def mark_book_opened(book_id: str, opened_at: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE books
            SET last_opened_at = ?, open_count = COALESCE(open_count, 0) + 1
            WHERE id = ?
            """,
            (opened_at, book_id),
        )


def delete_book(book_id: str) -> Optional[Dict[str, Any]]:
    book = get_book(book_id)
    if not book:
        return None
    with get_connection() as conn:
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
    for key in ("file_path", "text_path"):
        path = book.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return book


def clear_library() -> None:
    books = list_books()
    for book in books:
        for key in ("file_path", "text_path"):
            path = book.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    with get_connection() as conn:
        conn.execute("DELETE FROM query_history")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM books")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('chunks', 'query_history')")


def log_query(
    query_text: str,
    intent: str,
    executed_at: str,
    result_count: int,
    latency_ms: int,
    used_llm: bool,
    notes: str = "",
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_history (query_text, intent, executed_at, result_count, latency_ms, used_llm, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_text,
                intent,
                executed_at,
                int(result_count),
                int(latency_ms),
                1 if used_llm else 0,
                notes,
            ),
        )


def list_recent_queries(limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY executed_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
