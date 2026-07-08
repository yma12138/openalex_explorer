"""SQLite-backed paper store.

Persists paper metadata and pipeline state across stages.
No dependency beyond Python's built-in sqlite3.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models import PaperMeta


@dataclass
class PaperRecord:
    """A paper record in the store, spanning the full pipeline."""

    # --- Identity (immutable after insert) ---
    id: str  # ArXiv ID, e.g. "1706.03762"
    title: str
    authors: list[str]  # stored as JSON in DB
    abstract: str
    year: int
    source: str = "arxiv"
    pdf_url: Optional[str] = None
    categories: list[str] = field(default_factory=list)  # e.g. ["cs.CV", "cs.AI"]
    keywords: list[str] = field(
        default_factory=list
    )  # e.g. ["transformer", "attention"]
    doi: Optional[str] = None
    journal_ref: Optional[str] = None

    # --- Pipeline state (mutable) ---
    status: str = "searched"
    # status flow:
    #   searched → relevanced → quality_passed → summarized → ready
    #        ↓          ↓              ↓
    #       rejected  rejected      rejected

    # --- Scores (filled progressively) ---
    relevance_score: float = 0.0  # 0-10, filled in relevance_filter
    relevance_reason: str = ""  # 相关性筛选理由
    quality_score: float = 0.0  # 0-100, filled in quality evaluation
    quality_reason: str = ""  # 质量审核理由

    # --- Summary (filled after quality pass) ---
    summary_json: str = ""  # JSON, PaperSummary 序列化

    # --- File paths (filled in PDF/parse stages) ---
    pdf_path: Optional[str] = None
    md_path: Optional[str] = None

    # --- Metadata ---
    search_query: str = ""  # which query found this paper
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT NOT NULL,          -- JSON array
    abstract        TEXT NOT NULL DEFAULT '',
    year            INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL DEFAULT 'arxiv',
    pdf_url         TEXT,
    categories      TEXT DEFAULT '[]',      -- JSON array
    keywords        TEXT DEFAULT '[]',      -- JSON array
    doi             TEXT,
    journal_ref     TEXT,

    status          TEXT NOT NULL DEFAULT 'searched',
    relevance_score REAL NOT NULL DEFAULT 0.0,
    relevance_reason TEXT DEFAULT '',
    quality_score   REAL NOT NULL DEFAULT 0.0,
    quality_reason  TEXT DEFAULT '',
    summary_json    TEXT DEFAULT '',

    pdf_path        TEXT,
    md_path         TEXT,

    search_query    TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_relevance ON papers(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_quality ON papers(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_year ON papers(year DESC);
"""


class PaperStore:
    """SQLite-backed store for papers across the pipeline."""

    def __init__(self, db_path: str | Path = "papers.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # ---- Write operations ----

    def insert_paper(self, paper: PaperMeta, search_query: str = "") -> str:
        """Insert a paper from an API result. Returns the paper ID.

        If the paper already exists (same ID), updates metadata but
        preserves pipeline state.
        """
        now = datetime.now().isoformat()

        categories = json.dumps(paper.categories or [])
        keywords = json.dumps(paper.keywords or [])

        self._conn.execute(
            """
            INSERT INTO papers
                (id, title, authors, abstract, year, source, pdf_url,
                 categories, keywords, status, search_query, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'searched', ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title       = excluded.title,
                authors     = excluded.authors,
                abstract    = excluded.abstract,
                year        = excluded.year,
                pdf_url     = excluded.pdf_url,
                categories  = excluded.categories,
                keywords    = excluded.keywords,
                updated_at  = excluded.updated_at
            """,
            (
                paper.id,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                paper.year,
                paper.source,
                paper.pdf_url or "",
                categories,
                keywords,
                search_query,
                now,
                now,
            ),
        )
        self._conn.commit()
        return paper.id

    def insert_papers(
        self, papers: list[PaperMeta], search_query: str = ""
    ) -> list[str]:
        """Insert multiple papers. Returns list of IDs."""
        ids = []
        for p in papers:
            ids.append(self.insert_paper(p, search_query))
        return ids

    def update_status(self, paper_id: str, status: str, **extra) -> None:
        """Update a paper's status and optionally extra fields.

        Args:
            paper_id: Paper ID to update.
            status: New status value.
            **extra: Additional fields to update, e.g.
                relevance_score=4.5, pdf_path="output/paper.pdf"
        """
        now = datetime.now().isoformat()
        fields = {"status": status, "updated_at": now}
        fields.update(extra)

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [paper_id]

        self._conn.execute(f"UPDATE papers SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    # ---- Read operations ----

    def get_paper(self, paper_id: str) -> Optional[PaperRecord]:
        """Get a single paper by ID."""
        row = self._conn.execute(
            "SELECT * FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_papers(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "relevance_score",
    ) -> list[PaperRecord]:
        """List papers, optionally filtered by status."""
        allowed_sort = {
            "relevance": "relevance_score DESC",
            "quality": "quality_score DESC",
            "year": "year DESC",
            "created": "created_at DESC",
        }
        order = allowed_sort.get(sort_by, "relevance_score DESC")

        if status:
            rows = self._conn.execute(
                f"SELECT * FROM papers WHERE status = ? "
                f"ORDER BY {order} LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT * FROM papers ORDER BY {order} LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        return [self._row_to_record(r) for r in rows]

    def count(self, status: Optional[str] = None) -> int:
        """Count papers, optionally filtered by status."""
        if status:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM papers WHERE status = ?",
                (status,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) as cnt FROM papers").fetchone()
        return row["cnt"] if row else 0

    # ---- Utility ----

    def close(self):
        self._conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> PaperRecord:
        authors = json.loads(row["authors"]) if row["authors"] else []
        categories = json.loads(row["categories"]) if row["categories"] else []
        keywords = json.loads(row["keywords"]) if row["keywords"] else []

        return PaperRecord(
            id=row["id"],
            title=row["title"],
            authors=authors,
            abstract=row["abstract"],
            year=row["year"],
            source=row["source"],
            pdf_url=row["pdf_url"] or None,
            categories=categories,
            keywords=keywords,
            doi=row["doi"],
            journal_ref=row["journal_ref"],
            relevance_reason=row["relevance_reason"] or "",
            quality_reason=row["quality_reason"] or "",
            summary_json=row["summary_json"] or "",
            status=row["status"],
            relevance_score=row["relevance_score"],
            quality_score=row["quality_score"],
            pdf_path=row["pdf_path"],
            md_path=row["md_path"],
            search_query=row["search_query"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
