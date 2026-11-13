"""
Shared configuration — loading .env and database/storage settings
=================================================================

.env is loaded here, once, for every script of the package. load_dotenv does
not overwrite variables that are already set: a variable exported in the
shell takes precedence over .env.
"""

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Search for .env in current directory, project directory, and parent directory
_project_root = Path(__file__).resolve().parent.parent.parent
_candidates = [
    Path.cwd() / ".env",
    _project_root / ".env",
    _project_root.parent / ".env",
]
for _p in _candidates:
    if _p.exists():
        load_dotenv(_p, override=False)


def add_db_url_argument(parser: argparse.ArgumentParser) -> None:
    """Adds --db_url, optional. If unset or 'local', the built-in local vector store is used."""
    default = os.getenv("DB_URL", "local")
    parser.add_argument(
        "--db_url", type=str, default=default, required=False,
        help="PostgreSQL URL (e.g. postgresql://user:pass@localhost:5432/ragdb) "
             "or 'local' to use built-in local vector storage (default: DB_URL or 'local')",
    )


def get_db_connection(db_url: str | None) -> Any:
    """Returns a PostgreSQL connection if db_url is a postgresql:// URL, or 'local'."""
    if not db_url or db_url == "local" or not db_url.startswith("postgresql://"):
        return "local"
    try:
        import psycopg2
        return psycopg2.connect(db_url)
    except Exception:
        return "local"
