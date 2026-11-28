"""Run a query on a SQLite database read-only, with a time limit."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def run_sql(db: Path | str, sql: str, timeout: float = 60.0) -> tuple[str, int | None, float]:
    """Returns (status, rows, seconds); status is "ok", "timeout" or "error: <message>"."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    t0 = time.monotonic()
    try:
        n = 0
        cur = con.execute(sql)
        while chunk := cur.fetchmany(10000):
            n += len(chunk)
        return "ok", n, time.monotonic() - t0
    except sqlite3.OperationalError as e:
        return ("timeout" if str(e) == "interrupted" else f"error: {e}"), None, time.monotonic() - t0
    except Exception as e:  # noqa: BLE001  any failure drops the gold
        return f"error: {type(e).__name__}: {e}", None, time.monotonic() - t0
    finally:
        con.close()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
