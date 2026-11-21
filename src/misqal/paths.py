"""Paths to files shipped in the repository, independent of the working directory."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def shipped(rel: str) -> Path:
    """``rel`` under the repository root (e.g. ``benchmarks/column-roles.jsonl``); falls back to the
    working directory when the package is not run from a source checkout."""
    p = REPO / rel
    return p if p.exists() else Path(rel)
