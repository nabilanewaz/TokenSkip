"""
utils/io.py
-----------
Shared I/O helpers used across all phases:
    load_jsonl / save_jsonl   — JSONL data files
    load_config               — protocol.yaml
    write_json                — metrics and result files
"""

from __future__ import annotations

import json
import pathlib
from typing import Any


# ── YAML loading (optional at import, required at call time) ──────────────────

def _get_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading.  "
            "Install with: pip install pyyaml"
        )


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: str | pathlib.Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts. Blank lines are skipped."""
    data: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str | pathlib.Path) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data):>6} examples → {path}")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def write_json(data: Any, path: str | pathlib.Path) -> None:
    """Write a dict/list to a pretty-printed JSON file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | pathlib.Path) -> Any:
    """Read a JSON file and return its contents."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str | pathlib.Path | None = None) -> dict:
    """
    Load ``configs/protocol.yaml``.

    Search order:
    1. ``config_path`` if explicitly provided
    2. ``research/configs/protocol.yaml`` relative to this file's parent-parent
    3. ``configs/protocol.yaml`` relative to cwd

    Returns an empty dict if nothing is found (graceful degradation).
    """
    yaml = _get_yaml()

    candidates: list[pathlib.Path] = []
    if config_path is not None:
        candidates.append(pathlib.Path(config_path))

    # Relative to utils/ → go up two levels to research/ root
    research_root = pathlib.Path(__file__).resolve().parent.parent
    candidates.append(research_root / "configs" / "protocol.yaml")
    candidates.append(pathlib.Path("configs") / "protocol.yaml")

    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    return {}
