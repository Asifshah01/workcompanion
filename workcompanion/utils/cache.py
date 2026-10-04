"""Small disk-backed TTL cache.

Used to cut token cost (LLM responses) and latency (embeddings, paper
analyses).  Entries are stored as individual JSON files so the cache survives
restarts and never needs to be loaded in full.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["DiskCache", "cache_key"]


def cache_key(namespace: str, *parts: Any) -> str:
    """Build a deterministic cache key from a namespace and arbitrary parts."""
    raw = "|".join([namespace, *(str(p) for p in parts)])
    return f"{namespace}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


class DiskCache:
    """TTL cache persisted as JSON files under a single directory."""

    def __init__(self, directory: str | Path, *, ttl_seconds: int = 86_400, enabled: bool = True):
        self.directory = Path(directory)
        self.ttl_seconds = int(ttl_seconds)
        self.enabled = bool(enabled) and self.ttl_seconds > 0
        self._lock = threading.RLock()
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _path(self, key: str) -> Path:
        safe = "".join(c for c in key if c.isalnum() or c in "-_.")[:120]
        return self.directory / f"{safe}.json"

    def get(self, key: str, default: Any = None) -> Any:
        """Return a cached value, or ``default`` when missing/expired."""
        if not self.enabled:
            return default
        path = self._path(key)
        try:
            if not path.is_file():
                return default
            if self.ttl_seconds > 0 and (time.time() - path.stat().st_mtime) > self.ttl_seconds:
                path.unlink(missing_ok=True)
                return default
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)["value"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Store a value (JSON serialisable)."""
        if not self.enabled:
            return
        path = self._path(key)
        payload = {"key": key, "ts": time.time(), "value": value}
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
                ) as handle:
                    json.dump(payload, handle, ensure_ascii=False, default=str)
                    temp_path = handle.name
                os.replace(temp_path, path)
            except (OSError, TypeError, ValueError):
                # Cache writes are best-effort and must never break the app.
                return

    def delete(self, key: str) -> None:
        if not self.enabled:
            return
        self._path(key).unlink(missing_ok=True)

    def clear(self) -> int:
        """Delete every cache entry; returns the number of files removed."""
        if not self.directory.is_dir():
            return 0
        removed = 0
        with self._lock:
            for item in self.directory.glob("*.json"):
                try:
                    item.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def stats(self) -> dict[str, Any]:
        """Cache size / entry count, for the Settings page."""
        if not self.directory.is_dir():
            return {"entries": 0, "size_bytes": 0, "enabled": self.enabled}
        files = list(self.directory.glob("*.json"))
        return {
            "entries": len(files),
            "size_bytes": sum(f.stat().st_size for f in files if f.exists()),
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
        }