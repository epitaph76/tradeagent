from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|authorization|secret|password|bearer)", re.IGNORECASE)
SECRET_VALUE_RE = re.compile(r"(Bearer\s+[A-Za-z0-9._\-]+|pza_[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{8,})")


def redact(value: Any, *, key: str = "") -> Any:
    if SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_RE.sub("[REDACTED]", value)
    return value


class JsonlLogger:
    def __init__(self, logs_dir: str | Path, *, stdout: bool | None = None):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stdout = (
            os.environ.get("ARENA_LOG_STDOUT", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
            if stdout is None
            else stdout
        )
        self._lock = threading.Lock()

    def write(self, event: str, payload: Mapping[str, Any] | None = None, *, stream: str = "arena_live") -> None:
        now = datetime.now()
        row = {
            "ts": now.isoformat(timespec="seconds"),
            "event": event,
            **redact(dict(payload or {})),
        }
        safe_stream = re.sub(r"[^A-Za-z0-9_\-]+", "_", stream).strip("_") or "arena_live"
        path = self.logs_dir / f"{safe_stream}_{now:%Y%m%d}.jsonl"
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if self.stdout:
                print(line, flush=True)

    def error(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self.write(event, payload, stream="errors")

