"""Rotating debug log for the enhancement pipeline.

Host-independent (stdlib only). Writes timestamped lines to a file and
rotates to a single backup once the primary grows past the size limit.
Thread-safe: the WebUI main thread and the server's stderr pipe drainer
write concurrently.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Rotate the debug log once it grows past this size (bytes).
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class RotatingLog:
    """Thread-safe appending debug log with one-generation rotation."""

    __slots__ = ("path", "_backup", "_max_bytes", "_lock")

    def __init__(self, path: Path, max_bytes: int = DEFAULT_MAX_BYTES):
        self.path = path
        self._backup = path.with_name(path.name + ".1")
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def _rotate_if_needed(self) -> None:
        """Rotate to the backup once the primary passes the size limit."""
        try:
            if self.path.exists() and self.path.stat().st_size > self._max_bytes:
                if self._backup.exists():
                    self._backup.unlink()
                self.path.rename(self._backup)
        except OSError:
            logger.warning("Could not rotate debug log", exc_info=True)

    def write(self, msg: str) -> None:
        """Append a timestamped message (rotation happens under the lock)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {msg}\n"
        with self._lock:
            self._rotate_if_needed()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
