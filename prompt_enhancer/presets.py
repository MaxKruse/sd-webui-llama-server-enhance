"""Preset file discovery and loading."""

from __future__ import annotations

from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def list_presets() -> list[str]:
    """Return sorted list of preset names (filenames without .txt extension)."""
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.txt"))


def load_preset(name: str) -> str | None:
    """Load a preset file by name (without extension). Returns None if not found."""
    preset_path = PRESETS_DIR / f"{name}.txt"
    if not preset_path.is_file():
        return None
    return preset_path.read_text(encoding="utf-8").strip()
