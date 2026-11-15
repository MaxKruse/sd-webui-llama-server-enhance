"""Preset file discovery and loading.

Supports automatic preset selection based on the Forge-Neo UI preset
(shared.opts.forge_preset) and the loaded checkpoint name.
"""

from __future__ import annotations

from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"

# Mapping from Forge-Neo UI preset names (PresetArch) to our LLM preset filenames.
# Keys are lowercase forge preset names; values are preset names (without .txt).
FORGE_TO_PRESET_MAP: dict[str, str] = {
    "flux": "flux-dev",
    "zit": "z-image-turbo",
    "anima": "anima",
}

# Checkpoint name substrings (case-insensitive) that override the base flux preset
# to a more specific variant. Checked in order — first match wins.
FLUX_CHECKPOINT_OVERRIDES: list[tuple[str, str]] = [
    ("krea", "flux-krea-dev"),
]

# Presets whose system prompt explicitly forbids negative prompts (FLUX and
# Z-Image models have no negative-prompt conditioning). Enhancing a negative
# prompt against these presets contradicts the preset, so the negative step
# is skipped with a console notice instead.
NEGATIVES_IGNORED_PRESETS: frozenset[str] = frozenset(
    {"flux-dev", "flux-krea-dev", "z-image-turbo"}
)


def negatives_ignored(name: str) -> bool:
    """True if the named preset's model ignores negative prompts."""
    return name in NEGATIVES_IGNORED_PRESETS


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


def resolve_auto_preset(
    *,
    forge_preset: str | None = None,
    checkpoint_name: str | None = None,
) -> str | None:
    """Resolve the preset from the current Forge-Neo UI preset (always auto).

    Args:
        forge_preset: Current Forge-Neo UI preset (shared.opts.forge_preset).
        checkpoint_name: Current checkpoint name (shared.opts.sd_model_checkpoint).

    Returns:
        The resolved preset name (e.g. "flux-dev"), or None if the Forge
        preset has no mapping.
    """
    if not forge_preset:
        return None

    forge_key = forge_preset.lower()

    # Direct mapping (e.g. "zit" → "z-image-turbo")
    if forge_key in FORGE_TO_PRESET_MAP:
        return FORGE_TO_PRESET_MAP[forge_key]

    # Flux family: check checkpoint name for variant override
    if forge_key == "flux" and checkpoint_name:
        ckpt_lower = checkpoint_name.lower()
        for substring, preset_name in FLUX_CHECKPOINT_OVERRIDES:
            if substring in ckpt_lower:
                return preset_name
        # Default flux preset
        return FORGE_TO_PRESET_MAP.get("flux")

    return None
