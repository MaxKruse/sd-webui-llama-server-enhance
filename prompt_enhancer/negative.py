"""Negative-prompt enhancement: prompt-building and output validation.

Pure logic, host-independent (stdlib only). Consumed by
scripts/prompt_enhancer.py and by test_negative.py.

Two responsibilities:
1. Build the negative-enhancement system-prompt block and the user message.
2. Validate the LLM's output: reject echoes of the positive prompt or
   user-message markers, before it reaches the generation pipeline.
"""

from __future__ import annotations

# Legit enhanced negatives are a single line of comma-separated tags
# (~100-250 chars). Anything longer is almost certainly an echo.
_MAX_NEGATIVE_CHARS = 300

# Minimum length of a positive-prompt fragment that counts as an echo probe.
_ECHO_PROBE_CHARS = 60


def build_negative_instruction() -> str:
    """Build the negative-prompt-enhancement instruction block.

    Appended to the preset (and its resolution/DP notes) for negative
    calls. The LLM receives the enhanced positive prompt for context and
    the original negative prompt to enhance.
    """
    return (
        "\n\n--- Negative Prompt Enhancement ---\n"
        "You are now enhancing a NEGATIVE prompt. The user message contains:\n"
        "  - 'POSITIVE: ' followed by the already-enhanced positive prompt (context only - NEVER repeat it, quote it, paraphrase it, or describe it)\n"
        "  - 'NEGATIVE: ' followed by the original negative prompt to enhance\n\n"
        "Your job: extend the original negative prompt to best complement this specific positive scene.\n"
        "- Keep every tag already in the original negative prompt.\n"
        "- Make sure the standard base list is present: artist name, watermark, signature, text, blurry, jpeg artifacts, chromatic aberration.\n"
        "- Add scene-relevant unwanted elements for THIS scene: things that would ruin the composition, subject, lighting, or atmosphere described in the positive prompt.\n"
        "- NEVER output the positive prompt or any part of it. NEVER repeat the user message format ('POSITIVE: ', 'NEGATIVE: '). NEVER add quality tags (masterpiece, best quality).\n"
        "- Output ONLY the enhanced negative prompt as a single line of comma-separated tags. No natural language, no sentences, no explanations, no quotes, no markdown.\n\n"
        "Example:\n"
        "POSITIVE: 1girl, solo, astronaut, ikumi <lora:Ikumi-Anima-v2-phase2:1>. Ikumi looks at the viewer while standing inside a spacecraft cabin. Warm blue lighting fills the scene.\n"
        "NEGATIVE: worst quality, low quality, score_1-3, artist name\n"
        "Output: worst quality, low quality, score_1-3, artist name, watermark, signature, text, blurry, jpeg artifacts, chromatic aberration, bad anatomy, deformed hands, extra fingers, missing fingers, extra limbs\n"
    )


def build_negative_user_prompt(positive: str, negative: str) -> str:
    """Format a user message for the negative-prompt-enhancement call."""
    return f"POSITIVE: {positive}\n\nNEGATIVE: {negative}"


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_echo_of_positive(enhanced: str, positive: str) -> bool:
    """True if the enhanced output contains a substantial fragment of the positive."""
    n_positive = _normalize(positive)
    n_enhanced = _normalize(enhanced)
    if not n_positive or not n_enhanced:
        return False
    if n_positive in n_enhanced:
        return True
    # Slide a window along the negative output: any substantial fragment
    # pasted from the positive counts as an echo.
    probe = min(_ECHO_PROBE_CHARS, len(n_enhanced))
    if probe < 30:
        return False
    step = max(1, probe // 2)
    for start in range(0, len(n_enhanced) - probe + 1, step):
        if n_enhanced[start:start + probe] in n_positive:
            return True
    return False


def _strip_leading_marker(text: str) -> tuple[str, str | None]:
    """Strip a stray user-message marker the model copied from its input.

    Returns (cleaned_text, reason) where reason is set when the result is
    empty after the strip.
    """
    for marker in ("POSITIVE:", "NEGATIVE:"):
        if text.upper().startswith(marker):
            text = text[len(marker):].strip()
            break
    if not text:
        return "", "empty response after marker strip"
    return text, None


def validate_negative_output(
    positive: str,
    original_negative: str,
    enhanced: str | None,
) -> tuple[str | None, str | None]:
    """Validate an LLM-enhanced negative prompt before it reaches the pipeline.

    Args:
        positive: The enhanced positive prompt sent to the model (for echo detection).
        original_negative: The original negative prompt (outputs identical to it are rejected).
        enhanced: The raw content the model returned.

    Returns:
        (cleaned_output, None) on success, or (None, reason) on rejection.
        The cleaned output has a stray leading marker stripped, so it can be
        used as-is.
    """
    if not enhanced or not enhanced.strip():
        return None, "empty response"

    text, dead_reason = _strip_leading_marker(enhanced.strip())
    if dead_reason:
        return None, dead_reason

    # A legit negative is a single line of comma-separated tags.
    if "\n" in text or "\r" in text:
        return None, "multi-line output"

    # Model pasted the positive prompt (or a large fragment) as the negative.
    if _is_echo_of_positive(text, positive):
        return None, "output contains the positive prompt"

    # Markers/echoes sometimes sneak into the middle of the output.
    upper = text.upper()
    if "POSITIVE:" in upper or "NEGATIVE:" in upper:
        return None, "output contains the user-message format"

    if len(text) > _MAX_NEGATIVE_CHARS:
        return None, f"output too long ({len(text)} chars)"

    # A negative that is just the original (or worse) is not an enhancement.
    # Keep it only if it actually adds something.
    n_original = _normalize(original_negative)
    if n_original and _normalize(text) == n_original:
        return None, "output is identical to the original negative"

    return text, None
