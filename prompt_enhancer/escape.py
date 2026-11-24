"""Dynamic Prompts escaping: keep LLM output parseable by the DP parser.

Host-independent (stdlib only). The DP parser expands variant blocks
({red|blue}) and wildcards (__name__) AFTER enhancement, so variant syntax
the LLM invented on its own must be escaped (pipes -> commas) before
write-back. Tokens that were already in the user's original prompt, and
variants containing wildcards (the LLM was instructed to preserve them),
are always kept as-is.

Separators are passed explicitly - the entry reads them from the DP
settings - so this module is testable outside the WebUI host.
"""

from __future__ import annotations

DEFAULT_VARIANT_START = "{"
DEFAULT_VARIANT_END = "}"
DEFAULT_WILDCARD_WRAP = "__"


def contains_wildcard(text: str, wc_wrap: str) -> bool:
    """Check if text contains a wildcard token (wc_wrap + name + wc_wrap)."""
    if not wc_wrap:
        return False
    # Look for pattern: wc_wrap ... wc_wrap where there's at least one char between
    wc_len = len(wc_wrap)
    if wc_len * 2 + 1 > len(text):
        return False
    first = text.find(wc_wrap)
    if first < 0:
        return False
    second = text.find(wc_wrap, first + wc_len)
    return second > first + wc_len  # At least one char between the wraps


def find_matching_variant_end(text: str, start: int, v_start: str, v_end: str) -> int | None:
    """Find the position of the matching variant_end, accounting for nesting.

    Returns the index of the first character of variant_end, or None if not found.

    When v_start == v_end (e.g. both are '###'), we check v_end FIRST so the
    closing token is matched before the opening token, keeping depth correct.
    """
    same_delim = v_start == v_end
    depth = 1
    i = start
    limit = len(text) - len(v_end)
    while i <= limit:
        if same_delim:
            # When delimiters are identical, check end first so we close before opening
            if text[i:i + len(v_end)] == v_end:
                depth -= 1
                if depth == 0:
                    return i
                i += len(v_end)
                continue
            # Also acts as start (nesting)
            if text[i:i + len(v_start)] == v_start:
                depth += 1
                i += len(v_start)
                continue
        else:
            if text[i:i + len(v_start)] == v_start:
                depth += 1
                i += len(v_start)
                continue
            if text[i:i + len(v_end)] == v_end:
                depth -= 1
                if depth == 0:
                    return i
                i += len(v_end)
                continue
        i += 1
    return None


def extract_variant_blocks(text: str, v_start: str, v_end: str, out: set[str]) -> None:
    """Extract all variant blocks from text and add them to the out set."""
    i = 0
    while i < len(text):
        if text[i:i + len(v_start)] == v_start:
            end_pos = find_matching_variant_end(text, i + len(v_start), v_start, v_end)
            if end_pos is not None:
                full_block = text[i:end_pos + len(v_end)]
                out.add(full_block)
                i = end_pos + len(v_end)
                continue
        i += 1


def escape_accidental_dp_syntax(
    original: str,
    enhanced: str,
    *,
    v_start: str = DEFAULT_VARIANT_START,
    v_end: str = DEFAULT_VARIANT_END,
    wc_wrap: str = DEFAULT_WILDCARD_WRAP,
) -> str:
    """Escape variant pipes in LLM-generated text that wasn't in the original prompt.

    When the LLM generates text containing DP variant syntax (e.g. {red|blue})
    that wasn't present in the user's original prompt, those pipes would be
    misinterpreted by the Dynamic Prompts parser. This function detects such
    accidental syntax and replaces the pipes with commas.

    Args:
        original: The user's original prompt (before enhancement).
        enhanced: The LLM-enhanced prompt.
        v_start/v_end: Variant block delimiters (from the DP settings).
        wc_wrap: Wildcard wrapper (from the DP settings).

    Returns:
        The enhanced prompt with accidental variant pipes escaped.
    """
    # Collect all original variant blocks for comparison
    original_variants: set[str] = set()
    extract_variant_blocks(original, v_start, v_end, original_variants)

    result: list[str] = []
    i = 0
    while i < len(enhanced):
        # Check for variant start at position i
        if enhanced[i:i + len(v_start)] == v_start:
            # Find the matching end (handles nested variants)
            end_pos = find_matching_variant_end(enhanced, i + len(v_start), v_start, v_end)
            if end_pos is not None:
                full_block = enhanced[i:end_pos + len(v_end)]
                inner = full_block[len(v_start):-len(v_end)]

                # Skip if the variant contains wildcard syntax (e.g. {__style__|realistic})
                # The LLM was instructed to preserve wildcards, so variants containing
                # them are intentional and should be kept as-is.
                if contains_wildcard(inner, wc_wrap):
                    result.append(full_block)
                    i = end_pos + len(v_end)
                    continue

                # If this exact block was in the original, preserve it
                if full_block in original_variants:
                    result.append(full_block)
                    i = end_pos + len(v_end)
                    continue

                # Accidental DP syntax - escape pipes inside
                inner_escaped = inner.replace("|", ",")
                result.append(f"{v_start}{inner_escaped}{v_end}")
                i = end_pos + len(v_end)
                continue

        result.append(enhanced[i])
        i += 1

    return "".join(result)
