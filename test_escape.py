"""Standalone tests for _escape_accidental_dp_syntax logic."""

from __future__ import annotations


def _find_matching_variant_end(text: str, start: int, v_start: str, v_end: str) -> int | None:
    same_delim = v_start == v_end
    depth = 1
    i = start
    limit = len(text) - len(v_end)
    while i <= limit:
        if same_delim:
            if text[i:i + len(v_end)] == v_end:
                depth -= 1
                if depth == 0:
                    return i
                i += len(v_end)
                continue
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


def _extract_variant_blocks(text: str, v_start: str, v_end: str, out: set[str]) -> None:
    i = 0
    while i < len(text):
        if text[i:i + len(v_start)] == v_start:
            end_pos = _find_matching_variant_end(text, i + len(v_start), v_start, v_end)
            if end_pos is not None:
                full_block = text[i:end_pos + len(v_end)]
                out.add(full_block)
                i = end_pos + len(v_end)
                continue
        i += 1


def _contains_wildcard(text: str, wc_wrap: str) -> bool:
    if not wc_wrap:
        return False
    wc_len = len(wc_wrap)
    if wc_len * 2 + 1 > len(text):
        return False
    first = text.find(wc_wrap)
    if first < 0:
        return False
    second = text.find(wc_wrap, first + wc_len)
    return second > first + wc_len


def _escape_accidental_dp_syntax(
    original: str,
    enhanced: str,
    *,
    v_start: str = "{",
    v_end: str = "}",
    wc_wrap: str = "__",
) -> str:
    original_variants: set[str] = set()
    _extract_variant_blocks(original, v_start, v_end, original_variants)

    result: list[str] = []
    i = 0
    while i < len(enhanced):
        if enhanced[i:i + len(v_start)] == v_start:
            end_pos = _find_matching_variant_end(enhanced, i + len(v_start), v_start, v_end)
            if end_pos is not None:
                full_block = enhanced[i:end_pos + len(v_end)]
                inner = full_block[len(v_start):-len(v_end)]

                if _contains_wildcard(inner, wc_wrap):
                    result.append(full_block)
                    i = end_pos + len(v_end)
                    continue

                if full_block in original_variants:
                    result.append(full_block)
                    i = end_pos + len(v_end)
                    continue

                inner_escaped = inner.replace("|", ",")
                result.append(f"{v_start}{inner_escaped}{v_end}")
                i = end_pos + len(v_end)
                continue

        result.append(enhanced[i])
        i += 1

    return "".join(result)


# --- Test harness ---

passed = 0
failed = 0


def check(name: str, actual: str, expected: str) -> None:
    global passed, failed
    if actual == expected:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        print(f"    expected: {expected!r}")
        print(f"    actual:   {actual!r}")
        failed += 1


print("=== Default separators: v_start='{' v_end='}' wc_wrap='__' ===\n")

# Basic: original variant preserved
check(
    "original variant preserved",
    _escape_accidental_dp_syntax("a {red|blue} cat", "a beautiful {red|blue} cat sitting on a windowsill"),
    "a beautiful {red|blue} cat sitting on a windowsill",
)

# Accidental variant escaped
check(
    "accidental variant escaped",
    _escape_accidental_dp_syntax("a cat", "a {red|blue} cat sitting {indoor|outdoor}"),
    "a {red,blue} cat sitting {indoor,outdoor}",
)

# Wildcard inside variant preserved
check(
    "wildcard inside variant preserved",
    _escape_accidental_dp_syntax("a cat", "a {__style__|realistic} cat"),
    "a {__style__|realistic} cat",
)

# No variants at all
check(
    "no variants",
    _escape_accidental_dp_syntax("a cat", "a beautiful cat"),
    "a beautiful cat",
)

# Nested variants (original preserved)
check(
    "nested variant preserved",
    _escape_accidental_dp_syntax(
        "a {outer {inner1|inner2} end} cat",
        "a {outer {inner1|inner2} end} cat with extra",
    ),
    "a {outer {inner1|inner2} end} cat with extra",
)

# Nested variant + accidental
check(
    "nested original + accidental new",
    _escape_accidental_dp_syntax(
        "a {outer {inner1|inner2} end} cat",
        "a {outer {inner1|inner2} end} cat with {new|stuff}",
    ),
    "a {outer {inner1|inner2} end} cat with {new,stuff}",
)

# Mixed: original + accidental
check(
    "mixed original + accidental",
    _escape_accidental_dp_syntax("{red|blue}", "{red|blue} and {fast|slow} and {big|small}"),
    "{red|blue} and {fast,slow} and {big,small}",
)

# Empty strings
check(
    "empty strings",
    _escape_accidental_dp_syntax("", ""),
    "",
)

# Unmatched brace (no closing)
check(
    "unmatched opening brace",
    _escape_accidental_dp_syntax("a cat", "a {red cat"),
    "a {red cat",
)

# Pipe outside of variant braces (should NOT be touched)
check(
    "pipe outside braces untouched",
    _escape_accidental_dp_syntax("a cat", "a cat | dog | bird"),
    "a cat | dog | bird",
)

# Wildcard standalone (not inside variant)
check(
    "wildcard standalone preserved",
    _escape_accidental_dp_syntax("a __style__ cat", "a beautiful __style__ cat"),
    "a beautiful __style__ cat",
)

# Multiple wildcards + accidental variant
check(
    "multiple wildcards + accidental variant",
    _escape_accidental_dp_syntax(
        "a __style__ and __artist__",
        "a __style__ painting in __artist__ style, {cinematic|dramatic} lighting",
    ),
    "a __style__ painting in __artist__ style, {cinematic,dramatic} lighting",
)

# Original has wildcard, enhanced preserves it
check(
    "original wildcard preserved in enhanced",
    _escape_accidental_dp_syntax("a __style__ cat", "a beautiful __style__ cat with fur"),
    "a beautiful __style__ cat with fur",
)

# Variant with wildcard as one option, preserved
check(
    "variant with wildcard as one option",
    _escape_accidental_dp_syntax("a cat", "a {__style__|photorealistic|3d render} cat"),
    "a {__style__|photorealistic|3d render} cat",
)

# Multiple original variants, all preserved
check(
    "multiple original variants preserved",
    _escape_accidental_dp_syntax(
        "{red|blue} {cat|dog}",
        "a {red|blue} {cat|dog} sitting down",
    ),
    "a {red|blue} {cat|dog} sitting down",
)

# Same variant appears multiple times in original
check(
    "repeated original variant preserved",
    _escape_accidental_dp_syntax(
        "{red|blue} and also {red|blue}",
        "a {red|blue} thing and also {red|blue} other thing",
    ),
    "a {red|blue} thing and also {red|blue} other thing",
)

# Empty variant {}
check(
    "empty variant",
    _escape_accidental_dp_syntax("a cat", "a {} cat"),
    "a {} cat",  # empty inner has no pipes, so no change
)

# Variant with only one option (no pipe)
check(
    "variant with one option (no pipe)",
    _escape_accidental_dp_syntax("a cat", "a {red} cat"),
    "a {red} cat",  # no pipes to escape
)

print()
print("=== Custom separator: v_start='[' v_end=']' wc_wrap='__' ===\n")

check(
    "custom brackets - original preserved",
    _escape_accidental_dp_syntax(
        "a [red|blue] cat",
        "a beautiful [red|blue] cat sitting [indoor|outdoor]",
        v_start="[", v_end="]",
    ),
    "a beautiful [red|blue] cat sitting [indoor,outdoor]",
)

check(
    "custom brackets - no false match on curly braces",
    _escape_accidental_dp_syntax(
        "a [red|blue] cat",
        "a [red|blue] cat with {curly} braces",
        v_start="[", v_end="]",
    ),
    "a [red|blue] cat with {curly} braces",  # curly braces untouched
)

print()
print("=== Custom separator: v_start='###' v_end='###' wc_wrap='___' ===\n")

check(
    "triple-hash variants, triple-underscore wildcards - original preserved",
    _escape_accidental_dp_syntax(
        "a ###red|blue### cat",
        "a beautiful ###red|blue### cat sitting ###indoor|outdoor###",
        v_start="###", v_end="###", wc_wrap="___",
    ),
    "a beautiful ###red|blue### cat sitting ###indoor,outdoor###",
)

check(
    "triple-hash + triple-underscore wildcard inside variant preserved",
    _escape_accidental_dp_syntax(
        "a cat",
        "a ###___style___|realistic### cat",
        v_start="###", v_end="###", wc_wrap="___",
    ),
    "a ###___style___|realistic### cat",
)

check(
    "triple-underscore wildcard standalone",
    _escape_accidental_dp_syntax(
        "a ___style___ cat",
        "a beautiful ___style___ cat",
        v_start="###", v_end="###", wc_wrap="___",
    ),
    "a beautiful ___style___ cat",
)

print()
print("=== Custom separator: v_start='(' v_end=')' wc_wrap='**' ===\n")

check(
    "parentheses variants, star wildcards - original preserved",
    _escape_accidental_dp_syntax(
        "a (red|blue) cat",
        "a beautiful (red|blue) cat sitting (indoor|outdoor)",
        v_start="(", v_end=")",
    ),
    "a beautiful (red|blue) cat sitting (indoor,outdoor)",
)

# Parentheses are tricky because they're common in natural language
check(
    "parentheses - natural language parens with no pipe left alone",
    _escape_accidental_dp_syntax(
        "a cat (on a mat)",
        "a beautiful cat (on a mat) looking cute",
        v_start="(", v_end=")",
    ),
    "a beautiful cat (on a mat) looking cute",  # no pipe, so no escaping needed
)

check(
    "parentheses - accidental pipe in parens escaped",
    _escape_accidental_dp_syntax(
        "a cat",
        "a (red|blue) cat (on a mat)",
        v_start="(", v_end=")",
    ),
    "a (red,blue) cat (on a mat)",
)

print()
print("=== Edge cases with multi-char separators ===\n")

# When variant_start is a substring of variant_end or vice versa
check(
    "v_start='{' v_end='}}}' (asymmetric multi-char)",
    _escape_accidental_dp_syntax(
        "a {red|blue}}} cat",
        "a {red|blue}}} cat {indoor|outdoor}}}",
        v_start="{", v_end="}}}",
    ),
    "a {red|blue}}} cat {indoor,outdoor}}}",
)

# Single-char wrap
check(
    "single-char wildcard wrap '*'",
    _escape_accidental_dp_syntax(
        "a *style* cat",
        "a beautiful *style* cat with {cinematic|dramatic} lighting",
        wc_wrap="*",
    ),
    "a beautiful *style* cat with {cinematic,dramatic} lighting",
)

# Wildcard wrap that appears inside a word
check(
    "wildcard wrap 'x' - tricky since x is common",
    _escape_accidental_dp_syntax(
        "a cat",
        "a {xstylx|realistic} cat",
        wc_wrap="x",
    ),
    "a {xstylx|realistic} cat",  # contains wildcard pattern, preserved
)

# Same delimiter nesting with ###
check(
    "triple-hash nested variant preserved",
    _escape_accidental_dp_syntax(
        "a ###outer ###inner1|inner2### end### cat",
        "a ###outer ###inner1|inner2### end### cat with ###new|stuff###",
        v_start="###", v_end="###",
    ),
    "a ###outer ###inner1|inner2### end### cat with ###new,stuff###",
)

# Same delimiter: default {} still works with nesting
check(
    "default {} with nested still works",
    _escape_accidental_dp_syntax(
        "{a {b|c} d}",
        "{a {b|c} d} and {e|f}",
    ),
    "{a {b|c} d} and {e,f}",
)

# v_start is substring of v_end: { and }}
check(
    "v_start='{' v_end='}}' - asymmetric",
    _escape_accidental_dp_syntax(
        "a {red|blue}} cat",
        "a {red|blue}} cat {indoor|outdoor}}",
        v_start="{", v_end="}}",
    ),
    "a {red|blue}} cat {indoor,outdoor}}",
)

# v_end is substring of v_start: } and ###}
check(
    "v_end is substring of v_start",
    _escape_accidental_dp_syntax(
        "a ###red|blue} cat",
        "a ###red|blue} cat ###indoor|outdoor}",
        v_start="###", v_end="}",
    ),
    "a ###red|blue} cat ###indoor,outdoor}",
)

# Wildcard wrap that's also inside variant brackets
check(
    "wildcard wrap inside variant - both preserved",
    _escape_accidental_dp_syntax(
        "{__style__|realistic}",
        "a beautiful {__style__|realistic} painting",
    ),
    "a beautiful {__style__|realistic} painting",
)

# Empty wc_wrap (disabled wildcards)
check(
    "empty wc_wrap - no wildcard check",
    _escape_accidental_dp_syntax(
        "a cat",
        "a {__style__|realistic} cat",
        wc_wrap="",
    ),
    "a {__style__,realistic} cat",  # no wildcard wrap, so pipes escaped
)

# Only whitespace in variant
check(
    "variant with only whitespace",
    _escape_accidental_dp_syntax(
        "a cat",
        "a { } cat",
    ),
    "a { } cat",  # no pipes, no change
)

# Variant with weights (DP feature: 2::red|1::blue)
check(
    "variant-like with weight syntax",
    _escape_accidental_dp_syntax(
        "a cat",
        "a {2::red|1::blue} cat",
    ),
    "a {2::red,1::blue} cat",  # pipes escaped (not original)
)

print()
print("=== Regression: ensure original behavior still works ===\n")

# Long prompt with multiple elements
long_original = "a {red|blue} __style__ painting of a {forest|mountain} scene"
long_enhanced = "a stunning {red|blue} __style__ painting of a {forest|mountain} landscape with {golden|silver} lighting and {misty|clear} atmosphere"
check(
    "long prompt - originals preserved, accidentals escaped",
    _escape_accidental_dp_syntax(long_original, long_enhanced),
    "a stunning {red|blue} __style__ painting of a {forest|mountain} landscape with {golden,silver} lighting and {misty,clear} atmosphere",
)

print()
print(f"=== Results: {passed} passed, {failed} failed ===")
if failed:
    exit(1)
