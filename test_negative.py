"""Standalone tests for prompt_enhancer/negative.py (negative-prompt enhancement).

Runs with the host venv's python (stdlib only - no host imports). Imports
the real module from the extension root, so the tests exercise the same
code the pipeline uses.

Usage: python test_negative.py
Exit code 0 on pass.
"""

import os
import sys

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXT_ROOT)

from prompt_enhancer.negative import (
    build_negative_instruction,
    build_negative_user_prompt,
    validate_negative_output,
)

checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"ok {checks}: {msg}")


POSITIVE = (
    "1girl, solo, masterpiece, best quality, red eyes, purple hair, virtual youtuber, hair ornament, "
    "astronaut, ikumi <lora:Ikumi-Anima-v2-phase2:1>. Ikumi looks directly at the viewer with a calm "
    "confident expression while standing in a relaxed pose inside a spacecraft cabin. She has long "
    "purple hair with red eyes and wears a white astronaut suit with a purple visor. Soft blue ambient "
    "lighting from the viewport creates a calm yet adventurous atmosphere."
)
ORIGINAL_NEGATIVE = "worst quality, low quality, score_1-3, artist name"

# A legitimate enhanced negative.
GOOD = (
    "worst quality, low quality, score_1-3, artist name, watermark, signature, text, blurry, "
    "jpeg artifacts, chromatic aberration, bad anatomy, deformed hands, extra fingers, missing fingers, extra limbs"
)

# --- build_negative_instruction -------------------------------------------------

instruction = build_negative_instruction()
check("NEGATIVE: '" in instruction or "NEGATIVE: " in instruction, "instruction mentions the user-message format")
check("POSITIVE: " in instruction, "instruction mentions the POSITIVE marker")
check("NEVER" in instruction.upper(), "instruction forbids repeating the positive")
check("Output:" in instruction, "instruction includes a concrete example")
check("comma-separated" in instruction.lower(), "instruction requires a comma-separated line")
check("no line breaks" in instruction.lower(), "instruction forbids line breaks in the output")

# --- build_negative_user_prompt -------------------------------------------------

msg = build_negative_user_prompt("some positive", "some negative")
check(msg == "POSITIVE: some positive\n\nNEGATIVE: some negative", "user prompt is POSITIVE/NEGATIVE blocks")

# --- validate_negative_output ---------------------------------------------------

# Legit output passes.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, GOOD)
check(reason is None and cleaned == GOOD, "legit negative passes validation")

# Empty / whitespace input is rejected.
for payload in (None, "", "   "):
    cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, payload)
    check(cleaned is None and reason == "empty response", f"empty input rejected ({payload!r})")

# Full echo of the positive (the reported prompt-8 failure) is rejected.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, POSITIVE)
check(cleaned is None and "positive" in reason, "full positive echo is rejected")

# Partial echo (model pastes the middle of the positive) is rejected.
fragment = POSITIVE[200:260]  # 60-char fragment
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, fragment)
check(cleaned is None and "positive" in reason, "partial positive echo is rejected")

# A negative that repeats the user-message format is rejected.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, "POSITIVE: " + POSITIVE)
check(cleaned is None, "POSITIVE marker echo is rejected")

# A stray marker prefix is stripped and the rest is kept.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, "NEGATIVE: " + GOOD)
check(cleaned == GOOD, "stray NEGATIVE: marker is stripped")

# Markers mid-output are rejected.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, "blurry, NEGATIVE: text")
check(cleaned is None, "marker mid-output is rejected")

# Multi-line output is rejected.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, GOOD + "\nextra line")
check(cleaned is None and "multi-line" in reason, "multi-line output is rejected")

# No length cap: a long scene-tailored single-line negative is accepted
# (the echo check covers the real risk; scene-tailored negatives are
# naturally 300-600 chars).
long_negative = (
    "worst quality, low quality, score_1-3, artist name, watermark, signature, text, blurry, "
    "jpeg artifacts, chromatic aberration, bad anatomy, deformed hands, extra fingers, "
    "missing fingers, extra limbs, deformed face, asymmetric eyes, bad teeth, cropped, "
    "out of frame, bad proportions, long neck, error, fingerprint, lowres, bad feet, "
    "extra digits, bad joints, gross proportions"
)
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, long_negative)
check(reason is None and cleaned == long_negative, "long scene-tailored single-line negative is accepted")

# Output identical to the original negative is rejected.
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, ORIGINAL_NEGATIVE)
check(cleaned is None and "identical" in reason, "output identical to the original is rejected")

# Output with different whitespace than the original (still the same content) is rejected.
cleaned, reason = validate_negative_output(
    POSITIVE, ORIGINAL_NEGATIVE, "worst quality, low quality, score_1-3, artist name"
)
check(cleaned is None, "whitespace-insensitive identical output is rejected")

# A negative that only adds a little beyond the original is kept.
extended = ORIGINAL_NEGATIVE + ", watermark"
cleaned, reason = validate_negative_output(POSITIVE, ORIGINAL_NEGATIVE, extended)
check(reason is None and cleaned == extended, "extended negative passes validation")

# Short positives: full-echo detection still works via containment.
short_positive = "a cat"
cleaned, reason = validate_negative_output(short_positive, ORIGINAL_NEGATIVE, "a cat")
check(cleaned is None, "short positive echo is rejected")

print(f"\nAll {checks} checks passed.")
