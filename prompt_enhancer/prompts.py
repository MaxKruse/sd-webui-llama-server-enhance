"""System-prompt assembly for LLM enhancement.

Host-independent (stdlib only): hires-fix dimension math, the resolution
context block, and the base assembly shared by the positive and negative
enhancement calls. The caller reads settings (hires fix state) and
passes the values in.
"""

from __future__ import annotations

import math


def get_effective_dimensions(p) -> tuple[int, int]:
    """Get the final output dimensions, accounting for hires fix.

    Returns (width, height) of the final rendered image.
    Mirrors the logic in StableDiffusionProcessingTxt2Img.calculate_target_resolution().
    """
    if not getattr(p, "enable_hr", False):
        return (p.width, p.height)

    hr_resize_x = getattr(p, "hr_resize_x", 0)
    hr_resize_y = getattr(p, "hr_resize_y", 0)

    if hr_resize_x == 0 and hr_resize_y == 0:
        hr_scale = getattr(p, "hr_scale", 2.0)
        return (round(p.width * hr_scale), round(p.height * hr_scale))

    if hr_resize_y == 0:
        return (hr_resize_x, round(hr_resize_x * (p.height / p.width)))
    elif hr_resize_x == 0:
        return (round(hr_resize_y * (p.width / p.height)), hr_resize_y)

    return (hr_resize_x, hr_resize_y)


def build_resolution_instruction(
    width: int,
    height: int,
    *,
    hires: bool = False,
    base_width: int = 0,
    base_height: int = 0,
) -> str:
    """Build a resolution hint block for the LLM system prompt.

    Tells the LLM the target image dimensions and orientation so it can
    frame compositions, aspect ratios, and layout descriptions appropriately.
    Includes hires fix info when applicable.
    """
    gcd = math.gcd(width, height)
    ratio_w, ratio_h = width // gcd, height // gcd
    orientation = "portrait" if height > width else "landscape" if width > height else "square"

    block = (
        f"\n\n--- Image Resolution ---\n"
        f"Final output resolution: {width}x{height} ({orientation}, {ratio_w}:{ratio_h} aspect ratio)\n"
        f"Frame your description to suit this orientation."
    )

    if hires and base_width > 0 and base_height > 0:
        base_gcd = math.gcd(base_width, base_height)
        base_ratio_w, base_ratio_h = base_width // base_gcd, base_height // base_gcd
        block += (
            f"\nBase generation resolution: {base_width}x{base_height} ({base_ratio_w}:{base_ratio_h}, hires fix upscaling to final)"
        )

    return block


def assemble_system_prompt(
    preset_content: str,
    *,
    width: int = 0,
    height: int = 0,
    hires: bool = False,
    base_width: int = 0,
    base_height: int = 0,
    negative_instruction: str | None = None,
) -> tuple[str, str | None]:
    """Assemble the system prompts for the enhancement calls.

    Returns (positive_system_prompt, negative_system_prompt). The negative
    prompt is the same base plus the negative-enhancement instruction block,
    or None when ``negative_instruction`` is None (negative enhancement off).
    """
    result = preset_content

    # Append resolution info if dimensions provided
    if width > 0 and height > 0:
        result += build_resolution_instruction(
            width, height,
            hires=hires,
            base_width=base_width,
            base_height=base_height,
        )

    if negative_instruction is None:
        return result, None
    return result, result + negative_instruction
