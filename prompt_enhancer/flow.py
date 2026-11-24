"""Generation flow: the Once and Per-image enhancement pipelines.

Host-independent (stdlib + the rest of this package): given the resolved
system prompts and the user's prompts, spawn a transient llama-server
(via the ctx.spawn callable the entry wires from settings), send the
positive (and negative) prompts, apply DP escaping and negative-output
validation, and return the enhanced prompt lists. Failures degrade per
index - the original prompt is kept.

The entry (scripts/prompt_enhancer.py) injects the host-dependent pieces
through FlowContext: the settings-backed spawn/kill, the DP separators,
and the output sinks (debug log, console print, console logger).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from prompt_enhancer.batching import _batch_chat_completions
from prompt_enhancer.client import ChatResult, chat_completion_request
from prompt_enhancer.escape import escape_accidental_dp_syntax
from prompt_enhancer.negative import build_negative_user_prompt, validate_negative_output
from prompt_enhancer.server import (
    _ServerProcess,
    _discover_model_name,
    _discover_slot_count,
    _warmup_chat_endpoint,
)

# Bucket-size dropdown choices for "Per image" mode. The Auto choice means
# "use the server's parallel slot count" (discovered from /slots at run time).
BUCKET_SIZE_AUTO = "Auto (server slots)"
BUCKET_SIZE_CHOICES = [BUCKET_SIZE_AUTO, "1", "2", "3", "4", "5", "6", "8", "10", "16"]


def parse_bucket_size(raw) -> int:
    """Parse the bucket-size dropdown value; 0 means auto or invalid."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(1, value)


@dataclass
class FlowContext:
    """Per-generation configuration and output sinks for the flow functions.

    The callables keep this module host-independent and testable against a
    fake local server (test_flow.py):
      spawn(label="") - settings-backed spawn + health wait
      kill(sp)        - terminate + drainer cleanup
      log(msg)        - rotating debug log
      out(msg)        - console print
      info(msg)       - console logger line
    """

    system_prompt: str
    negative_system_prompt: str | None
    request_timeout: float
    bucket_size: str
    enhance_negative: bool
    skip_negatives: bool
    v_start: str
    v_end: str
    wc_wrap: str
    spawn: Callable[..., _ServerProcess | None]
    kill: Callable[[_ServerProcess | None], None]
    log: Callable[[str], None]
    out: Callable[[str], None]
    info: Callable[[str], None]

    def escape(self, original: str, enhanced: str) -> str:
        """Escape variant pipes the LLM newly invented, using the user's DP separators."""
        return escape_accidental_dp_syntax(
            original, enhanced, v_start=self.v_start, v_end=self.v_end, wc_wrap=self.wc_wrap
        )


# ---------------------------------------------------------------------------
# Once mode
# ---------------------------------------------------------------------------


def run_once(ctx: FlowContext, prompts: list[str], negatives: list[str]) -> tuple[list[str] | None, list[str] | None]:
    """Once mode: enhance the first prompt, replicate it across the batch.
    One shared server serves both positive and negative requests. Returns
    (new_prompts, new_negatives), or (None, None) when nothing was enhanced
    (the caller keeps the originals and logs the output state)."""
    original = prompts[0] if prompts else ""
    if not original.strip():
        ctx.log("  → Once mode: original prompt empty, skipping")
        return None, None

    original_negative = negatives[0] if negatives else ""
    sp = ctx.spawn()
    if sp is None:
        return None, None

    try:
        # Step 1: Enhance positive prompt
        ctx.log(f"  → Once mode: enhancing positive ({len(original)} chars)")
        pos_result: ChatResult = chat_completion_request(
            sp, ctx.system_prompt, original,
            request_timeout=ctx.request_timeout,
        )

        if not pos_result.content:
            ctx.log("  → Once mode: positive enhancement failed, keeping original")
            ctx.info("Using original prompt (enhancement failed)")
            ctx.out("  Using original prompt (enhancement failed)")
            return None, None

        # Escape accidental DP syntax in the enhanced prompt
        enhanced_positive = ctx.escape(original, pos_result.content)
        ctx.log(
            f"  → Once mode: positive enhanced ({len(enhanced_positive)} chars, "
            f"{pos_result.completion_tokens} tokens, {pos_result.generation_tps:.1f} tok/s, "
            f"{pos_result.total_ms:.0f}ms total)"
        )
        ctx.out(f"  Enhanced prompt (full): {enhanced_positive}")

        # Step 2: Enhance negative prompt (if enabled and the preset
        # actually uses negative prompts)
        enhanced_negative = original_negative
        if ctx.enhance_negative and not ctx.skip_negatives:
            enhanced_negative = _enhance_negative_once(ctx, sp, enhanced_positive, original_negative)

        ctx.info(f"Prompt enhanced (Once mode): {original[:60]} → {enhanced_positive[:60]}")
        ctx.info(f"Negative: {original_negative[:60]} → {enhanced_negative[:60]}")
        return [enhanced_positive] * len(prompts), [enhanced_negative] * len(negatives)
    finally:
        ctx.kill(sp)


def _enhance_negative_once(
    ctx: FlowContext,
    sp: _ServerProcess,
    enhanced_positive: str,
    original_negative: str,
) -> str:
    """Once mode: enhance the negative prompt; the original is kept when the
    request fails or the output fails validation."""
    ctx.log(f"  → Once mode: enhancing negative ({len(original_negative)} chars)")
    neg_user_prompt = build_negative_user_prompt(enhanced_positive, original_negative)
    neg_result: ChatResult = chat_completion_request(
        sp, ctx.negative_system_prompt, neg_user_prompt,
        request_timeout=ctx.request_timeout,
        label="Enhanced negative prompt",
    )

    if not neg_result.content:
        return original_negative

    cleaned_negative, reject_reason = validate_negative_output(
        enhanced_positive, original_negative, neg_result.content
    )
    if cleaned_negative is None:
        ctx.log(f"  → Once mode: negative rejected ({reject_reason}), keeping original")
        ctx.out(f"  Negative enhancement rejected ({reject_reason}) - keeping original")
        return original_negative

    # Escape accidental DP syntax in the enhanced negative prompt
    enhanced_negative = ctx.escape(original_negative, cleaned_negative)
    ctx.log(
        f"  → Once mode: negative enhanced ({len(enhanced_negative)} chars, "
        f"{neg_result.completion_tokens} tokens, {neg_result.generation_tps:.1f} tok/s, "
        f"{neg_result.total_ms:.0f}ms total)"
    )
    ctx.out(f"  Enhanced negative (full): {enhanced_negative}")
    return enhanced_negative


# ---------------------------------------------------------------------------
# Per image mode
# ---------------------------------------------------------------------------


def run_per_image(ctx: FlowContext, prompts: list[str], negatives: list[str]) -> tuple[list[str] | None, list[str] | None]:
    """Per image mode: one server, positives in strict buckets, then the
    negatives of the successful indices (same bucketing, no variance
    reference). Returns (new_prompts, new_negatives), or (None, None)
    when nothing was enhanced (the caller keeps the originals and logs
    the output state).
    """
    ctx.log(f"  → Per image mode: {len(prompts)} prompt(s)")

    # Collect prompts that need enhancement (skip empty ones)
    prompts_to_enhance: list[tuple[int, str]] = []
    skipped_indices: set[int] = set()
    for idx, original in enumerate(prompts):
        if not original.strip():
            ctx.log(f"  → prompt {idx}: empty, keeping as-is")
            skipped_indices.add(idx)
        else:
            prompts_to_enhance.append((idx, original))

    if not prompts_to_enhance:
        ctx.log("  → all prompts empty, skipping enhancement")
        return None, None

    # Start a single server instance (the helper drains the pipes,
    # streams stderr into the debug log, and waits for health)
    sp = ctx.spawn(label=f", {len(prompts_to_enhance)} prompt(s)")
    if sp is None:
        return None, None

    try:
        model_name, bucket = _prepare_session(ctx, sp, len(prompts_to_enhance))
        enhanced_prompts, successful_indices = _apply_positive_results(
            ctx, sp, model_name, bucket, prompts, prompts_to_enhance, skipped_indices
        )
        enhanced_negatives = _pad_negatives(negatives, len(enhanced_prompts))
        if ctx.enhance_negative and not ctx.skip_negatives and successful_indices:
            enhanced_negatives = _apply_negative_results(
                ctx, sp, model_name, bucket, negatives, enhanced_prompts,
                successful_indices, enhanced_negatives,
            )
        return enhanced_prompts, enhanced_negatives
    finally:
        # Always kill the server: an unexpected exception above must not
        # leak a GPU-resident llama-server (or its port) until restart.
        ctx.kill(sp)
        stderr_tail = sp.stderr_tail(3000)
        if stderr_tail.strip():
            ctx.log(f"  llama-server stderr (last 3000 chars):\n{stderr_tail}")


def _prepare_session(ctx: FlowContext, sp: _ServerProcess, num_prompts: int) -> tuple[str, int]:
    """Discover model/slots, resolve the bucket size, warm up the chat
    endpoint. Returns (model_name, bucket_size)."""
    base_url = sp.base_url
    model_name = _discover_model_name(base_url)
    ctx.log(f"  Discovered model: {model_name}")

    slot_count = _discover_slot_count(base_url, model_name)
    ctx.log(f"  Discovered {slot_count} slot(s)")

    # The dropdown's Auto choice maps to the slot count
    bucket = parse_bucket_size(ctx.bucket_size)
    if bucket == 0:
        bucket = slot_count
        ctx.log(f"  bucket size: auto → {slot_count} (server slots)")
    else:
        suffix = f" (server has {slot_count} slots - requests will queue)" if bucket > slot_count else ""
        ctx.log(f"  bucket size: user choice {bucket}{suffix}")

    ctx.out(f"  Model: {model_name}, Slots: {slot_count}, Bucket: {bucket}")

    # Warm up the chat endpoint — /health returns 200 before parallel slots
    # are initialized, causing HTTP 400 on the first real request.
    _warmup_chat_endpoint(base_url, model_name, num_slots=slot_count)
    return model_name, bucket


def _apply_positive_results(
    ctx: FlowContext,
    sp: _ServerProcess,
    model_name: str,
    bucket: int,
    prompts: list[str],
    prompts_to_enhance: list[tuple[int, str]],
    skipped_indices: set[int],
) -> tuple[list[str], list[int]]:
    """Batch-enhance the positive prompts and map results back by index.

    Returns (enhanced_prompts, successful_indices): the per-index enhanced
    list (originals kept where enhancement failed or the prompt was empty)
    and the indices that got a new enhanced positive.
    """
    # Prompts run in strict buckets. Each bucket after the first receives
    # the previous bucket's prompts as a "do not repeat, vary" reference.
    num_buckets = (len(prompts_to_enhance) + bucket - 1) // bucket
    ctx.out(
        f"  Sending {len(prompts_to_enhance)} positive prompt(s) "
        f"({bucket} concurrent, {num_buckets} bucket(s)) ..."
    )
    ctx.log(f"  → positive batch: {len(prompts_to_enhance)} prompt(s) in {num_buckets} bucket(s) of {bucket}, cross-bucket variance enabled")
    pos_results = _batch_chat_completions(
        sp.base_url, model_name, ctx.system_prompt, prompts_to_enhance,
        max_concurrent=bucket,
        request_timeout=ctx.request_timeout,
        vary_across_buckets=True,
    )
    pos_result_map: dict[int, ChatResult] = {idx: result for idx, result in pos_results}

    enhanced_prompts: list[str] = [""] * len(prompts)
    successful_indices: list[int] = []
    for idx, original in enumerate(prompts):
        if idx in skipped_indices:
            enhanced_prompts[idx] = original
        else:
            _merge_positive_result(ctx, idx, original, pos_result_map.get(idx), enhanced_prompts, successful_indices)
    return enhanced_prompts, successful_indices


def _merge_positive_result(
    ctx: FlowContext,
    idx: int,
    original: str,
    result: ChatResult | None,
    enhanced_prompts: list[str],
    successful_indices: list[int],
) -> None:
    """Apply one positive result: escape + write on success, keep original
    and announce on failure."""
    if result and result.content:
        # Escape accidental DP syntax in the enhanced prompt
        escaped_content = ctx.escape(original, result.content)
        ctx.log(
            f"  → prompt {idx}: positive enhanced ({len(result.content)} chars, "
            f"{result.completion_tokens} tokens, {result.generation_tps:.1f} tok/s, "
            f"{result.total_ms:.0f}ms total)"
        )
        ctx.info(f"Prompt {idx + 1} enhanced: {original[:60]} → {escaped_content[:60]}")
        ctx.out(f"  Enhanced prompt {idx + 1} (full): {escaped_content}")
        enhanced_prompts[idx] = escaped_content
        successful_indices.append(idx)
        return

    error_detail = result.error if result and result.error else "no result"
    ctx.log(f"  → prompt {idx}: positive enhancement failed ({error_detail}), keeping original")
    ctx.info(f"Prompt {idx + 1}: using original (enhancement failed: {error_detail})")
    ctx.out(f"  Prompt {idx + 1}: using original (enhancement failed: {error_detail})")
    enhanced_prompts[idx] = original


def _pad_negatives(negatives: list[str], count: int) -> list[str]:
    """Original negatives as the baseline, padded with empty strings."""
    enhanced = list(negatives)
    while len(enhanced) < count:
        enhanced.append("")
    return enhanced


def _apply_negative_results(
    ctx: FlowContext,
    sp: _ServerProcess,
    model_name: str,
    bucket: int,
    negatives: list[str],
    enhanced_prompts: list[str],
    successful_indices: list[int],
    enhanced_negatives: list[str],
) -> list[str]:
    """Batch-enhance the negatives of the successful indices, with validation
    before write-back; failed/rejected indices keep their baseline original."""
    neg_prompts: list[tuple[int, str]] = []
    for idx in successful_indices:
        original_negative = negatives[idx] if idx < len(negatives) else ""
        neg_prompts.append((idx, build_negative_user_prompt(enhanced_prompts[idx], original_negative)))

    ctx.out(f"  Sending {len(neg_prompts)} negative prompt(s) ({bucket} concurrent) ...")
    neg_results = _batch_chat_completions(
        sp.base_url, model_name, ctx.negative_system_prompt, neg_prompts,
        max_concurrent=bucket,
        request_timeout=ctx.request_timeout,
    )
    neg_result_map: dict[int, ChatResult] = {idx: result for idx, result in neg_results}
    for idx in successful_indices:
        original_negative = negatives[idx] if idx < len(negatives) else ""
        result = neg_result_map.get(idx)
        if not (result and result.content):
            continue  # keep original negative (already set as baseline)

        cleaned_negative, reject_reason = validate_negative_output(
            enhanced_prompts[idx], original_negative, result.content
        )
        if cleaned_negative is None:
            ctx.log(f"  → prompt {idx}: negative rejected ({reject_reason}), keeping original")
            ctx.out(f"  Negative {idx + 1} rejected ({reject_reason}) - keeping original")
            continue

        # Escape accidental DP syntax in the enhanced negative prompt
        escaped_negative = ctx.escape(original_negative, cleaned_negative)
        ctx.log(
            f"  → prompt {idx}: negative enhanced ({len(cleaned_negative)} chars, "
            f"{result.completion_tokens} tokens, {result.generation_tps:.1f} tok/s, "
            f"{result.total_ms:.0f}ms total)"
        )
        ctx.info(f"Negative {idx + 1} enhanced: {original_negative[:60]} → {escaped_negative[:60]}")
        ctx.out(f"  Enhanced negative {idx + 1} (full): {escaped_negative}")
        enhanced_negatives[idx] = escaped_negative

    return enhanced_negatives
