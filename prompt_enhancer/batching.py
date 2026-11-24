"""Batched chat completions against llama-server (aiohttp, strict buckets).

Sends multiple prompts in strict buckets of ``max_concurrent``: a whole
bucket must finish before the next one starts, so the server's parallel
slots stay busy without unbounded overlap. With ``vary_across_buckets``,
every bucket after the first receives the previous bucket's prompts
(enhanced text where available, original after a failure) as a "do not
repeat, vary" reference appended to the system prompt.

Host-independent, but requires ``aiohttp``. Single-request logic and the
shared retry policy live in ``prompt_enhancer.client``.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from prompt_enhancer.client import (
    DEFAULT_REQUEST_TIMEOUT,
    ChatResult,
    _MAX_CHAT_RETRIES,
    _completion_payload,
    _describe_error,
    _is_transient_error,
    _print_lock,
    _retry_backoff,
    _clean_http_error,
)

logger = logging.getLogger(__name__)


async def _send_chat_request_async(
    session: aiohttp.ClientSession,
    base_url: str,
    payload: dict,
    request_timeout: float,
) -> tuple[ChatResult | None, str | None, BaseException | None]:
    """Single aiohttp attempt against /v1/chat/completions (no retry).

    Returns (ChatResult, None, None) on success; on failure the error is
    pre-described (the HTTP body is only readable while the response is
    open) and returned as (None, message, exception).
    """
    body = ""
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer no-key"},
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError:
                # Response is still open: read the body for diagnostics.
                # (ClientResponseError does not expose it.) Read enough to
                # cover the full JSON error object so _clean_http_error can
                # parse it (localhost traffic - size is irrelevant).
                try:
                    body = (await resp.read()).decode("utf-8", errors="replace")[:2000]
                except Exception:
                    body = "<unreadable>"
                raise
            result = await resp.json(content_type=None)
            content = result["choices"][0]["message"]["content"]

            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", usage.get("predicted_n", 0))
            timings = result.get("timings", {})
            return ChatResult(
                content=content.strip() if content else None,
                completion_tokens=completion_tokens,
                prompt_ms=timings.get("prompt_ms", 0),
                predicted_ms=timings.get("predicted_ms", 0),
            ), None, None
    except aiohttp.ClientResponseError as exc:
        return None, _clean_http_error(exc.status, body), exc
    except Exception as exc:
        return None, _describe_error(exc), exc


def _print_final_status(index: int, result: ChatResult, last_error: str | None) -> None:
    """Print the one-line status for a finished request."""
    with _print_lock:
        if result.content:
            print(
                f"  Prompt {index + 1} response received ({len(result.content)} chars, "
                f"{result.completion_tokens} tokens, {result.generation_tps:.1f} tok/s)"
            )
        else:
            reason = last_error or "unknown"
            print(f"  Prompt {index + 1}: empty/failed response ({reason})")


async def _chat_completion_async(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    index: int,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> tuple[int, ChatResult]:
    """Send one chat completion via aiohttp with the shared retry policy.
    Returns (index, ChatResult); every status line carries the index
    because bucket requests fail concurrently."""
    payload = _completion_payload(model_name, system_prompt, user_prompt)

    result = ChatResult()
    last_error: str | None = None

    for attempt in range(1, _MAX_CHAT_RETRIES + 1):
        chat_result, last_error, exc = await _send_chat_request_async(
            session, base_url, payload, request_timeout
        )
        if chat_result is not None:
            result = chat_result
            break

        logger.error("Prompt %d: request failed (attempt %d/%d): %s", index + 1, attempt, _MAX_CHAT_RETRIES, last_error)
        with _print_lock:
            print(f"  Prompt {index + 1}: request failed (attempt {attempt}/{_MAX_CHAT_RETRIES}): {last_error}")

        if not _is_transient_error(exc):
            logger.error("Error is not transient - not retrying")
            break

        # Retry with exponential backoff (skip sleep on last attempt).
        if attempt < _MAX_CHAT_RETRIES:
            backoff = _retry_backoff(attempt)
            logger.info("Retrying in %.1fs...", backoff)
            with _print_lock:
                print(f"  Retrying in {backoff:.0f}s...")
            await asyncio.sleep(backoff)

    if not result.content:
        result.error = last_error or "all retries failed"

    _print_final_status(index, result, last_error)
    return index, result


def _plan_buckets(prompts: list[tuple[int, str]], bucket_size: int) -> list[list[tuple[int, str]]]:
    """Split prompts into consecutive buckets of bucket_size (last may be shorter)."""
    bucket_size = max(1, bucket_size)
    return [prompts[i:i + bucket_size] for i in range(0, len(prompts), bucket_size)]


def _variation_reference_block(previous_texts: list[str]) -> str:
    """Build the system-prompt addendum that references the previous bucket's prompts.

    The LLM sees what the previous bucket already produced, so it can vary
    mood, lighting, environment, pose, and composition instead of repeating
    the same scene. Whitespace is collapsed so each reference stays on one
    numbered line.
    """
    numbered = "\n".join(
        f"{n}. {' '.join(text.split())}" for n, text in enumerate(previous_texts, start=1)
    )
    return (
        "\n\n## Previous Batch Reference\n"
        "The following prompts were already enhanced and generated in the immediately preceding batch of this job:\n"
        f"{numbered}\n"
        "Do not repeat any of them exactly, and avoid near-duplicates of their mood, lighting, "
        "environment, pose, action, and composition. Introduce fresh variation while staying "
        "faithful to the user's original intent in the current request."
    )


async def _run_bucket(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    bucket_system: str,
    bucket: list[tuple[int, str]],
    request_timeout: float,
    bucket_num: int,
    bucket_count: int,
    has_reference: bool,
) -> list[tuple[int, ChatResult]]:
    """Run one strict bucket: all its requests finish before the next starts."""
    logger.info(
        "Bucket %d/%d: %d prompt(s)%s",
        bucket_num + 1, bucket_count, len(bucket),
        " (with previous-bucket reference)" if has_reference else "",
    )
    return list(await asyncio.gather(
        *[
            _chat_completion_async(
                session, base_url, model_name, bucket_system, prompt, idx,
                request_timeout=request_timeout,
            )
            for idx, prompt in bucket
        ]
    ))


def _batch_chat_completions(
    base_url: str,
    model_name: str,
    system_prompt: str,
    prompts: list[tuple[int, str]],
    *,
    max_concurrent: int | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    vary_across_buckets: bool = False,
) -> list[tuple[int, ChatResult]]:
    """Send multiple chat completion requests concurrently using aiohttp.

    See the module docstring for the bucket semantics. Returns a list of
    (original_index, ChatResult) tuples, in input order.
    """
    bucket_size = max_concurrent if max_concurrent else max(len(prompts), 1)
    buckets = _plan_buckets(prompts, bucket_size)

    async def _run() -> list[tuple[int, ChatResult]]:
        connector = aiohttp.TCPConnector(limit=bucket_size, force_close=False)
        all_results: list[tuple[int, ChatResult]] = []
        previous_texts: list[str] = []

        async with aiohttp.ClientSession(connector=connector) as session:
            for bucket_num, bucket in enumerate(buckets):
                has_reference = vary_across_buckets and bool(previous_texts)
                bucket_system = (
                    system_prompt + _variation_reference_block(previous_texts)
                    if has_reference
                    else system_prompt
                )
                results = await _run_bucket(
                    session, base_url, model_name, bucket_system, bucket,
                    request_timeout, bucket_num, len(buckets), has_reference,
                )
                all_results.extend(results)
                # What the next bucket should reference: the enhanced text
                # actually used, or the original prompt where enhancement failed.
                previous_texts = [
                    chat_result.content or original
                    for (_, original), (_, chat_result) in zip(bucket, results)
                ]

        return all_results

    return asyncio.run(_run())
