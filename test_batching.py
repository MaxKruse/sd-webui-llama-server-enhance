"""Standalone tests for the batched chat-completion pipeline.

Covers the pure helpers (_plan_buckets, _variation_reference_block) and the
end-to-end bucket behavior of _batch_chat_completions against a fake local
llama-server (real aiohttp client, canned LLM responses):

- prompts run in strict buckets of max_concurrent (a bucket must finish
  before the next one starts)
- with vary_across_buckets, each bucket after the first receives the
  previous bucket's prompts (enhanced where available, originals after a
  failure) as a "do not repeat, vary" reference in the system prompt
- by default no reference is added

Run:  python test_batching.py   (exit code 0 on pass)
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Resolve paths relative to this script's directory
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from prompt_enhancer.llm import (
    _batch_chat_completions,
    _plan_buckets,
    _variation_reference_block,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}{(' - ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Fake llama-server: real HTTP, canned chat completions.
# Echoes the request's system + user content into the completion so tests
# can inspect exactly what the client sent.
# ---------------------------------------------------------------------------

class _FakeLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        server = self.server
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        system = next(m["content"] for m in body["messages"] if m["role"] == "system")
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")

        with server.lock:
            server.in_flight += 1
            server.peak = max(server.peak, server.in_flight)
            started_at = time.monotonic()
        time.sleep(server.delay)
        finished_at = time.monotonic()

        record = {
            "system": system,
            "user": user,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        with server.lock:
            server.in_flight -= 1
            server.requests.append(record)

        payload: dict
        status = 200
        if server.fail_user is not None and user == server.fail_user:
            # 400 is non-transient: the client must fail fast, no retries
            payload = {"error": "boom"}
            status = 400
        else:
            payload = {
                "choices": [{"message": {"role": "assistant", "content": f"SYS:[{system}] USER:[{user}]"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "timings": {"prompt_ms": 1.0, "predicted_ms": 1.0},
            }

        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start_fake_server(delay: float = 0.05, fail_user: str | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLMHandler)
    server.lock = threading.Lock()
    server.requests: list[dict] = []
    server.in_flight = 0
    server.peak = 0
    server.delay = delay
    server.fail_user = fail_user
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


REF_MARKER = "Previous Batch Reference"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_plan_buckets():
    print("\n=== _plan_buckets ===")
    prompts = [(i, f"p{i}") for i in range(8)]
    check("8 prompts / size 4 -> [4, 4]", _plan_buckets(prompts, 4) == [prompts[0:4], prompts[4:8]])
    check("5 prompts / size 4 -> [4, 1]", _plan_buckets(prompts[:5], 4) == [prompts[:4], [prompts[4]]])
    check("3 prompts / size 4 -> [3]", _plan_buckets(prompts[:3], 4) == [prompts[:3]])
    check("empty -> []", _plan_buckets([], 4) == [])
    check("size 1 -> singletons", _plan_buckets(prompts, 1) == [[p] for p in prompts])
    check("size 0 clamps to 1", _plan_buckets(prompts[:3], 0) == [[prompts[0]], [prompts[1]], [prompts[2]]])


def test_reference_block():
    print("\n=== _variation_reference_block ===")
    block = _variation_reference_block(["alpha", "beta"])
    check("starts with blank line", block.startswith("\n"))
    check("has section marker", REF_MARKER in block)
    check("numbered entries", "1. alpha" in block and "2. beta" in block)
    check("do-not-repeat instruction", "Do not repeat any of them exactly" in block)
    check("vary instruction", "fresh variation" in block)
    check("original-intent instruction", "original intent" in block)
    check("collapses whitespace in entries", "1. line1 line2 x" in _variation_reference_block(["line1\nline2  x"]))


# ---------------------------------------------------------------------------
# End-to-end against the fake server
# ---------------------------------------------------------------------------

def test_bucket_barrier_and_reference():
    print("\n=== Bucket barrier + cross-bucket reference ===")
    base_url, server = _start_fake_server(delay=0.05)
    try:
        prompts = [(i, f"prompt {i}") for i in range(8)]
        results = _batch_chat_completions(
            base_url, "fake-model", "BASE SYSTEM", prompts,
            max_concurrent=4, request_timeout=10, vary_across_buckets=True,
        )
    finally:
        server.shutdown()

    check("8 results", len(results) == 8)
    check("results in input order", [idx for idx, _ in results] == list(range(8)))
    check("all succeeded", all(r.content for _, r in results))
    check("server saw 8 requests", len(server.requests) == 8)
    check("peak concurrency <= 4", server.peak <= 4, f"peak={server.peak}")

    plain = [r for r in server.requests if REF_MARKER not in r["system"]]
    varied = [r for r in server.requests if REF_MARKER in r["system"]]
    check("exactly 4 bucket-1 requests without reference", len(plain) == 4, f"got {len(plain)}")
    check("exactly 4 bucket-2 requests with reference", len(varied) == 4, f"got {len(varied)}")
    check("bucket-1 system prompt untouched", all(r["system"] == "BASE SYSTEM" for r in plain))

    barrier_ok = bool(varied) and bool(plain) and min(r["started_at"] for r in varied) >= max(r["finished_at"] for r in plain)
    check("strict bucket barrier (no overlap between buckets)", barrier_ok)

    # Every bucket-1 enhanced prompt (echoed by the fake server) must appear
    # in the reference block of every bucket-2 request.
    for r in varied:
        missing = [p["user"] for p in plain if f"SYS:[BASE SYSTEM] USER:[{p['user']}]" not in r["system"]]
        if missing:
            check("reference contains all bucket-1 enhanced prompts", False, f"missing {missing}")
            break
    else:
        check("reference contains all bucket-1 enhanced prompts", True)


def test_no_reference_by_default():
    print("\n=== Default: no cross-bucket reference ===")
    base_url, server = _start_fake_server(delay=0.02)
    try:
        prompts = [(i, f"prompt {i}") for i in range(5)]
        results = _batch_chat_completions(
            base_url, "fake-model", "BASE SYSTEM", prompts,
            max_concurrent=2, request_timeout=10,
        )
    finally:
        server.shutdown()

    check("5 results", len(results) == 5)
    check("server saw 5 requests", len(server.requests) == 5)
    check("no request carries a reference", all(REF_MARKER not in r["system"] for r in server.requests))
    check("all systems untouched", all(r["system"] == "BASE SYSTEM" for r in server.requests))


def test_failed_bucket_falls_back_to_original():
    print("\n=== Failure in previous bucket: reference falls back to original ===")
    base_url, server = _start_fake_server(delay=0.02, fail_user="prompt 1")
    try:
        prompts = [(i, f"prompt {i}") for i in range(6)]
        results = _batch_chat_completions(
            base_url, "fake-model", "BASE SYSTEM", prompts,
            max_concurrent=2, request_timeout=10, vary_across_buckets=True,
        )
    finally:
        server.shutdown()

    check("6 results", len(results) == 6)
    failed_result = dict(results).get(1)
    check("failed prompt reports error", failed_result is not None and bool(failed_result.error) and not failed_result.content)
    check("other prompts succeeded", all(r.content for idx, r in results if idx != 1))

    plain = [r for r in server.requests if REF_MARKER not in r["system"]]
    varied = [r for r in server.requests if REF_MARKER in r["system"]]
    check("bucket-1 has 2 requests, buckets 2-3 carry references", len(plain) == 2 and len(varied) == 4,
          f"plain={len(plain)} varied={len(varied)}")
    if varied:
        check("failed slot referenced by its original prompt", ". prompt 1" in varied[0]["system"])
        check("succeeded slot referenced by its enhanced prompt", "SYS:[BASE SYSTEM] USER:[prompt 0]" in varied[0]["system"])
        # bucket 3 must reference bucket 2's outputs (the only place "USER:[prompt 2]" can appear in a system prompt)
        later = [r for r in server.requests if REF_MARKER in r["system"] and "USER:[prompt 2]" in r["system"] and "USER:[prompt 3]" in r["system"]]
        check("bucket 3 references bucket 2's outputs", len(later) == 2, f"got {len(later)}")


def main():
    test_plan_buckets()
    test_reference_block()
    test_bucket_barrier_and_reference()
    test_no_reference_by_default()
    test_failed_bucket_falls_back_to_original()

    print()
    print(f"=== Results: {passed} passed, {failed} failed ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
