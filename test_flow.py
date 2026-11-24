"""Offline tests for prompt_enhancer/flow.py against a fake local llama-server.

Drives run_once() and run_per_image() with the real HTTP/aiohttp client
talking to a canned-response local server (no model, no GPU). Covers the
pipeline glue that test_batching.py does not: the Once and Per-image entry
points, empty-prompt skipping, failure fallback to the original prompt,
server-kill on all paths, and the negative-output validation (the reported
bug: the model echoing the positive prompt into the enhanced negative).

Run with the host venv's python (needs aiohttp, like test_batching.py):
    python test_flow.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prompt_enhancer.flow import FlowContext, run_once, run_per_image

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
# Fake llama-server: real HTTP, canned chat completions per user message.
# Unregistered user messages get HTTP 400 so unexpected requests (e.g. a
# negative call that should not happen) are caught by the assertions.
# ---------------------------------------------------------------------------


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"data": [{"id": "fake-model.gguf"}]})
        elif self.path.startswith("/slots"):
            self._send(200, [1, 2])
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        self.server.seen.append(user)
        if user not in self.server.respond:
            self._send(400, {"error": f"unexpected user message: {user!r}"})
            return
        content, status = self.server.respond[user]
        if status != 200:
            self._send(status, {"error": content})
            return
        self._send(200, {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "timings": {"prompt_ms": 1.0, "predicted_ms": 1.0},
        })


def _start_fake_server(respond: dict):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.respond = respond
    server.seen: list[str] = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


class _FakeSp:
    """Stand-in for server._ServerProcess: just the surface flow.py touches."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def stderr_tail(self, max_chars: int = 3000) -> str:
        return ""


def _make_ctx(
    base_url: str,
    *,
    enhance_negative: bool = True,
    skip_negatives: bool = False,
    bucket_size: str = "Auto (server slots)",
    spawn=None,
    log=None,
    kill=None,
) -> FlowContext:
    return FlowContext(
        system_prompt="SYS",
        negative_system_prompt="NEGSYS",
        request_timeout=10.0,
        bucket_size=bucket_size,
        enhance_negative=enhance_negative,
        skip_negatives=skip_negatives,
        v_start="{",
        v_end="}",
        wc_wrap="__",
        spawn=spawn or (lambda label="": _FakeSp(base_url)),
        kill=kill or (lambda sp: None),
        log=log or (lambda msg: None),
        out=lambda msg: None,
        info=lambda msg: None,
    )


WARMUP = "hi"  # chat user content used by _warmup_chat_endpoint


# ---------------------------------------------------------------------------
# Once mode
# ---------------------------------------------------------------------------


def test_once_happy():
    print("\n[run_once] happy path: enhance positive + negative, escape applied")
    base_url, server = _start_fake_server({
        WARMUP: ("ok", 200),
        "a cat": ("a beautiful cat in a {red|blue} meadow", 200),
        "POSITIVE: a beautiful cat in a {red,blue} meadow\n\nNEGATIVE: worst quality": (
            "worst quality, low quality, watermark, artist name", 200
        ),
    })
    calls = {"spawn": 0, "kill": 0}
    ctx = _make_ctx(
        base_url,
        spawn=lambda label="": (calls.__setitem__("spawn", calls["spawn"] + 1) or _FakeSp(base_url)),
        kill=lambda sp: calls.__setitem__("kill", calls["kill"] + 1),
    )
    try:
        new_prompts, new_negatives = run_once(ctx, ["a cat"], ["worst quality"])
        check("positive enhanced + escaped", new_prompts == ["a beautiful cat in a {red,blue} meadow"], repr(new_prompts))
        check("negative enhanced + applied", new_negatives == ["worst quality, low quality, watermark, artist name"], repr(new_negatives))
        check("request order: positive warmup+call, then negative warmup+call",
              server.seen == [WARMUP, "a cat", WARMUP, "POSITIVE: a beautiful cat in a {red,blue} meadow\n\nNEGATIVE: worst quality"],
              repr(server.seen))
        check("spawn + kill exactly once each", calls == {"spawn": 1, "kill": 1}, repr(calls))
    finally:
        server.shutdown()
        server.server_close()


def test_once_negative_echo_rejected():
    print("\n[run_once] reported bug: model echoes the positive into the negative")
    base_url, server = _start_fake_server({
        WARMUP: ("ok", 200),
        "a cat": ("a beautiful cat", 200),
        "POSITIVE: a beautiful cat\n\nNEGATIVE: worst quality": ("a beautiful cat", 200),
    })
    log_lines: list[str] = []
    ctx = _make_ctx(base_url, log=log_lines.append)
    try:
        new_prompts, new_negatives = run_once(ctx, ["a cat"], ["worst quality"])
        check("positive still applied", new_prompts == ["a beautiful cat"], repr(new_prompts))
        check("echoed negative rejected, original kept", new_negatives == ["worst quality"], repr(new_negatives))
        check("rejection logged", any("negative rejected (output contains the positive prompt)" in l for l in log_lines), repr(log_lines))
    finally:
        server.shutdown()
        server.server_close()


def test_once_positive_failure():
    print("\n[run_once] positive failure: original kept, server killed")
    base_url, server = _start_fake_server({
        WARMUP: ("ok", 200),
        "a cat": ("boom", 400),
    })
    calls = {"kill": 0}
    log_lines: list[str] = []
    ctx = _make_ctx(base_url, log=log_lines.append, kill=lambda sp: calls.__setitem__("kill", calls["kill"] + 1))
    try:
        new_prompts, new_negatives = run_once(ctx, ["a cat"], ["worst quality"])
        check("nothing enhanced", (new_prompts, new_negatives) == (None, None))
        check("kill called on failure path", calls == {"kill": 1}, repr(calls))
        check("failure logged", any("positive enhancement failed" in l for l in log_lines), repr(log_lines))
    finally:
        server.shutdown()
        server.server_close()


def test_once_skip_negatives():
    print("\n[run_once] skip_negatives: no negative request is sent")
    base_url, server = _start_fake_server({
        WARMUP: ("ok", 200),
        "a cat": ("a beautiful cat", 200),
    })
    ctx = _make_ctx(base_url, skip_negatives=True)
    try:
        new_prompts, new_negatives = run_once(ctx, ["a cat"], ["worst quality"])
        check("positive enhanced", new_prompts == ["a beautiful cat"], repr(new_prompts))
        check("original negative kept", new_negatives == ["worst quality"], repr(new_negatives))
        check("no negative call", server.seen == [WARMUP, "a cat"], repr(server.seen))
    finally:
        server.shutdown()
        server.server_close()


def test_once_empty_prompt_and_no_spawn():
    print("\n[run_once] empty first prompt / spawn failure: no enhancement")
    ctx = _make_ctx("http://127.0.0.1:1")
    check("empty first prompt -> (None, None)",
          run_once(ctx, ["   "], ["n"]) == (None, None))
    check("spawn returning None -> (None, None)",
          run_once(_make_ctx("http://127.0.0.1:1", spawn=lambda label="": None), ["a cat"], ["n"]) == (None, None))


# ---------------------------------------------------------------------------
# Per image mode
# ---------------------------------------------------------------------------


def test_per_image_mixed():
    print("\n[run_per_image] mixed batch: success, empty prompt, failure; negative of success only")
    base_url, server = _start_fake_server({
        WARMUP: ("ok", 200),
        "alpha": ("enhanced {a|b} alpha", 200),
        "beta": ("boom", 400),
        "POSITIVE: enhanced {a,b} alpha\n\nNEGATIVE: neg1": ("bad {x|y} stuff, neg1", 200),
    })
    ctx = _make_ctx(base_url)
    try:
        new_prompts, new_negatives = run_per_image(ctx, ["alpha", "", "beta"], ["neg1", "neg2", "neg3"])
        check("success: enhanced + escaped", new_prompts[0] == "enhanced {a,b} alpha", repr(new_prompts))
        check("empty prompt kept as-is", new_prompts[1] == "", repr(new_prompts))
        check("failed prompt kept original", new_prompts[2] == "beta", repr(new_prompts))
        check("negative of success enhanced + escaped", new_negatives[0] == "bad {x,y} stuff, neg1", repr(new_negatives))
        check("other negatives untouched", new_negatives[1:] == ["neg2", "neg3"], repr(new_negatives))
        check("no negative call for the failed positive",
              all("NEGATIVE: neg3" not in m for m in server.seen), repr(server.seen))
        check("exactly one negative call",
              [m for m in server.seen if m.startswith("POSITIVE:")] == ["POSITIVE: enhanced {a,b} alpha\n\nNEGATIVE: neg1"],
              repr(server.seen))
        check("warmup ran for the slot count", server.seen.count(WARMUP) == 2, repr(server.seen))
    finally:
        server.shutdown()
        server.server_close()


def test_per_image_negative_echo_rejected():
    print("\n[run_per_image] reported bug on the batch path: echoed negative rejected")
    base_url, server = _start_fake_server({
        WARMUP: ("ok", 200),
        "alpha": ("enhanced alpha", 200),
        "POSITIVE: enhanced alpha\n\nNEGATIVE: neg1": ("enhanced alpha", 200),
    })
    log_lines: list[str] = []
    ctx = _make_ctx(base_url, log=log_lines.append)
    try:
        new_prompts, new_negatives = run_per_image(ctx, ["alpha"], ["neg1"])
        check("positive still applied", new_prompts == ["enhanced alpha"], repr(new_prompts))
        check("echoed negative rejected, original kept", new_negatives == ["neg1"], repr(new_negatives))
        check("rejection logged", any("negative rejected (output contains the positive prompt)" in l for l in log_lines), repr(log_lines))
    finally:
        server.shutdown()
        server.server_close()


def test_per_image_all_empty_and_no_spawn():
    print("\n[run_per_image] all prompts empty / spawn failure: no enhancement")
    base_url, server = _start_fake_server({WARMUP: ("ok", 200)})
    try:
        check("all empty -> (None, None)",
              run_per_image(_make_ctx(base_url), ["", "   "], ["n1", "n2"]) == (None, None))
        check("no requests sent for all-empty", server.seen == [], repr(server.seen))
        check("spawn returning None -> (None, None)",
              run_per_image(_make_ctx(base_url, spawn=lambda label="": None), ["a"], ["n"]) == (None, None))
    finally:
        server.shutdown()
        server.server_close()


def main():
    print("flow tests (fake local llama-server)")
    test_once_happy()
    test_once_negative_echo_rejected()
    test_once_positive_failure()
    test_once_skip_negatives()
    test_once_empty_prompt_and_no_spawn()
    test_per_image_mixed()
    test_per_image_negative_echo_rejected()
    test_per_image_all_empty_and_no_spawn()
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
