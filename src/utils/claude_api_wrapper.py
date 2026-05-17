"""
claude_api_wrapper.py — Calls Claude via proxy (Colab) or CLI (local).

Priority:
  1. CLAUDE_PROXY_URL env var → HTTP call to local proxy server
  2. Local Claude CLI pipe mode (`claude -p`)

The proxy approach lets the Colab GPU server route LLM calls back to your
local machine where Claude Code is authenticated with your Max subscription.
"""

import subprocess
import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TIMEOUT = 120

def _get_proxy_url() -> str:
    """Get proxy URL at call time (not import time) so env changes are picked up."""
    return os.environ.get("CLAUDE_PROXY_URL", "").rstrip("/")


def _call_proxy(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call Claude via the HTTP proxy server."""
    import requests
    proxy_url = _get_proxy_url()

    resp = requests.post(
        f"{proxy_url}/v1/claude",
        json={
            "prompt": prompt,
            "system": system,
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
        timeout=timeout + 10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("ok"):
        return data["text"]
    raise RuntimeError(f"Proxy error: {data.get('error', 'unknown')}")


def _call_cli(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call Claude using the local Claude Code CLI pipe mode."""
    cmd = ["claude", "-p", "--model", model]
    if system:
        cmd.extend(["--system-prompt", system])

    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError(f"Claude CLI error (rc={result.returncode}): {result.stderr[:300]}")


def call_claude(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call Claude — uses proxy if CLAUDE_PROXY_URL is set, otherwise CLI.

    Args:
        prompt: The user message to send.
        system: Optional system prompt.
        model: Model identifier.
        max_tokens: Maximum tokens in response.
        timeout: Subprocess timeout in seconds.

    Returns:
        The text response from Claude.
    """
    proxy_url = _get_proxy_url()
    if proxy_url:
        return _call_proxy(prompt, system, model, max_tokens, timeout)

    try:
        return _call_cli(prompt, system, model, timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out ({timeout}s)")
    except FileNotFoundError:
        raise RuntimeError("Claude CLI not found. Is Claude Code installed?")


def call_claude_json(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Call Claude and parse the response as JSON."""
    text = call_claude(prompt, system, model, max_tokens, timeout)
    return {"text": text, "usage": {}, "model": model, "cost_usd": 0, "duration_ms": 0}


if __name__ == "__main__":
    print(f"Proxy URL: {_get_proxy_url() or '(not set — using CLI)'}")
    print("Testing Claude wrapper...")
    response = call_claude("Reply with exactly: WRAPPER_TEST_OK")
    print(f"Response: {response}")
    if "WRAPPER_TEST_OK" in response:
        print("TEST: PASS")
    else:
        print("TEST: FAIL")
