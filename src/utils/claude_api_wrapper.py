"""
claude_api_wrapper.py — Calls Claude using your Max subscription via CLI pipe mode.

Uses subprocess to invoke `claude -p` which is already authenticated.
No proxy needed. No API key needed. $0 extra cost.
"""

import subprocess
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TIMEOUT = 120


def call_claude(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call Claude using the authenticated Claude Code CLI pipe mode.

    Args:
        prompt: The user message to send.
        system: Optional system prompt.
        model: Model identifier.
        max_tokens: Maximum tokens in response.
        timeout: Subprocess timeout in seconds.

    Returns:
        The text response from Claude.
    """
    cmd = ["claude", "-p", "--model", model]

    if system:
        cmd.extend(["--system-prompt", system])

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Claude CLI error (rc={result.returncode}): {result.stderr[:300]}")

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
    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
    ]

    if system:
        cmd.extend(["--system-prompt", system])

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error (rc={result.returncode}): {result.stderr[:300]}")

        data = json.loads(result.stdout)
        return {
            "text": data.get("result", ""),
            "usage": data.get("usage", {}),
            "model": model,
            "cost_usd": data.get("total_cost_usd", 0),
            "duration_ms": data.get("duration_ms", 0),
        }

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out ({timeout}s)")
    except FileNotFoundError:
        raise RuntimeError("Claude CLI not found. Is Claude Code installed?")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON output: {e}")


if __name__ == "__main__":
    print("Testing Claude CLI wrapper...")
    response = call_claude("Reply with exactly: WRAPPER_TEST_OK")
    print(f"Response: {response}")
    if "WRAPPER_TEST_OK" in response:
        print("TEST: PASS")
    else:
        print("TEST: FAIL")
