"""
claude_proxy.py — Local proxy server that exposes Claude CLI over HTTP.

Run this on your local machine (where Claude Code is authenticated).
The Colab GPU server calls this proxy for all LLM requests, routing
them through your Max subscription at zero extra cost.

Usage:
    python scripts/claude_proxy.py                  # starts on port 8100
    python scripts/claude_proxy.py --port 8200      # custom port
    python scripts/claude_proxy.py --ngrok           # auto-create ngrok tunnel
"""

import argparse
import subprocess
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-proxy")

app = FastAPI(title="Claude CLI Proxy", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class ClaudeRequest(BaseModel):
    prompt: str
    system: str = ""
    model: str = DEFAULT_MODEL
    max_tokens: int = 1000
    timeout: int = 120


class ClaudeResponse(BaseModel):
    text: str
    ok: bool
    error: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "service": "claude-proxy"}


@app.post("/v1/claude", response_model=ClaudeResponse)
def call_claude(req: ClaudeRequest):
    cmd = ["claude", "-p", "--model", req.model]
    if req.system:
        cmd.extend(["--system-prompt", req.system])

    try:
        result = subprocess.run(
            cmd,
            input=req.prompt,
            capture_output=True,
            text=True,
            timeout=req.timeout,
        )
        if result.returncode == 0:
            logger.info(f"OK model={req.model} len={len(result.stdout)}")
            return ClaudeResponse(text=result.stdout.strip(), ok=True)
        else:
            err = result.stderr[:300]
            logger.error(f"CLI error rc={result.returncode}: {err}")
            return ClaudeResponse(text="", ok=False, error=err)

    except subprocess.TimeoutExpired:
        return ClaudeResponse(text="", ok=False, error=f"Timed out ({req.timeout}s)")
    except FileNotFoundError:
        return ClaudeResponse(text="", ok=False, error="Claude CLI not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude CLI Proxy Server")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--ngrok", action="store_true", help="Create ngrok tunnel")
    args = parser.parse_args()

    if args.ngrok:
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(args.port, "http")
            url = str(tunnel).split('"')[1] if '"' in str(tunnel) else str(tunnel)
            print("=" * 60)
            print("  CLAUDE PROXY IS LIVE")
            print("=" * 60)
            print(f"  Local:  http://localhost:{args.port}")
            print(f"  Public: {url}")
            print()
            print("  Set this on Colab before starting the server:")
            print(f'  os.environ["CLAUDE_PROXY_URL"] = "{url}"')
            print("=" * 60)
        except ImportError:
            print("pyngrok not installed. Run: pip install pyngrok")
            print(f"Starting without tunnel on port {args.port}")
    else:
        print(f"Claude proxy starting on http://localhost:{args.port}")
        print("Tip: use --ngrok to create a public tunnel for Colab")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
