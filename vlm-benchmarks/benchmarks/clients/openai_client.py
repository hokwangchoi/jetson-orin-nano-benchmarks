"""
OpenAI-compatible HTTP client for any inference runtime that speaks the
/v1/chat/completions API — llama.cpp's llama-server, vLLM, TRT Edge-LLM
behind an OpenAI shim, etc.

Uses server-sent-events streaming to measure TTFT accurately: we time the
gap between "request sent" and "first non-empty delta received".
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import urllib.request
import urllib.error


class OpenAIClient:
    def __init__(self, base_url: str = "http://localhost:8000",
                 model: str = "Cosmos-Reason2-2B-Q4_K_M"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def ping(self) -> None:
        """Raises if server not reachable."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5) as r:
                r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Server not reachable at {self.base_url}. "
                f"Start it with device/scripts/11_run_llamacpp_server.sh (or equivalent). ({e})"
            )

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert our workload format to OpenAI chat format, inlining images."""
        out = []
        for msg in messages:
            content = msg["content"]
            # Text-only shortcut
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue

            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    # Read local file → data URL
                    p = Path(part["image"]).expanduser()
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                elif part["type"] == "video":
                    # Pass path/URL through; server-specific handling
                    parts.append({
                        "type": "video_url",
                        "video_url": {"url": part["video"]},
                    })
                else:
                    raise ValueError(f"Unknown part type: {part['type']}")
            out.append({"role": msg["role"], "content": parts})
        return out

    def infer(self, messages: List[Dict], max_tokens: int = 128,
              temperature: float = 0.0) -> Dict[str, Any]:
        """Single streaming inference. Returns latency dict."""
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_sent = time.perf_counter()
        t_first = None
        t_last = None
        n_tokens = 0
        text_acc = []

        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text_acc.append(piece)
                    n_tokens += 1  # token ≈ chunk; good enough for throughput

        if t_first is None:
            raise RuntimeError("Server returned no tokens")

        ttft_ms = (t_first - t_sent) * 1000
        decode_s = max(t_last - t_first, 1e-6)
        # TPOT undefined for n_tokens == 1 (no decode gap measured)
        tpot_ms = (decode_s / max(n_tokens - 1, 1)) * 1000 if n_tokens > 1 else 0.0
        tps = n_tokens / decode_s if decode_s > 0 else 0.0

        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "tps": tps,
            "e2e_ms": (t_last - t_sent) * 1000,
            "n_tokens": n_tokens,
            "text": "".join(text_acc)[:200],  # truncate for log sanity
        }
