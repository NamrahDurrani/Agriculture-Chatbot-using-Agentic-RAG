"""
Remote Qwen3.5 client — OpenAI-compatible endpoint (vLLM / SGLang / transformers serve)
==========================================================================================
Talks to {base_url}/v1/chat/completions — the OpenAI chat schema vLLM/SGLang
expose. Same .call(system_prompt, user_prompt, max_tokens, temperature)
-> (text, usage_dict) interface as the other clients in this project, so
rag_pipeline.py needs no changes regardless of which backend is active.

Setup:
    QWEN_REMOTE_BASE_URL = current ngrok URL (no trailing slash needed —
                            stripped automatically). Must be refreshed
                            whenever the tunnel restarts.
    QWEN_REMOTE_MODEL     = e.g. "Qwen/Qwen3.5-2B". Loose variants like
                            "qwen3.5-2b" or "qwen3.5:2b" auto-correct to
                            the server's real registered name — see
                            _resolve_model() below.
"""

import os
import requests
from typing import Dict, Tuple

QWEN_REMOTE_BASE_URL = os.environ.get("QWEN_REMOTE_BASE_URL", "").rstrip("/")
QWEN_REMOTE_MODEL     = os.environ.get("QWEN_REMOTE_MODEL", "Qwen/Qwen3.5-2B")


def _resolve_model(base_url: str, requested_model: str) -> str:
    """
    Confirms the server is reachable and finds the exact registered model
    name, tolerating naming mismatches like "qwen3.5-2b" vs the server's
    real "Qwen/Qwen3.5-2B" (this exact mismatch has broken every backend
    swap so far — worth the few lines to auto-correct it).
    """
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Can't reach {base_url} ({type(e).__name__}). The ngrok "
            f"tunnel is likely dead/expired or the server isn't running — "
            f"get a fresh link and retry."
        ) from None

    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} from {base_url}/v1/models — "
                            f"tunnel may be dead. Response: {r.text[:200]}")

    available = [m["id"] for m in r.json().get("data", [])]
    if requested_model in available:
        return requested_model

    def normalize(s):
        s = s.split("/", 1)[1] if "/" in s else s
        return s.lower().translate(str.maketrans("", "", "-:_ "))

    for candidate in available:
        if normalize(candidate) == normalize(requested_model):
            print(f"[QWEN-REMOTE] Using '{candidate}' (closest match to "
                  f"'{requested_model}')")
            return candidate

    raise RuntimeError(
        f"Model '{requested_model}' not found. Available: {available}"
    )


class QwenRemoteClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or QWEN_REMOTE_BASE_URL).rstrip("/")
        if not self.base_url:
            raise ValueError("QWEN_REMOTE_BASE_URL is not set.")
        self.model = _resolve_model(self.base_url, model or QWEN_REMOTE_MODEL)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Remote Qwen call failed: {e}") from None

        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()

        raw_usage = data.get("usage", {})
        usage = {
            "prompt_tokens":     raw_usage.get("prompt_tokens", 0),
            "completion_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens":      raw_usage.get("total_tokens", 0),
        }
        return text, usage


if __name__ == "__main__":
    client = QwenRemoteClient()
    text, usage = client.call(
        system_prompt="You are a concise assistant. Answer in one sentence.",
        user_prompt="What are onion thrips?",
    )
    print(text, usage)