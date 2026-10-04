"""Benchmark the free-tier Groq models on what this app actually needs.

Two things decide the default model: latency (the UI shows it) and whether the
model reliably returns the JSON envelopes every agent prompt asks for.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

CANDIDATES = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]

PROMPT = (
    "Answer strictly as JSON with keys: answer (string), confidence (number 0-1).\n"
    "Question: In one sentence, what does the second law of thermodynamics say?"
)


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GROQ_API_KEY="):
            return line.partition("=")[2].strip()
    raise SystemExit("GROQ_API_KEY not found in .env")


def run(client: httpx.Client, key: str, model: str, runs: int = 3) -> dict[str, object]:
    latencies: list[float] = []
    json_ok = 0
    sample = ""
    for _ in range(runs):
        start = time.perf_counter()
        try:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                },
                timeout=90.0,
            )
        except httpx.HTTPError as exc:
            return {"error": type(exc).__name__}
        latencies.append(time.perf_counter() - start)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:120]}"}
        text = response.json()["choices"][0]["message"]["content"]
        try:
            json.loads(text)
            json_ok += 1
        except ValueError:
            pass
        sample = text[:90]

    return {
        "median_s": statistics.median(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "json_ok": f"{json_ok}/{runs}",
        "sample": sample,
    }


def main() -> int:
    key = api_key()
    print(f"{'model':<26} {'median':>8} {'min':>8} {'max':>8}  json")
    print("-" * 66)
    with httpx.Client() as client:
        for model in CANDIDATES:
            result = run(client, key, model)
            if "error" in result:
                print(f"{model:<26} ERROR {result['error']}")
                continue
            print(
                f"{model:<26} {result['median_s']:>7.2f}s {result['min_s']:>7.2f}s "
                f"{result['max_s']:>7.2f}s  {result['json_ok']}"
            )
            print(f"    {result['sample']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())