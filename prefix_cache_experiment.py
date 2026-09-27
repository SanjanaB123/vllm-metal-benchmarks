"""
prefix_cache_experiment.py - isolate the effect of prefix caching.

The idea:
  - SHARED case: every request begins with the SAME long prefix (e.g. a big
    system prompt). After the first request caches that prefix, all others
    reuse it -> high hit rate -> low TTFT (prefill is mostly skipped).
  - UNIQUE case: every request has a DIFFERENT prefix -> nothing to reuse
    -> ~0% hit rate -> higher TTFT (full prefill every time).

The TTFT gap between the two IS the prefix-caching benefit.

This experiment uses TWO server runs for a true A/B. From inside the project folder:

  Run A (prefix caching ON - the default):
      vllm serve Qwen/Qwen3-0.6B --max-model-len 2048
      python prefix_cache_experiment.py on

  Run B (prefix caching OFF - restart the server with it disabled):
      vllm serve Qwen/Qwen3-0.6B --max-model-len 2048 --no-enable-prefix-caching
      python prefix_cache_experiment.py off

  (If --no-enable-prefix-caching is rejected, run `vllm serve --help | grep -i prefix`
   to find the exact flag name for your build.)

Reads/writes the ./results folder next to this script.
Produces results/02_prefix_cache_<state>.json
"""

import sys
import time
import json
import asyncio
import os
import aiohttp

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 50
N_REQUESTS = 20          # requests per condition
SAMPLING = {"temperature": 0.0}

# results folder sits next to this script, wherever the project lives
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# A long shared prefix - the kind of big system prompt that prefix caching
# is designed to help with. ~identical across all SHARED requests.
LONG_PREFIX = (
    "You are an expert AI assistant with deep knowledge across many domains. "
    "Always answer carefully, precisely, and concisely. Consider edge cases, "
    "state assumptions explicitly, and prefer clarity over cleverness. "
    "You have been given the following extensive background context that you "
    "must keep in mind for every answer you produce in this session. "
) * 8  # repeated to make the prefix genuinely long (~hundreds of tokens)


async def one_request(session, content):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content + " /no_think"}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        **SAMPLING,
    }
    t0 = time.perf_counter()
    first_token_t = None
    async with session.post(BASE_URL, json=payload) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") and first_token_t is None:
                first_token_t = time.perf_counter()
    if first_token_t is None:
        first_token_t = time.perf_counter()
    return first_token_t - t0   # TTFT


async def run_condition(shared: bool):
    """SHARED: same prefix every time. UNIQUE: different prefix every time."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        ttfts = []
        # send sequentially so the cache has a chance to populate & be reused
        for i in range(N_REQUESTS):
            if shared:
                content = LONG_PREFIX + " Question: What is 2+2?"
            else:
                # unique prefix each time -> defeats the cache
                content = (f"Session {i}, id {i*7919}. " + LONG_PREFIX +
                           f" Question number {i}: What is 2+2?")
            ttft = await one_request(session, content)
            ttfts.append(ttft)
        return ttfts


async def main(state):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running SHARED-prefix requests (should benefit from cache)...")
    shared = await run_condition(shared=True)
    print("Running UNIQUE-prefix requests (cannot use cache)...")
    unique = await run_condition(shared=False)

    # skip the very first request of each (cold start / cache warm-up)
    shared_avg = sum(shared[1:]) / len(shared[1:])
    unique_avg = sum(unique[1:]) / len(unique[1:])

    result = {
        "server_prefix_caching": state,
        "shared_prefix_avg_ttft_ms": shared_avg * 1000,
        "unique_prefix_avg_ttft_ms": unique_avg * 1000,
        "speedup": unique_avg / shared_avg if shared_avg else 0,
        "shared_ttfts_ms": [t * 1000 for t in shared],
        "unique_ttfts_ms": [t * 1000 for t in unique],
    }

    print(f"\n--- Prefix caching {state.upper()} on server ---")
    print(f"Shared prefix  avg TTFT: {shared_avg*1000:7.1f} ms")
    print(f"Unique prefix  avg TTFT: {unique_avg*1000:7.1f} ms")
    print(f"Shared is {result['speedup']:.2f}x faster to first token")

    out = os.path.join(RESULTS_DIR, f"02_prefix_cache_{state}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    state = sys.argv[1] if len(sys.argv) > 1 else "on"
    asyncio.run(main(state))
