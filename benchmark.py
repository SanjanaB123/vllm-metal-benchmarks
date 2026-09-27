"""
benchmark.py - concurrency sweep against a running vLLM server.

Run the server first (separate terminal):
    vllm serve Qwen/Qwen3-0.6B --max-model-len 2048

Then, from inside the project folder:
    cd ~/Desktop/vllm-metal-benchmarks
    ulimit -n 8192
    python benchmark.py

Reads/writes the ./results folder next to this script.
Produces results/01_batching.json for plots.py.
"""

import time
import json
import asyncio
import os
import aiohttp

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Explain what gradient descent is."
MAX_TOKENS = 100
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]

# results folder sits next to this script, wherever the project lives
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Pin sampling so token counts are comparable across runs.
SAMPLING = {"temperature": 0.0}


async def one_request(session, prompt, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + " /no_think"}],
        "max_tokens": max_tokens,
        "stream": True,
        **SAMPLING,
    }
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
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
            if delta.get("content"):
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                n_tokens += 1
    t_end = time.perf_counter()
    if first_token_t is None:
        first_token_t = t_end
    return {
        "ttft": first_token_t - t0,
        "total": t_end - t0,
        "decode": t_end - first_token_t,
        "tokens": n_tokens,
    }


async def run_load(n_users):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        wall_start = time.perf_counter()
        results = await asyncio.gather(
            *[one_request(session, PROMPT, MAX_TOKENS) for _ in range(n_users)]
        )
        wall = time.perf_counter() - wall_start

    total_tokens = sum(r["tokens"] for r in results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    tpots = [r["decode"] / (r["tokens"] - 1)
             for r in results if r["tokens"] > 1]
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    return {
        "n_users": n_users,
        "wall_time": wall,
        "total_tokens": total_tokens,
        "throughput": total_tokens / wall if wall else 0.0,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "avg_latency": sum(r["total"] for r in results) / len(results),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    print(f"{'users':>6} {'thruput':>10} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'latency(s)':>10}")
    print("-" * 48)
    for n in CONCURRENCY_LEVELS:
        time.sleep(2)
        r = asyncio.run(run_load(n))
        all_results.append(r)
        print(f"{r['n_users']:>6} {r['throughput']:>9.1f}  "
              f"{r['avg_ttft']*1000:>8.0f} {r['avg_tpot']*1000:>8.1f} "
              f"{r['avg_latency']:>9.2f}")

    out = os.path.join(RESULTS_DIR, "01_batching.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
