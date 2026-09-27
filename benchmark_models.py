"""
benchmark_models.py - run the concurrency sweep for MULTIPLE models and
save each to its own results file, so you can compare where the bottleneck
moves (compute-bound small model vs memory-bound larger model).

You run ONE model per server session. The server only holds one model at a
time, so the workflow is:

  # small model
  vllm serve Qwen/Qwen3-0.6B --max-model-len 2048
  python benchmark_models.py qwen3-0.6b

  # then Ctrl+C the server, start the bigger one, and run again:
  vllm serve mlx-community/Qwen2.5-7B-Instruct-4bit --max-model-len 1024
  python benchmark_models.py qwen2.5-7b-4bit

The label you pass is just used to name the output file and the model field.
IMPORTANT: pass the SAME model string the server is running (see MODELS below).

Reads/writes the ./results folder next to this script.
Produces results/03_model_<label>.json
"""

import sys
import time
import json
import asyncio
import os
import aiohttp

BASE_URL = "http://localhost:8000/v1/chat/completions"

# label -> the model string the server serves.
# Add/edit as you try more models.
MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen2.5-7b-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "smollm3-3b-4bit": "mlx-community/SmolLM3-3B-4bit",
}

PROMPT = "Explain what gradient descent is."
MAX_TOKENS = 100
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
SAMPLING = {"temperature": 0.0}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


async def one_request(session, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + " /no_think"}],
        "max_tokens": max_tokens,
        "stream": True,
        **SAMPLING,
    }
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
    try:
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
    except Exception as e:
        # a failed request at high concurrency IS data (the ceiling)
        return {"ttft": 0, "total": 0, "decode": 0, "tokens": 0, "error": str(e)}
    t_end = time.perf_counter()
    if first_token_t is None:
        first_token_t = t_end
    return {
        "ttft": first_token_t - t0,
        "total": t_end - t0,
        "decode": t_end - first_token_t,
        "tokens": n_tokens,
        "error": None,
    }


async def run_load(model, n_users):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        wall_start = time.perf_counter()
        results = await asyncio.gather(
            *[one_request(session, model, PROMPT, MAX_TOKENS)
              for _ in range(n_users)]
        )
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if not r["error"] and r["tokens"] > 0]
    errors = len(results) - len(ok)
    total_tokens = sum(r["tokens"] for r in ok)
    avg_ttft = sum(r["ttft"] for r in ok) / len(ok) if ok else 0.0
    tpots = [r["decode"] / (r["tokens"] - 1) for r in ok if r["tokens"] > 1]
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    return {
        "n_users": n_users,
        "wall_time": wall,
        "total_tokens": total_tokens,
        "throughput": total_tokens / wall if wall else 0.0,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "avg_latency": sum(r["total"] for r in ok) / len(ok) if ok else 0.0,
        "failed_requests": errors,
    }


def main(label):
    if label not in MODELS:
        print(f"Unknown label '{label}'. Known: {list(MODELS)}")
        sys.exit(1)
    model = MODELS[label]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Model: {model}  (label: {label})")
    print(f"{'users':>6} {'thruput':>10} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'fail':>5}")
    print("-" * 46)
    all_results = []
    for n in CONCURRENCY_LEVELS:
        time.sleep(2)
        r = asyncio.run(run_load(model, n))
        all_results.append(r)
        print(f"{r['n_users']:>6} {r['throughput']:>9.1f}  "
              f"{r['avg_ttft']*1000:>8.0f} {r['avg_tpot']*1000:>8.1f} "
              f"{r['failed_requests']:>5}")
        # if a whole level failed, we've hit the ceiling; stop early
        if r["failed_requests"] == r["n_users"]:
            print("  (all requests failed at this level - stopping, ceiling reached)")
            break

    out = os.path.join(RESULTS_DIR, f"03_model_{label}.json")
    with open(out, "w") as f:
        json.dump({"model": model, "label": label, "sweep": all_results}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
    main(label)
