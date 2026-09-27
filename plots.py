"""
plots.py - turn results/01_batching.json into graphs.

Run from inside the project folder:
    cd ~/Desktop/vllm-metal-benchmarks
    python plots.py

Reads and writes in the ./results folder next to this script, so it works
wherever the project folder lives.

Produces:
    results/throughput_vs_users.png   (headline batching curve, peaks then falls)
    results/latency_vs_users.png      (the tradeoff: TTFT & TPOT rising)
"""

import os
import json
import matplotlib.pyplot as plt

# results folder sits next to this script, wherever the project lives
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "01_batching.json")) as f:
    data = json.load(f)

users = [d["n_users"] for d in data]
throughput = [d["throughput"] for d in data]
ttft_ms = [d["avg_ttft"] * 1000 for d in data]
tpot_ms = [d["avg_tpot"] * 1000 for d in data]

# --- Graph 1: throughput vs concurrency (the headline) ---
plt.figure(figsize=(8, 5))
plt.plot(users, throughput, marker="o", linewidth=2)
peak_i = throughput.index(max(throughput))
plt.annotate(f"peak {throughput[peak_i]:.0f} tok/s\n@ {users[peak_i]} users",
             xy=(users[peak_i], throughput[peak_i]),
             xytext=(-20, -40), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", alpha=0.6))
plt.xlabel("Concurrent requests")
plt.ylabel("Throughput (tokens/sec, all requests)")
plt.title("vLLM on Apple Silicon (M3): throughput vs concurrency\nQwen3-0.6B, vllm-metal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "throughput_vs_users.png"), dpi=150)
print("Saved -> results/throughput_vs_users.png")

# --- Graph 2: the tradeoff (latency rises as concurrency grows) ---
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(users, ttft_ms, marker="o", color="tab:red", label="TTFT")
ax1.set_xlabel("Concurrent requests")
ax1.set_ylabel("Time to first token (ms)", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.plot(users, tpot_ms, marker="s", color="tab:blue", label="TPOT")
ax2.set_ylabel("Time per output token (ms)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("The tradeoff: per-request latency rises with concurrency\nQwen3-0.6B, vllm-metal")
fig.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "latency_vs_users.png"), dpi=150)
print("Saved -> results/latency_vs_users.png")
