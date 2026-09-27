"""
plot_models.py - overlay the throughput curves of multiple models on one graph
to show where the bottleneck moves (compute-bound vs memory-bound).

Run after you've produced two or more results/03_model_*.json files:
    python plot_models.py

Produces:
    results/model_comparison_throughput.png
"""

import os
import glob
import json
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

files = sorted(glob.glob(os.path.join(RESULTS_DIR, "03_model_*.json")))
if not files:
    print("No results/03_model_*.json files found. Run benchmark_models.py first.")
    raise SystemExit

plt.figure(figsize=(8, 5))
for path in files:
    with open(path) as f:
        d = json.load(f)
    sweep = d["sweep"]
    users = [s["n_users"] for s in sweep]
    thr = [s["throughput"] for s in sweep]
    plt.plot(users, thr, marker="o", linewidth=2, label=d["label"])

plt.xlabel("Concurrent requests")
plt.ylabel("Throughput (tokens/sec, all requests)")
plt.title("Where the bottleneck moves: throughput vs concurrency by model\nvllm-metal on M3 (16GB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "model_comparison_throughput.png")
plt.savefig(out, dpi=150)
print(f"Saved -> {out}")
