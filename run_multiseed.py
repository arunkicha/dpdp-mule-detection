# run_multiseed.py
import json
import numpy as np
from pathlib import Path

SEEDS = [42, 123, 2024]  # add 7, 999 if time allows
CONFIGS = {
    "centralized": run_centralized_gnn,   # wrap your existing gnn_baseline.py main() 
    "federated": run_federated,            # wrap federated.py main()
    "dp_noise_0.5": lambda seed: run_dp_federated(seed, noise=0.5),
    "dp_noise_1.0": lambda seed: run_dp_federated(seed, noise=1.0),
    # add more noise levels once you extend the sweep
}

results = {name: [] for name in CONFIGS}

for name, run_fn in CONFIGS.items():
    for seed in SEEDS:
        metrics = run_fn(seed)  # returns {"roc_auc":..., "pr_auc":..., "f1":...}
        metrics["seed"] = seed
        results[name].append(metrics)
        print(f"{name} seed={seed}: {metrics}")

# Aggregate
summary = {}
for name, runs in results.items():
    arr = {k: np.array([r[k] for r in runs]) for k in ["roc_auc", "pr_auc", "f1"]}
    summary[name] = {
        k: {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
        for k, v in arr.items()
    }

Path("results/multiseed_summary.json").write_text(json.dumps(summary, indent=2))
Path("results/multiseed_raw.json").write_text(json.dumps(results, indent=2))