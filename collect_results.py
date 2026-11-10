"""Scan results/metrics/*.pt and write results/summary.csv (one row per run).
Run from the repo root:  python collect_results.py
"""
import csv, glob, math, os, re
import torch

METRICS_DIR = os.path.join("results", "metrics")
OUT_CSV = os.path.join("results", "summary.csv")


def epsilon(noise, rounds, delta):
    """Gaussian RDP, full client participation (q=1), composed over `rounds`."""
    if not noise:
        return float("inf")
    best = float("inf")
    # same alpha grid as the training code (1.1, 1.2, ...), so values match the logs
    # (e.g. 50.3912 at sigma=0.5, 2315.1293 at sigma=0.05)
    for i in range(11, 2560):
        a = i / 10
        best = min(best, rounds * a / (2 * noise ** 2) + math.log(1 / delta) / (a - 1))
    return best


def n_features(ckpt):
    sd = ckpt.get("model_state_dict", ckpt)
    for k, v in sd.items():
        if k.endswith("conv1.lin_l.weight"):
            return int(v.shape[1])
    return None


rows = []
for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.pt"))):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("skip", name, e)
        continue
    if not isinstance(ck, dict) or "test_metrics" not in ck:
        continue  # weights-only file
    cfg = ck.get("dp_config", {})
    tm = ck["test_metrics"]
    ab = ck.get("alert_budget_metrics", {}) or {}
    m = re.search(r"(?:_s|_seed)(\d+)$", name)
    row = {
        "file": name,
        "seed": int(m.group(1)) if m else "",
        "n_features": n_features(ck),
        "noise_multiplier": cfg.get("noise_multiplier"),
        "personalized": cfg.get("personalized", ""),
        "rounds": cfg.get("rounds"),
        "local_epochs": cfg.get("local_epochs"),
        "delta": cfg.get("delta"),
        "epsilon": (round(epsilon(cfg["noise_multiplier"], cfg["rounds"], cfg["delta"]), 4)
                    if cfg else ""),
        "test_auc": tm.get("auc"), "test_f1": tm.get("f1"),
        "test_precision": tm.get("precision"), "test_recall": tm.get("recall"),
        "test_pr_auc": tm.get("pr_auc"),
    }
    for k in ("0.01", "0.05", "0.1"):
        d = ab.get(float(k)) or ab.get(k) or {}
        row[f"top{int(float(k)*100)}_P"] = d.get("precision", "")
        row[f"top{int(float(k)*100)}_R"] = d.get("recall", "")
    if name.startswith("check_"): row["family"] = "check"
    elif name.startswith("federated_plain"): row["family"] = "federated_plain"
    elif name.startswith("dp_uniform"): row["family"] = "current_uniform"
    elif name.startswith("dp_personalized_s"): row["family"] = "current_personalized"
    elif re.match(r"dp_nm[\d.]+_seed", name): row["family"] = "frozen_early"
    else: row["family"] = "exploratory"
    rows.append(row)

os.makedirs("results", exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT_CSV} ({len(rows)} runs)")
