"""Evaluation figures. Run from repo root after collect_results.py:
    python make_figures.py
Reads results/summary.csv and external_metrics.json; writes PNGs (300 dpi) to results/figures_eval/.
No in-figure titles: put them in the paper captions.
"""
import json, math, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.join("results", "figures_eval")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join("results", "summary.csv"))
ext = json.load(open("external_metrics.json"))
POS = ext.get("test_positive_rate", 0.00177)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Okabe-Ito colour-blind-safe palette; markers/linestyles also differ so it survives greyscale print
C = {"centralized_gnn": ("#0072B2", "o", "-"), "federated_gnn": ("#E69F00", "s", "-"),
     "logistic_regression": ("#009E73", "^", "--"), "hist_gradient_boosting": ("#D55E00", "D", "-."),
     "dp": ("#000000", "X", ":")}
NAMES = {"centralized_gnn": "Centralized GNN", "federated_gnn": "Federated GNN",
         "logistic_regression": "Logistic regression", "hist_gradient_boosting": "HGB (tabular)"}


def eps_of(noise, rounds=10, delta=1e-5):          # same alpha grid as the training code
    if not noise:
        return float("inf")
    return min(rounds * (i / 10) / (2 * noise ** 2) + math.log(1 / delta) / (i / 10 - 1) for i in range(11, 2560))


def fam(name):                                      # derived from file name, so a stale CSV column is harmless
    if name.startswith("check_"): return "check"
    if name.startswith("federated_plain"): return "federated_plain"
    if name.startswith("dp_personalized_s"): return "current_personalized"
    if name.startswith("dp_uniform"): return "current_uniform"
    if re.match(r"dp_nm[\d.]+_seed", name): return "frozen_early"
    return "exploratory"


df["family"] = df.file.map(fam)
df["seed"] = pd.to_numeric(df.seed, errors="coerce")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def eps_label(e):
    return "∞" if not np.isfinite(e) else (f"{e/1000:.1f}k" if e >= 1000 else f"{e:.3g}")


# ---- model table: external (centralized/LR/HGB) + federated recomputed from summary.csv
models = {k: dict(v) for k, v in ext.items() if isinstance(v, dict) and "alert" in v}
fed = df[df.family == "federated_plain"]
if len(fed):
    f = models.setdefault("federated_gnn", {"alert": {}})
    f["seeds"] = fed.test_auc.tolist()
    f["roc_auc_mean"], f["roc_auc_std"] = fed.test_auc.mean(), fed.test_auc.std(ddof=1)
    if fed.top1_P.notna().all():
        f["alert"] = {"0.01": [fed.top1_P.mean(), fed.top1_R.mean()],
                      "0.05": [fed.top5_P.mean(), fed.top5_R.mean()],
                      "0.1": [fed.top10_P.mean(), fed.top10_R.mean()]}
    print(f"federated: n={len(fed)} mean={f['roc_auc_mean']:.3f} std={f['roc_auc_std']:.3f}")

# ---- Fig 1: ROC-AUC vs noise multiplier
d = df[(df.n_features == 14) & df.noise_multiplier.notna() & (df.family == "current_uniform")].copy()
levels = sorted(d.noise_multiplier.unique())
STY = {"current_uniform": ("#0072B2", "o", "uniform clipping")}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for j, (fm, (c, mk, lab)) in enumerate(STY.items()):
    first = True
    for i, nm in enumerate(levels):
        v = d[(d.noise_multiplier == nm) & (d.family == fm)].test_auc.values
        if not len(v):
            continue
        x = i
        ax.scatter(np.full(len(v), x), v, color=c, marker=mk, alpha=.75, s=24, zorder=3, label=lab if first else None)
        first = False
        if len(v) >= 3:
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="_", color="black", capsize=3, ms=10, zorder=4)
ax.axhline(0.5, ls="--", color="gray", lw=1)
ax.text(-0.45, 0.503, "chance", color="gray", fontsize=7)
for key, col in (("centralized_gnn", C["centralized_gnn"][0]), ("federated_gnn", C["federated_gnn"][0])):
    m = models.get(key, {}).get("roc_auc_mean")
    if m:
        ax.axhline(m, color=col, lw=1)
        ax.text(len(levels) - .5, m + .004, NAMES[key], color=col, fontsize=7, ha="right")
ax.set_xticks(range(len(levels)))
ax.set_xticklabels([f"{nm:g}\n{eps_label(eps_of(nm))}" for nm in levels], fontsize=8)
ax.set_xlabel("noise multiplier σ   /   RDP ε (δ = 1e-5, below)")
ax.set_ylabel("test ROC-AUC")
ax.legend(frameon=False, fontsize=7, loc="lower left")
save(fig, "fig_roc_vs_noise.png")

# ---- Fig 2: personalized vs uniform (paired by seed, noise 0.5)
u = df[(df.family == "current_uniform") & (df.noise_multiplier == 0.5)].set_index("seed").test_auc
p = df[df.family == "current_personalized"].set_index("seed").test_auc
seeds = sorted(set(u.index) & set(p.index))
if seeds:
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for s in seeds:
        up = p[s] >= u[s]
        ax.plot([0, 1], [u[s], p[s]], "-o", color="#0072B2" if up else "#D55E00", alpha=.85)
        ax.text(1.04, p[s], f"seed {int(s)}", fontsize=7, va="center")
    ax.plot([0, 1], [u[seeds].mean(), p[seeds].mean()], "-s", color="black", lw=2)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.text(-0.18, 0.503, "chance", color="gray", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["uniform", "personalized"]); ax.set_xlim(-.2, 1.45)
    ax.set_ylabel("test ROC-AUC")
    ax.legend(handles=[Line2D([], [], color="#0072B2", marker="o", label="personalized higher"),
                       Line2D([], [], color="#D55E00", marker="o", label="personalized lower"),
                       Line2D([], [], color="black", marker="s", lw=2, label="mean")],
              frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig_personalized_vs_uniform.png")

# ---- DP overlay (seed-42 alert-budget run at noise 0.5)
dpr = df[df.file == "dp_uniform_s42"]
dp_alert = None
if False and len(dpr) and dpr.top1_P.notna().all():
    r = dpr.iloc[0]
    dp_alert = ([r.top1_P, r.top5_P, r.top10_P], [r.top1_R, r.top5_R, r.top10_R])
ks = ["0.01", "0.05", "0.1"]
X = [1, 5, 10]


def legend_below(ax):
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)


# ---- Fig 3: alert-budget recall
fig, ax = plt.subplots(figsize=(5, 3.4))
for k, v in models.items():
    c, mk, ls = C[k]
    ax.plot(X, [v["alert"][q][1] for q in ks], marker=mk, ls=ls, color=c, label=NAMES[k])
if dp_alert:
    ax.plot(X, dp_alert[1], marker=C["dp"][1], ls=C["dp"][2], color="black", label="DP-federated (σ=0.5, seed 42)")
ax.plot(X, [.01, .05, .10], "--", color="gray", lw=1, label="random flagging")
ax.set_xticks(X); ax.set_xticklabels(["1%", "5%", "10%"])
ax.set_xlabel("review budget (top-k% of transactions)"); ax.set_ylabel("recall")
legend_below(ax)
save(fig, "fig_alert_budget_recall.png")

# ---- Fig 4: precision lift over random
fig, ax = plt.subplots(figsize=(5, 3.4))
for k, v in models.items():
    c, mk, ls = C[k]
    ax.plot(X, [v["alert"][q][0] / POS for q in ks], marker=mk, ls=ls, color=c, label=NAMES[k])
if dp_alert:
    ax.plot(X, [x / POS for x in dp_alert[0]], marker=C["dp"][1], ls=C["dp"][2], color="black",
            label="DP-federated (σ=0.5, seed 42)")
ax.axhline(1, ls="--", color="gray", lw=1, label="random flagging (lift = 1)")
ax.set_xticks(X); ax.set_xticklabels(["1%", "5%", "10%"])
ax.set_xlabel("review budget (top-k% of transactions)"); ax.set_ylabel("precision lift over random")
legend_below(ax)
save(fig, "fig_alert_budget_lift.png")

# ---- Fig 5: non-private ROC-AUC (dots = seeds where available, bar = mean ± std)
rows = [(k, v["roc_auc_mean"], v.get("roc_auc_std") or 0, v.get("seeds", []))
        for k, v in models.items() if v.get("roc_auc_mean")]
if rows:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, (k, m, sd, sds) in enumerate(rows):
        ax.errorbar(i, m, yerr=sd, fmt="_", color="black", capsize=5, ms=18, lw=1.5, zorder=3)
        if sds:
            ax.scatter(np.full(len(sds), i) + np.linspace(-.12, .12, len(sds)), sds, color=C[k][0],
                       marker=C[k][1], s=26, alpha=.85, zorder=4)
        else:
            ax.scatter([i], [m], color=C[k][0], marker=C[k][1], s=26, zorder=4)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.text(ax.get_xlim()[1], 0.505, "chance", color="gray", fontsize=7, ha="right")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([NAMES[r[0]] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("test ROC-AUC"); ax.set_ylim(0.4, 0.8)
    save(fig, "fig_roc_auc_models.png")