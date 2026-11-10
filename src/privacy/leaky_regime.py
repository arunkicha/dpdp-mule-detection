"""Membership-inference attacks in a small-hospital ("leaky") regime.

In the full-scale setting the model generalises (train AUC ~ test AUC), so no
attack finds leakage even without noise. Here the defender is a small hospital
network: a few thousand training admissions and a wider network trained without
weight decay, which memorises. The same federated pipeline is then run with

    no_dp                plain training (no clipping, no noise)
    noise0_sanity_check  DP-SGD machinery on, noise 0 (clipping only)
    flat_dp              same noise for every patient
    tiered               stronger noise for the stigma-sensitive tier
    tiered_strong        even stronger noise for the stigma-sensitive tier

and attacked three ways (all score "higher = member"):

    loss        negative per-example loss of the released model
    calibrated  loss z-scored per class using attacker reference data
    reference   loss of attacker-trained reference models minus target loss

Outputs: results/metrics/leaky_regime_{attacks,utility,epsilons}.csv
Run:     python -m src.privacy.leaky_regime
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import VITAL_FEATURES
from src.models.dp_federated import (
    build_features,
    get_flat_params,
    load_data,
    set_flat_params,
    standardize,
    weighted_average_params,
)
from src.privacy.accounting import epsilon_from_history
from src.privacy.tiered_dp import (
    DELTA,
    MAX_GRAD_NORM,
    NOISE_MULTIPLIER_FLAT,
    NOISE_MULTIPLIER_NON_SENSITIVE,
    NOISE_MULTIPLIER_STIGMA_SENSITIVE,
)

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "no_dp": {"stigma": None, "non_sensitive": None},
    "noise0_sanity_check": {"stigma": 0.0, "non_sensitive": 0.0},
    "flat_dp": {"stigma": NOISE_MULTIPLIER_FLAT, "non_sensitive": NOISE_MULTIPLIER_FLAT},
    "tiered": {
        "stigma": NOISE_MULTIPLIER_STIGMA_SENSITIVE,
        "non_sensitive": NOISE_MULTIPLIER_NON_SENSITIVE,
    },
    "tiered_strong": {"stigma": 2.0, "non_sensitive": NOISE_MULTIPLIER_NON_SENSITIVE},
}
ATTACKS = ["loss", "calibrated", "reference"]
GROUPS = ["overall", "stigma_sensitive", "non_sensitive"]
MIN_SUBSET = 8
MIN_GROUP = 10


class WideMLP(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--n-members", type=int, default=2000)
    p.add_argument("--n-nonmembers", type=int, default=2000)
    p.add_argument("--n-reference", type=int, default=2000)
    p.add_argument("--n-ref-models", type=int, default=4)
    p.add_argument("--ref-epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quick", action="store_true", help="tiny smoke-test settings")
    args = p.parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.n_members, args.n_nonmembers, args.n_reference = 600, 600, 600
        args.rounds, args.local_epochs, args.n_ref_models, args.ref_epochs = 4, 3, 2, 20
    return args


@torch.no_grad()
def per_sample_loss(model, X, y):
    model.eval()
    logits = model(torch.as_tensor(X))
    return F.binary_cross_entropy_with_logits(
        logits, torch.as_tensor(y, dtype=torch.float32), reduction="none"
    ).numpy()


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(X))).numpy()


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def train_subset(base_params, X, y, noise, n_features, args):
    n = len(X)
    if n < MIN_SUBSET:
        return None

    model = WideMLP(n_features, args.hidden)
    set_flat_params(model, base_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loader = DataLoader(TensorDataset(X, y), batch_size=max(1, n // 4), shuffle=True)

    engine = None
    if noise is not None:
        engine = PrivacyEngine()
        model, optimizer, loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise,
            max_grad_norm=MAX_GRAD_NORM,
        )

    model.train()
    for _ in range(args.local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()

    params = [p.detach().clone() for p in model.parameters()]
    history, round_eps = None, None
    if engine is not None and noise > 0:
        history = list(engine.accountant.history)
        try:
            round_eps = float(engine.get_epsilon(delta=DELTA))
        except Exception:
            round_eps = None
    return params, n, history, round_eps


def run_federated(config, subsets, client_ids, n_features, args):
    global_model = WideMLP(n_features, args.hidden)
    histories = {}
    last_round_eps = {}

    for rnd in range(args.rounds):
        client_params, client_weights = [], []
        for cid in client_ids:
            base = get_flat_params(global_model)
            parts = []
            for tier, key in [("stigma_sensitive", "stigma"), ("non_sensitive", "non_sensitive")]:
                sub = subsets[(cid, tier)]
                out = train_subset(base, sub["X"], sub["y"], config[key], n_features, args)
                if out is None:
                    continue
                params, n, history, round_eps = out
                parts.append((params, n))
                if history is not None:
                    histories.setdefault((cid, tier), []).extend(history)
                    last_round_eps[(cid, tier)] = round_eps
            if not parts:
                continue
            client_params.append(weighted_average_params([p for p, _ in parts], [n for _, n in parts]))
            client_weights.append(sum(n for _, n in parts))
        set_flat_params(global_model, weighted_average_params(client_params, client_weights))

    eps_rows = []
    for tier in ["stigma_sensitive", "non_sensitive"]:
        cum = [
            epsilon_from_history(h, DELTA)
            for (cid, t), h in histories.items()
            if t == tier
        ]
        per_round = [e for (cid, t), e in last_round_eps.items() if t == tier and e is not None]
        cum = [c for c in cum if c is not None]
        eps_rows.append(
            {
                "tier": tier,
                "epsilon_per_round_mean": float(np.mean(per_round)) if per_round else np.nan,
                "epsilon_cumulative_mean": float(np.mean(cum)) if cum else np.nan,
                "epsilon_cumulative_max": float(np.max(cum)) if cum else np.nan,
            }
        )
    return global_model, eps_rows


def train_reference_models(X_ref, y_ref, n_features, args, seed):
    rng = np.random.default_rng(seed + 1000)
    models = []
    for k in range(args.n_ref_models):
        idx = rng.choice(len(X_ref), size=int(0.75 * len(X_ref)), replace=False)
        Xk, yk = torch.as_tensor(X_ref[idx]), torch.as_tensor(y_ref[idx])
        model = WideMLP(n_features, args.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        loader = DataLoader(TensorDataset(Xk, yk), batch_size=64, shuffle=True)
        model.train()
        for _ in range(args.ref_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(model(xb), yb).backward()
                opt.step()
        models.append(model)
    return models


def attack_scores(model, ref_models, X_mem, y_mem, X_non, y_non, X_ref, y_ref):
    loss_mem = per_sample_loss(model, X_mem, y_mem)
    loss_non = per_sample_loss(model, X_non, y_non)
    loss_ref_target = per_sample_loss(model, X_ref, y_ref)

    mu, sd = {}, {}
    for c in (0.0, 1.0):
        vals = loss_ref_target[y_ref == c]
        if len(vals) < 2:
            vals = loss_ref_target
        mu[c], sd[c] = vals.mean(), vals.std() + 1e-8

    def calibrate(loss, y):
        return np.array([-(l - mu[c]) / sd[c] for l, c in zip(loss, y)])

    ref_mem = np.mean([per_sample_loss(m, X_mem, y_mem) for m in ref_models], axis=0)
    ref_non = np.mean([per_sample_loss(m, X_non, y_non) for m in ref_models], axis=0)

    return {
        "loss": (-loss_mem, -loss_non),
        "calibrated": (calibrate(loss_mem, y_mem), calibrate(loss_non, y_non)),
        "reference": (ref_mem - loss_mem, ref_non - loss_non),
    }


def attack_metrics(s_mem, s_non):
    if len(s_mem) < MIN_GROUP or len(s_non) < MIN_GROUP:
        return {"balanced_accuracy": np.nan, "auc": np.nan,
                "tpr_at_1pct_fpr": np.nan, "tpr_at_5pct_fpr": np.nan}
    labels = np.r_[np.ones(len(s_mem)), np.zeros(len(s_non))]
    scores = np.r_[s_mem, s_non]
    fpr, tpr, _ = roc_curve(labels, scores)

    def tpr_at(f):
        ok = fpr <= f
        return float(tpr[ok].max()) if ok.any() else 0.0

    return {
        "balanced_accuracy": float(0.5 + 0.5 * np.max(tpr - fpr)),
        "auc": float(roc_auc_score(labels, scores)),
        "tpr_at_1pct_fpr": tpr_at(0.01),
        "tpr_at_5pct_fpr": tpr_at(0.05),
    }


def make_subsets(train_df, X_train, y_train, client_ids):
    subsets = {}
    for cid in client_ids:
        cmask = (train_df["client_id"] == cid).to_numpy()
        for tier in ["stigma_sensitive", "non_sensitive"]:
            m = cmask & (train_df["final_tier"] == tier).to_numpy()
            subsets[(cid, tier)] = {
                "X": torch.as_tensor(X_train[m]),
                "y": torch.as_tensor(y_train[m]),
            }
    return subsets


def run_seed(seed, df, args):
    rng_seed = seed
    train_pool = df[df["split"] == "train"]
    test_pool = df[df["split"] == "test"].sample(frac=1.0, random_state=rng_seed)

    n_non, n_ref = args.n_nonmembers, args.n_reference
    if n_non + n_ref > len(test_pool):
        n_non = n_ref = len(test_pool) // 2
        print(f"  note: test split too small, using {n_non} non-members and {n_ref} reference admissions")

    members = train_pool.sample(n=min(args.n_members, len(train_pool)), random_state=rng_seed).reset_index(drop=True)
    nonmembers = test_pool.iloc[:n_non].reset_index(drop=True)
    reference = test_pool.iloc[n_non: n_non + n_ref].reset_index(drop=True)
    test_all = df[df["split"] == "test"].reset_index(drop=True)

    medians = members[["age"] + VITAL_FEATURES].median(numeric_only=True)
    X_mem, cols = build_features(members, medians)
    mean, std = X_mem.mean(axis=0, keepdims=True), X_mem.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    def prep(frame):
        X, _ = build_features(frame, medians)
        return standardize(X, mean, std).astype(np.float32)

    X_mem = standardize(X_mem, mean, std).astype(np.float32)
    X_non, X_ref, X_test = prep(nonmembers), prep(reference), prep(test_all)
    y_mem = members["hospital_expire_flag"].to_numpy(dtype=np.float32)
    y_non = nonmembers["hospital_expire_flag"].to_numpy(dtype=np.float32)
    y_ref = reference["hospital_expire_flag"].to_numpy(dtype=np.float32)
    y_test = test_all["hospital_expire_flag"].to_numpy(dtype=np.float32)
    t_mem, t_non, t_test = (members["final_tier"].to_numpy(), nonmembers["final_tier"].to_numpy(),
                            test_all["final_tier"].to_numpy())

    n_features = len(cols)
    client_ids = sorted(members["client_id"].dropna().unique())
    subsets = make_subsets(members, X_mem, y_mem, client_ids)

    torch.manual_seed(seed)
    np.random.seed(seed)
    ref_models = train_reference_models(X_ref, y_ref, n_features, args, seed)

    attack_rows, utility_rows, eps_rows = [], [], []
    for name in args.configs:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, eps = run_federated(CONFIGS[name], subsets, client_ids, n_features, args)

        p_mem, p_test = predict_proba(model, X_mem), predict_proba(model, X_test)
        for group in GROUPS:
            gm = np.ones(len(y_mem), bool) if group == "overall" else t_mem == group
            gt = np.ones(len(y_test), bool) if group == "overall" else t_test == group
            utility_rows.append({
                "config": name, "seed": seed, "group": group,
                "train_auc": safe_auc(y_mem[gm], p_mem[gm]),
                "test_auc": safe_auc(y_test[gt], p_test[gt]),
            })

        scores = attack_scores(model, ref_models, X_mem, y_mem, X_non, y_non, X_ref, y_ref)
        for attack in ATTACKS:
            s_mem, s_non = scores[attack]
            for group in GROUPS:
                gm = np.ones(len(s_mem), bool) if group == "overall" else t_mem == group
                gn = np.ones(len(s_non), bool) if group == "overall" else t_non == group
                row = attack_metrics(s_mem[gm], s_non[gn])
                row.update({"config": name, "seed": seed, "attack": attack, "group": group,
                            "n_members": int(gm.sum()), "n_nonmembers": int(gn.sum())})
                attack_rows.append(row)

        for e in eps:
            eps_rows.append({"config": name, "seed": seed, **e})
        print(f"  seed {seed} {name:<20} done in {time.time() - t0:5.1f}s")
    return attack_rows, utility_rows, eps_rows


def paired_report(attacks):
    print(f"\n{'=' * 74}\nPAIRED t-TESTS ON ATTACK BALANCED ACCURACY (across seeds)\n{'=' * 74}")
    pairs = [("no_dp", "flat_dp"), ("no_dp", "tiered"), ("flat_dp", "tiered"), ("flat_dp", "tiered_strong")]
    for attack in ATTACKS:
        for group in GROUPS:
            sub = attacks[(attacks["attack"] == attack) & (attacks["group"] == group)]
            piv = sub.pivot(index="seed", columns="config", values="balanced_accuracy")
            for a, b in pairs:
                if a not in piv.columns or b not in piv.columns or len(piv) < 3:
                    continue
                d = (piv[a] - piv[b]).dropna()
                if len(d) < 3 or d.std() == 0:
                    continue
                t = stats.ttest_rel(piv[a], piv[b], nan_policy="omit")
                print(f"{attack:<10} {group:<17} {a} - {b}: diff={d.mean():+.4f}  p={t.pvalue:.4f}")


def main():
    args = parse_args()
    print("Noise multipliers:", {k: (v["stigma"], v["non_sensitive"]) for k, v in CONFIGS.items()})
    print(f"Regime: members={args.n_members}, hidden={args.hidden}, rounds={args.rounds}, "
          f"local_epochs={args.local_epochs}, seeds={args.seeds}\n")

    df = load_data()
    all_attacks, all_util, all_eps = [], [], []
    for seed in args.seeds:
        print(f"--- seed {seed} ---")
        a, u, e = run_seed(seed, df, args)
        all_attacks += a
        all_util += u
        all_eps += e

    attacks, util, eps = pd.DataFrame(all_attacks), pd.DataFrame(all_util), pd.DataFrame(all_eps)
    attacks.to_csv(RESULTS_DIR / "leaky_regime_attacks.csv", index=False)
    util.to_csv(RESULTS_DIR / "leaky_regime_utility.csv", index=False)
    eps.to_csv(RESULTS_DIR / "leaky_regime_epsilons.csv", index=False)

    order = [c for c in CONFIGS if c in args.configs]
    print(f"\n{'=' * 74}\nUTILITY AND GENERALISATION GAP (mean over seeds)\n{'=' * 74}")
    u = util.groupby(["config", "group"])[["train_auc", "test_auc"]].mean()
    u["gap"] = u["train_auc"] - u["test_auc"]
    print(u.reindex(order, level=0).round(4).to_string())

    print(f"\n{'=' * 74}\nATTACK BALANCED ACCURACY (mean +/- std; 0.5 = chance)\n{'=' * 74}")
    for attack in ATTACKS:
        sub = attacks[attacks["attack"] == attack]
        table = sub.groupby(["config", "group"])["balanced_accuracy"].agg(["mean", "std"]).round(4)
        print(f"\n[{attack}]\n{table.reindex(order, level=0).to_string()}")

    print(f"\n{'=' * 74}\nATTACK TPR AT 5% FPR (mean)\n{'=' * 74}")
    tp = attacks.groupby(["attack", "config", "group"])["tpr_at_5pct_fpr"].mean().round(4)
    print(tp.to_string())

    print(f"\n{'=' * 74}\nEPSILON: per-round (what the main pipeline reports) vs cumulative over {args.rounds} rounds\n{'=' * 74}")
    print(eps.groupby(["config", "tier"])[["epsilon_per_round_mean", "epsilon_cumulative_mean",
                                           "epsilon_cumulative_max"]].mean().reindex(order, level=0).round(2).to_string())

    paired_report(attacks)
    print(f"\nSaved to {RESULTS_DIR}/leaky_regime_*.csv")


if __name__ == "__main__":
    main()
