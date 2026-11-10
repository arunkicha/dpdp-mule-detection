"""Centralized controls with the SAME model and training loop as federated.py.
Save as src/models/pooled_control.py and run from the repo root:

  python -m src.models.pooled_control --mode pooled --steps 20  --seed 42
  python -m src.models.pooled_control --mode pooled --steps 60  --seed 42
  python -m src.models.pooled_control --mode full   --steps 20  --seed 42
  python -m src.models.pooled_control --mode full   --steps 60  --seed 42
  python -m src.models.pooled_control --mode full   --steps 0 --init_noise 3.3 --seed 42

mode pooled : train on the union (deduplicated) of the ten clients' training edges
mode full   : train on all training edges
steps       : full-batch Adam steps (federated.py uses 10 rounds x 2 epochs = 20 steps per client;
              gnn_baseline.py uses 60)
init_noise  : std of Gaussian noise added to every parameter before training (0 = off)
Model: FederatedGNN(14 node feats, 15 edge feats, hidden 64, emb 32) -- same as federated/DP.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from src.models.federated import (FederatedGNN, build_node_features, edge_index_from_df,
                                  normalize_features, predict, calculate_metrics,
                                  find_best_threshold, set_seed)


def alert_budget(y, p, fracs=(0.01, 0.05, 0.10)):
    order = np.argsort(-p); out = {}
    for f in fracs:
        k = int(round(f * len(y))); hit = int(y[order[:k]].sum())
        out[f] = (hit / k, hit / max(int(y.sum()), 1), hit)
    return out


def main(a):
    set_seed(a.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = a.processed_dir
    edges = pd.read_parquet(f"{P}/edges_full.parquet"); ef = pd.read_parquet(f"{P}/edge_features.parquet")
    nb = pd.read_parquet(f"{P}/node_bank_lookup.parquet"); n_nodes = int(nb["node_id"].max()) + 1
    clients = pd.read_csv("results/federated_clients.csv")["bank"].astype(int).tolist()
    tm, vm, sm = (edges["split"] == s for s in ("train", "val", "test"))
    tr, va, te = edges[tm].copy(), edges[vm].copy(), edges[sm].copy()
    tei, vei, sei = (edge_index_from_df(d) for d in (tr, va, te))
    f = lambda m: torch.tensor(ef[m].to_numpy(dtype=np.float32))
    tea, vea, sea = f(tm), f(vm), f(sm)
    x = build_node_features(tr, n_nodes)
    x, (tea, vea, sea) = normalize_features(x, tea, [tea, vea, sea])
    yv, ys = va["is_laundering"].to_numpy(), te["is_laundering"].to_numpy()

    fb, tb = tr["from_bank"].astype(int), tr["to_bank"].astype(int)
    if a.mode == "pooled":
        keep = (fb.isin(clients) | tb.isin(clients)).to_numpy()
        per_client = sum(int(((fb == b) | (tb == b)).sum()) for b in clients)
        print(f"pooled unique edges: {keep.sum():,} | sum of per-client sizes: {per_client:,} "
              f"| duplicated: {per_client - keep.sum():,}")
    else:
        keep = np.ones(len(tr), dtype=bool)
    idx = torch.tensor(np.where(keep)[0])
    ei, ea = tei[:, idx], tea[idx]
    y = torch.tensor(tr["is_laundering"].to_numpy()[keep], dtype=torch.long)
    print(f"training edges: {len(y):,} | positives: {int(y.sum()):,}")

    model = FederatedGNN(x.size(1), tea.size(1), a.hidden_dim, a.emb_dim).to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    if a.init_noise > 0:
        with torch.no_grad():
            for p in model.parameters(): p.add_(torch.randn_like(p) * a.init_noise)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    npos = int(y.sum()); w = torch.tensor([1.0, (len(y) - npos) / max(npos, 1)], device=dev)
    xd, eid, ead, yd = x.to(dev), ei.to(dev), ea.to(dev), y.to(dev)
    t0 = time.time()
    for s in range(1, a.steps + 1):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(xd, eid, eid, ead), yd, weight=w)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if s % 10 == 0 or s == a.steps: print(f"  step {s}/{a.steps} loss {loss.item():.4f}")

    pv = predict(model, x, tei, vei, vea, dev); thr, _ = find_best_threshold(yv, pv)
    ps = predict(model, x, tei, sei, sea, dev); m = calculate_metrics(ys, ps, thr)
    ab = alert_budget(ys, ps)
    print(f"\n=== CONTROL ({a.mode}, {a.steps} steps, init_noise {a.init_noise}, seed {a.seed}) ===")
    print(f"Validation ROC-AUC: {roc_auc_score(yv, pv):.4f}")
    print(f"Test ROC-AUC: {m['auc']:.4f} | F1 {m['f1']:.4f} | PR-AUC {m['pr_auc']:.4f}")
    for k, (pr, rc, h) in ab.items():
        print(f"  Top-{int(k*100)}%: Precision={pr:.4f}  Recall={rc:.4f}  ({h}/{int(ys.sum())})")
    os.makedirs("results", exist_ok=True)
    with open("results/pooled_control.jsonl", "a") as fh:
        fh.write(json.dumps({"mode": a.mode, "steps": a.steps, "init_noise": a.init_noise, "seed": a.seed,
                             "test_auc": float(m["auc"]),
                             "alert": {str(k): v[:2] for k, v in ab.items()}, "secs": time.time() - t0}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--mode", choices=["pooled", "full"], required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--init_noise", type=float, default=0.0)
    main(ap.parse_args())