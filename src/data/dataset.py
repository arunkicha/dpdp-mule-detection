"""
AMLworld HI-Small -- EDGE-LEVEL Dataset Loader (MAIN TASK)
=============================================================

Builds the transaction-level (edge) classification dataset used by
EVERY model in the main results table: tabular_baseline.py,
strong_account_baseline.py, and gnn_baseline.py.

Task: is this TRANSACTION laundering? (data.y = is_laundering, per edge)
This is the real IBM label -- not a derived/proxy label.

Node features (14-dim, log1p-scaled) are built from TRAIN-split edges
only, to avoid leaking validation/test structure into account-level
statistics.

Output: a single Data object with:
  - data.x            : [num_nodes, 14]  node (account) features
  - data.edge_index    : [2, num_edges]   all edges (src, dst)
  - data.edge_attr     : [num_edges, F]   edge (transaction) features
  - data.y             : [num_edges]      is_laundering (0/1), per edge
  - data.train_mask / val_mask / test_mask : [num_edges] bool, over EDGES

gnn_baseline.py already expects exactly this shape -- no changes needed
there. tabular_baseline.py / strong_account_baseline.py are rewritten
separately to build [x[src] || x[dst] || edge_attr] feature rows from
this same Data object, so all three models solve the identical task.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


NODE_FEATURE_COLS = [
    "out_degree", "in_degree", "total_degree",
    "out_unique_counterparties", "in_unique_counterparties",
    "out_amount_sum", "in_amount_sum",
    "out_amount_mean", "in_amount_mean",
    "out_amount_std", "in_amount_std",
    "out_bank_count", "in_bank_count",
    "out_in_degree_ratio",
]
# NOTE: no laundering-count features here -- including "how many
# laundering transactions has this account touched" as a NODE FEATURE
# would leak the label directly. Only structural/volume statistics.


def build_node_features(edges: pd.DataFrame, num_nodes: int) -> np.ndarray:
    """Built from TRAIN-split edges only (caller passes train_edges)."""
    feats = np.zeros((num_nodes, len(NODE_FEATURE_COLS)), dtype=np.float32)

    out_group = edges.groupby("src")
    in_group = edges.groupby("dst")

    out_degree = out_group.size()
    in_degree = in_group.size()
    out_unique = out_group["dst"].nunique()
    in_unique = in_group["src"].nunique()
    out_amount_sum = out_group["amount_paid"].sum()
    in_amount_sum = in_group["amount_received"].sum()
    out_amount_mean = out_group["amount_paid"].mean()
    in_amount_mean = in_group["amount_received"].mean()
    out_amount_std = out_group["amount_paid"].std().fillna(0)
    in_amount_std = in_group["amount_received"].std().fillna(0)
    out_bank_count = out_group["dst"].nunique()
    in_bank_count = in_group["src"].nunique()

    def fill(series, col_idx):
        idx = series.index.to_numpy()
        vals = series.to_numpy(dtype=np.float32)
        valid = (idx >= 0) & (idx < num_nodes)
        feats[idx[valid], col_idx] = vals[valid]

    fill(out_degree, 0)
    fill(in_degree, 1)
    feats[:, 2] = feats[:, 0] + feats[:, 1]
    fill(out_unique, 3)
    fill(in_unique, 4)
    fill(out_amount_sum, 5)
    fill(in_amount_sum, 6)
    fill(out_amount_mean, 7)
    fill(in_amount_mean, 8)
    fill(out_amount_std, 9)
    fill(in_amount_std, 10)
    fill(out_bank_count, 11)
    fill(in_bank_count, 12)
    feats[:, 13] = feats[:, 0] / np.maximum(feats[:, 1], 1)

    for col_idx in range(len(NODE_FEATURE_COLS)):
        feats[:, col_idx] = np.log1p(np.maximum(feats[:, col_idx], 0))

    return feats


def load_aml_data(processed_dir: str) -> Data:
    processed_dir = Path(processed_dir)

    edges = pd.read_parquet(processed_dir / "edges_full.parquet")
    edge_feats = pd.read_parquet(processed_dir / "edge_features.parquet")
    node_bank = pd.read_parquet(processed_dir / "node_bank_lookup.parquet")

    assert len(edges) == len(edge_feats), (
        "edges_full.parquet and edge_features.parquet row counts differ."
    )

    num_nodes = int(node_bank["node_id"].max()) + 1
    train_edges = edges[edges["split"] == "train"].copy()

    print(f"Building node features from {len(train_edges):,} TRAIN edges "
          f"over {num_nodes:,} nodes...")
    x = build_node_features(train_edges, num_nodes)

    edge_index = torch.tensor(
        np.stack([edges["src"].to_numpy(), edges["dst"].to_numpy()]),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(edge_feats.to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor(edges["is_laundering"].to_numpy(dtype=np.int64), dtype=torch.long)

    train_mask = torch.tensor((edges["split"] == "train").to_numpy())
    val_mask = torch.tensor((edges["split"] == "val").to_numpy())
    test_mask = torch.tensor((edges["split"] == "test").to_numpy())

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"Data object ready: {data.num_nodes:,} nodes, {data.num_edges:,} edges, "
          f"{data.x.size(1)} node features, {data.edge_attr.size(1)} edge features")
    print(f"  Positive (laundering) edges: {int(y.sum()):,} / {len(y):,} "
          f"({100 * y.float().mean().item():.4f}%)")
    print(f"  Split sizes -- train: {int(train_mask.sum()):,}, "
          f"val: {int(val_mask.sum()):,}, test: {int(test_mask.sum()):,}")

    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--save_to", default="data/processed/graph_data.pt")
    args = ap.parse_args()

    data = load_aml_data(args.processed_dir)
    torch.save(data, args.save_to)
    print(f"Saved Data object to {args.save_to}")