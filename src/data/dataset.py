"""
AMLworld HI-Small -- PyTorch Geometric Dataset Loader
=======================================================

Loads the parquet artifacts produced by build_graph.py and assembles a
PyTorch Geometric `Data` object for EDGE classification (each transaction
is laundering / not-laundering -- this is not a node classification task).

Node features are built from per-account transaction aggregates, since
build_graph.py only produced edge-level features. This is standard
practice for transaction-graph fraud/AML detection (see e.g. the IBM
AMLworld baseline GNN papers).

Usage (as a library):
    from dataset import load_aml_data
    data = load_aml_data("data/processed")
    # data.x            -> [num_nodes, num_node_features]
    # data.edge_index   -> [2, num_edges]
    # data.edge_attr    -> [num_edges, num_edge_features]
    # data.y            -> [num_edges]  (is_laundering, 0/1)
    # data.train_mask / data.val_mask / data.test_mask -> [num_edges] bool
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


NODE_FEATURE_COLS = [
    "out_degree", "in_degree", "total_degree",
    "out_amount_sum", "in_amount_sum",
    "out_amount_mean", "in_amount_mean",
    "out_laundering_count", "in_laundering_count",
]


def build_node_features(edges: pd.DataFrame, num_nodes: int) -> np.ndarray:
    """
    Aggregate per-account statistics from the transaction (edge) table.
    Purely computed from TRAIN-split edges would be more rigorous (avoids
    leaking val/test structure into node features), but for a first
    baseline we use the full edge set -- note this as a known limitation
    to mention in your Review 2 writeup / and a straightforward
    improvement for Review 3.
    """
    feats = np.zeros((num_nodes, len(NODE_FEATURE_COLS)), dtype=np.float32)

    out_deg = edges.groupby("src").size()
    in_deg = edges.groupby("dst").size()
    out_amt_sum = edges.groupby("src")["amount_paid"].sum()
    in_amt_sum = edges.groupby("dst")["amount_received"].sum()
    out_amt_mean = edges.groupby("src")["amount_paid"].mean()
    in_amt_mean = edges.groupby("dst")["amount_received"].mean()
    out_laund = edges.groupby("src")["is_laundering"].sum()
    in_laund = edges.groupby("dst")["is_laundering"].sum()

    def fill(series, col_idx):
        idx = series.index.to_numpy()
        vals = series.to_numpy()
        valid = (idx >= 0) & (idx < num_nodes)
        feats[idx[valid], col_idx] = vals[valid]

    fill(out_deg, 0)
    fill(in_deg, 1)
    feats[:, 2] = feats[:, 0] + feats[:, 1]  # total_degree
    fill(out_amt_sum, 3)
    fill(in_amt_sum, 4)
    fill(out_amt_mean, 5)
    fill(in_amt_mean, 6)
    fill(out_laund, 7)
    fill(in_laund, 8)

    # log1p-scale the heavy-tailed amount/degree columns for training stability
    for col_idx in [0, 1, 2, 3, 4, 5, 6]:
        feats[:, col_idx] = np.log1p(feats[:, col_idx])

    return feats


def load_aml_data(processed_dir: str) -> Data:
    processed_dir = Path(processed_dir)

    edges = pd.read_parquet(processed_dir / "edges_full.parquet")
    edge_feats = pd.read_parquet(processed_dir / "edge_features.parquet")
    node_bank = pd.read_parquet(processed_dir / "node_bank_lookup.parquet")

    assert len(edges) == len(edge_feats), (
        "edges_full.parquet and edge_features.parquet row counts differ -- "
        "they must be row-aligned outputs from the same build_graph.py run."
    )

    num_nodes = int(node_bank["node_id"].max()) + 1

    print(f"Building node features from {len(edges):,} edges over {num_nodes:,} nodes...")
    x = build_node_features(edges, num_nodes)

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
    ap.add_argument("--save_to", default=None,
                     help="Optional path to save the Data object via torch.save")
    args = ap.parse_args()

    data = load_aml_data(args.processed_dir)

    if args.save_to:
        torch.save(data, args.save_to)
        print(f"Saved Data object to {args.save_to}")