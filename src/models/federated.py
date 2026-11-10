import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from src.metrics_alert_budget import alert_budget_metrics, format_alert_budget_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class EdgeClassifier(nn.Module):
    def __init__(self, node_emb_dim, edge_feat_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, node_emb, edge_index, edge_attr):
        src, dst = edge_index
        pair = torch.cat([node_emb[src], node_emb[dst], edge_attr], dim=1)
        return self.mlp(pair)


class FederatedGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, emb_dim=32):
        super().__init__()
        self.encoder = GraphSAGEEncoder(node_feat_dim, hidden_dim, emb_dim)
        self.classifier = EdgeClassifier(emb_dim, edge_feat_dim, hidden_dim)

    def forward(self, x, message_edge_index, classify_edge_index, edge_attr):
        node_emb = self.encoder(x, message_edge_index)
        return self.classifier(node_emb, classify_edge_index, edge_attr)


def build_node_features(edges, num_nodes):
    """
    14-feature builder -- matches src/data/dataset.py exactly, so the
    federated/DP-federated models see the same feature richness as the
    centralized GNN baseline. Previously this used only 7 basic features,
    which understated federated/DP performance for reasons unrelated to
    federation or privacy.
    """
    features = np.zeros((num_nodes, 14), dtype=np.float32)

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

    def fill(series, column):
        idx = series.index.to_numpy()
        values = series.to_numpy(dtype=np.float32)
        valid = (idx >= 0) & (idx < num_nodes)
        features[idx[valid], column] = values[valid]

    fill(out_degree, 0)
    fill(in_degree, 1)
    features[:, 2] = features[:, 0] + features[:, 1]
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
    features[:, 13] = features[:, 0] / np.maximum(features[:, 1], 1)

    for column in range(14):
        features[:, column] = np.log1p(np.maximum(features[:, column], 0))

    return torch.tensor(features, dtype=torch.float)


def edge_index_from_df(edges):
    return torch.tensor(
        np.stack([edges["src"].to_numpy(), edges["dst"].to_numpy()]),
        dtype=torch.long
    )


def normalize_features(x, train_edge_attr, edge_attrs):
    node_mean = x.mean(dim=0)
    node_std = x.std(dim=0).clamp(min=1e-6)

    edge_mean = train_edge_attr.mean(dim=0)
    edge_std = train_edge_attr.std(dim=0).clamp(min=1e-6)

    x = (x - node_mean) / node_std

    normalized_edges = []
    for edge_attr in edge_attrs:
        normalized_edges.append((edge_attr - edge_mean) / edge_std)

    return x, normalized_edges


def fedavg(states, sizes):
    total_size = sum(sizes)
    result = {}

    for key in states[0]:
        result[key] = states[0][key].float() * sizes[0] / total_size
        for i in range(1, len(states)):
            result[key] += states[i][key].float() * sizes[i] / total_size

    return result


def train_local(global_model, x, local_edge_index, local_edge_attr, labels, args, device):
    model = copy.deepcopy(global_model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)

    x = x.to(device)
    local_edge_index = local_edge_index.to(device)
    local_edge_attr = local_edge_attr.to(device)
    labels = labels.to(device)

    for _ in range(args.local_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(x, local_edge_index, local_edge_index, local_edge_attr)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return model.state_dict(), len(labels), n_pos


def predict(model, x, message_edge_index, classify_edge_index, edge_attr, device):
    model.eval()
    with torch.no_grad():
        logits = model(
            x.to(device),
            message_edge_index.to(device),
            classify_edge_index.to(device),
            edge_attr.to(device)
        )
        probabilities = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probabilities


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, probabilities)
    except ValueError:
        pr_auc = float("nan")

    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc, "pr_auc": pr_auc}


def find_best_threshold(y_true, probabilities):
    best_threshold, best_metrics, best_f1 = 0.5, None, -1.0

    for threshold in np.arange(0.05, 1.00, 0.01):
        metrics = calculate_metrics(y_true, probabilities, threshold)
        if metrics["f1"] > best_f1:
            best_f1, best_threshold, best_metrics = metrics["f1"], threshold, metrics

    return best_threshold, best_metrics


def main(args):
    set_seed(args.seed)  # ADDED -- must run before model init, right at the top of main()

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")  # ADDED

    processed = args.processed_dir
    client_file = "results/federated_clients.csv"

    if not os.path.exists(client_file):
        raise FileNotFoundError(
            f"Client file not found: {client_file}\nRun select_federated_clients.py first."
        )

    print("\nLoading transaction data...")
    edges = pd.read_parquet(f"{processed}/edges_full.parquet")
    edge_features = pd.read_parquet(f"{processed}/edge_features.parquet")
    node_bank = pd.read_parquet(f"{processed}/node_bank_lookup.parquet")
    clients_df = pd.read_csv(client_file)

    clients = clients_df["bank"].astype(int).tolist()
    num_nodes = int(node_bank["node_id"].max()) + 1

    train_mask = edges["split"] == "train"
    val_mask = edges["split"] == "val"
    test_mask = edges["split"] == "test"

    train_edges = edges[train_mask].copy()
    val_edges = edges[val_mask].copy()
    test_edges = edges[test_mask].copy()

    print(f"Nodes: {num_nodes:,}")
    print(f"Transactions: {len(edges):,}")
    print(f"Training transactions: {len(train_edges):,}")
    print(f"Validation transactions: {len(val_edges):,}")
    print(f"Test transactions: {len(test_edges):,}")
    print(f"\nFederated clients: {len(clients)}")
    print("Clients:")
    print(clients)

    train_edge_index = edge_index_from_df(train_edges)
    val_edge_index = edge_index_from_df(val_edges)
    test_edge_index = edge_index_from_df(test_edges)

    train_edge_attr = torch.tensor(
        edge_features[train_mask].to_numpy(dtype=np.float32), dtype=torch.float
    )
    val_edge_attr = torch.tensor(
        edge_features[val_mask].to_numpy(dtype=np.float32), dtype=torch.float
    )
    test_edge_attr = torch.tensor(
        edge_features[test_mask].to_numpy(dtype=np.float32), dtype=torch.float
    )

    y_val = val_edges["is_laundering"].to_numpy(dtype=np.int64)
    y_test = test_edges["is_laundering"].to_numpy(dtype=np.int64)

    x = build_node_features(train_edges, num_nodes)

    x, normalized_edges = normalize_features(
        x, train_edge_attr, [train_edge_attr, val_edge_attr, test_edge_attr]
    )
    train_edge_attr, val_edge_attr, test_edge_attr = normalized_edges

    train_edges["from_bank"] = train_edges["from_bank"].astype(int)
    train_edges["to_bank"] = train_edges["to_bank"].astype(int)

    client_data = {}
    print("\nPreparing client datasets...")

    for bank in clients:
        local = train_edges[
            (train_edges["from_bank"] == bank) | (train_edges["to_bank"] == bank)
        ].copy()

        if len(local) == 0:
            print(f"WARNING: Bank {bank} has no training transactions.")
            continue

        local_positions = train_edges.index.get_indexer(local.index)
        local_edge_index = train_edge_index[:, local_positions]
        local_edge_attr = train_edge_attr[local_positions]

        local_labels = torch.tensor(
            local["is_laundering"].to_numpy(dtype=np.int64), dtype=torch.long
        )

        client_data[bank] = {
            "edge_index": local_edge_index,
            "edge_attr": local_edge_attr,
            "labels": local_labels
        }

        print(f"  Bank {bank}: {len(local):,} local records | {int(local_labels.sum()):,} positives")

    clients = [bank for bank in clients if bank in client_data]
    if len(clients) == 0:
        raise RuntimeError("No valid federated clients found.")

    model = FederatedGNN(
        node_feat_dim=x.size(1),
        edge_feat_dim=train_edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim
    ).to(device)

    for round_number in range(1, args.rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"FEDERATED ROUND {round_number}/{args.rounds}")
        print(f"{'=' * 60}")

        states, sizes = [], []

        for bank in clients:
            data = client_data[bank]
            state, size, n_pos = train_local(
                model, x, data["edge_index"], data["edge_attr"], data["labels"], args, device
            )
            states.append({key: value.cpu() for key, value in state.items()})
            sizes.append(size)
            print(f"  Bank {bank}: {size:,} records trained | {n_pos:,} positives")

        global_state = fedavg(states, sizes)
        model.load_state_dict(global_state)

        val_probabilities = predict(
            model, x, train_edge_index, val_edge_index, val_edge_attr, device
        )
        val_metrics = calculate_metrics(y_val, val_probabilities, 0.5)

        print("\nValidation:")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {val_metrics['auc']:.4f}")
        print(f"  PR-AUC: {val_metrics['pr_auc']:.4f}")

    threshold, best_val_metrics = find_best_threshold(y_val, val_probabilities)

    print("\n=== Validation Threshold Selection ===")
    print(f"Best threshold: {threshold:.2f}")
    print(f"Validation Precision: {best_val_metrics['precision']:.4f}")
    print(f"Validation Recall: {best_val_metrics['recall']:.4f}")
    print(f"Validation F1: {best_val_metrics['f1']:.4f}")
    print(f"Validation ROC-AUC: {best_val_metrics['auc']:.4f}")
    print(f"Validation PR-AUC: {best_val_metrics['pr_auc']:.4f}")

    test_probabilities = predict(
        model, x, train_edge_index, test_edge_index, test_edge_attr, device
    )
    test_metrics = calculate_metrics(y_test, test_probabilities, threshold)

    print("\n=== FINAL FEDERATED TEST RESULTS ===")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['auc']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Threshold: {threshold:.2f}")

    # --- Alert-budget metrics (fixed review-capacity framing) ---
    alert_results = alert_budget_metrics(y_test, test_probabilities)
    print(format_alert_budget_metrics(alert_results, title="Federated GraphSAGE"))

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print("\nGlobal model saved to:")
        print(args.save_model)

    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--save_model", default="results/metrics/federated_gnn.pt")
    parser.add_argument("--seed", type=int, default=42)  # ADDED

    args = parser.parse_args()
    main(args)