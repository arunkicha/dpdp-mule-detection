import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import random
import time

import numpy as np
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
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
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
    def __init__(self, node_emb_dim: int, edge_feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = 2 * node_emb_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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


class AMLBaselineGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, emb_dim=32):
        super().__init__()
        self.encoder = GraphSAGEEncoder(node_feat_dim, hidden_dim, emb_dim)
        self.classifier = EdgeClassifier(emb_dim, edge_feat_dim, hidden_dim)

    def forward(self, x, message_edge_index, classify_edge_index, edge_attr):
        node_emb = self.encoder(x, message_edge_index)
        return self.classifier(node_emb, classify_edge_index, edge_attr)


def get_predictions(model, data, mask, message_edge_index):
    model.eval()

    with torch.no_grad():
        classify_edge_index = data.edge_index[:, mask]
        classify_edge_attr = data.edge_attr[mask]

        logits = model(
            data.x,
            message_edge_index,
            classify_edge_index,
            classify_edge_attr
        )

        probs = F.softmax(logits, dim=1)[:, 1]

        y_true = data.y[mask].cpu().numpy()
        y_prob = probs.cpu().numpy()

    return y_true, y_prob


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "pr_auc": pr_auc
    }


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0

    thresholds = [i / 100 for i in range(1, 100)]

    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_prob, threshold)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Loading {args.data}...")

    data = torch.load(args.data, weights_only=False)

    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")

    # REVERTED: no add_reverse_edges() -- plain directed message passing only
    train_edge_index = data.edge_index[:, data.train_mask]

    print(f"Training message-passing edges: {train_edge_index.size(1):,}")

    x_mean = data.x.mean(dim=0)
    x_std = data.x.std(dim=0).clamp(min=1e-6)
    data.x = (data.x - x_mean) / x_std

    edge_mean = data.edge_attr[data.train_mask].mean(dim=0)
    edge_std = data.edge_attr[data.train_mask].std(dim=0).clamp(min=1e-6)
    data.edge_attr = (data.edge_attr - edge_mean) / edge_std

    data = data.to(device)

    train_edge_index = data.edge_index[:, data.train_mask]

    model = AMLBaselineGNN(
        node_feat_dim=data.x.size(1),
        edge_feat_dim=data.edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    n_pos = int(data.y[data.train_mask].sum())
    n_neg = int(data.train_mask.sum()) - n_pos

    # REVERTED: plain weighted cross-entropy instead of focal loss.
    # Inverse-frequency class weighting so the rare positive class isn't
    # drowned out, without focal loss's extra alpha/gamma hyperparameters.
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)

    print(
        f"Train class balance -- pos: {n_pos:,}, "
        f"neg: {n_neg:,} "
        f"(weighted cross-entropy, pos_weight={pos_weight:.2f})"
    )

    best_val_pr_auc = -1.0
    best_state = None
    # Tracking "best" by PR-AUC (threshold-independent) rather than F1@0.5 --
    # kept from the focal-loss debugging even though focal loss itself
    # was reverted, since PR-AUC tracking is strictly more robust either way.

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_edge_attr = data.edge_attr[data.train_mask]

        logits = model(
            data.x,
            train_edge_index,
            data.edge_index[:, data.train_mask],
            train_edge_attr
        )

        loss = F.cross_entropy(
            logits,
            data.y[data.train_mask],
            weight=class_weights
        )

        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            y_val, val_prob = get_predictions(
                model,
                data,
                data.val_mask,
                train_edge_index
            )

            val_metrics = calculate_metrics(
                y_val,
                val_prob,
                threshold=0.5
            )

            print(
                f"Epoch {epoch:3d} | "
                f"loss {loss.item():.4f} | "
                f"val P {val_metrics['precision']:.3f} "
                f"R {val_metrics['recall']:.3f} "
                f"F1 {val_metrics['f1']:.3f} "
                f"AUC {val_metrics['auc']:.3f} "
                f"PR-AUC {val_metrics['pr_auc']:.3f}"
            )

            if val_metrics["pr_auc"] > best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    y_val, val_prob = get_predictions(
        model,
        data,
        data.val_mask,
        train_edge_index
    )

    best_threshold, threshold_metrics = find_best_threshold(
        y_val,
        val_prob
    )

    print("\n=== Validation Threshold Selection ===")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation Precision: {threshold_metrics['precision']:.4f}")
    print(f"Validation Recall: {threshold_metrics['recall']:.4f}")
    print(f"Validation F1: {threshold_metrics['f1']:.4f}")
    print(f"Validation ROC-AUC: {threshold_metrics['auc']:.4f}")
    print(f"Validation PR-AUC: {threshold_metrics['pr_auc']:.4f}")

    y_test, test_prob = get_predictions(
        model,
        data,
        data.test_mask,
        train_edge_index
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold=best_threshold
    )

    print("\n=== Final Test Metrics (validation-selected threshold) ===")

    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"  threshold: {best_threshold:.2f}")

    # --- Alert-budget metrics (fixed review-capacity framing) ---
    alert_results = alert_budget_metrics(y_test, test_prob)
    print(format_alert_budget_metrics(alert_results, title="Centralized GNN Baseline"))

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"\nModel saved to {args.save_model}")

    return test_metrics, alert_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/graph_data.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--save_model", default="results/baseline_gnn.pt")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    start = time.time()
    train(args)

    print(f"\nTotal time: {time.time() - start:.1f}s")