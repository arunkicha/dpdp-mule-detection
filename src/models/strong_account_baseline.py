"""
Edge-level HistGradientBoosting baseline (MAIN TASK)
========================================================

Same task/labels/splits as tabular_baseline.py and gnn_baseline.py:
predicts is_laundering PER TRANSACTION using
[src account features, dst account features, transaction features].

Usage:
    python strong_account_baseline.py --data data/processed/graph_data.pt
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from src.metrics_alert_budget import alert_budget_metrics, format_alert_budget_metrics


def build_edge_feature_matrix(data):
    src, dst = data.edge_index
    x_src = data.x[src].numpy()
    x_dst = data.x[dst].numpy()
    edge_attr = data.edge_attr.numpy()
    return np.concatenate([x_src, x_dst, edge_attr], axis=1)


def evaluate(model, X, y, mask, threshold=0.5):
    prob = model.predict_proba(X[mask])[:, 1]
    pred = (prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y[mask], pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y[mask], prob)
    pr_auc = average_precision_score(y[mask], prob)

    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc, "pr_auc": pr_auc}, prob


def find_best_threshold(y_true, prob):
    best_threshold, best_f1, best_metrics = 0.5, -1.0, None
    for t in np.arange(0.01, 1.00, 0.01):
        pred = (prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best_metrics


def main(args):
    print(f"Loading {args.data}...")
    data = torch.load(args.data, weights_only=False)

    print("Building per-transaction feature matrix "
          "[src_features || dst_features || edge_features]...")
    X = build_edge_feature_matrix(data)
    y = data.y.numpy()

    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    print(f"Transactions: {len(X):,} | Features per transaction: {X.shape[1]}")

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )

    print("\nTraining HistGradientBoosting...")
    model.fit(X[train_mask], y[train_mask])

    val_metrics, val_prob = evaluate(model, X, y, val_mask)
    print("\n=== Validation Metrics ===")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    best_threshold, best_val = find_best_threshold(y[val_mask], val_prob)
    print("\n=== Validation Threshold Selection ===")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation Precision: {best_val['precision']:.4f}")
    print(f"Validation Recall: {best_val['recall']:.4f}")
    print(f"Validation F1: {best_val['f1']:.4f}")

    test_metrics, test_prob = evaluate(model, X, y, test_mask, threshold=best_threshold)
    print("\n=== Final Test Metrics (edge-level, is_laundering) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  threshold: {best_threshold:.2f}")

    # --- Alert-budget metrics (fixed review-capacity framing) ---
    alert_results = alert_budget_metrics(y[test_mask], test_prob)
    print(format_alert_budget_metrics(alert_results, title="HistGradientBoosting Baseline"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/graph_data.pt")
    args = parser.parse_args()
    main(args)