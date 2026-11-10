import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from src.metrics_alert_budget import alert_budget_metrics, format_alert_budget_metrics
from src.models.federated import (
    FederatedGNN,
    build_node_features,
    edge_index_from_df,
    normalize_features,
    train_local,
    predict,
)

from src.privacy.dp_mechanism import (
    privatize_federated_round,
    apply_update,
)
from src.privacy.dp_accountant import get_privacy_spent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = float("nan")

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )
    except ValueError:
        pr_auc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "pr_auc": pr_auc,
    }


def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0

    for threshold in np.arange(0.05, 1.00, 0.01):
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def main(args):
    set_seed(args.seed)

    start_time = time.time()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")

    processed_dir = args.processed_dir

    client_file = "results/federated_clients.csv"

    if not os.path.exists(client_file):
        raise FileNotFoundError(
            f"Missing {client_file}. "
            "Run select_federated_clients.py first."
        )

    print("\nLoading transaction data...")

    edges = pd.read_parquet(
        f"{processed_dir}/edges_full.parquet"
    )

    edge_features = pd.read_parquet(
        f"{processed_dir}/edge_features.parquet"
    )

    node_bank = pd.read_parquet(
        f"{processed_dir}/node_bank_lookup.parquet"
    )

    clients_df = pd.read_csv(client_file)

    clients = (
        clients_df["bank"]
        .astype(int)
        .tolist()
    )

    num_nodes = (
        int(node_bank["node_id"].max()) + 1
    )

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

    print(f"\nDP Federated clients: {len(clients)}")
    print("Clients:")
    print(clients)

    train_edge_index = edge_index_from_df(train_edges)
    val_edge_index = edge_index_from_df(val_edges)
    test_edge_index = edge_index_from_df(test_edges)

    train_edge_attr = torch.tensor(
        edge_features[train_mask].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )

    val_edge_attr = torch.tensor(
        edge_features[val_mask].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )

    test_edge_attr = torch.tensor(
        edge_features[test_mask].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )

    y_val = val_edges["is_laundering"].to_numpy(
        dtype=np.int64
    )

    y_test = test_edges["is_laundering"].to_numpy(
        dtype=np.int64
    )

    x = build_node_features(
        train_edges,
        num_nodes,
    )

    x, normalized_edges = normalize_features(
        x,
        train_edge_attr,
        [
            train_edge_attr,
            val_edge_attr,
            test_edge_attr,
        ],
    )

    (
        train_edge_attr,
        val_edge_attr,
        test_edge_attr,
    ) = normalized_edges

    train_edges["from_bank"] = (
        train_edges["from_bank"].astype(int)
    )

    train_edges["to_bank"] = (
        train_edges["to_bank"].astype(int)
    )

    client_data = {}

    print("\nPreparing client datasets...")

    for bank in clients:
        local = train_edges[
            (train_edges["from_bank"] == bank)
            | (train_edges["to_bank"] == bank)
        ].copy()

        if len(local) == 0:
            print(
                f"WARNING: Bank {bank} has no training data."
            )
            continue

        positions = train_edges.index.get_indexer(
            local.index
        )

        local_edge_index = train_edge_index[:, positions]
        local_edge_attr = train_edge_attr[positions]

        local_labels = torch.tensor(
            local["is_laundering"].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

        client_data[bank] = {
            "edge_index": local_edge_index,
            "edge_attr": local_edge_attr,
            "labels": local_labels,
        }

        print(
            f"  Bank {bank}: "
            f"{len(local):,} records | "
            f"{int(local_labels.sum()):,} positives"
        )

    clients = [
        bank for bank in clients
        if bank in client_data
    ]

    if not clients:
        raise RuntimeError(
            "No valid federated clients."
        )

    model = FederatedGNN(
        node_feat_dim=x.size(1),
        edge_feat_dim=train_edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
    ).to(device)

    print("\n" + "=" * 60)
    print("DP CONFIGURATION")
    print("=" * 60)

    print(f"Clipping norm:     {args.max_norm}")
    print(f"Noise multiplier:  {args.noise_multiplier}")
    print(f"Delta:             {args.delta}")
    print(f"Rounds:            {args.rounds}")
    print(f"Local epochs:      {args.local_epochs}")
    print(f"Personalized DP:   {args.personalized}")
    if args.personalized:
        print(f"  min_scale:       {args.min_scale}")
        print(f"  max_scale:       {args.max_scale}")
    print(
        "Privacy mechanism: "
        "per-client update clipping + "
        "weighted FedAvg + "
        "Gaussian noise on aggregated update"
    )

    if args.noise_multiplier == 0:
        print("\nWARNING:")
        print(
            "noise_multiplier=0 disables Gaussian noise."
        )
        print("This is a NON-PRIVATE sanity-check run.")

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    last_val_probabilities = None
    # Track sensitivity actually used each round, for the RDP accountant
    # call after training. Under uniform clipping this equals max(weights)
    # * max_norm every round (constant given fixed clients/sizes); under
    # personalized clipping it's max_i(weight_i * max_norm_i), also
    # constant across rounds since client_sizes don't change round to
    # round -- but we record it explicitly rather than assume, in case
    # that assumption ever stops holding (e.g. client dropout is added
    # later).
    last_sensitivity = None

    for round_number in range(1, args.rounds + 1):
        print("\n" + "=" * 60)
        print(
            f"DP-FEDERATED ROUND "
            f"{round_number}/{args.rounds}"
        )
        print("=" * 60)

        client_states = []
        client_sizes = []

        for bank in clients:
            data = client_data[bank]

            model.load_state_dict(global_state)

            local_state, size, n_pos = train_local(
                model,
                x,
                data["edge_index"],
                data["edge_attr"],
                data["labels"],
                args,
                device,
            )

            local_state = {
                key: value.detach().cpu()
                for key, value in local_state.items()
            }

            client_states.append(local_state)
            client_sizes.append(size)

            print(
                f"  Bank {bank}: "
                f"{size:,} records | "
                f"{n_pos:,} positives"
            )

        noisy_update, dp_diagnostics = (
            privatize_federated_round(
                client_states,
                global_state,
                client_sizes,
                args.max_norm,
                args.noise_multiplier,
                personalized=args.personalized,
                min_scale=args.min_scale,
                max_scale=args.max_scale,
            )
        )

        last_sensitivity = dp_diagnostics["sensitivity"]

        global_state = apply_update(
            global_state,
            noisy_update,
        )

        model.load_state_dict(global_state)

        last_val_probabilities = predict(
            model,
            x,
            train_edge_index,
            val_edge_index,
            val_edge_attr,
            device,
        )

        val_metrics = calculate_metrics(
            y_val,
            last_val_probabilities,
            0.5,
        )

        print("\nValidation:")
        print(
            f"  Precision: {val_metrics['precision']:.4f}"
        )
        print(
            f"  Recall: {val_metrics['recall']:.4f}"
        )
        print(
            f"  F1: {val_metrics['f1']:.4f}"
        )
        print(
            f"  ROC-AUC: {val_metrics['auc']:.4f}"
        )
        print(
            f"  PR-AUC: {val_metrics['pr_auc']:.4f}"
        )
        print(
            f"  Average update norm: "
            f"{dp_diagnostics['average_original_norm']:.4f}"
        )
        print(
            f"  Clients clipped: "
            f"{dp_diagnostics['num_clipped']}/"
            f"{dp_diagnostics['num_clients']}"
        )
        print(
            f"  Sensitivity (max_i weight_i*max_norm_i): "
            f"{dp_diagnostics['sensitivity']:.6f}"
        )
        print(
            f"  Noise std: "
            f"{dp_diagnostics['noise_std']:.6f}"
        )
        if args.personalized:
            print(
                f"  Per-client max_norms: "
                f"{[round(n, 4) for n in dp_diagnostics['per_client_max_norms']]}"
            )
            print(
                f"  Per-client scales: "
                f"{[round(s, 4) for s in dp_diagnostics['personalization_scales']]}"
            )

    if last_val_probabilities is None:
        raise RuntimeError(
            "No validation predictions produced."
        )

    threshold, val_best = find_best_threshold(
        y_val,
        last_val_probabilities,
    )

    print("\n=== Validation Threshold Selection ===")
    print(f"Best threshold: {threshold:.2f}")
    print(
        f"Validation Precision: "
        f"{val_best['precision']:.4f}"
    )
    print(
        f"Validation Recall: "
        f"{val_best['recall']:.4f}"
    )
    print(
        f"Validation F1: "
        f"{val_best['f1']:.4f}"
    )
    print(
        f"Validation ROC-AUC: "
        f"{val_best['auc']:.4f}"
    )
    print(
        f"Validation PR-AUC: "
        f"{val_best['pr_auc']:.4f}"
    )

    test_probabilities = predict(
        model,
        x,
        train_edge_index,
        test_edge_index,
        test_edge_attr,
        device,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\n=== FINAL DP-FEDERATED TEST RESULTS ===")
    print(
        f"Precision: {test_metrics['precision']:.4f}"
    )
    print(
        f"Recall: {test_metrics['recall']:.4f}"
    )
    print(
        f"F1: {test_metrics['f1']:.4f}"
    )
    print(
        f"ROC-AUC: {test_metrics['auc']:.4f}"
    )
    print(
        f"PR-AUC: {test_metrics['pr_auc']:.4f}"
    )
    print(f"Threshold: {threshold:.2f}")

    # --- Alert-budget metrics (fixed review-capacity framing) ---
    alert_title = (
        "DP-Federated (personalized)"
        if args.personalized
        else "DP-Federated (uniform)"
    )
    alert_results = alert_budget_metrics(y_test, test_probabilities)
    print(format_alert_budget_metrics(alert_results, title=alert_title))

    print("\n=== PRIVACY CONFIGURATION ===")
    print(f"Delta: {args.delta:.2e}")
    print(
        f"Noise multiplier: "
        f"{args.noise_multiplier:.4f}"
    )
    print(
        f"Clipping norm (base): "
        f"{args.max_norm:.4f}"
    )
    print(f"Rounds: {args.rounds}")
    print(f"Personalized: {args.personalized}")

    print("\nPrivacy mechanism:")
    print(
        "Per-client L2 clipping (uniform or personalized) followed by "
        "weighted FedAvg and Gaussian noise on "
        "the aggregated update."
    )

    if args.noise_multiplier > 0:
        epsilon, delta_used, best_alpha = get_privacy_spent(
            args.noise_multiplier, args.rounds, args.delta
        )
        print(f"\nFormal privacy guarantee (RDP accountant):")
        print(f"  (epsilon, delta) = ({epsilon:.4f}, {delta_used:.2e})")
        print(f"  achieved at RDP order alpha={best_alpha}")
        print(f"  Mechanism: client-level DP-FedAvg, {args.rounds} rounds, "
              f"full participation (no subsampling)")
        print(
            f"  NOTE: this epsilon formula depends only on noise_multiplier "
            f"and rounds, NOT on sensitivity/max_norm -- it is identical "
            f"for uniform and personalized clipping (noise_std = "
            f"noise_multiplier * sensitivity already normalizes this out). "
            f"Final sensitivity actually used this run: {last_sensitivity:.6f}"
        )
    else:
        print("\nnoise_multiplier=0: no privacy guarantee (non-private sanity check).")

    if args.save_model:
        save_dir = os.path.dirname(args.save_model)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "model_state_dict": global_state,
                "threshold": threshold,
                "dp_config": {
                    "noise_multiplier":
                        args.noise_multiplier,
                    "max_norm":
                        args.max_norm,
                    "delta":
                        args.delta,
                    "rounds":
                        args.rounds,
                    "local_epochs":
                        args.local_epochs,
                    "personalized":
                        args.personalized,
                    "min_scale":
                        args.min_scale,
                    "max_scale":
                        args.max_scale,
                    "sensitivity":
                        last_sensitivity,
                },
                "clients": clients,
                "validation_metrics": val_best,
                "test_metrics": test_metrics,
                "alert_budget_metrics": alert_results,
            },
            args.save_model,
        )

        print("\nGlobal DP model saved to:")
        print(args.save_model)

    print(
        f"\nTotal time: "
        f"{time.time() - start_time:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed_dir",
        default="data/processed",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--emb_dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--save_model",
        default="results/metrics/dp_federated_gnn.pt",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --- NEW: personalized/degree-aware DP flags ---
    parser.add_argument(
        "--personalized",
        action="store_true",
        help="Use degree-aware (client-size-aware) personalized clipping "
             "instead of one shared max_norm for every client.",
    )

    parser.add_argument(
        "--min_scale",
        type=float,
        default=0.5,
        help="Minimum per-client clip-norm scale factor under --personalized.",
    )

    parser.add_argument(
        "--max_scale",
        type=float,
        default=2.0,
        help="Maximum per-client clip-norm scale factor under --personalized.",
    )

    args = parser.parse_args()

    main(args)