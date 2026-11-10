import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score


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


class NodeClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, embeddings):
        return self.mlp(embeddings)


class MuleAccountGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            node_feat_dim,
            hidden_dim,
            embedding_dim
        )
        self.classifier = NodeClassifier(
            embedding_dim,
            hidden_dim
        )

    def forward(self, x, edge_index):
        embeddings = self.encoder(x, edge_index)
        return self.classifier(embeddings)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, probabilities)
    except ValueError:
        pr_auc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
        "pr_auc": pr_auc
    }


def get_predictions(model, data, edge_index, mask, labels):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, edge_index)
        probabilities = F.softmax(logits, dim=1)[:, 1]
        y_true = labels[mask].cpu().numpy()
        y_prob = probabilities[mask].cpu().numpy()

    return y_true, y_prob


def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0

    for i in range(1, 100):
        threshold = i / 100
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Loading {args.data}...")

    data = torch.load(
        args.data,
        weights_only=False
    )

    print(f"Nodes: {data.num_nodes:,}")
    print(f"Training edges: {data.train_edge_index.size(1):,}")
    print(f"Validation history edges: {data.val_edge_index.size(1):,}")
    print(f"Test history edges: {data.test_edge_index.size(1):,}")

    x = data.x.to(device)
    train_edge_index = data.train_edge_index.to(device)
    val_edge_index = data.val_edge_index.to(device)
    test_edge_index = data.test_edge_index.to(device)

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    y_train = data.y_train.to(device)
    y_val = data.y_val.to(device)
    y_test = data.y_test.to(device)

    model = MuleAccountGNN(
        node_feat_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.emb_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    n_pos = int(y_train[train_mask].sum())
    n_total = int(train_mask.sum())
    n_neg = n_total - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    class_weights = torch.tensor(
        [1.0, pos_weight],
        dtype=torch.float,
        device=device
    )

    print(
        f"Train class balance -- pos: {n_pos:,}, "
        f"neg: {n_neg:,}, "
        f"pos_weight: {pos_weight:.2f}"
    )

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(
            x,
            train_edge_index
        )

        loss = F.cross_entropy(
            logits[train_mask],
            y_train[train_mask],
            weight=class_weights
        )

        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            y_val_true, val_prob = get_predictions(
                model,
                data,
                val_edge_index,
                val_mask,
                y_val
            )

            val_metrics = calculate_metrics(
                y_val_true,
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

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    y_val_true, val_prob = get_predictions(
        model,
        data,
        val_edge_index,
        val_mask,
        y_val
    )

    best_threshold, val_threshold_metrics = find_best_threshold(
        y_val_true,
        val_prob
    )

    print("\n=== Validation Threshold Selection ===")
    print(f"Best threshold: {best_threshold:.2f}")
    print(
        f"Validation Precision: "
        f"{val_threshold_metrics['precision']:.4f}"
    )
    print(
        f"Validation Recall: "
        f"{val_threshold_metrics['recall']:.4f}"
    )
    print(
        f"Validation F1: "
        f"{val_threshold_metrics['f1']:.4f}"
    )
    print(
        f"Validation ROC-AUC: "
        f"{val_threshold_metrics['auc']:.4f}"
    )
    print(
        f"Validation PR-AUC: "
        f"{val_threshold_metrics['pr_auc']:.4f}"
    )

    y_test_true, test_prob = get_predictions(
        model,
        data,
        test_edge_index,
        test_mask,
        y_test
    )

    test_metrics = calculate_metrics(
        y_test_true,
        test_prob,
        threshold=best_threshold
    )

    print("\n=== Final Test Metrics (validation-selected threshold) ===")

    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"  threshold: {best_threshold:.2f}")

    if args.save_model:
        torch.save(
            model.state_dict(),
            args.save_model
        )
        print(f"\nModel saved to {args.save_model}")

    return test_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/graph_node_data_v2.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--save_model", default="results/metrics/mule_node_gnn.pt")

    args = ap.parse_args()

    start = time.time()
    train(args)

    print(f"\nTotal time: {time.time() - start:.1f}s")