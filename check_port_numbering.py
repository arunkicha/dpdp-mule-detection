"""
Checks whether the edge_features.parquet / graph_data.pt currently on disk
include the `out_port` feature (Egressy et al. port numbering), and whether
that matches the "reverted" narrative in the handoff report.

Usage:
    python check_port_numbering.py
"""

from pathlib import Path
import pandas as pd
import torch

PROCESSED_DIR = Path("data/processed")


def main():
    print("=" * 70)
    print("PORT-NUMBERING FEATURE AUDIT")
    print("=" * 70)

    edge_feat_path = PROCESSED_DIR / "edge_features.parquet"
    graph_data_path = PROCESSED_DIR / "graph_data.pt"

    # --- 1. Check edge_features.parquet directly ---
    if edge_feat_path.exists():
        edge_features = pd.read_parquet(edge_feat_path)

        print(f"\n{edge_feat_path} columns:")
        print(list(edge_features.columns))

        has_port = "out_port" in edge_features.columns

        print(
            f"\n'out_port' column present in edge_features.parquet: {has_port}"
        )

        if has_port:
            print("\nout_port summary stats:")
            print(edge_features["out_port"].describe().to_string())
            print(
                f"\nFraction of edges with out_port > 0 "
                f"(i.e. NOT the first transaction between that src/dst pair): "
                f"{(edge_features['out_port'] > 0).mean():.4f}"
            )
    else:
        print(f"\n{edge_feat_path} not found.")

    # --- 2. Check graph_data.pt (what gnn_baseline.py / strong_account_baseline.py /
    #     tabular_baseline.py actually load and train on for the frozen table) ---
    print("\n" + "=" * 70)
    print(f"CHECKING {graph_data_path} (what the frozen Section 4 models used)")
    print("=" * 70)

    if graph_data_path.exists():
        data = torch.load(graph_data_path, weights_only=False)

        print(f"\ndata.edge_attr shape: {tuple(data.edge_attr.shape)}")
        print(f"Number of edge features: {data.edge_attr.size(1)}")

        # edge_features.parquet columns, if available, tell us which index
        # corresponds to out_port -- otherwise we can only report the count.
        if edge_feat_path.exists() and "out_port" in edge_features.columns:
            col_idx = list(edge_features.columns).index("out_port")
            print(
                f"\n'out_port' is column index {col_idx} in edge_features.parquet."
            )
            if col_idx < data.edge_attr.size(1):
                port_column_in_tensor = data.edge_attr[:, col_idx]
                print(
                    "Corresponding column in data.edge_attr -- summary stats "
                    "(post train-normalization, so won't match raw out_port, "
                    "but nonzero variance confirms the column is real signal, "
                    "not an artifact of column misalignment):"
                )
                print(
                    f"  mean={port_column_in_tensor.mean().item():.4f}  "
                    f"std={port_column_in_tensor.std().item():.4f}  "
                    f"min={port_column_in_tensor.min().item():.4f}  "
                    f"max={port_column_in_tensor.max().item():.4f}"
                )
    else:
        print(f"\n{graph_data_path} not found.")

    # --- 3. File modification times, to sanity check which was built first ---
    print("\n" + "=" * 70)
    print("FILE TIMESTAMPS (to check build order)")
    print("=" * 70)

    for p in [edge_feat_path, graph_data_path]:
        if p.exists():
            import datetime
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
            print(f"{p}: last modified {mtime}")
        else:
            print(f"{p}: does not exist")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        """
If 'out_port' IS present in edge_features.parquet AND graph_data.pt was
built (or rebuilt) from it AFTER that column was added to build_graph.py,
then your frozen Section 4 baseline/GNN/DP results already include port
numbering as a base feature -- available to every model equally (LR, HGB,
GNN). In that case, the Section 5 claim that your F1 gap vs. Egressy et al.
is explained by "missing port numbering" needs to be corrected to just
"missing reverse-edge message passing and ego IDs".

If 'out_port' is present in the CURRENT edge_features.parquet but
graph_data.pt predates that column being added, the frozen numbers do NOT
include it, and you'd need to note the file timestamps as evidence for
that in a footnote, or rebuild graph_data.pt and re-verify the frozen
numbers before submitting.
"""
    )


if __name__ == "__main__":
    main()
