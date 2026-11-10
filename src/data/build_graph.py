import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",
        "To Bank": "to_bank",
        "Account.1": "to_account",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_patterns(path: str) -> pd.DataFrame:
    pattern_type_re = re.compile(r"LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+)")

    rows = []
    current_type = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("BEGIN"):
                m = pattern_type_re.search(line)
                current_type = m.group(1) if m else "UNKNOWN"
                continue

            if line.startswith("END"):
                current_type = None
                continue

            if current_type is not None:
                fields = line.split(",")

                if len(fields) >= 11:
                    rows.append({
                        "timestamp": fields[0],
                        "from_bank": fields[1],
                        "from_account": fields[2],
                        "to_bank": fields[3],
                        "to_account": fields[4],
                        "amount_received": fields[5],
                        "receiving_currency": fields[6],
                        "amount_paid": fields[7],
                        "payment_currency": fields[8],
                        "payment_format": fields[9],
                        "is_laundering": fields[10],
                        "pattern_type": current_type,
                    })

    pat_df = pd.DataFrame(rows)

    if not pat_df.empty:
        pat_df["timestamp"] = pd.to_datetime(pat_df["timestamp"])

    return pat_df


def build_node_edge_tables(df: pd.DataFrame):
    df = df.copy()

    df["from_node"] = (
        df["from_bank"].astype(str) + "_" +
        df["from_account"].astype(str)
    )

    df["to_node"] = (
        df["to_bank"].astype(str) + "_" +
        df["to_account"].astype(str)
    )

    all_nodes = pd.Index(
        pd.concat([df["from_node"], df["to_node"]]).unique()
    )

    node_id_map = {node: i for i, node in enumerate(all_nodes)}

    df["src"] = df["from_node"].map(node_id_map)
    df["dst"] = df["to_node"].map(node_id_map)

    node_bank = pd.concat([
        df[["from_node", "from_bank"]].rename(
            columns={"from_node": "node", "from_bank": "bank"}
        ),
        df[["to_node", "to_bank"]].rename(
            columns={"to_node": "node", "to_bank": "bank"}
        ),
    ]).drop_duplicates("node")

    node_bank["node_id"] = node_bank["node"].map(node_id_map)
    node_bank = node_bank.sort_values("node_id").reset_index(drop=True)

    return df, node_id_map, node_bank


def build_edge_features(df: pd.DataFrame):
    """
    NOTE: this must be called AFTER build_node_edge_tables, since it uses
    the "from_node"/"to_node" columns added there (df is reassigned to
    that returned df in main()).
    """
    feats = pd.DataFrame()

    feats["amount_paid"] = df["amount_paid"].astype(float)
    feats["amount_received"] = df["amount_received"].astype(float)
    feats["amount_diff"] = feats["amount_paid"] - feats["amount_received"]
    feats["same_currency"] = (
        df["payment_currency"] == df["receiving_currency"]
    ).astype(int)
    feats["same_bank"] = (df["from_bank"] == df["to_bank"]).astype(int)
    feats["hour"] = df["timestamp"].dt.hour
    feats["dayofweek"] = df["timestamp"].dt.dayofweek

    # --- PORT NUMBERING (Egressy et al., 2024, AAAI) ---
    # A directed multigraph can have many transactions between the same
    # (src, dst) account pair. Plain message-passing GNNs treat all of
    # them as interchangeable -- they cannot tell "the 1st transfer A->B"
    # from "the 5th". Egressy et al. show this is a key reason plain GNNs
    # miss laundering cycles/fan-out/gather-scatter patterns on this exact
    # dataset. Port numbering gives each edge a local sequence index among
    # same-pair edges (ordered by time), so the GNN can use that ordering
    # as a signal.
    df_sorted = df.sort_values("timestamp")
    port = df_sorted.groupby(["from_node", "to_node"]).cumcount()
    feats["out_port"] = port.reindex(df.index).astype(float)
    feats["out_port"] = np.log1p(feats["out_port"])  # a few pairs repeat a lot

    fmt_dummies = pd.get_dummies(df["payment_format"], prefix="fmt")

    feats = pd.concat([feats, fmt_dummies], axis=1)

    return feats


def temporal_split(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    split = np.array(["train"] * n, dtype=object)
    split[train_end:val_end] = "val"
    split[val_end:] = "test"

    df["split"] = split

    return df


def partition_by_bank(df: pd.DataFrame, node_bank: pd.DataFrame, n_clients: int = 8):
    bank_counts = df["from_bank"].value_counts()
    top_banks = set(bank_counts.head(n_clients).index)

    def bucket(b):
        if b in top_banks:
            return b
        return "OTHER"

    df = df.copy()
    df["client_bucket"] = df["from_bank"].apply(bucket)

    partitions = {
        b: g
        for b, g in df.groupby("client_bucket")
    }

    return partitions


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--trans", required=True)
    ap.add_argument("--patterns", required=False)
    ap.add_argument("--outdir", default="./processed")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading transactions...")

    df = load_transactions(args.trans)

    print(
        f"  {len(df):,} transactions | "
        f"{df['is_laundering'].sum():,} labeled laundering "
        f"({100 * df['is_laundering'].mean():.4f}%)"
    )

    if args.patterns:
        print("Loading laundering patterns...")

        pat_df = load_patterns(args.patterns)

        if not pat_df.empty:
            print(
                f"  {len(pat_df):,} pattern-tagged transactions across "
                f"{pat_df['pattern_type'].nunique()} typologies:"
            )

            print(pat_df["pattern_type"].value_counts().to_string())

            pat_df.to_csv(
                outdir / "patterns_parsed.csv",
                index=False
            )

    print("Building node/edge tables...")

    df, node_id_map, node_bank = build_node_edge_tables(df)

    print(f"  {len(node_id_map):,} unique accounts (nodes)")

    print("Building edge features (including port numbering)...")

    edge_feats = build_edge_features(df)

    print("Applying temporal 60/20/20 split...")

    df = temporal_split(df)

    print(df["split"].value_counts().to_string())

    print("Partitioning by bank for federated simulation...")

    partitions = partition_by_bank(
        df,
        node_bank,
        n_clients=8
    )

    print(
        f"  {len(partitions)} client buckets (banks): "
        f"{list(partitions.keys())[:10]}"
        f"{'...' if len(partitions) > 10 else ''}"
    )

    df.to_parquet(
        outdir / "edges_full.parquet",
        index=False
    )

    edge_feats.to_parquet(
        outdir / "edge_features.parquet",
        index=False
    )

    node_bank.to_parquet(
        outdir / "node_bank_lookup.parquet",
        index=False
    )

    for bank, g in partitions.items():
        safe_name = str(bank).replace("/", "_")

        g.to_parquet(
            outdir / f"partition_{safe_name}.parquet",
            index=False
        )

    print(f"\nDone. Artifacts written to: {outdir.resolve()}")

    print(
        "Next: load edges_full.parquet + "
        "edge_features.parquet into a PyTorch Geometric "
        "Data object."
    )


if __name__ == "__main__":
    main()