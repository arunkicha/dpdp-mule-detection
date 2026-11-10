import pandas as pd

DATA = "data/processed/edges_full.parquet"


def main():

    print("Loading transaction data...", flush=True)

    df = pd.read_parquet(DATA)

    print(
        f"Loaded {len(df):,} transactions.",
        flush=True
    )

    print("Calculating bank statistics...", flush=True)

    from_stats = (
        df.groupby("from_bank")
        .agg(
            sent_transactions=("from_bank", "size"),
            sent_laundering=("is_laundering", "sum"),
            sent_accounts=("from_account", "nunique"),
        )
    )

    to_stats = (
        df.groupby("to_bank")
        .agg(
            received_transactions=("to_bank", "size"),
            received_laundering=("is_laundering", "sum"),
            received_accounts=("to_account", "nunique"),
        )
    )

    intra = (
        df[df["from_bank"] == df["to_bank"]]
        .groupby("from_bank")
        .size()
        .rename("intra_bank")
    )

    cross_sent = (
        df[df["from_bank"] != df["to_bank"]]
        .groupby("from_bank")
        .size()
        .rename("cross_bank_sent")
    )

    cross_received = (
        df[df["from_bank"] != df["to_bank"]]
        .groupby("to_bank")
        .size()
        .rename("cross_bank_received")
    )

    result = (
        from_stats
        .join(to_stats, how="outer")
        .join(intra, how="outer")
        .join(cross_sent, how="outer")
        .join(cross_received, how="outer")
        .fillna(0)
    )

    result["transactions"] = (
        result["sent_transactions"]
        + result["received_transactions"]
        - result["intra_bank"]
    )

    result["laundering"] = (
        result["sent_laundering"]
        + result["received_laundering"]
    )

    result["accounts"] = (
        result["sent_accounts"]
        + result["received_accounts"]
    )

    result["cross_bank"] = (
        result["cross_bank_sent"]
        + result["cross_bank_received"]
    )

    result = result.reset_index()

    if "bank" not in result.columns:
        result = result.rename(
            columns={result.columns[0]: "bank"}
        )

    numeric_columns = [
        "transactions",
        "laundering",
        "accounts",
        "intra_bank",
        "cross_bank",
    ]

    for column in numeric_columns:
        result[column] = result[column].astype(int)

    print()
    print("=" * 70)
    print("FEDERATED BANK DISTRIBUTION AUDIT")
    print("=" * 70)

    print(
        f"\nBanks: {len(result):,}"
    )

    print(
        f"Total transactions: "
        f"{len(df):,}"
    )

    print(
        f"Cross-bank transactions: "
        f"{(df['from_bank'] != df['to_bank']).sum():,}"
    )

    print(
        f"Laundering transactions: "
        f"{int(df['is_laundering'].sum()):,}"
    )

    print(
        "\nBanks with laundering transactions:"
    )

    print(
        int((result["laundering"] > 0).sum())
    )

    print(
        "\nBanks with >=10 laundering transactions:"
    )

    print(
        int((result["laundering"] >= 10).sum())
    )

    print(
        "\nBanks with >=50 laundering transactions:"
    )

    print(
        int((result["laundering"] >= 50).sum())
    )

    print(
        "\nBanks with >=100 laundering transactions:"
    )

    print(
        int((result["laundering"] >= 100).sum())
    )

    print(
        "\nTransaction distribution per bank:"
    )

    print(
        result["transactions"]
        .describe()
        .to_string()
    )

    print(
        "\nLaundering distribution per bank:"
    )

    print(
        result["laundering"]
        .describe()
        .to_string()
    )

    print(
        "\nTop 20 banks by transaction volume:"
    )

    top_volume = (
        result
        .sort_values(
            "transactions",
            ascending=False
        )
        [
            [
                "bank",
                "transactions",
                "laundering",
                "intra_bank",
                "cross_bank",
                "accounts",
            ]
        ]
        .head(20)
    )

    print(
        top_volume.to_string(
            index=False
        )
    )

    print(
        "\nTop 20 banks by laundering transactions:"
    )

    top_laundering = (
        result
        .sort_values(
            "laundering",
            ascending=False
        )
        [
            [
                "bank",
                "transactions",
                "laundering",
                "intra_bank",
                "cross_bank",
                "accounts",
            ]
        ]
        .head(20)
    )

    print(
        top_laundering.to_string(
            index=False
        )
    )

    result.to_csv(
        "results/federated_bank_distribution.csv",
        index=False
    )

    print(
        "\nSaved to "
        "results/federated_bank_distribution.csv"
    )


if __name__ == "__main__":
    main()