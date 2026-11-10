import os
import pandas as pd


DATA = "data/processed/edges_full.parquet"
OUTPUT = "results/federated_clients.csv"

NUM_CLIENTS = 10

MIN_TRANSACTIONS = 1000
MIN_LAUNDERING = 5

EXCLUDE_BANKS = {70}


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

    for column in [
        "transactions",
        "laundering",
        "accounts",
        "intra_bank",
        "cross_bank",
    ]:
        result[column] = result[column].astype(int)

    eligible = result[
        (result["transactions"] >= MIN_TRANSACTIONS)
        & (result["laundering"] >= MIN_LAUNDERING)
        & (~result["bank"].isin(EXCLUDE_BANKS))
    ].copy()

    print(
        f"\nEligible banks: {len(eligible):,}"
    )

    print(
        f"Criteria:"
        f"\n  Transactions >= {MIN_TRANSACTIONS:,}"
        f"\n  Laundering transactions >= {MIN_LAUNDERING}"
        f"\n  Excluded banks: {sorted(EXCLUDE_BANKS)}"
    )

    high = (
        eligible[
            eligible["transactions"]
            >= eligible["transactions"].quantile(0.75)
        ]
        .sort_values(
            ["laundering", "transactions"],
            ascending=False
        )
    )

    medium = (
        eligible[
            (eligible["transactions"]
             < eligible["transactions"].quantile(0.75))
            &
            (eligible["transactions"]
             >= eligible["transactions"].quantile(0.25))
        ]
        .sort_values(
            ["laundering", "transactions"],
            ascending=False
        )
    )

    laundering_active = (
        eligible
        .sort_values(
            ["laundering", "transactions"],
            ascending=False
        )
    )

    selected = []

    selected.extend(
        high.head(3)["bank"].tolist()
    )

    selected.extend(
        medium.head(3)["bank"].tolist()
    )

    selected.extend(
        laundering_active.head(4)["bank"].tolist()
    )

    selected = list(
        dict.fromkeys(selected)
    )

    if len(selected) < NUM_CLIENTS:

        remaining = eligible[
            ~eligible["bank"].isin(selected)
        ].sort_values(
            ["transactions", "laundering"],
            ascending=False
        )

        selected.extend(
            remaining.head(
                NUM_CLIENTS - len(selected)
            )["bank"].tolist()
        )

    selected = selected[:NUM_CLIENTS]

    clients = result[
        result["bank"].isin(selected)
    ].copy()

    clients["client_group"] = "selected"

    clients.loc[
        clients["bank"].isin(
            high.head(3)["bank"]
        ),
        "client_group"
    ] = "high_activity"

    clients.loc[
        clients["bank"].isin(
            medium.head(3)["bank"]
        ),
        "client_group"
    ] = "medium_activity"

    clients.loc[
        clients["bank"].isin(
            laundering_active.head(4)["bank"]
        ),
        "client_group"
    ] = "laundering_active"

    clients = clients[
        [
            "bank",
            "transactions",
            "laundering",
            "accounts",
            "intra_bank",
            "cross_bank",
            "client_group",
        ]
    ].sort_values(
        "transactions",
        ascending=False
    )

    print()
    print("=" * 70)
    print("SELECTED FEDERATED CLIENTS")
    print("=" * 70)

    print(
        clients.to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected clients: {len(clients)}"
    )

    print(
        f"Total client transactions: "
        f"{clients['transactions'].sum():,}"
    )

    print(
        f"Total client laundering: "
        f"{clients['laundering'].sum():,}"
    )

    print(
        f"Client laundering rate: "
        f"{100 * clients['laundering'].sum() / clients['transactions'].sum():.4f}%"
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    clients.to_csv(
        OUTPUT,
        index=False
    )

    print(
        f"\nSaved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()