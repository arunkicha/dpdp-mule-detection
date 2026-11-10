import pandas as pd

PATTERN_FILE = "data/processed/patterns_parsed.csv"
EDGE_FILE = "data/processed/edges_full.parquet"


def make_account_id(bank, account):
    return (
        str(bank).strip()
        + "_"
        + str(account).strip()
    )


def main():

    patterns = pd.read_csv(PATTERN_FILE)

    patterns["src"] = [
        make_account_id(b, a)
        for b, a in zip(
            patterns["from_bank"],
            patterns["from_account"]
        )
    ]

    patterns["dst"] = [
        make_account_id(b, a)
        for b, a in zip(
            patterns["to_bank"],
            patterns["to_account"]
        )
    ]

    stats = {}

    for _, row in patterns.iterrows():

        for account, role in [
            (row["src"], "sender"),
            (row["dst"], "receiver")
        ]:

            if account not in stats:
                stats[account] = {
                    "sent": 0,
                    "received": 0,
                    "transactions": 0,
                    "types": set()
                }

            stats[account]["transactions"] += 1
            stats[account]["types"].add(
                row["pattern_type"]
            )

            if role == "sender":
                stats[account]["sent"] += 1
            else:
                stats[account]["received"] += 1

    rows = []

    for account, s in stats.items():

        if (
            s["sent"] >= 1
            and s["received"] >= 1
            and s["transactions"] >= 3
        ):

            bank = account.split("_")[0]

            rows.append({
                "account": account,
                "bank": bank,
                "sent": s["sent"],
                "received": s["received"],
                "transactions": s["transactions"],
                "pattern_types": len(
                    s["types"]
                )
            })

    df = pd.DataFrame(rows)

    print("=" * 70)
    print("REPEATED INTERMEDIARY DISTRIBUTION")
    print("=" * 70)

    print(
        f"\nRepeated intermediary accounts: "
        f"{len(df):,}"
    )

    if len(df) == 0:
        print("\nNo candidate accounts found.")
        return

    print(
        "\nAccounts by bank:"
    )

    bank_counts = (
        df.groupby("bank")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        bank_counts.to_string()
    )

    print(
        f"\nNumber of banks containing "
        f"candidate accounts: "
        f"{df['bank'].nunique()}"
    )

    print(
        f"Average candidates per bank: "
        f"{df.groupby('bank').size().mean():.2f}"
    )

    print(
        "\nDistribution:"
    )

    print(
        f"  Maximum in one bank: "
        f"{bank_counts.max():,}"
    )

    print(
        f"  Minimum in one bank: "
        f"{bank_counts.min():,}"
    )

    print(
        "\nTop banks:"
    )

    print(
        bank_counts.head(20).to_string()
    )

    print(
        "\nPattern-type diversity:"
    )

    print(
        f"  Mean pattern types/account: "
        f"{df['pattern_types'].mean():.2f}"
    )

    print(
        f"  Accounts with >=2 pattern types: "
        f"{(df['pattern_types'] >= 2).sum():,}"
    )

    print(
        f"  Accounts with >=3 pattern types: "
        f"{(df['pattern_types'] >= 3).sum():,}"
    )

    print(
        "\nRepeated activity:"
    )

    print(
        f"  Mean pattern transactions: "
        f"{df['transactions'].mean():.2f}"
    )

    print(
        f"  Median pattern transactions: "
        f"{df['transactions'].median():.0f}"
    )

    print(
        f"  Maximum pattern transactions: "
        f"{df['transactions'].max():,}"
    )

    print(
        "\nTop candidate accounts:"
    )

    print(
        df.sort_values(
            [
                "transactions",
                "pattern_types"
            ],
            ascending=False
        )
        .head(20)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()