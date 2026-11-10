from pathlib import Path
from collections import defaultdict
import pandas as pd


PATTERN_FILE = Path(
    "data/processed/patterns_parsed.csv"
)


def account_id(bank, account):
    return (
        str(bank).strip()
        + "_"
        + str(account).strip()
    )


def main():

    patterns = pd.read_csv(
        PATTERN_FILE
    )

    patterns["src"] = [
        account_id(b, a)
        for b, a in zip(
            patterns["from_bank"],
            patterns["from_account"]
        )
    ]

    patterns["dst"] = [
        account_id(b, a)
        for b, a in zip(
            patterns["to_bank"],
            patterns["to_account"]
        )
    ]

    print("=" * 70)
    print("MULE-ROLE / PATTERN STRUCTURE AUDIT")
    print("=" * 70)

    print(
        f"\nPattern records: {len(patterns):,}"
    )

    print(
        f"Unique patterns/types: "
        f"{patterns['pattern_type'].nunique()}"
    )

    # ---------------------------------------------------------
    # ACCOUNT ROLE STATISTICS
    # ---------------------------------------------------------

    accounts = set(
        patterns["src"]
    ) | set(
        patterns["dst"]
    )

    stats = defaultdict(
        lambda: {
            "sent": 0,
            "received": 0,
            "patterns": set(),
            "types": set(),
            "counterparties": set()
        }
    )

    for _, row in patterns.iterrows():

        src = row["src"]
        dst = row["dst"]
        ptype = row["pattern_type"]

        stats[src]["sent"] += 1
        stats[dst]["received"] += 1

        stats[src]["patterns"].add(
            ptype
        )

        stats[dst]["patterns"].add(
            ptype
        )

        stats[src]["types"].add(
            ptype
        )

        stats[dst]["types"].add(
            ptype
        )

        stats[src]["counterparties"].add(
            dst
        )

        stats[dst]["counterparties"].add(
            src
        )

    rows = []

    for account in accounts:

        s = stats[account]

        rows.append({
            "account": account,
            "sent": s["sent"],
            "received": s["received"],
            "total": (
                s["sent"]
                + s["received"]
            ),
            "pattern_types": len(
                s["types"]
            ),
            "counterparties": len(
                s["counterparties"]
            )
        })

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # ROLE GROUPS
    # ---------------------------------------------------------

    sender_only = (
        (df["sent"] > 0)
        & (df["received"] == 0)
    )

    receiver_only = (
        (df["received"] > 0)
        & (df["sent"] == 0)
    )

    intermediary = (
        (df["sent"] > 0)
        & (df["received"] > 0)
    )

    print("\n" + "=" * 70)
    print("ACCOUNT ROLES")
    print("=" * 70)

    print(
        f"Pattern accounts:                 "
        f"{len(df):,}"
    )

    print(
        f"Sender only:                       "
        f"{sender_only.sum():,}"
    )

    print(
        f"Receiver only:                     "
        f"{receiver_only.sum():,}"
    )

    print(
        f"Both sender + receiver:            "
        f"{intermediary.sum():,}"
    )

    # ---------------------------------------------------------
    # ACTIVITY DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATTERN ACTIVITY")
    print("=" * 70)

    print(
        f"Median pattern transactions/account: "
        f"{df['total'].median():.0f}"
    )

    print(
        f"Mean pattern transactions/account:   "
        f"{df['total'].mean():.2f}"
    )

    print(
        f"Maximum pattern transactions/account:"
        f" {df['total'].max():,}"
    )

    print(
        f"\nAccounts with >=2 pattern transactions: "
        f"{(df['total'] >= 2).sum():,}"
    )

    print(
        f"Accounts with >=3 pattern transactions: "
        f"{(df['total'] >= 3).sum():,}"
    )

    print(
        f"Accounts with >=5 pattern transactions: "
        f"{(df['total'] >= 5).sum():,}"
    )

    # ---------------------------------------------------------
    # INTERMEDIARY STRENGTH
    # ---------------------------------------------------------

    intermediary_df = df[
        intermediary
    ].copy()

    intermediary_df["flow_balance"] = (
        intermediary_df["sent"]
        + intermediary_df["received"]
    )

    print("\n" + "=" * 70)
    print("INTERMEDIARY ACCOUNTS")
    print("=" * 70)

    print(
        f"Intermediary accounts: "
        f"{len(intermediary_df):,}"
    )

    print(
        f"Intermediaries with >=2 total pattern txns: "
        f"{(intermediary_df['total'] >= 2).sum():,}"
    )

    print(
        f"Intermediaries with >=3 total pattern txns: "
        f"{(intermediary_df['total'] >= 3).sum():,}"
    )

    print(
        f"Intermediaries with >=5 total pattern txns: "
        f"{(intermediary_df['total'] >= 5).sum():,}"
    )

    # ---------------------------------------------------------
    # PATTERN TYPE BY ROLE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATTERN TYPE TRANSACTION COUNTS")
    print("=" * 70)

    print(
        patterns["pattern_type"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # TOP ACCOUNTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP PATTERN-PARTICIPATING ACCOUNTS")
    print("=" * 70)

    top = df.sort_values(
        ["total", "counterparties"],
        ascending=False
    ).head(20)

    print(
        top[
            [
                "account",
                "sent",
                "received",
                "total",
                "pattern_types",
                "counterparties"
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # PROPOSED CANDIDATE LABELS
    # ---------------------------------------------------------

    label_a = intermediary

    label_b = (
        intermediary
        & (df["total"] >= 2)
    )

    label_c = (
        intermediary
        & (df["total"] >= 3)
    )

    print("\n" + "=" * 70)
    print("CANDIDATE MULE PROXIES")
    print("=" * 70)

    print(
        f"A: Pattern intermediary:             "
        f"{label_a.sum():,}"
    )

    print(
        f"B: Intermediary + >=2 pattern txns:  "
        f"{label_b.sum():,}"
    )

    print(
        f"C: Intermediary + >=3 pattern txns:  "
        f"{label_c.sum():,}"
    )

    print("\n")
    print(
        "These are candidate proxies only."
    )
    print(
        "We will NOT select a final mule label "
        "until we inspect these distributions."
    )


if __name__ == "__main__":
    main()