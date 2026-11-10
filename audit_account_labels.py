from pathlib import Path
import pandas as pd


TRANS_FILE = Path("data/raw/HI-Small_Trans.csv")
PATTERN_FILE = Path("data/processed/patterns_parsed.csv")


def account_id(bank, account):
    return (
        bank.astype(str).str.strip()
        + "::"
        + account.astype(str).str.strip()
    )


def main():

    print("=" * 70)
    print("ACCOUNT LABEL STRATEGY AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # L1: Accounts involved in laundering transactions
    # ---------------------------------------------------------

    laundering_accounts = set()

    chunks = []

    for chunk in pd.read_csv(
        TRANS_FILE,
        chunksize=250_000
    ):

        laundering = (
            chunk["Is Laundering"]
            .astype(str)
            .str.strip()
            == "1"
        )

        x = chunk[laundering]

        if len(x) == 0:
            continue

        src = account_id(
            x["From Bank"],
            x["Account"]
        )

        dst = account_id(
            x["To Bank"],
            x["Account.1"]
        )

        laundering_accounts.update(src)
        laundering_accounts.update(dst)

    # ---------------------------------------------------------
    # L2: Accounts involved in laundering patterns
    # ---------------------------------------------------------

    patterns = pd.read_csv(
        PATTERN_FILE
    )

    pattern_accounts = set()

    pattern_senders = set()
    pattern_receivers = set()

    if "pattern_type" not in patterns.columns:

        raise ValueError(
            "pattern_type column not found in patterns_parsed.csv"
        )

    src = account_id(
        patterns["from_bank"],
        patterns["from_account"]
    )

    dst = account_id(
        patterns["to_bank"],
        patterns["to_account"]
    )

    pattern_senders.update(src)
    pattern_receivers.update(dst)

    pattern_accounts.update(src)
    pattern_accounts.update(dst)

    # ---------------------------------------------------------
    # L3: Pattern + role
    # ---------------------------------------------------------

    pattern_both_roles = (
        pattern_senders
        & pattern_receivers
    )

    # ---------------------------------------------------------
    # OVERLAP
    # ---------------------------------------------------------

    l1 = laundering_accounts
    l2 = pattern_accounts
    l3 = pattern_both_roles

    print("\nL1 — Laundering Transaction Involvement")
    print("-" * 70)

    print(
        f"Positive accounts: {len(l1):,}"
    )

    print("\nL2 — Laundering Pattern Involvement")
    print("-" * 70)

    print(
        f"Positive accounts: {len(l2):,}"
    )

    print("\nL3 — Pattern + Both Sender/Receiver Role")
    print("-" * 70)

    print(
        f"Positive accounts: {len(l3):,}"
    )

    # ---------------------------------------------------------
    # SET OVERLAP
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERLAP")
    print("=" * 70)

    print(
        f"L1 ∩ L2: {len(l1 & l2):,}"
    )

    print(
        f"L1 only:  {len(l1 - l2):,}"
    )

    print(
        f"L2 only:  {len(l2 - l1):,}"
    )

    print(
        f"L3 ∩ L1: {len(l3 & l1):,}"
    )

    # ---------------------------------------------------------
    # PATTERN TYPES
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATTERN TYPES")
    print("=" * 70)

    print(
        patterns["pattern_type"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # ROLE COUNTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATTERN ACCOUNT ROLES")
    print("=" * 70)

    print(
        f"Pattern senders:              "
        f"{len(pattern_senders):,}"
    )

    print(
        f"Pattern receivers:            "
        f"{len(pattern_receivers):,}"
    )

    print(
        f"Both sender and receiver:      "
        f"{len(pattern_both_roles):,}"
    )

    # ---------------------------------------------------------
    # CONCLUSION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        """
L1 = account participates in a labelled laundering transaction.

L2 = account participates in a labelled laundering pattern.

L3 = account participates in a laundering pattern as both
    sender and receiver.

These are dataset-derived suspicious-account proxies.
They are NOT official IBM mule-account ground-truth labels.

Do not modify the GNN until the most defensible label strategy
has been selected.
"""
    )


if __name__ == "__main__":
    main()