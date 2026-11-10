from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
TRANS_FILE = RAW_DIR / "HI-Small_Trans.csv"
PATTERN_FILE = Path("data/processed/patterns_parsed.csv")


def make_account_id(bank, account):
    return (
        bank.astype(str).str.strip()
        + "::"
        + account.astype(str).str.strip()
    )


def main():

    print("=" * 70)
    print("IBM AML HI-SMALL ACCOUNT-LEVEL AUDIT")
    print("=" * 70)

    total_transactions = 0
    laundering_transactions = 0

    all_accounts = set()
    laundering_accounts = set()

    laundering_senders = set()
    laundering_receivers = set()

    laundering_sender_counts = {}
    laundering_receiver_counts = {}

    chunk_size = 250_000

    print("\nReading transaction data...")

    for chunk in pd.read_csv(
        TRANS_FILE,
        chunksize=chunk_size
    ):

        total_transactions += len(chunk)

        src = make_account_id(
            chunk["From Bank"],
            chunk["Account"]
        )

        dst = make_account_id(
            chunk["To Bank"],
            chunk["Account.1"]
        )

        all_accounts.update(src)
        all_accounts.update(dst)

        laundering = (
            chunk["Is Laundering"].astype(str).str.strip() == "1"
        )

        laundering_chunk = chunk[laundering]

        laundering_transactions += len(
            laundering_chunk
        )

        if len(laundering_chunk) > 0:

            lsrc = make_account_id(
                laundering_chunk["From Bank"],
                laundering_chunk["Account"]
            )

            ldst = make_account_id(
                laundering_chunk["To Bank"],
                laundering_chunk["Account.1"]
            )

            laundering_senders.update(lsrc)
            laundering_receivers.update(ldst)

            laundering_accounts.update(lsrc)
            laundering_accounts.update(ldst)

            for account in lsrc:
                laundering_sender_counts[account] = (
                    laundering_sender_counts.get(account, 0) + 1
                )

            for account in ldst:
                laundering_receiver_counts[account] = (
                    laundering_receiver_counts.get(account, 0) + 1
                )

    # -------------------------------------------------------------
    # BASIC STATISTICS
    # -------------------------------------------------------------

    sender_only = (
        laundering_senders
        - laundering_receivers
    )

    receiver_only = (
        laundering_receivers
        - laundering_senders
    )

    both = (
        laundering_senders
        & laundering_receivers
    )

    positive_rate = (
        len(laundering_accounts)
        / max(len(all_accounts), 1)
        * 100
    )

    transaction_rate = (
        laundering_transactions
        / max(total_transactions, 1)
        * 100
    )

    print("\n" + "=" * 70)
    print("TRANSACTION-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"Total transactions:              "
        f"{total_transactions:,}"
    )

    print(
        f"Laundering transactions:         "
        f"{laundering_transactions:,}"
    )

    print(
        f"Laundering transaction rate:     "
        f"{transaction_rate:.4f}%"
    )

    print("\n" + "=" * 70)
    print("ACCOUNT-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"Total unique accounts:            "
        f"{len(all_accounts):,}"
    )

    print(
        f"Accounts involved in laundering:  "
        f"{len(laundering_accounts):,}"
    )

    print(
        f"Account positive rate:             "
        f"{positive_rate:.4f}%"
    )

    print("\n" + "=" * 70)
    print("LAUNDERING ACCOUNT ROLES")
    print("=" * 70)

    print(
        f"Laundering senders:               "
        f"{len(laundering_senders):,}"
    )

    print(
        f"Laundering receivers:             "
        f"{len(laundering_receivers):,}"
    )

    print(
        f"Sender only:                       "
        f"{len(sender_only):,}"
    )

    print(
        f"Receiver only:                     "
        f"{len(receiver_only):,}"
    )

    print(
        f"Both sender and receiver:          "
        f"{len(both):,}"
    )

    # -------------------------------------------------------------
    # NUMBER OF LAUNDERING TRANSACTIONS PER ACCOUNT
    # -------------------------------------------------------------

    laundering_counts = {}

    for account, count in laundering_sender_counts.items():
        laundering_counts[account] = (
            laundering_counts.get(account, 0)
            + count
        )

    for account, count in laundering_receiver_counts.items():
        laundering_counts[account] = (
            laundering_counts.get(account, 0)
            + count
        )

    count_values = list(
        laundering_counts.values()
    )

    if count_values:

        counts = pd.Series(
            count_values
        )

        print("\n" + "=" * 70)
        print("LAUNDERING INVOLVEMENT PER ACCOUNT")
        print("=" * 70)

        print(
            f"Minimum:                          "
            f"{counts.min():.0f}"
        )

        print(
            f"Median:                            "
            f"{counts.median():.0f}"
        )

        print(
            f"Mean:                              "
            f"{counts.mean():.2f}"
        )

        print(
            f"Maximum:                           "
            f"{counts.max():.0f}"
        )

        print(
            f"Accounts with >=2 laundering txns: "
            f"{(counts >= 2).sum():,}"
        )

        print(
            f"Accounts with >=3 laundering txns: "
            f"{(counts >= 3).sum():,}"
        )

        print(
            f"Accounts with >=5 laundering txns: "
            f"{(counts >= 5).sum():,}"
        )

    # -------------------------------------------------------------
    # PATTERN DATA
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATTERN DATA")
    print("=" * 70)

    if PATTERN_FILE.exists():

        patterns = pd.read_csv(
            PATTERN_FILE
        )

        print(
            f"Pattern records:                  "
            f"{len(patterns):,}"
        )

        print(
            f"Pattern columns:                  "
            f"{list(patterns.columns)}"
        )

        if "pattern_type" in patterns.columns:

            print(
                "\nPattern types:"
            )

            print(
                patterns["pattern_type"]
                .value_counts()
                .to_string()
            )

        elif "pattern" in patterns.columns:

            print(
                "\nPattern types:"
            )

            print(
                patterns["pattern"]
                .value_counts()
                .to_string()
            )

        else:

            print(
                "\nNo recognised pattern_type column found."
            )

    else:

        print(
            f"{PATTERN_FILE} not found."
        )

    # -------------------------------------------------------------
    # PRELIMINARY CONCLUSION
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRELIMINARY ASSESSMENT")
    print("=" * 70)

    print(
        "\nThe account-level positive class is defined here only as "
        "an account involved in at least one transaction labelled "
        "as laundering."
    )

    print(
        "This is a DATASET-DERIVED SUSPICIOUS-ACCOUNT PROXY, "
        "not an official IBM 'mule_account' ground-truth label."
    )

    print(
        "\nDo NOT use this proxy as the final research label until "
        "we inspect the resulting class distribution and pattern "
        "participation."
    )


if __name__ == "__main__":
    main()