"""
Alert-budget evaluation metrics (Nayak & Kumar-style).

Standard F1/precision/recall at a probability threshold answers "how good
is the model at the operating point we happened to pick." An analyst team
doesn't pick a threshold -- they have a fixed REVIEW CAPACITY (they can
manually check N transactions/day) and want to know: if I only ever look
at my top-k% highest-risk transactions, how many of the real laundering
cases do I actually catch, and how much of my reviewers' time is wasted
on false positives?

This reframes the extremely-low F1 numbers (0.001-0.01, expected at
0.1% base rate) into a practically interpretable statement: "reviewing
the top 1% of flagged transactions catches X% of laundering activity,
at Y% precision."
"""

import numpy as np


def alert_budget_metrics(y_true, y_prob, fractions=(0.01, 0.05, 0.10)):
    """
    y_true: array-like of 0/1 labels
    y_prob: array-like of predicted probabilities (same order as y_true)
    fractions: review-capacity fractions to evaluate, e.g. 0.01 = top 1%

    Returns: dict {fraction: {"k", "precision", "recall", "flagged_positives",
                               "total_positives"}}
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    total_positives = int(y_true.sum())

    # Descending sort by predicted probability
    order = np.argsort(-y_prob)

    results = {}

    for frac in fractions:
        k = max(1, int(round(frac * n)))
        top_k_idx = order[:k]

        flagged_positives = int(y_true[top_k_idx].sum())

        precision = flagged_positives / k
        recall = (
            flagged_positives / total_positives
            if total_positives > 0
            else float("nan")
        )

        results[frac] = {
            "k": k,
            "precision": precision,
            "recall": recall,
            "flagged_positives": flagged_positives,
            "total_positives": total_positives,
        }

    return results


def format_alert_budget_metrics(results, title="Alert-budget metrics"):
    lines = [f"\n{title} (fixed review-capacity framing):"]

    for frac in sorted(results.keys()):
        m = results[frac]
        lines.append(
            f"  Top-{frac * 100:.0f}% "
            f"({m['k']:,} transactions reviewed): "
            f"Precision={m['precision']:.4f}  "
            f"Recall={m['recall']:.4f}  "
            f"({m['flagged_positives']}/{m['total_positives']} "
            f"laundering transactions caught)"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    rng = np.random.default_rng(0)
    n = 100_000
    y_true = (rng.random(n) < 0.001).astype(int)
    y_prob = y_true * 0.7 + rng.random(n) * 0.3  # imperfect but informative

    results = alert_budget_metrics(y_true, y_prob)
    print(format_alert_budget_metrics(results, title="Sanity check"))