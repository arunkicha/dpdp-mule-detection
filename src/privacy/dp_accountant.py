"""
Rényi Differential Privacy (RDP) Accountant
===============================================

Computes a formal (epsilon, delta)-DP guarantee for the DP-FedAvg
mechanism implemented in dp_mechanism.py.

WHAT THIS MECHANISM ACTUALLY IS (read before using):
  Each round, every one of the N=10 selected clients' full model update
  is clipped to L2 norm <= max_norm (dp_mechanism.clip_update). The
  clipped updates are weighted-averaged (weighted_average_updates), which
  by convexity keeps the aggregate's L2 norm <= max_norm. Gaussian noise
  with std = noise_multiplier * max_norm is then added ONCE to that
  aggregate (add_aggregated_gaussian_noise). ALL clients participate
  EVERY round -- there is no random subsampling of clients per round.

  This is therefore a CLIENT-LEVEL DP mechanism (McMahan et al., 2017,
  "Learning Differentially Private Recurrent Language Models"): the
  privacy unit being protected is one client's full local update within
  a single round. Sensitivity = max_norm, by construction of the
  clip-then-average step.

  Because there is no subsampling, there is no "privacy amplification by
  subsampling" to invoke (unlike standard DP-SGD accountants, e.g.
  Opacus's default, which assume Poisson-subsampled minibatches -- that
  assumption does NOT hold here and using it would silently understate
  epsilon). Composition across rounds is therefore plain RDP composition
  of R independent applications of the same Gaussian mechanism.

METHOD:
  1. RDP of a single Gaussian mechanism application at order alpha, with
     noise calibrated as std = noise_multiplier * sensitivity, is the
     standard closed form (Mironov, 2017, "Renyi Differential Privacy"):
         RDP(alpha) = alpha / (2 * noise_multiplier^2)
  2. RDP composes additively across R independent rounds:
         RDP_total(alpha) = R * alpha / (2 * noise_multiplier^2)
  3. Convert RDP to (epsilon, delta)-DP (Mironov, 2017, Prop. 3):
         epsilon = min_alpha [ RDP_total(alpha) + log(1/delta) / (alpha - 1) ]

This gives a correct, if not maximally tight, epsilon for exactly the
mechanism as implemented -- not an estimate borrowed from a different
setting.
"""

import math

# Standard RDP order grid (same range used by TensorFlow Privacy / Opacus
# for their own accountants) -- fine enough to get a near-tight bound.
DEFAULT_ORDERS = (
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(11, 64))
    + [128, 256, 512]
)


def rdp_gaussian(noise_multiplier: float, orders=DEFAULT_ORDERS):
    """
    RDP of ONE application of the Gaussian mechanism (sensitivity=1,
    i.e. noise_multiplier is std already normalized by sensitivity --
    exactly matching add_aggregated_gaussian_noise's noise_std = 
    noise_multiplier * max_norm convention).

    Returns: dict {alpha: rdp_value}
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0 for a valid DP guarantee.")

    return {alpha: alpha / (2.0 * noise_multiplier ** 2) for alpha in orders}


def compose_rounds(noise_multiplier: float, rounds: int, orders=DEFAULT_ORDERS):
    """
    RDP after composing `rounds` independent applications of the same
    Gaussian mechanism (full client participation each round -- no
    subsampling, so this is plain additive RDP composition).
    """
    per_round = rdp_gaussian(noise_multiplier, orders)
    return {alpha: rounds * val for alpha, val in per_round.items()}


def rdp_to_dp(rdp: dict, delta: float):
    """
    Convert an RDP curve {alpha: rdp_value} to a single (epsilon, delta)
    guarantee, minimizing over all available orders (Mironov, 2017).

    Returns: (best_epsilon, best_alpha)
    """
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1).")

    best_eps = float("inf")
    best_alpha = None

    for alpha, rdp_val in rdp.items():
        if alpha <= 1:
            continue
        eps = rdp_val + math.log(1.0 / delta) / (alpha - 1)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha

    return best_eps, best_alpha


def get_privacy_spent(noise_multiplier: float, rounds: int, delta: float,
                       orders=DEFAULT_ORDERS):
    """
    Full pipeline: composed RDP -> (epsilon, delta)-DP for this exact
    client-level DP-FedAvg mechanism (see module docstring for the
    precise assumptions this relies on).

    Returns: (epsilon, delta, best_alpha)
    """
    rdp = compose_rounds(noise_multiplier, rounds, orders)
    eps, best_alpha = rdp_to_dp(rdp, delta)
    return eps, delta, best_alpha


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute (epsilon, delta) for DP-FedAvg.")
    ap.add_argument("--noise_multiplier", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--delta", type=float, default=1e-5)
    args = ap.parse_args()

    eps, delta, best_alpha = get_privacy_spent(
        args.noise_multiplier, args.rounds, args.delta
    )

    print(f"Mechanism: client-level DP-FedAvg, full participation, no subsampling")
    print(f"Noise multiplier: {args.noise_multiplier}")
    print(f"Rounds: {args.rounds}")
    print(f"Delta: {args.delta:.2e}")
    print(f"==> Epsilon: {eps:.4f}  (achieved at RDP order alpha={best_alpha})")