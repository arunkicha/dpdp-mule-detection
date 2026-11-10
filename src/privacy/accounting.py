"""Privacy accounting helpers: cumulative epsilon across federated rounds.

The main pipeline creates a fresh Opacus PrivacyEngine for every local training
call, so the epsilon it reports covers a single round only. A record that stays
in the training set is used in every round, so its total privacy loss is the
composition of all rounds. These helpers compose the recorded (noise, sample
rate, steps) history to give the cumulative epsilon.
"""

from opacus.accountants import create_accountant

DEFAULT_ACCOUNTANT = "prv"


def merge_history(history):
    merged = []
    for sigma, q, steps in history:
        if merged and merged[-1][0] == sigma and abs(merged[-1][1] - q) < 1e-12:
            merged[-1] = (sigma, q, merged[-1][2] + steps)
        else:
            merged.append((sigma, q, steps))
    return merged


def epsilon_from_history(history, delta, accountant=DEFAULT_ACCOUNTANT):
    if not history:
        return None
    acc = create_accountant(accountant)
    acc.history = merge_history(history)
    return float(acc.get_epsilon(delta=delta))


def epsilon_for(noise_multiplier, sample_rate, steps, delta, accountant=DEFAULT_ACCOUNTANT):
    if noise_multiplier <= 0 or steps <= 0:
        return None
    return epsilon_from_history([(noise_multiplier, sample_rate, steps)], delta, accountant)
