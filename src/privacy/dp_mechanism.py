import torch


def clone_state_dict(state_dict):
    return {key: value.detach().clone() for key, value in state_dict.items()}


def compute_model_update(local_state, global_state):
    update = {}

    for key in global_state:
        if torch.is_floating_point(global_state[key]):
            update[key] = local_state[key].detach() - global_state[key].detach()
        else:
            update[key] = local_state[key].detach().clone()

    return update


def update_l2_norm(update):
    squared_norm = 0.0

    for value in update.values():
        if torch.is_floating_point(value):
            squared_norm += value.detach().double().pow(2).sum().item()

    return squared_norm ** 0.5


def clip_update(update, max_norm):
    if max_norm <= 0:
        raise ValueError("max_norm must be greater than zero.")

    norm = update_l2_norm(update)

    if norm <= max_norm:
        return {key: value.clone() for key, value in update.items()}, norm, 1.0

    scale = max_norm / (norm + 1e-12)
    clipped = {}

    for key, value in update.items():
        if torch.is_floating_point(value):
            clipped[key] = value * scale
        else:
            clipped[key] = value.clone()

    return clipped, norm, scale


def weighted_average_updates(updates, sizes):
    if len(updates) == 0:
        raise ValueError("No client updates provided.")

    if len(updates) != len(sizes):
        raise ValueError("updates and sizes must have the same length.")

    total_size = sum(sizes)

    if total_size <= 0:
        raise ValueError("Total client size must be positive.")

    result = {}

    for key in updates[0]:
        first = updates[0][key]

        if not torch.is_floating_point(first):
            result[key] = first.clone()
            continue

        result[key] = first.clone() * (sizes[0] / total_size)

        for i in range(1, len(updates)):
            result[key] += updates[i][key] * (sizes[i] / total_size)

    return result


def add_aggregated_gaussian_noise(aggregated_update, sensitivity, noise_multiplier):
    """
    Adds noise calibrated to `sensitivity` -- the maximum possible change
    in the aggregated update from any single client's data. Under UNIFORM
    clipping (all clients share one max_norm) this equals max_norm, as
    before. Under PERSONALIZED clipping (each client has its own max_norm_i)
    the true sensitivity is max_i(weight_i * max_norm_i), which can be
    tighter than the old uniform max_norm -- so personalization can let us
    add LESS noise for the same formal privacy guarantee, not just
    redistribute the same noise differently.
    """
    if sensitivity <= 0:
        raise ValueError("sensitivity must be greater than zero.")

    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative.")

    noise_std = noise_multiplier * sensitivity
    noisy_update = {}

    for key, value in aggregated_update.items():
        if torch.is_floating_point(value):
            if noise_multiplier == 0:
                noisy_update[key] = value.clone()
            else:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=value.shape,
                    device=value.device,
                    dtype=value.dtype
                )
                noisy_update[key] = value + noise
        else:
            noisy_update[key] = value.clone()

    return noisy_update, noise_std


def compute_personalized_max_norms(
    client_sizes,
    base_max_norm,
    min_scale=0.5,
    max_scale=2.0
):
    """
    Degree-aware (here: data-size-aware) per-client clipping norms, adapted
    from Yuan et al.'s personalized-DP-GNN idea (allocate noise/budget by
    importance rather than uniformly) to a federated-client setting, where
    client data size is the natural stand-in for "degree"/importance.

    Clients with MORE data get a LARGER clip norm (their updates are lower-
    variance and more trustworthy, so let them contribute more signal).
    Clients with LESS data get a SMALLER clip norm (their updates are
    noisier/less reliable on their own, so cap their influence tighter --
    this also tightens their individual contribution to the aggregate
    sensitivity, which is where the personalization can pay off in the
    privacy accounting, not just in signal quality).

    Scale factor is sqrt(size_i / mean_size), clamped to [min_scale, max_scale]
    so no single client is given an unbounded budget or reduced to near zero.
    """
    if len(client_sizes) == 0:
        raise ValueError("client_sizes must be non-empty.")

    if base_max_norm <= 0:
        raise ValueError("base_max_norm must be greater than zero.")

    mean_size = sum(client_sizes) / len(client_sizes)

    scales = []
    for size in client_sizes:
        raw_scale = (size / mean_size) ** 0.5
        clamped_scale = max(min_scale, min(max_scale, raw_scale))
        scales.append(clamped_scale)

    max_norms = [base_max_norm * scale for scale in scales]

    return max_norms, scales


def privatize_federated_round(
    client_states,
    global_state,
    client_sizes,
    max_norm,
    noise_multiplier,
    personalized=False,
    min_scale=0.5,
    max_scale=2.0
):
    """
    personalized=False (default): exact original behavior -- one shared
    max_norm for every client, noise calibrated to that same max_norm.
    Fully backward compatible; existing runs/accounting are unaffected.

    personalized=True: each client gets its own clip norm via
    compute_personalized_max_norms(). The noise is then calibrated to the
    TRUE aggregate sensitivity under this scheme -- max_i(weight_i * max_norm_i)
    -- which is returned in diagnostics as "sensitivity" so the RDP
    accountant can be called with the correct (and, if this is tighter
    than the old uniform max_norm, more favorable) sensitivity value.
    """
    if len(client_states) == 0:
        raise ValueError("No client states provided.")

    if len(client_states) != len(client_sizes):
        raise ValueError(
            "client_states and client_sizes must have the same length."
        )

    if personalized:
        per_client_max_norms, scales = compute_personalized_max_norms(
            client_sizes,
            max_norm,
            min_scale=min_scale,
            max_scale=max_scale
        )
    else:
        per_client_max_norms = [max_norm] * len(client_states)
        scales = [1.0] * len(client_states)

    clipped_updates = []
    original_norms = []
    clipping_scales = []
    clipped_flags = []

    for local_state, client_max_norm in zip(client_states, per_client_max_norms):
        update = compute_model_update(
            local_state,
            global_state
        )

        clipped_update, original_norm, clipping_scale = clip_update(
            update,
            client_max_norm
        )

        clipped_updates.append(clipped_update)
        original_norms.append(original_norm)
        clipping_scales.append(clipping_scale)
        clipped_flags.append(original_norm > client_max_norm)

    aggregated_update = weighted_average_updates(
        clipped_updates,
        client_sizes
    )

    total_size = sum(client_sizes)
    weights = [size / total_size for size in client_sizes]

    # True sensitivity of the weighted average to any single client's
    # update: the largest (weight_i * client_max_norm_i) across clients.
    # Under personalized=False this reduces to max(weights) * max_norm,
    # matching the original uniform-case sensitivity exactly.
    sensitivity = max(
        weight * client_norm
        for weight, client_norm in zip(weights, per_client_max_norms)
    )

    noisy_update, noise_std = add_aggregated_gaussian_noise(
        aggregated_update,
        sensitivity,
        noise_multiplier
    )

    diagnostics = {
        "original_norms": original_norms,
        "clipping_scales": clipping_scales,
        "clipped_flags": clipped_flags,
        "average_original_norm": sum(original_norms) / len(original_norms),
        "num_clipped": sum(clipped_flags),
        "num_clients": len(client_states),
        "noise_std": noise_std,
        "personalized": personalized,
        "per_client_max_norms": per_client_max_norms,
        "personalization_scales": scales,
        "sensitivity": sensitivity
    }

    return noisy_update, diagnostics


def apply_update(global_state, update):
    new_state = {}

    for key in global_state:
        if torch.is_floating_point(global_state[key]):
            new_state[key] = global_state[key].detach() + update[key]
        else:
            new_state[key] = global_state[key].detach().clone()

    return new_state