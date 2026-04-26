from __future__ import annotations

"""Pauli exclusion helpers for DTES/RL-style state-space constraints."""


def pauli_valid(config):
    """
    config: iterable of particle positions, lattice sites, or occupation labels.
    Returns True iff all occupied one-particle states are distinct.
    """
    seen = set()
    for x in config:
        key = tuple(x) if hasattr(x, "__iter__") and not isinstance(x, str) else x
        if key in seen:
            return False
        seen.add(key)
    return True


def pauli_mask(configs):
    """
    configs: list of configurations.
    Returns boolean list: True for Pauli-valid states.
    """
    return [pauli_valid(c) for c in configs]


def pauli_penalty(config, epsilon=1e-3):
    """Debug-only soft Pauli relaxation; final models should use hard masking."""
    penalty = 0.0
    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            dist = abs(config[i] - config[j])
            penalty += 1.0 / (dist + epsilon)
    return penalty


def dtes_energy(state, base_energy):
    if not pauli_valid(state.particles):
        return float("inf")
    return base_energy(state)


def edge_weight(s, s_next, base_weight):
    if not pauli_valid(s_next.particles):
        return 0.0
    return base_weight


def valid_actions_pauli(env, state, actions):
    valid = []
    for action in actions:
        ns = env.transition_preview(state, action)
        valid.append(pauli_valid(ns.particles))
    return valid
