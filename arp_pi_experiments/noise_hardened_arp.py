"""Noise-hardened ARP baseline harness.

This module implements a simple annealed-noise baseline for ARP-style
TSP solving.  It injects Langevin noise into the point coordinates and
applies an Ornstein–Uhlenbeck jitter during the search.  The default
parameters reflect the p=1.0 specification discussed in the project
notes.
"""

from __future__ import annotations

import argparse
import numpy as np
from tsp_opt import solve_tsp


def noise_hardened_arp(
    n: int = 1000,
    steps: int = 250,
    alpha: float = 2.0,
    mu: float = 0.1,
    sigma0: float = 0.1,
    p: float = 1.0,
    eta0: float = 0.01,
    p_drop: float = 0.02,
    p_dither: float = 0.05,
    seed: int = 0,
):
    """Run a basic NH-ARP search on ``n`` random points."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))

    # median distance used for noise scaling
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    d_med = float(np.median(dists))
    sigma0_scaled = sigma0 * d_med

    ou_state = np.zeros_like(coords)
    best_cost = np.inf
    best_order = list(range(n))

    for t in range(steps):
        frac = 1.0 - t / steps
        sigma_t = sigma0_scaled * (frac ** p)
        eta_t = eta0 * frac

        # OU jitter on point locations
        ou_state += -eta_t * ou_state + rng.normal(scale=eta_t, size=ou_state.shape)

        # Langevin noise + dropout/dither
        jitter = rng.normal(scale=sigma_t, size=coords.shape)
        drop_mask = rng.random(n) < p_drop
        jitter[drop_mask] += rng.normal(scale=sigma0_scaled, size=(drop_mask.sum(), 2))
        jitter += rng.normal(scale=sigma_t * p_dither, size=coords.shape)
        noisy_coords = coords + jitter + ou_state

        order = solve_tsp(list(map(tuple, noisy_coords)))
        path = noisy_coords[order]
        cost = float(np.linalg.norm(path - np.roll(path, -1, axis=0), axis=1).sum())
        if cost < best_cost:
            best_cost = cost
            best_order = order

    return best_order, best_cost


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NH-ARP baseline runner")
    p.add_argument("--n", type=int, default=1000, help="number of cities")
    p.add_argument("--steps", type=int, default=250, help="annealing steps")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    order, cost = noise_hardened_arp(n=args.n, steps=args.steps, seed=args.seed)
    print(f"NH-ARP tour cost: {cost:.3f}")
