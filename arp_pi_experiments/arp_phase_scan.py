"""
ARP-πₐ Phase Scan
-----------------
Sweeps over alpha/mu ratios to see where phase flips, hysteresis,
or threshold behaviors emerge in ARP-evolving πₐ manifolds.

Author: Ryan McKenna
"""

import numpy as np
from adaptive_pi_tsp import CurvatureField, Obstacles, tsp_solve

from arp_adaptive_pi_sim import ARPAdaptivePiField

def _sample_points_outside_obstacles(n: int, obstacles: Obstacles, seed: int):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.random(2)
        if not obstacles.contains(p[0], p[1]):
            pts.append(p)
    return np.array(pts)

def phase_scan(alphas, mus, steps=10, r=0.1, seed=0):
    np.random.seed(seed)
    results = []

    for alpha in alphas:
        for mu in mus:
            sim = ARPAdaptivePiField(alpha=alpha, mu=mu, r=r)
            points = _sample_points_outside_obstacles(10, sim.obstacles, seed)
            last_cost = None
            deltas = []
            for t in range(steps):
                sim.step(points)
                order, cost = tsp_solve(points, r=r,
                                        field=sim.to_field(),
                                        obstacles=sim.obstacles)
                if last_cost is not None:
                    deltas.append(cost - last_cost)
                last_cost = cost
            avg_delta = np.mean(deltas) if deltas else 0.0
            results.append({
                "alpha": alpha,
                "mu": mu,
                "final_cost": last_cost,
                "avg_delta": avg_delta
            })
            print(f"α={alpha}, μ={mu} => Final cost {last_cost:.4f}, Δavg={avg_delta:.4f}")
    return results

if __name__ == "__main__":
    alphas = [0.5, 1.0, 2.0]
    mus = [0.05, 0.1, 0.2]
    phase_scan(alphas, mus, steps=12, r=0.1, seed=42)
