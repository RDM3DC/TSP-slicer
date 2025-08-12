"""
ARP + Adaptive Pi Geometry Simulator
------------------------------------
This script evolves curvature K(x,t) on a πₐ manifold using
the Adaptive Resistance Principle (ARP) and computes TSP tours
over time-evolving geodesics.

Author: Ryan McKenna
"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_pi_tsp import CurvatureField, Obstacles, tsp_solve, tsp_plot

class ARPAdaptivePiField:
    def __init__(self, alpha=1.0, mu=0.05, r=0.1):
        self.alpha = alpha
        self.mu = mu
        self.r = r
        # Grid for curvature memory
        self.grid_res = 64
        self.X, self.Y = np.meshgrid(
            np.linspace(0,1,self.grid_res),
            np.linspace(0,1,self.grid_res)
        )
        self.K_grid = CurvatureField.default().K(self.X, self.Y)
        self.M_grid = np.zeros_like(self.K_grid)
        self.obstacles = Obstacles.default()

    def step(self, points, dt=0.1):
        """Update curvature field via ARP dynamics."""
        # Compute geodesic path to deposit "current"
        order, _ = tsp_solve(points, r=self.r,
                             field=self.to_field(),
                             obstacles=self.obstacles)
        path_pts = np.array([points[i] for i in order] + [points[order[0]]])

        # Mark visited areas on grid
        for (p,q) in zip(path_pts[:-1], path_pts[1:]):
            xs = np.linspace(p[0], q[0], 50)
            ys = np.linspace(p[1], q[1], 50)
            for x,y in zip(xs, ys):
                ix = int(x*(self.grid_res-1))
                iy = int(y*(self.grid_res-1))
                # note: array rows correspond to y (iy), columns to x (ix)
                self.M_grid[iy, ix] += 1.0 * dt

        # ARP PDE update: dK/dt = alpha * |I| - mu * K
        I = self.M_grid
        self.K_grid += dt*(self.alpha*np.abs(I) - self.mu*self.K_grid)

        # Decay memory slightly each step
        self.M_grid *= (1.0 - 0.01)

    def to_field(self):
        """Convert current K_grid to CurvatureField object."""
        # Sample grid back to gaussian-like bumps
        terms = []
        for _ in range(3):
            mx = np.random.rand()
            my = np.random.rand()
            sx = 0.1 + 0.1*np.random.rand()
            sy = 0.1 + 0.1*np.random.rand()
            cx = int(mx*(self.grid_res-1))
            cy = int(my*(self.grid_res-1))
            A = self.K_grid[cx, cy]
            terms.append((A, mx, my, sx, sy))
        return CurvatureField(terms)

def _sample_points_outside_obstacles(n: int, obstacles: Obstacles, seed: int):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.random(2)
        if not obstacles.contains(p[0], p[1]):
            pts.append(p)
    return np.array(pts)

def run_sim(alpha=1.0, mu=0.05, steps=20, r=0.1, seed=0):
    np.random.seed(seed)
    sim = ARPAdaptivePiField(alpha=alpha, mu=mu, r=r)
    # ensure points are outside obstacles
    points = _sample_points_outside_obstacles(12, sim.obstacles, seed)

    for t in range(steps):
        sim.step(points)
        if (t+1) % 5 == 0:
            print(f"[Step {t+1}] Running TSP solve...")
            order, cost = tsp_solve(points, r=r,
                                    field=sim.to_field(),
                                    obstacles=sim.obstacles)
            print(f"Cost at step {t+1}: {cost:.4f}")
            tsp_plot(points, order, field=sim.to_field(),
                     obstacles=sim.obstacles,
                     r=r,
                     title=f"ARP-πₐ Step {t+1} (α={alpha}, μ={mu})")

if __name__ == "__main__":
    run_sim(alpha=2.0, mu=0.1, steps=15, r=0.1)
