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
from dataclasses import dataclass
import argparse
from typing import Tuple


# Noise parameters for ARPAdaptivePiField
@dataclass
class NoiseParams:
    sigma0: float = 0.1          # initial Langevin noise scale (relative to grid units)
    p_anneal: float = 1.0        # anneal exponent p
    eta0: float = 0.01           # initial OU diffusion strength
    theta: float = 0.2           # OU mean reversion rate
    p_drop: float = 0.02         # edge dropout probability during deposition
    dither_std: float = 0.01     # std for coordinate dither during deposition (in [0,1] units)
    c_mu: float = 0.3            # noise-aware hysteresis coefficient

# --- Utils ---
def _softmax_rows(G: np.ndarray) -> np.ndarray:
    Gmax = G.max(axis=1, keepdims=True)
    exp = np.exp(G - Gmax)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)

def _erdos_renyi_adj(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.random((n, n)) < density
    A = np.triu(A, 1)
    A = A | A.T
    np.fill_diagonal(A, False)
    return A

class ARPAdaptivePiField:
    def __init__(self, alpha=1.0, mu=0.05, r=0.1, grid_res=64, tmax=1000, seed: int | None = None, noise: NoiseParams | None = None):
        self.alpha = alpha
        self.mu = mu
        self.r = r
        self.grid_res = grid_res
        self.t = 0  # internal step counter
        self.tmax = max(int(tmax), 1)
        self.noise = noise if noise is not None else NoiseParams()
        # RNG
        self.rng = np.random.default_rng(seed)
        # Grid for curvature memory
        self.X, self.Y = np.meshgrid(
            np.linspace(0,1,self.grid_res),
            np.linspace(0,1,self.grid_res)
        )
        self.K_grid = CurvatureField.default().K(self.X, self.Y)
        self.M_grid = np.zeros_like(self.K_grid)
        self.obstacles = Obstacles.default()
        # OU target (mean curvature level)
        self.K_bar = np.zeros_like(self.K_grid)

    def step(self, points, dt=0.1):
        """Update curvature field via ARP dynamics with annealed noise and OU jitter."""
        # 1) Compute current schedules
        # Avoid division by zero if tmax==0
        frac = min(self.t / max(self.tmax, 1), 1.0)
        sigma_t = self.noise.sigma0 * (1.0 + self.t / max(self.tmax, 1)) ** (-self.noise.p_anneal)
        eta_t   = self.noise.eta0   * (1.0 + self.t / max(self.tmax, 1)) ** (-self.noise.p_anneal)
        mu_t    = self.mu * (1.0 + self.noise.c_mu * (sigma_t / max(self.noise.sigma0, 1e-12)))

        # 2) Geodesic path to deposit "current"
        order, _ = tsp_solve(points, r=self.r, field=self.to_field(), obstacles=self.obstacles)
        path_pts = np.array([points[i] for i in order] + [points[order[0]]])

        # 3) Mark visited areas on grid with dropout & dither
        for (p, q) in zip(path_pts[:-1], path_pts[1:]):
            # Edge dropout: skip a fraction of edges each step
            if self.rng.random() < self.noise.p_drop:
                continue
            xs = np.linspace(p[0], q[0], 50)
            ys = np.linspace(p[1], q[1], 50)
            for x, y in zip(xs, ys):
                # Coordinate dither (keeps corridors from overfitting exact lattice)
                if self.noise.dither_std > 0:
                    x = float(np.clip(x + self.rng.normal(0.0, self.noise.dither_std), 0.0, 1.0))
                    y = float(np.clip(y + self.rng.normal(0.0, self.noise.dither_std), 0.0, 1.0))
                ix = int(np.clip(x * (self.grid_res - 1), 0, self.grid_res - 1))
                iy = int(np.clip(y * (self.grid_res - 1), 0, self.grid_res - 1))
                # note: array rows correspond to y (iy), columns to x (ix)
                self.M_grid[iy, ix] += 1.0 * dt

        # 4) ARP PDE core update with noise-aware hysteresis
        I = self.M_grid
        # Deterministic part
        dK_det = self.alpha * np.abs(I) - mu_t * self.K_grid
        # (a) Langevin noise on K
        xi = self.rng.normal(0.0, 1.0, size=self.K_grid.shape)
        dK_langevin = sigma_t * xi
        # (b) OU jitter on K towards K_bar
        dW = self.rng.normal(0.0, 1.0, size=self.K_grid.shape)
        dK_ou = (self.noise.theta * (self.K_bar - self.K_grid) + eta_t * dW) * dt

        # Apply updates
        self.K_grid += dt * dK_det + dK_langevin + dK_ou

        # 5) Decay memory slightly each step (as before)
        self.M_grid *= (1.0 - 0.01)

        # 6) Increment internal time
        self.t += 1

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
            # Note: rows are y (cy), cols are x (cx)
            A = float(self.K_grid[cy, cx])
            terms.append((A, mx, my, sx, sy))
        return CurvatureField(terms)


# --- CMA Graph Coloring Solver ---
class ARPColoringSolver:
    """k-Coloring via CMA corridors with NH-noise, entropy gating, reheats."""
    def __init__(self, n: int, k: int, density: float = 0.1, alpha: float = 2.0, mu: float = 0.1,
                 steps: int = 2000, p_anneal: float = 0.7, seed: int = 0,
                 noise: NoiseParams | None = None):
        self.n, self.k = n, k
        self.alpha, self.mu = alpha, mu
        self.steps = steps
        self.noise = noise if noise is not None else NoiseParams()
        self.noise.p_anneal = p_anneal
        self.rng = np.random.default_rng(seed)
        self.A = _erdos_renyi_adj(n, density, self.rng)
        # corridors G[v, c]
        self.G = np.zeros((n, k), dtype=float)
        self.t = 0
        self.tmax = max(int(steps), 1)

    def _schedules(self) -> Tuple[float, float, float]:
        sigma_t = self.noise.sigma0 * (1.0 + self.t / self.tmax) ** (-self.noise.p_anneal)
        eta_t   = self.noise.eta0   * (1.0 + self.t / self.tmax) ** (-self.noise.p_anneal)
        mu_t    = self.mu * (1.0 + self.noise.c_mu * (sigma_t / max(self.noise.sigma0, 1e-12)))
        return sigma_t, eta_t, mu_t

    def _entropy_gate(self, P: np.ndarray) -> np.ndarray:
        # entropy per node over k colors
        H = -np.sum(np.clip(P,1e-12,None) * np.log(np.clip(P,1e-12,None)), axis=1)
        H = (H - H.min()) / max(H.ptp(), 1e-12)
        return 1.0 + 0.3 * H  # lambda=0.3

    def step(self):
        sigma_t, _, mu_t = self._schedules()
        P = _softmax_rows(self.G)
        # conflicts[v, c] = sum over neighbors u of P[u, c]
        conflicts = self.A @ P
        # current encourages colors with fewer conflicts
        I = -conflicts
        # normalize I per-node to zero mean (avoid drift)
        I = I - I.mean(axis=1, keepdims=True)
        # entropy-gated alpha
        alpha_t = self.alpha * self._entropy_gate(P)[:, None]
        # Langevin noise on corridors
        noise = self.rng.normal(0.0, 1.0, size=self.G.shape) * sigma_t
        # Edge dropout: randomly ignore a fraction of neighbors when computing conflicts
        if self.noise.p_drop > 0:
            mask = self.rng.random(self.A.shape) >= self.noise.p_drop
            A_mask = np.where(mask, self.A, False)
            I = -(A_mask @ P)
            I = I - I.mean(axis=1, keepdims=True)
        # Update
        self.G += alpha_t * I - mu_t * self.G + noise
        # reheats at 60% and 85%
        if self.t in (int(0.6 * self.tmax), int(0.85 * self.tmax)):
            self.G += self.rng.normal(0.0, self.noise.sigma0 * 1.5, size=self.G.shape)
        self.t += 1

    def assignment(self) -> np.ndarray:
        return self.G.argmax(axis=1)

    def valid(self) -> bool:
        col = self.assignment()
        # Check no edge connects same colors
        u, v = np.where(np.triu(self.A, 1))
        return bool(np.all(col[u] != col[v]))

    def run(self) -> Tuple[bool, int]:
        for s in range(self.steps):
            self.step()
            if (s + 1) % 25 == 0:
                if self.valid():
                    return True, s + 1
        return self.valid(), self.steps

def _sample_points_outside_obstacles(n: int, obstacles: Obstacles, seed: int):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.random(2)
        if not obstacles.contains(p[0], p[1]):
            pts.append(p)
    return np.array(pts)

def run_sim(alpha=1.0, mu=0.05, steps=20, r=0.1, seed=0, n_points=12, show_plots=True, disable_obstacles=False):
    np.random.seed(seed)
    sim = ARPAdaptivePiField(alpha=alpha, mu=mu, r=r, tmax=steps, seed=seed)
    if disable_obstacles:
        sim.obstacles = Obstacles(ellipses=[])
    # ensure points are outside obstacles
    points = _sample_points_outside_obstacles(n_points, sim.obstacles, seed)

    for t in range(steps):
        sim.step(points)
        if (t+1) % 5 == 0:
            print(f"[Step {t+1}] Running TSP solve...")
            order, cost = tsp_solve(points, r=r, field=sim.to_field(), obstacles=sim.obstacles)
            print(f"Cost at step {t+1}: {cost:.4f}")
            if show_plots:
                tsp_plot(points, order, field=sim.to_field(), obstacles=sim.obstacles, r=r, title=f"ARP-πₐ Step {t+1} (α={alpha}, μ={mu})")


# --- Run a coloring experiment ---
def run_coloring(k=4, n=500, density=0.1, steps=2000, p_anneal=0.7, seed=0):
    print(f"[Coloring] n={n}, k={k}, d={density}, steps={steps}, p={p_anneal}, seed={seed}")
    solver = ARPColoringSolver(n=n, k=k, density=density, steps=steps, p_anneal=p_anneal, seed=seed,
                               alpha=2.0, mu=0.1)
    ok, used = solver.run()
    print(f"[Coloring] valid={ok}, steps_used={used}")
    return ok, used

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARP + Adaptive Pi Geometry Simulator & Graph Coloring (CMA)")
    sub = parser.add_subparsers(dest="mode", required=False)

    p_sim = sub.add_parser("sim", help="Run ARP-πₐ TSP field simulation")
    p_sim.add_argument("--alpha", type=float, default=2.0)
    p_sim.add_argument("--mu", type=float, default=0.1)
    p_sim.add_argument("--steps", type=int, default=25)
    p_sim.add_argument("--r", type=float, default=0.1)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--n_points", type=int, default=50)
    p_sim.add_argument("--no-plot", action="store_true")
    p_sim.add_argument("--no-obstacles", action="store_true")

    p_col = sub.add_parser("coloring", help="Run k-coloring with CMA + NH-noise")
    p_col.add_argument("--k", type=int, default=4)
    p_col.add_argument("--n", type=int, default=500)
    p_col.add_argument("--density", type=float, default=0.1)
    p_col.add_argument("--steps", type=int, default=2000)
    p_col.add_argument("--p_anneal", type=float, default=0.7)
    p_col.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.mode == "coloring":
        run_coloring(k=args.k, n=args.n, density=args.density, steps=args.steps, p_anneal=args.p_anneal, seed=args.seed)
    else:
        # Default mirrors noise‑hardened baseline; tweak n_points/steps as desired
    run_sim(alpha=getattr(args, 'alpha', 2.0), mu=getattr(args, 'mu', 0.1), steps=getattr(args, 'steps', 25),
        r=getattr(args, 'r', 0.1), seed=getattr(args, 'seed', 0), n_points=getattr(args, 'n_points', 50),
        show_plots=not getattr(args, 'no_plot', False), disable_obstacles=getattr(args, 'no_obstacles', False))
