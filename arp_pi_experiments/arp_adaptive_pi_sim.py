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
                self.M_grid[ix, iy] += 1.0 * dt

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

def run_sim(alpha=1.0, mu=0.05, steps=20, r=0.1, seed=0):
    np.random.seed(seed)
    points = np.random.rand(12,2)
    sim = ARPAdaptivePiField(alpha=alpha, mu=mu, r=r)

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
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

from adaptive_pi_tsp import tsp_solve, CurvatureField, Obstacles


@dataclass
class GridFieldAdapter:
    """CurvatureField-compatible adapter using a K(x,y) grid with bilinear interp."""
    K_grid: np.ndarray
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]

    def _xy_to_ij(self, x: float, y: float):
        H, W = self.K_grid.shape
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        u = (x - xmin) / (xmax - xmin) * (W - 1)
        v = (y - ymin) / (ymax - ymin) * (H - 1)
        i = int(np.clip(np.floor(v), 0, H-2))
        j = int(np.clip(np.floor(u), 0, W-2))
        a = u - j
        b = v - i
        return i, j, a, b

    def K(self, x: float, y: float) -> float:
        i, j, a, b = self._xy_to_ij(x, y)
        k00 = self.K_grid[i, j]
        k10 = self.K_grid[i+1, j]
        k01 = self.K_grid[i, j+1]
        k11 = self.K_grid[i+1, j+1]
        return (1-a)*(1-b)*k00 + (1-a)*b*k10 + a*(1-b)*k01 + a*b*k11

    def phi(self, x: float, y: float, r: float = 0.1) -> float:
        K = self.K(x, y)
        denom = 1.0 - (K * (r**2) / 6.0)
        denom = max(denom, 0.02)
        return 1.0 / denom


def laplace_2d(K: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = K.shape
    L = np.zeros_like(K)
    L[1:-1,1:-1] = ((K[1:-1,2:] - 2*K[1:-1,1:-1] + K[1:-1,:-2]) / (dx*dx) +
                    (K[2:,1:-1] - 2*K[1:-1,1:-1] + K[:-2,1:-1]) / (dy*dy))
    L[0,:]  = L[1,:]
    L[-1,:] = L[-2,:]
    L[:,0]  = L[:,1]
    L[:,-1] = L[:,-2]
    return L


def rasterize_tour_intensity(points: np.ndarray, order: List[int], H: int, W: int,
                             sigma_pix: float = 1.5) -> np.ndarray:
    I = np.zeros((H, W), dtype=float)
    path = np.array([points[i] for i in order] + [points[order[0]]])
    for p,q in zip(path[:-1], path[1:]):
        v = q - p
        L = np.linalg.norm(v)
        if L == 0: continue
        steps = max(6, int(100*L))
        ts = np.linspace(0,1,steps)
        xs = p[0] + ts*v[0]
        ys = p[1] + ts*v[1]
        js = np.clip((xs * (W-1)).astype(int), 0, W-1)
        is_ = np.clip((ys * (H-1)).astype(int), 0, H-1)
        I[is_, js] += 1.0/steps
    rad = int(3*sigma_pix)
    x = np.arange(-rad, rad+1)
    g = np.exp(-0.5*(x/sigma_pix)**2); g /= g.sum()
    I = np.apply_along_axis(lambda m: np.convolve(m, g, mode='same'), axis=1, arr=I)
    I = np.apply_along_axis(lambda m: np.convolve(m, g, mode='same'), axis=0, arr=I)
    return I


def local_entropy(field: np.ndarray, eps: float = 1e-9, win: int = 7) -> np.ndarray:
    H, W = field.shape
    pad = win//2
    F = np.pad(field, pad, mode='reflect')
    E = np.zeros_like(field)
    for i in range(H):
        for j in range(W):
            patch = F[i:i+win, j:j+win]
            p = patch - patch.min()
            s = p.sum() + eps
            if s > 0:
                p = p / s
                p = np.clip(p, eps, 1.0)
                e = -(p*np.log(p)).sum()
            else:
                e = 0.0
            E[i,j] = e
    E -= E.min(); rng = E.max() - E.min() + eps
    E /= rng
    return E


from dataclasses import dataclass


@dataclass
class ARPConfig:
    steps: int = 20
    dt: float = 1.0
    alpha: float = 2.0
    mu: float = 0.1
    D: float = 0.001
    rho: float = 0.9
    eta: float = 1.0
    sigma_pix: float = 1.5
    r_local: float = 0.1
    lambda_entropy: float = 0.0
    entropy_window: int = 7


def evolve_with_arp(points: np.ndarray,
                    obstacles: Obstacles,
                    K0: np.ndarray,
                    cfg: ARPConfig,
                    seed: int = 0):
    rng = np.random.default_rng(seed)
    H, W = K0.shape
    dx = 1.0/(W-1); dy = 1.0/(H-1)
    K = K0.copy()
    M = np.zeros_like(K)
    history = dict(K=[], M=[], tours=[], costs=[])

    for t in range(cfg.steps):
        field = GridFieldAdapter(K_grid=K, xlim=(0,1), ylim=(0,1))
        order, cost = tsp_solve(points, r=cfg.r_local, field=field, obstacles=obstacles, max_iter=120)
        history["K"].append(K.copy())
        history["M"].append(M.copy())
        history["tours"].append(order.copy())
        history["costs"].append(cost)
        I = rasterize_tour_intensity(points, order, H, W, sigma_pix=cfg.sigma_pix)
        if cfg.lambda_entropy != 0.0:
            Hmap = local_entropy(I, win=cfg.entropy_window)
            alpha_eff = cfg.alpha * (1.0 + cfg.lambda_entropy * Hmap)
        else:
            alpha_eff = cfg.alpha
        M = cfg.rho * M + cfg.eta * I
        K = K + cfg.dt * (cfg.D * laplace_2d(K, dx, dy) + alpha_eff * M - cfg.mu * K)
    return history


def quick_demo(n_points: int = 28, seed: int = 1,
               alpha: float = 2.0, mu: float = 0.1,
               lambda_entropy: float = 0.15):
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))
    obs = Obstacles.default()
    mask = np.array([not obs.contains(x,y) for x,y in points])
    points = points[mask][:n_points]
    base = CurvatureField.default()
    H, W = 160, 160
    xs = np.linspace(0,1,W); ys = np.linspace(0,1,H)
    K0 = np.zeros((H,W))
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            K0[i,j] = base.K(x,y)
    cfg = ARPConfig(steps=12, alpha=alpha, mu=mu, lambda_entropy=lambda_entropy)
    hist = evolve_with_arp(points, obs, K0, cfg, seed=seed)
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(hist["K"][0].T, origin='lower', extent=[0,1,0,1]); axs[0].set_title("K (t=0)")
    axs[1].imshow(hist["K"][-1].T, origin='lower', extent=[0,1,0,1]); axs[1].set_title(f"K (t={cfg.steps})")
    axs[2].imshow(hist["K"][-1].T, origin='lower', extent=[0,1,0,1])
    tt = np.linspace(0,2*np.pi,200)
    for (cx,cy,ax,ay,th) in obs.ellipses:
        c,s = np.cos(th), np.sin(th)
        xw = cx + ax*np.cos(tt)*c - ay*np.sin(tt)*s
        yw = cy + ax*np.cos(tt)*s + ay*np.sin(tt)*c
        axs[2].plot(xw, yw)
    order = hist["tours"][-1]
    path = np.array([points[i] for i in order] + [points[order[0]]])
    axs[2].plot(path[:,0], path[:,1], linewidth=1.2)
    axs[2].scatter(points[:,0], points[:,1], s=10)
    axs[2].set_title("Final tour on K")
    for ax in axs:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    quick_demo(alpha=2.0, mu=0.1, lambda_entropy=0.15)
