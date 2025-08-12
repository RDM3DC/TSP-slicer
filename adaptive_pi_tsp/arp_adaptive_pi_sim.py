
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

# You need the package from: adaptive-pi-tsp-0.1.0.zip
from adaptive_pi_tsp import tsp_solve, tsp_plot
from adaptive_pi_tsp import CurvatureField, Obstacles

# -----------------------------
# Grid-based adaptive-π field
# -----------------------------

@dataclass
class GridFieldAdapter:
    """A CurvatureField-compatible adapter that holds K(x,y) on a grid and
    exposes K(x,y) and phi(x,y,r) by bilinear interpolation.
    """
    K_grid: np.ndarray           # shape (H, W)
    xlim: Tuple[float, float]    # (xmin, xmax)
    ylim: Tuple[float, float]    # (ymin, ymax)

    def _xy_to_ij(self, x: float, y: float) -> Tuple[int, int, float, float]:
        H, W = self.K_grid.shape
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        # normalize to [0, W-1]/[0, H-1]
        u = (x - xmin) / (xmax - xmin) * (W - 1)
        v = (y - ymin) / (ymax - ymin) * (H - 1)
        i = int(np.clip(np.floor(v), 0, H-2))
        j = int(np.clip(np.floor(u), 0, W-2))
        a = u - j
        b = v - i
        return i, j, a, b

    def K(self, x: float, y: float) -> float:
        H, W = self.K_grid.shape
        i, j, a, b = self._xy_to_ij(x, y)
        # bilinear
        k00 = self.K_grid[i, j]
        k10 = self.K_grid[i+1, j]
        k01 = self.K_grid[i, j+1]
        k11 = self.K_grid[i+1, j+1]
        return (1-a)*(1-b)*k00 + (1-a)*b*k10 + a*(1-b)*k01 + a*b*k11

    def phi(self, x: float, y: float, r: float = 0.1) -> float:
        K = self.K(x, y)
        denom = 1.0 - (K * (r**2) / 6.0)
        denom = max(denom, 0.02)  # clamp to avoid near-singularities
        return 1.0 / denom

# -----------------------------
# ARP + PDE evolution of K(x,t)
# -----------------------------

def laplace_2d(K: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """5-point Laplacian with Neumann boundaries."""
    H, W = K.shape
    L = np.zeros_like(K)
    # interior
    L[1:-1,1:-1] = ((K[1:-1,2:] - 2*K[1:-1,1:-1] + K[1:-1,:-2]) / (dx*dx) +
                    (K[2:,1:-1] - 2*K[1:-1,1:-1] + K[:-2,1:-1]) / (dy*dy))
    # simple Neumann (copy edges)
    L[0,:]    = L[1,:]
    L[-1,:]   = L[-2,:]
    L[:,0]    = L[:,1]
    L[:,-1]   = L[:,-2]
    return L

def rasterize_tour_intensity(points: np.ndarray, order: List[int], H: int, W: int,
                             sigma_pix: float = 1.5) -> np.ndarray:
    """Rasterize a tour as a blurred 'ink' map I(x) on the grid."""
    I = np.zeros((H, W), dtype=float)
    path = np.array([points[i] for i in order] + [points[order[0]]])
    # draw segments by sampling along each edge
    for p,q in zip(path[:-1], path[1:]):
        v = q - p
        L = np.linalg.norm(v)
        if L == 0: continue
        steps = max(6, int(100*L))
        ts = np.linspace(0,1,steps)
        xs = p[0] + ts*v[0]
        ys = p[1] + ts*v[1]
        # map to pixel indices in [0,W-1],[0,H-1] (domain assumed [0,1]^2)
        js = np.clip((xs * (W-1)).astype(int), 0, W-1)
        is_ = np.clip((ys * (H-1)).astype(int), 0, H-1)
        I[is_, js] += 1.0/steps
    # Gaussian blur via separable convolution
    rad = int(3*sigma_pix)
    x = np.arange(-rad, rad+1)
    g = np.exp(-0.5*(x/sigma_pix)**2); g /= g.sum()
    I = np.apply_along_axis(lambda m: np.convolve(m, g, mode='same'), axis=1, arr=I)
    I = np.apply_along_axis(lambda m: np.convolve(m, g, mode='same'), axis=0, arr=I)
    return I

def local_entropy(field: np.ndarray, eps: float = 1e-9, win: int = 7) -> np.ndarray:
    """Compute a simple local Shannon entropy map on a sliding window (normalized 0..1)."""
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
                # quantize to avoid log(0)
                p = np.clip(p, eps, 1.0)
                e = -(p*np.log(p)).sum()
            else:
                e = 0.0
            E[i,j] = e
    # normalize to [0,1]
    E -= E.min(); rng = E.max() - E.min() + eps
    E /= rng
    return E

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
    lambda_entropy: float = 0.0  # strength for e_a from entropy
    entropy_window: int = 7

def evolve_with_arp(points: np.ndarray,
                    obstacles: Obstacles,
                    K0: np.ndarray,
                    cfg: ARPConfig,
                    seed: int = 0):
    """Closed-loop ARP evolution.
    Returns: history dict with K_t, M_t, tours, costs
    """
    rng = np.random.default_rng(seed)
    H, W = K0.shape
    dx = 1.0/(W-1); dy = 1.0/(H-1)
    K = K0.copy()
    M = np.zeros_like(K)
    history = dict(K=[], M=[], tours=[], costs=[])

    for t in range(cfg.steps):
        # field adapter for current K
        field = GridFieldAdapter(K_grid=K, xlim=(0,1), ylim=(0,1))

        # solve TSP on current manifold (uses package solver with our field via monkey-patch)
        # We can't replace types, so we create thin wrappers to match expected interface.
        # The package only calls .phi(x,y,r) and .K(x,y), so GridFieldAdapter already matches.
        # We'll route obstacles through the package's default mechanisms.
        order, cost = tsp_solve(points, r=cfg.r_local, field=field, obstacles=obstacles, max_iter=120)

        history["K"].append(K.copy())
        history["M"].append(M.copy())
        history["tours"].append(order.copy())
        history["costs"].append(cost)

        # Rasterize intensity around tour
        I = rasterize_tour_intensity(points, order, H, W, sigma_pix=cfg.sigma_pix)

        # Optional entropy-driven dynamic constant e_a → scale alpha
        if cfg.lambda_entropy != 0.0:
            Hmap = local_entropy(I, win=cfg.entropy_window)
            alpha_eff = cfg.alpha * (1.0 + cfg.lambda_entropy * Hmap)
        else:
            alpha_eff = cfg.alpha

        # Update memory and curvature (explicit Euler)
        M = cfg.rho * M + cfg.eta * I
        K = K + cfg.dt * (cfg.D * laplace_2d(K, dx, dy) + alpha_eff * M - cfg.mu * K)

    return history

def quick_demo(n_points: int = 28, seed: int = 1,
               alpha: float = 2.0, mu: float = 0.1,
               lambda_entropy: float = 0.15):
    """End-to-end demo that runs the ARP loop and plots K_t snapshots + tours."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))

    obs = Obstacles.default()
    # ensure points are outside obstacles
    mask = np.array([not obs.contains(x,y) for x,y in points])
    points = points[mask][:n_points]

    # initial curvature: use CurvatureField.default() sampled to a grid
    base = CurvatureField.default()
    H, W = 160, 160
    xs = np.linspace(0,1,W); ys = np.linspace(0,1,H)
    K0 = np.zeros((H,W))
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            K0[i,j] = base.K(x,y)

    cfg = ARPConfig(steps=12, alpha=alpha, mu=mu, lambda_entropy=lambda_entropy)
    hist = evolve_with_arp(points, obs, K0, cfg, seed=seed)

    # Plot first and last K, plus last tour overlay
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(hist["K"][0].T, origin='lower', extent=[0,1,0,1]); axs[0].set_title("K (t=0)")
    axs[1].imshow(hist["K"][-1].T, origin='lower', extent=[0,1,0,1]); axs[1].set_title(f"K (t={cfg.steps})")
    axs[2].imshow(hist["K"][-1].T, origin='lower', extent=[0,1,0,1])
    # draw obstacles
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
    for ax in axs: ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example matching Grok's suggestion: alpha=2.0, mu=0.1
    quick_demo(alpha=2.0, mu=0.1, lambda_entropy=0.15)
