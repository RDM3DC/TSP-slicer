#!/usr/bin/env python3
# tsp_realignr_full.py
"""
Dielectric-breakdown + ARP TSP solver, GPU-ready (NumPy or CuPy).

Toggle GPU by:   USE_GPU = True
Requires CuPy >= 12 and a CUDA-capable card.
"""

import time, sys
USE_GPU = True          # ← flip to False for pure CPU
ETA      = 2.3          # exploration ↔ exploitation knob
TRIALS   = 256          # constructive passes
KWIN     = 30           # 2-opt window (positions apart)
BETA_TURN= 0.3          # curvature penalty strength
ARP_STEPS= 100          # conductance iterations
ALPHA    = 1e-2         # ARP α
MU       = 1e-3         # ARP μ

if USE_GPU:
    import cupy as xp
    from cupyx.scipy.fft import rfftn, irfftn
else:
    import numpy as xp
    from numpy.fft import rfftn, irfftn

# ───────────────────────── helpers ───────────────────────── #

def pairwise_dist(A, B):
    """All-pairs Euclidean distance, vectorised."""
    return xp.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

def fft_laplace_potential(mask_src, shape):
    """
    Solve ∇²φ = –ρ with ρ = mask_src (Dirichlet 0 at boundary).
    For our heuristic a simple FFT Poisson solve is enough.
    """
    rho = mask_src.astype(xp.float32)
    kx = xp.fft.fftfreq(shape[0])[:, None]
    ky = xp.fft.fftfreq(shape[1])[None, :]
    k2 = (kx**2 + ky**2)
    k2[0, 0] = 1.0            # avoid /0
    phi_k = rfftn(rho)
    phi_k /= (4 * xp.pi**2 * k2[:, :shape[1]//2+1])
    phi   = irfftn(phi_k, s=shape)
    return phi

# ───────────────── DBM constructive walk ───────────────── #

def dbm_walk(cities, eta=ETA, beta=BETA_TURN):
    N = len(cities)
    unvis = list(range(N))
    cur   = unvis.pop(xp.random.randint(N))
    tour  = [cur]

    while unvis:
        vecs   = cities[unvis] - cities[cur]
        dists  = xp.linalg.norm(vecs, axis=1)
        probs  = dists ** (-eta)
        probs /= probs.sum()

        # curvature bonus (look-ahead)
        if beta > 0 and len(tour) > 1:
            v_now   = cities[cur] - cities[tour[-2]]
            head    = xp.angle(v_now[0] + 1j*v_now[1])
            angles  = xp.angle(vecs[:,0] + 1j*vecs[:,1])
            penalty = xp.exp(-beta * (angles-head)**2)
            probs  *= penalty
            probs  /= probs.sum()

        nxt = int(xp.random.choice(len(unvis), p=probs))
        cur = unvis.pop(nxt)
        tour.append(cur)
    return tour

# ───────────────────────── 2-opt (window) ─────────────────── #

def two_opt_window(path, cities, k_win=KWIN, W=None):
    n = len(path)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            j_max = min(i + k_win, n-1)
            for j in range(i+2, j_max):
                if j-i == 1:              # neighbours
                    continue
                a,b,c,d = path[i-1], path[i], path[j-1], path[j]
                if W is None:
                    cost_o = xp.linalg.norm(cities[a]-cities[b]) + \
                             xp.linalg.norm(cities[c]-cities[d])
                    cost_n = xp.linalg.norm(cities[a]-cities[c]) + \
                             xp.linalg.norm(cities[b]-cities[d])
                else:
                    cost_o = W[a,b] + W[c,d]
                    cost_n = W[a,c] + W[b,d]
                if cost_n < cost_o - 1e-12:
                    path[i:j] = path[i:j][::-1]
                    improved = True
    return path

# ───────────── adaptive-resistance (ARP) refine ──────────── #

def arp_refine(path, cities, steps=ARP_STEPS, α=ALPHA, μ=MU):
    n = len(path)
    D = pairwise_dist(cities[path], cities[path])
    G = 1.0/(D + 1e-9)
    I = xp.zeros_like(G)
    I[0, n//2] = 1.0           # inject current
    for _ in range(steps):
        G += α*xp.abs(I) - μ*G
        G = xp.maximum(G, 1e-9)
    return 1.0/G               # weights

# ─────────────────── high-level TSP solve ────────────────── #

def solve_tsp(cities, trials=TRIALS, use_arp=True):
    best_L, best_path = xp.inf, None
    for _ in range(trials):
        path = dbm_walk(cities)
        path = two_opt_window(path, cities)
        L    = tour_len(path, cities)
        if L < best_L:
            best_L, best_path = L, path

    if use_arp:
        W = arp_refine(best_path, cities)
        best_path = two_opt_window(best_path, cities, W=W)
        best_L    = tour_len(best_path, cities)
    return best_path, best_L

def tour_len(path, cities):
    idx = xp.asarray(path + [path[0]])
    return xp.linalg.norm(cities[idx[:-1]] - cities[idx[1:]], axis=1).sum()

# ───────────────────────── demo run ───────────────────────── #

if __name__ == "__main__":
    seed = 0; xp.random.seed(seed)
    N    = 1000
    cities = xp.random.rand(N,2).astype(xp.float32)*100

    t0 = time.time()
    path, L = solve_tsp(cities)
    t1 = time.time()

    mode = "GPU" if USE_GPU else "CPU"
    print(f"{N}-city tour, {mode}:  length = {float(L):.2f},  time = {t1-t0:.2f} s")

    # ↓ plotting on GPU arrays requires transfer back to CPU
    import matplotlib.pyplot as plt
    tour_xy = cities[xp.asarray(path + [path[0]])].get() if USE_GPU else cities[path + [path[0]]]
    plt.figure(figsize=(6,6))
    plt.plot(tour_xy[:,0], tour_xy[:,1], lw=0.4)
    plt.scatter(cities.get()[:,0] if USE_GPU else cities[:,0],
                cities.get()[:,1] if USE_GPU else cities[:,1],
                s=6, c='red')
    plt.title(f"TSP tour, {N} cities ({mode})")
    plt.axis('equal'); plt.tight_layout(); plt.show()