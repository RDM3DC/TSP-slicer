#!/usr/bin/env python3
"""
Hybrid VTSP Solver for Smoothness-Constrained TSP
Combines tsp_realignr_full.py (DBM, 2-opt, ARP, GPU), CurveMemoryTSPSolver (turn/entropy penalties),
elastic TSP (Laplacian smoothing), and octagon solver (curvature learning).
Outputs smooth tours with G-code and visualization for CNC/UAV applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from itertools import combinations
import networkx as nx

# Toggle GPU (requires CuPy >= 12 and CUDA)
USE_GPU = True
if USE_GPU:
    import cupy as xp
    from cupyx.scipy.fft import rfftn, irfftn
else:
    import numpy as xp
    from numpy.fft import rfftn, irfftn

# Hyperparameters
ETA = 2.3              # DBM exploration-exploitation
TRIALS = 256           # Number of DBM trials
KWIN = 30              # 2-opt window size
BETA_TURN = 0.3        # DBM curvature penalty strength
MEMORY_WEIGHT = 0.7    # Turn penalty weight
ENTROPY_WEIGHT = 0.3   # Entropy penalty weight
ARP_STEPS = 100        # ARP iterations
ALPHA = 1e-2           # ARP conductance update
MU = 1e-3              # ARP decay
SMOOTH_STEPS = 50      # Laplacian smoothing steps
BETA_LAP = 0.2         # Laplacian smoothing strength
DELTA = 0.2            # Curvature update rate
EPSILON = 0.01         # Curvature decay rate
FEEDRATE = 1200.0      # G-code feedrate

# Helper Functions
def pairwise_dist(A, B):
    """All-pairs Euclidean distance, vectorized."""
    return xp.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

def fft_laplace_potential(mask_src, shape):
    """FFT-based Poisson solver for DBM potential field."""
    rho = mask_src.astype(xp.float32)
    kx = xp.fft.fftfreq(shape[0])[:, None]
    ky = xp.fft.fftfreq(shape[1])[None, :]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # Avoid division by zero
    phi_k = rfftn(rho)
    phi_k /= (4 * xp.pi**2 * k2[:, :shape[1]//2+1])
    phi = irfftn(phi_k, s=shape)
    return phi

def compute_smoothness_cost(prev, curr, nextp, memory_weight=MEMORY_WEIGHT, entropy_weight=ENTROPY_WEIGHT):
    """Smoothness cost from CurveMemoryTSPSolver (turn + entropy penalties)."""
    d = xp.linalg.norm(curr - nextp)
    if prev is None:
        turn_penalty = entropy = 0.0
    else:
        v1 = curr - prev
        v2 = nextp - curr
        cos_angle = xp.dot(v1, v2) / (xp