import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def gaussian_force(cities, pts, sigma2):
    diff = pts[:, None, :] - cities[None, :, :]
    r2   = (diff**2).sum(-1, keepdims=True)
    w    = np.exp(-r2 / (2 * sigma2)) / sigma2
    return (-w * diff).sum(1)           # shape (N,2)

def two_opt(path):
    def cross(a,b,c,d):
        return (np.cross(b-a, c-a)*np.cross(b-a, d-a) < 0 and
                np.cross(d-c, a-c)*np.cross(d-c, b-c) < 0)

    n = len(path)
    improved = True
    while improved:
        improved = False
        for i in range(n-3):
            for j in range(i+2, n-1):
                if cross(path[i], path[i+1], path[j], path[j+1]):
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    improved = True
                    break
            if improved: break
    return path

def elastic_tsp(cities, steps=300, σ0=5.0, α0=0.5, β0=0.2,
                decay=0.995, two_opt_every=25):
    n = len(cities)
    curve = cities + np.random.normal(0, 0.1, cities.shape)
    σ2, α, β = σ0**2, α0, β0

    for t in range(steps):
        tree = cKDTree(curve)
        nb   = tree.query(curve, k=8)[1]          # 7 neighbours + self
        lap  = curve[nb[:,1:]].mean(1) - curve    # discrete Laplacian
        force = gaussian_force(cities, curve, σ2)
        curve += α*force + β*lap

        if two_opt_every and t % two_opt_every == 0:
            curve = two_opt(curve)

        σ2 *= decay;  α *= decay;  β *= decay*0.9
    return curve

# demo -------------------------------------------------------------
if __name__ == "__main__":
    n = 200
    np.random.seed(0)
    cities = np.random.rand(n,2)*100
    tour   = elastic_tsp(cities)
    plt.plot(*tour.T, '-o', ms=4);  plt.scatter(*cities.T, c='red')
    plt.show()