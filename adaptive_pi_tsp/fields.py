
import numpy as np

class CurvatureField:
    def __init__(self, terms):
        self.terms = terms  # list of (A, mx, my, sx, sy)
    @staticmethod
    def default():
        return CurvatureField([
            (+1.2, 0.25, 0.30, 0.12, 0.15),
            (-1.5, 0.70, 0.60, 0.18, 0.12),
            (+0.8,  0.60, 0.20, 0.10, 0.10),
        ])
    def K(self, x, y):
        K = 0.0
        for A, mx, my, sx, sy in self.terms:
            K += A * np.exp(-((x-mx)**2/(2*sx**2) + (y-my)**2/(2*sy**2)))
        return K
    def phi(self, x, y, r=0.1):
        # φ = π/π_a with π_a ≈ π(1 - K r^2 / 6) -> 1/(1 - K r^2/6)
        K = self.K(x, y)
        denom = 1.0 - (K * (r**2) / 6.0)
        denom = np.maximum(denom, 0.02)
        return 1.0 / denom

class Obstacles:
    # Elliptic obstacles: (cx,cy,ax,ay,theta)
    def __init__(self, ellipses=None):
        self.ellipses = ellipses or []
    @staticmethod
    def default():
        return Obstacles([
            (0.50, 0.75, 0.18, 0.10, 0.2),
            (0.22, 0.25, 0.14, 0.09, -0.4),
        ])
    def contains(self, x, y):
        for (cx, cy, ax, ay, th) in self.ellipses:
            c, s = np.cos(th), np.sin(th)
            dx, dy = x - cx, y - cy
            xr =  c*dx + s*dy
            yr = -s*dx + c*dy
            if (xr/ax)**2 + (yr/ay)**2 <= 1.0:
                return True
        return False
