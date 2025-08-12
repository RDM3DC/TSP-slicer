
# adaptive-pi-tsp

Traveling Salesman solver on an adaptive-π manifold. Edge cost is the geodesic line integral
with φ = π/πₐ where πₐ ≈ π(1 - K r² / 6). Obstacles are infinite-cost zones.

## Quickstart

```python
import numpy as np
from adaptive_pi_tsp import CurvatureField, Obstacles, tsp_solve, tsp_plot

rng = np.random.default_rng(0)
points = rng.random((30,2))

# ensure points are outside obstacles for demo
obs = Obstacles.default()
mask = np.array([not obs.contains(x,y) for x,y in points])
points = points[mask][:30]

order, cost = tsp_solve(points, r=0.1, field=CurvatureField.default(), obstacles=obs)
tsp_plot(points, order, field=CurvatureField.default(), obstacles=obs, r=0.1)
```
