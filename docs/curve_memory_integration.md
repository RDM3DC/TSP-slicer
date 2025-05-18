# Curve Memory Integration Overview

This document summarises how Curve Memory can be incorporated into a 3D-printing workflow. The approach complements the existing TSP travel optimisation by smoothing travel paths based on local curvature information stored from previous layers.

---

## Mathematical Background

Given a 2D path parameterised by arc length \(s\), the curvature \(k(s)\) is

\[
  k(s) = \frac{|r'(s) \times r''(s)|}{|r'(s)|^3}
\]

where \(r(s)\) is the original path and the primes denote first and second derivatives.

The Curve Memory factor \(\hat{\theta}(s,t)\) evolves layer by layer:

\[
  \hat{\theta}(s,t) = \lambda |k(s,t)| - \mu\, \theta(s,t-1)
\]

- \(\lambda\) reinforces regions of high curvature.
- \(\mu\) controls how quickly memory decays between layers.

An updated path is then computed using a normal offset proportional to the stored memory:

\[
  r_{\text{new}}(s,t) = r_{\text{linear}}(s,t) + \pi\_\alpha(s,t)\,n(s)
\]

with \(n(s)\) the unit normal and \(\pi\_\alpha(s,t) = \pi\, \theta(s,t)\).

---

## Reference Pseudocode

```python
import numpy as np

# Parameters
lambda_curvature = 0.5
mu_decay = 0.1

def compute_curvature(path):
    dr = np.gradient(path, axis=0)
    d2r = np.gradient(dr, axis=0)
    num = np.abs(dr[:,0]*d2r[:,1] - dr[:,1]*d2r[:,0])
    denom = (dr[:,0]**2 + dr[:,1]**2)**(3/2) + 1e-6
    return num / denom

def update_memory(curv, prev):
    if prev is None:
        prev = np.zeros_like(curv)
    return lambda_curvature * curv - mu_decay * prev

def adapt_path(linear_path, memory):
    dr = np.gradient(linear_path, axis=0)
    tang = dr / np.linalg.norm(dr, axis=1, keepdims=True)
    normal = np.stack([-tang[:,1], tang[:,0]], axis=1)
    pi_alpha = np.pi * memory
    return linear_path + (pi_alpha[:, None] * normal)
```

The workflow processes each layer sequentially, updating memory values and applying the adaptive offset before G-code generation.

---

This outline is intended as a starting point for implementing Curve Memory alongside the existing TSP solver.
