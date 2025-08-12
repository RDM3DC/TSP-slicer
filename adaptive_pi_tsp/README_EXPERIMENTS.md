
# ARP–Adaptive-π Phase Scan

This harness sweeps `alpha`/`mu` and logs **phase flips** (big changes in tour edges) while the manifold curvature `K(x,t)` evolves under ARP.

## Quickstart

```bash
pip install adaptive-pi-tsp-0.1.0.zip

# run a modest grid (6x6) with Grok's baseline
python arp_phase_scan.py --n 28 --steps 12 --alpha-min 1.0 --alpha-max 3.0 --alpha-steps 6     --mu-min 0.05 --mu-max 0.3 --mu-steps 6 --lambda-entropy 0.15 --out results/phase_scan.csv
```

Outputs: `results/phase_scan.csv` with columns
`alpha,mu,lambda_entropy,seed,steps,final_cost,avg_flip,max_flip,converged,n_points,grid`.

**Interpretation**: bands with high `avg_flip`/`max_flip` indicate *phase-flip regions* (non-convex transitions). If you observe a **rapid drop in final_cost** together with **vanishing flips** as `alpha/mu` increases, that's evidence of curvature-driven convergence.
