# ARP–Adaptive-π Experiments

This folder contains:

- `arp_adaptive_pi_sim.py`: ARP + adaptive-π evolution wrapper over a grid-based curvature field K(x,y).
- `arp_phase_scan.py`: A harness to sweep alpha/mu and log tour phase flips and final costs.

Quickstart (from repo root):

```bash
# Optional: create venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy matplotlib

# Run demo
python arp_pi_experiments/arp_adaptive_pi_sim.py

# Phase scan
python arp_pi_experiments/arp_phase_scan.py --n 28 --steps 12 \
  --alpha-min 1.0 --alpha-max 3.0 --alpha-steps 6 \
  --mu-min 0.05 --mu-max 0.3 --mu-steps 6 \
  --lambda-entropy 0.15 --out results/phase_scan.csv
```

Results are written to `results/phase_scan.csv`.
