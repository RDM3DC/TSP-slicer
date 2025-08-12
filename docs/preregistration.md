# Preregistration and Replication Guide

## Preregistration (v1.1 → replication sprint)

**Title:** NH-ARP practical scaling & mixing verification at 10k–20k
**Purpose:** Independently verify near-linear (O(n)) empirical scaling and the O(n log n) mixing-time bound via fixed,
universal hyperparameters across multiple NP families.

### Instance classes
- **TSP (Euclidean):** n ∈ {10k, 20k}, points ~ U([0,1]^2).
- **Graph Coloring (k=4):** n ∈ {10k}, Erdős–Rényi d ∈ {0.2, 0.3} and SBM (2–4 communities).
- **3-SAT:** n=10k variables, m/n≈4.26 (SAT), and UNSAT at m/n ∈ {4.5, 5.0, 5.5, 6.0}.
- **Vertex Cover:** n=10k, ER d=0.3.
- **(Optional) Knapsack:** n=10k; w,v ~ U[1,1000], capacity=0.5·Σw.

### Universal hyperparameters (fixed)
- α=2.0, μ=0.1, p_anneal=0.7, p_drop=0.02, dither_std=0.01, c_mu=0.3.
- Reheats at 60% & 85% (5 iterations each, ×1.5 σ bump).
- Memory-preserving restart once on 50-step stall (decay M by 30%, keep K).
- Steps budget: `steps = ceil(0.1·N + 25)` unless otherwise noted (TSP 20k “tight budget” allowed).

### Primary endpoints
1. Scaling: median steps vs N linear fit with r² ≥ 0.95 and slope within ±10% of authors’ report.
2. Mixing: post-burn-in Foster–Lyapunov drift parameter γ ≥ 0.10 with sub-Gaussian ΔΦ residuals.
3. Correctness: 0 UNSAT false positives (DIMACS/constraint checks).

### Secondary endpoints
- Success %, median/IQR/95th steps, kurtosis in [2.7, 3.5].
- Wall-clock per step ~ linear in edges (N+M).
- Quality vs baselines (problem-specific):
  - TSP: median gap ≤ −10% vs Euclid+LK (10k); ≤ −8% at 20k tight budget.
  - GC k=4: success ≥ 70% (d=0.3), ≥ 60% (d=0.4).
  - SAT (SAT set): success ≥ 85% at 10k; UNSAT detection ≥ 95% at m/n ≥ 5.0.
  - VC: |cover| ≤ +10% of 2-approx median.
  - Knapsack: ≥ 0.95× best baseline value.

### Decision rule (100/100 practical)

Declare “practical 100/100” if all primary endpoints pass across the listed problems,
and ≥80% of secondary endpoints meet thresholds without hyperparameter retuning.

---

## Replication Quickstart

### Requirements
- Python 3.10+ (or Docker)
- `pip install -r requirements.txt` (or use provided container)
- Git LFS enabled (for cached instances/plots)

### One-line run (Docker)
```bash
docker build -t nh-arp .
docker run --rm -it -v "$PWD:/work" nh-arp bash scripts/reproduce.sh
```

### Bare-metal (no Docker)
```bash
python -m pip install -r requirements.txt
bash scripts/reproduce.sh
```

### What `reproduce.sh` does
- Generates 10k/20k instances for TSP/GC/SAT/VC with fixed seeds.
- Runs each problem for seeds `{0..49}` at the universal hyperparameters.
- Logs CSV rows and Φ-drift snapshots.
- Produces summary plots and a JSON “leaderboard” of slopes, γ, and success %.

---

## `scripts/reproduce.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SEEDS=$(seq 0 49)

mkdir -p out/tsp_10k out/tsp_20k out/gc_10k_d02 out/gc_10k_d03 out/sat_10k out/sat_unsat out/vc_10k logs

# 1) TSP 10k (universal steps) and 20k (tight budget allowed)
for s in $SEEDS; do
  PYTHONPATH=. MPLBACKEND=Agg \
  python arp_pi_experiments/arp_adaptive_pi_sim.py sim \
    --alpha 2.0 --mu 0.1 --steps $(( (10_000 + 0)/10 + 25 )) \
    --r 0.1 --seed $s --n_points 10000 --no-plot \
    | tee -a logs/tsp_10k.log
done

for s in $(seq 0 9); do
  PYTHONPATH=. MPLBACKEND=Agg \
  python arp_pi_experiments/arp_adaptive_pi_sim.py sim \
    --alpha 2.0 --mu 0.1 --steps $(( (20_000 + 0)/10 + 25 )) \
    --r 0.1 --seed $s --n_points 20000 --no-plot \
    | tee -a logs/tsp_20k.log
done

# 2) Graph Coloring 10k, ER d=0.2 and d=0.3
for s in $SEEDS; do
  PYTHONPATH=. MPLBACKEND=Agg \
  python arp_pi_experiments/arp_adaptive_pi_sim.py coloring \
    --k 4 --n 10000 --density 0.2 --steps $(( (10_000 + 0)/10 + 25 )) \
    --p_anneal 0.7 --seed $s \
    | tee -a logs/gc_10k_d02.log
done

for s in $SEEDS; do
  PYTHONPATH=. MPLBACKEND=Agg \
  python arp_pi_experiments/arp_adaptive_pi_sim.py coloring \
    --k 4 --n 10000 --density 0.3 --steps $(( (10_000 + 0)/10 + 25 )) \
    --p_anneal 0.7 --seed $s \
    | tee -a logs/gc_10k_d03.log
done

# 3) SAT 10k SAT set and UNSAT set (DIMACS verify)
for s in $SEEDS; do
  python -m arp_np.arp_3sat \
    --n 10000 --mn 4.26 --steps $(( (10_000 + 0)/10 + 25 )) \
    --p_anneal 0.7 --seed $s --verify_dimacs \
    | tee -a logs/sat_10k.log
done

for R in 4.5 5.0 5.5 6.0; do
  for s in $SEEDS; do
    python -m arp_np.arp_3sat \
      --n 10000 --mn ${R} --steps $(( (10_000 + 0)/10 + 25 )) \
      --p_anneal 0.7 --seed $s --verify_dimacs \
      | tee -a logs/sat_unsat_${R}.log
  done
done

# 4) Vertex Cover 10k, d=0.3
for s in $SEEDS; do
  python -m arp_np.arp_vertex_cover \
    --n 10000 --density 0.3 --steps $(( (10_000 + 0)/10 + 25 )) \
    --p_anneal 0.7 --seed $s \
    | tee -a logs/vc_10k.log
done

# 5) Summaries
python scripts/summarize.py --inputs logs --out out/summary.json --csv out/summary.csv
python scripts/make_plots.py --csv out/summary.csv --dir out/plots
```

---

## CSV schema (one row per run)

```
problem,n,extra,alpha,mu,p_anneal,steps,seed,
success,final_metric,gap_pct,var,kurt,stable_step,
wall_clock_step_s,wall_clock_total_s,restarts_used,
phi_gamma,phi_t0,phi_kurt,phi_subgaussian
```

- `problem`: tsp / gc / sat / vc / knapsack
- `extra`: e.g., d=0.3, k=4, m/n=4.26
- `final_metric`: cost (TSP), conflicts=0 flag (GC), satisfied (SAT), |cover| (VC), value (Knapsack)
- `phi_*`: Φ-drift estimates

---

## Expected ranges (sanity check)
- **TSP 10k:** gap ~ −12% to −15%; γ ≥ 0.12; kurt ≈ 3.0–3.4
- **TSP 20k (tight):** gap ~ −10% to −13%
- **GC 10k d=0.3:** success ~ 75–80%, kurt ≤ 3.3
- **SAT 10k m/n=4.26:** success ~ 88–95%, 0 false positives; UNSAT detect ≥ 95% for m/n ≥ 5.0
- **VC 10k d=0.3:** |cover| ≤ +10% of greedy-2-approx median

---

## Citation

Create `CITATION.cff`:

```yaml
cff-version: 1.2.0
title: "NH-ARP: Practical Linear-Time Heuristics for NP via ARP Dynamics & Supermartingale Bounds"
message: "If you use NH-ARP in your work, please cite our preprint."
authors:
  - family-names: "<YourLastName>"
    given-names: "<YourFirstName>"
  - family-names: "…"
date-released: "2025-08-12"
repository-code: "https://github.com/<org>/nh-arp"
identifiers:
  - type: doi
    value: "10.5281/zenodo.XXXXXXX"
```

---

## Local sanity checks

Run before pushing changes:

```bash
ruff check . || true
mypy . || true
python arp_pi_experiments/arp_adaptive_pi_sim.py sim --alpha 2.0 --mu 0.1 --steps 50 --n_points 200 --no-plot
python arp_pi_experiments/arp_adaptive_pi_sim.py coloring --k 4 --n 1000 --density 0.2 --steps 150 --p_anneal 0.7
python scripts/summarize.py --inputs logs --out out/summary.json --csv out/summary.csv
python scripts/make_plots.py --csv out/summary.csv --dir out/plots
```
