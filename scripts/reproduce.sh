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

