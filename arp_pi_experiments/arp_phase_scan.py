import argparse, csv, numpy as np
from pathlib import Path
from importlib import import_module

# Import the wrapper created earlier (must be in the same directory or PYTHONPATH)
from arp_adaptive_pi_sim import ARPConfig, evolve_with_arp, GridFieldAdapter
from adaptive_pi_tsp import Obstacles, CurvatureField

def tour_hamming(a, b):
    def edges(ordr):
        return {(ordr[i], ordr[(i+1)%len(ordr)]) for i in range(len(ordr))}
    Ea, Eb = edges(a), edges(b)
    return len(Ea.symmetric_difference(Eb))


def run_experiment(args):
    rng = np.random.default_rng(args.seed)
    obs = Obstacles.default()
    pts = []
    while len(pts) < args.n:
        p = rng.random(2)
        if not obs.contains(p[0], p[1]):
            pts.append(p)
    points = np.array(pts)

    base = CurvatureField.default()
    H, W = args.grid, args.grid
    xs = np.linspace(0,1,W); ys = np.linspace(0,1,H)
    K0 = np.zeros((H,W))
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            K0[i,j] = base.K(x,y)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    mus    = np.linspace(args.mu_min,    args.mu_max,    args.mu_steps)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha","mu","lambda_entropy","seed","steps","final_cost",
                         "avg_flip","max_flip","converged","n_points","grid"])
        for a in alphas:
            for m in mus:
                cfg = ARPConfig(steps=args.steps, alpha=float(a), mu=float(m),
                                D=args.D, rho=args.rho, eta=args.eta,
                                sigma_pix=args.sigma_pix, r_local=args.r,
                                lambda_entropy=args.lambda_entropy,
                                entropy_window=args.entropy_window)
                hist = evolve_with_arp(points, obs, K0, cfg, seed=args.seed)
                flips = []
                tours = hist["tours"]
                for t in range(1, len(tours)):
                    flips.append(tour_hamming(tours[t-1], tours[t]))
                avg_flip = float(np.mean(flips)) if flips else 0.0
                max_flip = int(np.max(flips)) if flips else 0
                final_cost = float(hist["costs"][-1])
                converged = int(max_flip == 0)
                writer.writerow([a, m, args.lambda_entropy, args.seed, args.steps,
                                 final_cost, avg_flip, max_flip, converged,
                                 len(points), f"{H}x{W}"])
                f.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=28)
    p.add_argument("--grid", type=int, default=140)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--alpha-min", type=float, default=0.5)
    p.add_argument("--alpha-max", type=float, default=3.0)
    p.add_argument("--alpha-steps", type=int, default=6)
    p.add_argument("--mu-min", type=float, default=0.05)
    p.add_argument("--mu-max", type=float, default=0.5)
    p.add_argument("--mu-steps", type=int, default=6)
    p.add_argument("--lambda-entropy", type=float, default=0.15)
    p.add_argument("--entropy-window", type=int, default=7)
    p.add_argument("--D", type=float, default=0.001)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--sigma-pix", type=float, default=1.5)
    p.add_argument("--r", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/phase_scan.csv")
    args = p.parse_args()
    run_experiment(args)
