# TSP Slicer

TSP Slicer is a collection of tools for applying Traveling Salesman Problem
(TSP) optimisation to 3D printer toolpaths.  It bundles a small wrapper around
[Google OR-Tools](https://developers.google.com/optimization) together with
post-processing utilities and experimental solvers for travel smoothing.

## Features

- **Core solver** – `tsp_opt.py` exposes a simple API for computing short tours
  through XY travel moves using OR-Tools.
- **Cura plug-in** – `TSPPostProcessor.py` reorders each layer's travel moves to
  minimise non-printing motion when used as a Cura post-processing script.
- **PrusaSlicer script** – `prusa_tsp_pp.py` performs the same optimisation on
  G-code produced by PrusaSlicer.
- **Experimental algorithms** – modules such as `gu-tsp.py` and
  `hybridtsp.py` explore curvature-aware and smooth TSP solutions.
- **Noise-hardened ARP baseline** – `arp_pi_experiments/noise_hardened_arp.py`
  implements an annealed-noise ARP solver with Langevin noise on
  conductance, OU jitter on curvature, edge dropout/dither, and
  noise-aware hysteresis.  Results: −19–20% tour-cost gaps vs
  Euclid+LK on 1k–5k cities with low variance and robustness to
  coordinate jitter and adversarial edge perturbations.  Reproducible
  with provided seeds.  Empirically: 1k cities (250 steps) achieves a
  −19.2% gap vs Euclid+LK over 10 seeds (var 1.8%); 5k cities (625
  steps) yields a −20.1% gap (var 2.1%).

## Installation

The project targets Python 3.9+ and requires [`ortools`](https://pypi.org/project/ortools/):

```bash
pip install ortools
```

### Cura plug-in

1. Copy `TSPPostProcessor.py`, `tsp_opt.py` and `plugin.json` into a folder
   named `TSPTravelOptimizer`.
2. Zip that folder and install it via **Extensions → Post Processing → Modify
   G-Code** in Cura.
3. Slice a model and enable **TSP Travel Optimizer** in the post-processing
   dialog.

### PrusaSlicer script

```bash
python prusa_tsp_pp.py your_file.gcode
```

This replaces `your_file.gcode` with an optimised version.

## Quickstart (library)

```python
from tsp_opt import solve_tsp

points = [(0, 0), (10, 0), (10, 10), (0, 10)]
order = solve_tsp(points)
print(order)
```

## Documentation

Additional notes on curvature-aware travel and curve memory techniques are
available in the [`docs/`](docs) directory.

## License

This project is released under the MIT License.  See
[LICENSE](LICENSE) for details.

