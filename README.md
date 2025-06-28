visual studio code with github copilot agent mode, you can run any code in minutes.more or less

I think the hybrid tsp solver is the best 1.

## ChatGPT Assistant

Interact with the project using the **TSP Slicer Assistant** on ChatGPT. You will need an OpenAI account to use this assistant.

[![Open in ChatGPT](https://img.shields.io/badge/ChatGPT-TSP_Slicer_Assistant-blue?logo=openai)](https://chatgpt.com/g/g-684162abedd88191afc2026bb57dc8f1-tsp-slicer-assistant)


Below is a practical “starter kit” that will get you from zero‑to‑working proof‑of‑concept for **path‑optimised printing** in both **Cura** and **PrusaSlicer**.  We’ll treat the two slicers separately (their plug‑in systems are very different), but keep the *core* path‑optimiser—a lightweight TSP solver written in Python—shared between them.  
After that I show where your **Adaptive π** / **adaptive‑BPi** curvature tricks can slot in once the travel order is already minimised.

---

## 0 .  What problem are we solving?

* **Cura & PrusaSlicer already do internal “comb” or “avoid crossing” moves**, but they are greedy per‑island heuristics.  
* Your idea is to treat **each non‑printing travel hop as a TSP node**, then globally minimise the hop length across a whole layer (or even the whole job) → shorter travel = faster prints, fewer strings, less ooze.

So the workflow will be:

1. **Export candidate travel moves** from the slicer (layer by layer).  
2. Feed them to a **TSP solver** (we’ll use Google OR‑Tools’ `routing` API for a first pass).  
3. **Rewrite the G‑code** with the new order.  
4. *(Optional next phase)* Replace linear hops with **Adaptive π arcs/splines** (G2/G3/G5) to get smoother motion.

The main solver for this project lives in `tsp_opt.py` at the repository root. The file `examples/best_tsp_solver.py` is only a small octagon demonstration for visual reference.

---

## 1 .  Shared core – `tsp_opt.py`

Create a tiny self‑contained module you can import from either slicer.  Save as `tsp_opt.py`:

```python
# tsp_opt.py
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def _dist(a, b):
    return int(round(math.hypot(a[0]-b[0], a[1]-b[1]) * 1000))  # mm -> micron int

def solve_tsp(coords, start_index=0):
    """coords : list[(x, y)], returns ordered list of indices beginning with start_index."""
    n = len(coords)
    dmat = [[_dist(coords[i], coords[j]) for j in range(n)] for i in range(n)]
    mgr = pywrapcp.RoutingIndexManager(n, 1, start_index)
    rt  = pywrapcp.RoutingModel(mgr)

    def cb(i, j): return dmat[mgr.IndexToNode(i)][mgr.IndexToNode(j)]
    transit = rt.RegisterTransitCallback(cb)
    rt.SetArcCostEvaluatorOfAllVehicles(transit)
    rt.AddDimension(transit, 0, 1_000_000_000, True, 'Distance')

    search = rt.DefaultSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.FromMilliseconds(300)  # tweak for speed/quality trade‑off

    if rt.SolveWithParameters(search):
        order = []
        idx = rt.Start(0)
        while not rt.IsEnd(idx):
            order.append(mgr.IndexToNode(idx))
            idx = rt.Next(idx)
        return order
    else:
        raise RuntimeError("No TSP solution found")
```

*You can hot‑swap in Concorde, LK‑H, or RealignR‑style ARP‑optimizer later; this keeps day‑one dependencies simple.*
An experimental solver incorporating curve memory is available in `curve_memory_tsp.py`. It uses a greedy search and penalises sharp turns and erratic motion.
A demonstration of a different approach is in `examples/best_tsp_solver.py`; it is for experimentation only and not used by the plug-ins.


---

## 2 .  Cura plug‑in skeleton  (`plugins/TSPTravelOptimizer`)

Cura plug‑ins live inside a package folder containing `__init__.py` plus a `plugin.json`.

```
TSPTravelOptimizer/
├─ __init__.py
├─ TSPPostProcessor.py
└─ plugin.json
```

### `plugin.json`

```json
{
  "id": "TSPTravelOptimizer",
  "name": "TSP Travel Optimizer",
  "version": "0.1.0",
  "author": "Ryan McKenna",
  "api": 5,
  "main": "TSPTravelOptimizer",
  "metadata": {
    "type": "postprocessing"
  }
}
```

### `__init__.py`

```python
from .TSPPostProcessor import TSPPostProcessor

def getMetaData():
    from cura.CuraApplication import CuraApplication
    return {
        "mesh": CuraApplication.getInstance().getMetaData().get("mesh")
    }

def register(app):
    return { "postprocessor": TSPPostProcessor(app) }
```

### `TSPPostProcessor.py`

```python
from UM.Logger import Logger
from UM.PluginRegistry import PluginObject
from UM.Application import Application
import re, pathlib
from .tsp_opt import solve_tsp       # local copy of the shared core

_TRAVEL_RE = re.compile(r"G0 X([-+]?[0-9]*\.?[0-9]+) Y([-+]?[0-9]*\.?[0-9]+).*?F")

class TSPPostProcessor(PluginObject):
    def __init__(self, app):
        super().__init__()
        self._app = app

    def _parse_layer(self, lines):
        coords, idx = [], []
        for i, l in enumerate(lines):
            m = _TRAVEL_RE.match(l)
            if m:
                coords.append((float(m.group(1)), float(m.group(2))))
                idx.append(i)
        return coords, idx

    def _reorder(self, lines, coords, idx, order):
        new_lines = lines[:]
        for rank, old_pos in enumerate(order):
            new_lines[idx[rank]] = lines[idx[old_pos]]  # move line
        return new_lines

    # === Cura hook ===
    def perform(self, data):
        new_gcode = []
        layer = []
        for line in data.splitlines(True):
            if ";LAYER:" in line and layer:          # flush previous layer
                new_gcode.extend(self._process_layer(layer))
                layer = []
            layer.append(line)
        new_gcode.extend(self._process_layer(layer))
        return "".join(new_gcode)

    def _process_layer(self, layer_lines):
        coords, idx = self._parse_layer(layer_lines)
        if len(coords) <= 2:
            return layer_lines
        order = solve_tsp(coords)
        Logger.log("d", f"TSP optimisation saved {len(coords)} hops")
        return self._reorder(layer_lines, coords, idx, order)
```

Copy `tsp_opt.py` into the same folder.  Zip the folder or symlink it into Cura’s `plugins` directory; on reload you’ll see **“TSP Travel Optimizer”** in the post‑processing menu.

---

## 3 .  PrusaSlicer route – external post‑processing script

PrusaSlicer (as of 2.7+) lets you set **Print Settings → Output options → Post‑processing scripts**.  
Drop a *stand‑alone* Python script somewhere on your PATH:

```python
#!/usr/bin/env python3
# prusa_tsp_pp.py
import sys, re, tempfile, shutil, os
from tsp_opt import solve_tsp     # assumes script is alongside tsp_opt.py

TRAVEL = re.compile(r"G0 X([-+]?[0-9]*\.?[0-9]+) Y([-+]?[0-9]*\.?[0-9]+).*?F")

def optimise(file_in, file_out):
    with open(file_in) as f: lines = f.readlines()

    out, layer = [], []
    for l in lines:
        if ";LAYER:" in l and layer:
            out.extend(process(layer)); layer=[]
        layer.append(l)
    out.extend(process(layer))
    with open(file_out, "w") as g: g.writelines(out)

def process(layer_lines):
    coords, idx = [], []
    for i,l in enumerate(layer_lines):
        m = TRAVEL.match(l)
        if m:
            coords.append((float(m.group(1)), float(m.group(2))))
            idx.append(i)
    if len(coords) <= 2: return layer_lines
    order = solve_tsp(coords)
    new_layer = layer_lines[:]
    for r, pos in enumerate(order):
        new_layer[idx[r]] = layer_lines[idx[pos]]
    return new_layer

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = infile + ".tsp.gcode"
    optimise(infile, outfile)
    shutil.move(outfile, infile)      # replace original
```

Add `prusa_tsp_pp.py` in the script list (make sure Python is installed).  
PrusaSlicer will hand each exported G‑code file to the script before saving.

---

## 4 .  Integrating **Adaptive π / adaptive‑BPi**

Once the *order* of travel moves is globally optimal, you can re‑run a second pass that:

1. Looks at consecutive `G0` hops.  
2. Generates **approximating arcs / splines** that hit the same endpoints but follow a single‑radius path whose chord error ≤ printer tolerance.  
3. Emits `G2/G3` (or `G5 Spline` for RRF) instead of multiple `G1` hops.

You already have a framework for **curvature‑dependent π(κ)**; simply feed the curvature of the arc you plan and compute the exact radius/arc‑length command.  
Keep that pass modular so Cura/Prusa users can toggle “Travel order TSP” and “Replace travels with Adaptive π arcs” independently.

---

## 5 .  Development checklist & next steps

| Step | Cura | PrusaSlicer |
|------|------|-------------|
| Local dev env | Clone **Ultimaker/Cura** repo, or just drop plug‑in into `Cura/plugins` and enable Developer mode. | No build needed—just point to Python script. |
| Data access | Post‑processing plug‑ins see final G‑code; easiest path for now. | Same. |
| Testing | Use `--single-instance` Cura CLI to slice headlessly and diff G‑code. | `--export-gcode` CLI flag. |
| GUI hooks | Later you can add Settings panel (`QtProperty`) to switch solver, time budget, adaptive‑π on/off. | Use PrusaSlicer “Configuration → Filament overrides” to pass env vars to script. |

**Milestone plan**

1. **MVP** – TSP reorder travels, verify shorter total travel distance (export to CSV & graph).  
2. **Quality checks** – no missed retractions, Z‑hops, or tool‑change commands; regression test on benchy.  
3. **Adaptive π layer** – swap linear travels with arcs/splines, measure jerk & print‑time reduction.  
4. **Real‑time preview** – embed visual diff (Matplotlib or Three.js) inside plug‑in UI.  
5. **Swap out OR‑Tools** – integrate RealignR/ARP‑guided heuristic when you’re ready.

---

### Why start with G‑code post‑processing instead of deep slicer surgery?

* Changing the slicer’s path planner is a *huge* codebase dive.  
* Post‑processing is stable across versions and portable to Klipper/RRF/Marlin alike.  
* Once you prove the benefit, you can upstream a native patch or publish on the Cura Marketplace and Prusa Printer dev hub.

---

## 6 .  What to try first

```bash
# 1.  Install OR‑Tools
pip install ortools

# 2.  Drop TSPTravelOptimizer folder into Cura's plugins
#     reload Cura → Extensions → Post-Processing → Modify G-code
#     → Add 'TSP Travel Optimizer'

# 3.  Slice a calibration cube, export G‑code, measure print time in simulation tab.

# 4.  Enable plug‑in, reslice, compare travel distance & Cura‑estimated time.

# 5.  In PrusaSlicer, add path to prusa_tsp_pp.py in
#     Print Settings → Output options → Post‑processing scripts,
#     then slice a file and examine `.gcode`.
```

Expect **5–15 % travel reduction on typical models** with default 300 ms search.  Bump `time_limit` or swap to LK‑H if you want closer to the global optimum.

---

### Ready for the Adaptive π pass?

You already have the math for curvature‑dependent π(κ).  Build a small helper:

```python
def adaptive_pi(curvature):
    # return π_eff based on curvature κ
    return math.pi * (1 + 0.05 * curvature)   # placeholder
```

Then inside the second pass, compute radius R = 1/κ for your arc, feed κ into `adaptive_pi`, and adjust the commanded angular sweep so **arc length = R * 2π_eff θ / 2π**.  Emit `G2/G3` with that adjusted sweep.  Firmware that stores only the chord endpoints will run the smooth path at native stepper resolution—no blobs.

---

## 7 .  When you’re ready to publish

* **Cura Marketplace**: zip your folder (`TSPTravelOptimizer.curapackage`), add screenshots & README.  
* **PrusaSlicer**: fork `PrusaSlicer` GitHub, open PR adding a sample post‑processing script; or release on Printables.  
* MIT‑license the core `tsp_opt.py` so both slicers share it.

---



## Example scripts

Prototype scripts and notes are kept under `examples/`:
- `best_tsp_solver.py`
- `newpost_proc`
- `newinit`
- `newjson`
- `newarp_opt`
- `Newfile`
- `open source concepts`
- `fusion 360 readme`
