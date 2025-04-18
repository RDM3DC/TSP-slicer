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
