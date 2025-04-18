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
