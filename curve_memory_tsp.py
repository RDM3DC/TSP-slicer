import numpy as np
from typing import Callable, List, Tuple

class CurveMemoryTSPSolver:
    def __init__(self,
                 points: np.ndarray,
                 memory_weight: float = 0.5,
                 entropy_weight: float = 0.2,
                 cost_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = None):
        """Memory-aware TSP solver.

        Args:
            points: (N,2) array of XY coordinates
            memory_weight: penalty for turning angles (0 = classic TSP)
            entropy_weight: penalty for path entropy
            cost_fn: custom cost function. Defaults to curve/entropy-aware cost.
        """
        self.points = points
        self.N = points.shape[0]
        self.memory_weight = memory_weight
        self.entropy_weight = entropy_weight
        self.cost_fn = cost_fn if cost_fn else self.default_cost_fn

    def default_cost_fn(self, prev, curr, nextp):
        """Adaptive cost combining distance, curvature, and entropy."""
        d = np.linalg.norm(curr - nextp)
        if prev is None:
            turn_penalty = 0.0
        else:
            v1 = curr - prev
            v2 = nextp - curr
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            turn_penalty = angle / np.pi
        entropy = 0.0
        if prev is not None:
            avg_vec = (v1 + v2) / 2
            deviation = np.linalg.norm(
                v2 / (np.linalg.norm(v2) + 1e-8) - avg_vec / (np.linalg.norm(avg_vec) + 1e-8)
            )
            entropy = deviation
        return d + self.memory_weight * turn_penalty + self.entropy_weight * entropy

    def solve(self, start_idx: int = 0) -> Tuple[List[int], float]:
        """Greedy memory-aware solver.

        Returns:
            path indices and total cost.
        """
        path = [start_idx]
        unused = set(range(self.N))
        unused.remove(start_idx)
        total_cost = 0.0
        prev = None
        curr_idx = start_idx
        while unused:
            costs = []
            for j in unused:
                cost = self.cost_fn(
                    prev=self.points[path[-2]] if len(path) > 1 else None,
                    curr=self.points[curr_idx],
                    nextp=self.points[j],
                )
                costs.append((cost, j))
            min_cost, next_idx = min(costs)
            path.append(next_idx)
            total_cost += min_cost
            unused.remove(next_idx)
            prev = self.points[curr_idx]
            curr_idx = next_idx
        total_cost += self.cost_fn(
            prev=self.points[path[-2]],
            curr=self.points[path[-1]],
            nextp=self.points[path[0]],
        )
        path.append(path[0])
        return path, total_cost

    def as_gcode(self, path: List[int], feedrate: float = 1200.0) -> str:
        """Return the path as G-code commands."""
        gcode_lines = []
        for idx in path:
            x, y = self.points[idx]
            gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feedrate}")
        return "\n".join(gcode_lines)

if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(20, 2) * 100
    solver = CurveMemoryTSPSolver(points, memory_weight=0.7, entropy_weight=0.3)
    path, cost = solver.solve()
    print("TSP Path:", path)
    print("Total Cost:", cost)
    print("\nG-code:\n", solver.as_gcode(path))
