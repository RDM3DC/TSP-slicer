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
