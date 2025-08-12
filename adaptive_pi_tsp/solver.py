
import numpy as np
import matplotlib.pyplot as plt
from .fields import CurvatureField, Obstacles

def edge_hits_obstacle(p, q, obstacles: Obstacles, steps=24):
    vec = q - p
    ts = np.linspace(0,1,steps)
    xs = p[0] + ts*vec[0]
    ys = p[1] + ts*vec[1]
    for xi, yi in zip(xs, ys):
        if obstacles.contains(xi, yi):
            return True
    return False

def segment_cost(p, q, field: CurvatureField, obstacles: Obstacles, r=0.1, samples=24):
    if edge_hits_obstacle(p, q, obstacles, steps=16):
        return np.inf
    vec = q - p
    L = np.linalg.norm(vec)
    if L == 0:
        return 0.0
    ts = np.linspace(0, 1, samples)
    xs = p[0] + ts * vec[0]
    ys = p[1] + ts * vec[1]
    phi_vals = np.array([field.phi(x, y, r=r) for x,y in zip(xs, ys)])
    return L * np.mean(phi_vals)

def tsp_cost(order, pts, field: CurvatureField, obstacles: Obstacles, r=0.1, samples=24, euclid=False):
    total = 0.0
    for i in range(len(order)):
        a = pts[order[i]]
        b = pts[order[(i+1)%len(order)]]
        if euclid:
            c = np.linalg.norm(a-b)
        else:
            c = segment_cost(a, b, field, obstacles, r=r, samples=samples)
        if not np.isfinite(c):
            return np.inf
        total += c
    return total

def two_opt(order, pts, field, obstacles, r=0.1, max_iter=120, samples=24, euclid=False):
    it = 0
    while it < max_iter:
        improved = False
        it += 1
        N = len(order)
        for i in range(N - 1):
            for k in range(i + 2, N if i > 0 else N - 1):
                a, b = order[i], order[i+1]
                c, d = order[k], order[(k+1) % N]
                if euclid:
                    cost_ab = np.linalg.norm(pts[a]-pts[b]); cost_cd = np.linalg.norm(pts[c]-pts[d])
                    cost_ac = np.linalg.norm(pts[a]-pts[c]); cost_bd = np.linalg.norm(pts[b]-pts[d])
                else:
                    if edge_hits_obstacle(pts[a], pts[c], obstacles) or edge_hits_obstacle(pts[b], pts[d], obstacles):
                        continue
                    cost_ab = segment_cost(pts[a], pts[b], field, obstacles, r=r, samples=samples)
                    cost_cd = segment_cost(pts[c], pts[d], field, obstacles, r=r, samples=samples)
                    cost_ac = segment_cost(pts[a], pts[c], field, obstacles, r=r, samples=samples)
                    cost_bd = segment_cost(pts[b], pts[d], field, obstacles, r=r, samples=samples)
                if not (np.isfinite(cost_ab) and np.isfinite(cost_cd) and np.isfinite(cost_ac) and np.isfinite(cost_bd)):
                    continue
                if cost_ac + cost_bd + 1e-12 < cost_ab + cost_cd:
                    order[i+1:k+1] = reversed(order[i+1:k+1])
                    improved = True
        if not improved:
            break
    return order

def obstacle_aware_seed(pts, field, obstacles, r=0.1):
    N = len(pts)
    unvisited = set(range(N))
    start = 0
    order = [start]
    unvisited.remove(start)
    curr = start
    while unvisited:
        choices = []
        for j in list(unvisited):
            c = segment_cost(pts[curr], pts[j], field, obstacles, r=r, samples=16)
            if np.isfinite(c):
                choices.append((c, j))
        if not choices:
            # random hop
            j = list(unvisited)[0]
            order.append(j); unvisited.remove(j); curr = j
        else:
            choices.sort()
            nxt = choices[0][1]
            order.append(nxt); unvisited.remove(nxt); curr = nxt
    return order

def repair(order, pts, field, obstacles, r=0.1, steps=200):
    def edges(ordr):
        return [(ordr[i], ordr[(i+1)%len(ordr)]) for i in range(len(ordr))]
    for _ in range(steps):
        bad = [(i,j) for (i,j) in edges(order) if edge_hits_obstacle(pts[i], pts[j], obstacles)]
        if not bad:
            return order
        a,b = bad[0]
        ia = order.index(a)
        # try reconnecting with nearest candidate
        N = len(order)
        improved = False
        for k in range(1, N-2):
            i = (ia + k) % N; j = (i + 1) % N
            c, d = order[i], order[j]
            if edge_hits_obstacle(pts[a], pts[c], obstacles) or edge_hits_obstacle(pts[b], pts[d], obstacles):
                continue
            old = segment_cost(pts[a], pts[b], field, obstacles, r=r) + segment_cost(pts[c], pts[d], field, obstacles, r=r)
            new = segment_cost(pts[a], pts[c], field, obstacles, r=r) + segment_cost(pts[b], pts[d], field, obstacles, r=r)
            if np.isfinite(new) and new + 1e-12 < old:
                ia = order.index(a); ic = order.index(c)
                i1 = (ia+1)%N; i2 = ic
                if i1 < i2:
                    order[i1:i2+1] = list(reversed(order[i1:i2+1]))
                else:
                    seg = list(reversed(order[i1:] + order[:i2+1]))
                    order = order[:i1] + seg[:len(order)-i1] + seg[len(order)-i1:]
                improved = True; break
        if not improved:
            order = order[1:] + order[:1]  # rotate
    return order

def tsp_solve(points, r=0.1, field=None, obstacles=None, max_iter=150):
    field = field or CurvatureField.default()
    obstacles = obstacles or Obstacles.default()
    order = obstacle_aware_seed(points, field, obstacles, r=r)
    order = repair(order, points, field, obstacles, r=r, steps=300)
    order = two_opt(order, points, field, obstacles, r=r, max_iter=max_iter, samples=24, euclid=False)
    cost = tsp_cost(order, points, field, obstacles, r=r, samples=24, euclid=False)
    return order, cost

def tsp_plot(points, order, field=None, obstacles=None, r=0.1, res=200, title=None):
    field = field or CurvatureField.default(); obstacles = obstacles or Obstacles.default()
    xs = np.linspace(0,1,res); ys = np.linspace(0,1,res)
    X,Y = np.meshgrid(xs,ys)
    K_grid = np.zeros_like(X)
    for i in range(res):
        for j in range(res):
            K_grid[i,j] = field.K(X[i,j], Y[i,j])
    plt.figure()
    plt.imshow(K_grid.T, origin='lower', extent=[0,1,0,1], aspect='equal')
    # draw obstacles
    tt = np.linspace(0, 2*np.pi, 200)
    for (cx, cy, ax, ay, th) in obstacles.ellipses:
        c, s = np.cos(th), np.sin(th)
        xw = cx + ax*np.cos(tt)*c - ay*np.sin(tt)*s
        yw = cy + ax*np.cos(tt)*s + ay*np.sin(tt)*c
        plt.plot(xw, yw)
    path = np.array([points[i] for i in order] + [points[order[0]]])
    plt.plot(path[:,0], path[:,1])
    plt.scatter(points[:,0], points[:,1], s=12)
    ttl = title or f"Adaptive-π TSP (φ=π/πₐ, r={r})"
    plt.title(ttl); plt.xlabel("x"); plt.ylabel("y")
    plt.show()
