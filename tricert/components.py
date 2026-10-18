"""Triangle components and the compression theorem (Section 2 of the paper).

All linear algebra is over F_3 with exact integer arithmetic.
"""
import itertools
from collections import deque

__all__ = ["triangles", "triangle_components", "propagate", "F3_nullspace_dim", "compression_count",
           "brute_force_count", "every_edge_in_triangle", "component_colourable"]


def triangles(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    out = []
    for u, v in edges:
        a, b = min(u, v), max(u, v)
        for w in adj[a] & adj[b]:
            if w > b:
                out.append((a, b, w))
    return sorted(out)


def every_edge_in_triangle(n, edges):
    T = triangles(n, edges)
    covered = {tuple(sorted(p)) for t in T for p in itertools.combinations(t, 2)}
    return all(tuple(sorted(e)) in covered for e in edges)


def triangle_components(n, edges):
    """Lists of triangles; two triangles are adjacent when they share an edge."""
    T = triangles(n, edges)
    by_edge = {}
    for i, t in enumerate(T):
        for p in itertools.combinations(t, 2):
            by_edge.setdefault(p, []).append(i)
    parent = list(range(len(T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for lst in by_edge.values():
        for i in lst[1:]:
            ri, r0 = find(i), find(lst[0])
            if ri != r0:
                parent[ri] = r0
    comps = {}
    for i in range(len(T)):
        comps.setdefault(find(i), []).append(T[i])
    return list(comps.values())


def propagate(component):
    """Theorem 2.2(c): colour the first triangle 0,1,2 and propagate c_w = -c_u - c_v along shared edges.
    Returns (colouring dict or None, conflict vertex or None)."""
    by_edge = {}
    for t in component:
        for p in itertools.combinations(t, 2):
            by_edge.setdefault(p, []).append(t)
    t0 = component[0]
    col = {t0[0]: 0, t0[1]: 1, t0[2]: 2}
    seen = {t0}
    queue = deque([t0])
    while queue:
        t = queue.popleft()
        for p in itertools.combinations(t, 2):
            for s in by_edge[p]:
                if s in seen:
                    continue
                seen.add(s)
                u, v = p
                w = next(x for x in s if x not in p)
                val = (-col[u] - col[v]) % 3
                if w in col and col[w] != val:
                    return None, w
                col[w] = val
                queue.append(s)
    return col, None


def component_colourable(component):
    return propagate(component)[0] is not None


def _rank_mod3(rows, ncols):
    rows = [[x % 3 for x in r] for r in rows]
    rank, col = 0, 0
    while rank < len(rows) and col < ncols:
        piv = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if piv is None:
            col += 1
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = 1 if rows[rank][col] == 1 else 2
        rows[rank] = [(x * inv) % 3 for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][col]:
                f = rows[i][col]
                rows[i] = [(a - f * b) % 3 for a, b in zip(rows[i], rows[rank])]
        rank += 1
        col += 1
    return rank


def F3_nullspace_dim(n, rows):
    return n - _rank_mod3(rows, n) if rows else n


def compression_count(n, edges):
    """Theorem 2.4(b): P(G,3) = sum_S (-1)^|S| 3^dim(H_S), for graphs in which every edge lies in a triangle.
    Returns (P(G,3), k, dims) where dims[i] = dim H_i and dim L = dims[-1]."""
    assert every_edge_in_triangle(n, edges)
    comps = triangle_components(n, edges)
    base = []
    for (u, v, w) in triangles(n, edges):
        r = [0] * n
        r[u] = r[v] = r[w] = 1
        base.append(r)
    const_rows = []
    for K in comps:
        verts = sorted({x for t in K for x in t})
        rows = []
        for a in verts[1:]:
            r = [0] * n
            r[verts[0]] = 1
            r[a] = -1
            rows.append(r)
        const_rows.append(rows)
    k = len(comps)
    total = 0
    for mask in range(1 << k):
        rows = list(base)
        for i in range(k):
            if mask >> i & 1:
                rows += const_rows[i]
        total += (-1) ** bin(mask).count("1") * 3 ** F3_nullspace_dim(n, rows)
    dims = [F3_nullspace_dim(n, base + const_rows[i]) for i in range(k)] + [F3_nullspace_dim(n, base)]
    return total, k, dims


def brute_force_count(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    col = [-1] * n

    def rec(v):
        if v == n:
            return 1
        tot = 0
        for c in range(3):
            if all(col[w] != c for w in adj[v] if w < v):
                col[v] = c
                tot += rec(v + 1)
        col[v] = -1
        return tot
    return rec(0)
