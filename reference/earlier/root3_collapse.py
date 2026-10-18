#!/usr/bin/env python3
"""
root3_collapse.py -- Root-3 Complex Geography and the Boundaries of Colorability
================================================================================

Standalone computational companion (imports nothing from any other project package).

    python root3_collapse.py            full report  (logs/report.txt is produced by redirection)
    python root3_collapse.py --quick    reduced report
    python -m pytest -q test_root3_collapse.py

Dependencies: Python >= 3.9, sympy >= 1.12, numpy.

Conventions
-----------
Omega_3 = {1, w, w^2}, w = exp(2 pi i / 3).  A colour is stored in log coordinates c in Z/3, a = w^c.
Root-3 (Bayer) ideal   I_G = < x_v^3 - 1  (v in V),   q_uv = x_u^2 + x_u x_v + x_v^2  (uv in E) >.
R = K[x] / < x_v^3 - 1 >  has the monomial basis x^e, e in {0,1,2}^n  (monomials are stored sparsely
as sorted tuples ((v, e_v), ...) with e_v in {1, 2}).
NS degree (proof-complexity convention): least d such that
        1 = sum_v alpha_v (x_v^3 - 1) + sum_e beta_e q_e,   deg(alpha_v) + 3 <= d,  deg(beta_e) + 2 <= d.
It equals the least d with 1 = sum_e beta_e q_e in R and deg(beta_e) + 2 <= d (draft, Lemma 4.1), which is
what `ns_degree` computes, by modular linear algebra, exact rational reconstruction and exact verification.

Parts
-----
  A  graphs, generators, gadgets, exact 3-colouring by backtracking
  B  the Root-3 encoding, the Vieta lemma, the Root-3 variety (Groebner with time limits)
  C  triangle propagation: Vieta transport, holonomy in Sym(Omega_3) = D_3, pinches, GF(3) collapse,
     the component-hyperplane count (inclusion-exclusion), the NP-hardness gadget G -> G^natural
  D  Nullstellensatz: exact degree-bounded certificate search, dual designs, the constant-degree
     certificate of the propagating regime, the degree-preserving substitution for G^natural
  E  rank-2 Gram matrices and Kneser witnesses
  F  directed vertex geography: resolvent recursion, outcomes, set potentials, back-arc cocycle,
     matching potentials by Groebner/Rabinowitsch saturation with hard time limits, widget suite
  G  monodromy of the triangle rescues: Chebyshev cubic, environment sextic, Y_3
  H  report driver
"""
from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as _mp
import random
import sys
import time
from collections import defaultdict, deque
from fractions import Fraction
from functools import lru_cache

import numpy as np
import sympy as sp

sys.setrecursionlimit(100000)

# =====================================================================================================
# Part A.  Graphs
# =====================================================================================================


class Graph:
    """Simple undirected graph on {0, ..., n-1}."""

    def __init__(self, n, edges, name=""):
        self.n = int(n)
        self.edges = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
        self.adj = [set() for _ in range(self.n)]
        for a, b in self.edges:
            self.adj[a].add(b)
            self.adj[b].add(a)
        self.name = name

    def __repr__(self):
        return "Graph(%s, n=%d, m=%d)" % (self.name, self.n, len(self.edges))

    def triangles(self):
        out = []
        for a, b in self.edges:
            for c in self.adj[a] & self.adj[b]:
                if c > b:
                    out.append((a, b, c))
        return out

    def every_edge_in_triangle(self):
        cov = set()
        for a, b, c in self.triangles():
            cov |= {(a, b), (a, c), (b, c)}
        return all(e in cov for e in self.edges)


def colour_search(G, count_all=True):
    """Proper 3-colourings in log coordinates, by backtracking with forward checking.
    Returns (number of colourings (all of them if count_all, else 0/1), one colouring or None)."""
    n, adj = G.n, [sorted(s) for s in G.adj]
    dom, col = [7] * n, [-1] * n
    state = {"count": 0, "first": None, "stop": False}

    def pick():
        best, bd, bdeg = -1, 9, -1
        for v in range(n):
            if col[v] < 0:
                d = (dom[v] & 1) + (dom[v] >> 1 & 1) + (dom[v] >> 2 & 1)
                if d < bd or (d == bd and len(adj[v]) > bdeg):
                    best, bd, bdeg = v, d, len(adj[v])
        return best

    def rec(k):
        if k == n:
            state["count"] += 1
            if state["first"] is None:
                state["first"] = col[:]
            if not count_all:
                state["stop"] = True
            return
        v = pick()
        for c in range(3):
            if state["stop"]:
                return
            if not dom[v] >> c & 1:
                continue
            changed, ok = [], True
            for w in adj[v]:
                if col[w] < 0 and dom[w] >> c & 1:
                    dom[w] &= ~(1 << c)
                    changed.append(w)
                    if dom[w] == 0:
                        ok = False
            if ok:
                col[v] = c
                rec(k + 1)
                col[v] = -1
            for w in changed:
                dom[w] |= 1 << c

    rec(0)
    return state["count"], state["first"]


def is_3colourable(G):
    return colour_search(G, count_all=False)[0] > 0


def complete(n):
    return Graph(n, itertools.combinations(range(n), 2), "K%d" % n)


def cycle_graph(n):
    return Graph(n, [(i, (i + 1) % n) for i in range(n)], "C%d" % n)


def wheel(k):
    """Hub 0 joined to the rim cycle 1..k (odd k: not 3-colourable)."""
    E = [(0, i) for i in range(1, k + 1)] + [(i, i % k + 1) for i in range(1, k + 1)]
    return Graph(k + 1, E, "W%d" % k)


def moser_spindle():
    E = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (0, 4), (0, 5), (4, 5), (4, 6), (5, 6), (3, 6)]
    return Graph(7, E, "Moser")


def octahedron():
    return Graph(6, [(a, b) for a, b in itertools.combinations(range(6), 2) if b != a ^ 1], "Octahedron")


def icosahedron():
    E = []
    up, lo = [1 + i for i in range(5)], [6 + i for i in range(5)]
    for i in range(5):
        E += [(0, up[i]), (up[i], up[(i + 1) % 5]), (up[i], lo[i]), (up[i], lo[(i + 1) % 5]),
              (lo[i], lo[(i + 1) % 5]), (11, lo[i])]
    return Graph(12, E, "Icosahedron")


def triangular_torus(p, q):
    """Quotient of the triangular lattice (edges (1,0),(0,1),(1,1)) by pZ x qZ; p, q >= 4."""
    idx = lambda i, j: (i % p) * q + (j % q)
    E = []
    for i in range(p):
        for j in range(q):
            E += [(idx(i, j), idx(i + 1, j)), (idx(i, j), idx(i, j + 1)), (idx(i, j), idx(i + 1, j + 1))]
    return Graph(p * q, E, "T(%d,%d)" % (p, q))


def groetzsch():
    E = [(i, (i + 1) % 5) for i in range(5)]
    E += [(5 + i, (i - 1) % 5) for i in range(5)] + [(5 + i, (i + 1) % 5) for i in range(5)]
    E += [(10, 5 + i) for i in range(5)]
    return Graph(11, E, "Groetzsch")


def kneser(n, k):
    V = list(itertools.combinations(range(n), k))
    E = [(i, j) for i, j in itertools.combinations(range(len(V)), 2) if not set(V[i]) & set(V[j])]
    return Graph(len(V), E, "KG(%d,%d)" % (n, k)), V


def random_triangle_graph(n, t, rng):
    """Union of t random triangles on n vertices: every edge lies in a triangle."""
    E = []
    for _ in range(t):
        a, b, c = rng.sample(range(n), 3)
        E += [(a, b), (b, c), (a, c)]
    return Graph(n, E, "rand_tri(%d,%d)" % (n, t))


def strip_gadget(G):
    """G -> G^natural: every edge uv of G is replaced by the triangles {u,p,q}, {p,q,s}, {s,v,t}.
    chi(G^nat) <= 3 iff chi(G) <= 3; every edge of G^nat lies in a triangle; its triangle components
    are the strips {u,p,q}+{p,q,s} and the single triangles {s,v,t} (2|E(G)| components, all trees)."""
    E, nxt, labels = [], G.n, {}
    for (u, v) in G.edges:
        p, q, s, t = nxt, nxt + 1, nxt + 2, nxt + 3
        nxt += 4
        labels[(u, v)] = (p, q, s, t)
        E += [(u, p), (u, q), (p, q), (p, s), (q, s), (s, v), (s, t), (v, t)]
    H = Graph(nxt, E, "%s^nat" % G.name)
    H.gadget_labels = labels
    return H


# =====================================================================================================
# Part B.  The Root-3 encoding
# =====================================================================================================

ZW = {0: (1, 0), 1: (0, 1), 2: (-1, -1)}  # w^c as a + b w  (w^2 = -1 - w)


def zw_add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def zw_mul(x, y):
    a, b = x
    c, d = y
    return (a * c - b * d, a * d + b * c - b * d)


def mono(*pairs):
    d = {}
    for v, e in pairs:
        k = (d.get(v, 0) + e) % 3
        if k:
            d[v] = k
        else:
            d.pop(v, None)
    return tuple(sorted(d.items()))


def m_mul(m1, m2):
    d = dict(m1)
    for v, e in m2:
        k = (d.get(v, 0) + e) % 3
        if k:
            d[v] = k
        else:
            d.pop(v, None)
    return tuple(sorted(d.items()))


def m_deg(m):
    return sum(e for _, e in m)


def q_poly(u, v):
    """q_uv in R (coefficients 1)."""
    return {mono((u, 2)): 1, mono((u, 1), (v, 1)): 1, mono((v, 2)): 1}


def eval_R(P, colour):
    """Value of P in Z[w] at a_v = w^{colour[v]} (exact)."""
    s = (0, 0)
    for m, c in P.items():
        k = sum(e * colour[v] for v, e in m) % 3
        a, b = ZW[k]
        s = (s[0] + c * a, s[1] + c * b)
    return s


def vieta_lemma_check():
    """Lemma 1.2 on all 27 points of Omega_3^3: proper <=> e1 = 0 <=> (e1,e2,e3) = (0,0,1) <=> all q vanish."""
    ok = True
    for c in itertools.product(range(3), repeat=3):
        w = [ZW[k] for k in c]
        e1 = zw_add(zw_add(w[0], w[1]), w[2])
        e2 = zw_add(zw_add(zw_mul(w[0], w[1]), zw_mul(w[1], w[2])), zw_mul(w[0], w[2]))
        e3 = zw_mul(zw_mul(w[0], w[1]), w[2])
        proper = len(set(c)) == 3
        qs = all(eval_R(q_poly(i, j), dict(enumerate(c))) == (0, 0) for i, j in [(0, 1), (0, 2), (1, 2)])
        ok &= (proper == (e1 == (0, 0)) == (e1 == (0, 0) and e2 == (0, 0) and e3 == (1, 0)) == qs)
    return ok


def root3_ideal_exprs(G):
    xs = sp.symbols("x0:%d" % G.n)
    gens = [x ** 3 - 1 for x in xs] + [xs[u] ** 2 + xs[u] * xs[v] + xs[v] ** 2 for u, v in G.edges]
    return xs, gens


# ----------------------------------------------------------------------------- time-limited subprocess
def run_with_timeout(func, args=(), timeout=60):
    """Run func(*args) in a forked process; return ('OK', value) | ('TIMEOUT', None) | ('ERROR', msg).
    value must be picklable.  This is the guard against Groebner computations that do not terminate
    in reasonable time."""
    ctx = _mp.get_context("fork")
    q = ctx.Queue()

    def worker():
        try:
            q.put(("OK", func(*args)))
        except Exception as exc:  # pragma: no cover
            q.put(("ERROR", repr(exc)))

    p = ctx.Process(target=worker)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return ("TIMEOUT", None)
    try:
        return q.get(timeout=5)
    except Exception:
        return ("ERROR", "no result")


def _variety_job(G):
    xs, gens = root3_ideal_exprs(G)
    GB = sp.groebner(gens, *xs, order="grevlex", domain=sp.QQ)
    if list(GB.exprs) == [1]:
        return {"unit": True, "points": 0}
    leads = [sp.Poly(g, *xs).monoms(order="grevlex")[0] for g in GB.exprs]
    cnt = 0
    for e in itertools.product(range(3), repeat=G.n):  # x_v^3 is a leading term: exponents <= 2
        if not any(all(e[i] >= L[i] for i in range(G.n)) for L in leads):
            cnt += 1
    return {"unit": False, "points": cnt, "basis_size": len(GB.exprs)}


def variety_report(G, timeout=60):
    """Groebner basis of I_G over Q (time-limited).  I_G is radical and zero-dimensional, so the number of
    standard monomials equals |V(I_G)| = P(G,3); the points themselves are the colourings (Omega_3-valued),
    extracted by backtracking and checked against every generator in Z[w]."""
    status, val = run_with_timeout(_variety_job, (G,), timeout)
    count, first = colour_search(G)
    pts_ok = True
    if first is not None:
        col = dict(enumerate(first))
        pts_ok = all(eval_R(q_poly(u, v), col) == (0, 0) for u, v in G.edges)
    return {"status": status, "groebner": val, "P(G,3)": count, "sample_point_ok": pts_ok}


def vieta_ideal_equals_triangle_ideal(timeout=60):
    def job():
        x0, x1, x2 = sp.symbols("x0:3")
        I1 = sp.groebner([x0 ** 3 - 1, x1 ** 3 - 1, x2 ** 3 - 1, x0 ** 2 + x0 * x1 + x1 ** 2,
                          x0 ** 2 + x0 * x2 + x2 ** 2, x1 ** 2 + x1 * x2 + x2 ** 2], x0, x1, x2, order="grevlex")
        I2 = sp.groebner([x0 + x1 + x2, x0 * x1 + x1 * x2 + x0 * x2, x0 * x1 * x2 - 1], x0, x1, x2, order="grevlex")
        return sorted(map(str, I1.exprs)) == sorted(map(str, I2.exprs))
    return run_with_timeout(job, (), timeout)


# =====================================================================================================
# Part C.  Triangle propagation, holonomy, GF(3) collapse, component hyperplanes
# =====================================================================================================


def triangle_components(G):
    """Connected components of the triangle graph (triangles adjacent iff they share an edge)."""
    T = G.triangles()
    by_edge = defaultdict(list)
    for i, (a, b, c) in enumerate(T):
        for e in ((a, b), (a, c), (b, c)):
            by_edge[e].append(i)
    parent = list(range(len(T)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lst in by_edge.values():
        for j in lst[1:]:
            parent[find(j)] = find(lst[0])
    comps = defaultdict(list)
    for i in range(len(T)):
        comps[find(i)].append(T[i])
    return list(comps.values())


def _edges_of(t):
    a, b, c = t
    return ((a, b), (a, c), (b, c))


def _third(t, e):
    return next(x for x in t if x not in e)


def propagate(G, comp):
    """Vieta transport a_w = -a_u - a_v (log: c_w = -c_u - c_v) along shared edges, from the base colouring
    (0,1,2).  Returns (colouring dict or None, first conflict or None)."""
    by_edge = defaultdict(list)
    for t in comp:
        for e in _edges_of(t):
            by_edge[e].append(t)
    t0 = comp[0]
    col = {t0[0]: 0, t0[1]: 1, t0[2]: 2}
    done, queue, conflict = {t0}, deque([t0]), None
    while queue:
        t = queue.popleft()
        for e in _edges_of(t):
            for t2 in by_edge[e]:
                if t2 in done:
                    continue
                done.add(t2)
                w = _third(t2, e)
                cw = (-col[e[0]] - col[e[1]]) % 3
                if w in col and col[w] != cw and conflict is None:
                    conflict = (w, col[w], cw, t2)
                col.setdefault(w, cw)
                queue.append(t2)
    return (None if conflict else col), conflict


def _compose(s, t):  # (s o t)(c) = s[t[c]]
    return tuple(s[t[c]] for c in range(3))


def _closure(gens):
    G = {(0, 1, 2)}
    frontier = list(G)
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                k = _compose(h, g)
                if k not in G:
                    G.add(k)
                    new.append(k)
        frontier = new
    return G


def group_name(H):
    if len(H) == 1:
        return "1"
    if len(H) == 2:
        return "C2 (reflection)"
    if len(H) == 3:
        return "C3 = mu_3 (rotations x -> w x)"
    return "S3 = D3"


def holonomy(G, comp):
    """Holonomy of the principal Sym(Omega_3)-bundle of triangle colourings over the triangle graph of comp.
    Fibres are transported along a BFS tree; every non-tree adjacency yields sigma with T(c_t) = sigma o c_t2.
    Returns (holonomy group as set of permutations of Z/3, number of pinch defects).  Theorem 3.1: comp is
    colourable iff the group is trivial and there is no pinch defect."""
    by_edge = defaultdict(list)
    for t in comp:
        for e in _edges_of(t):
            by_edge[e].append(t)
    t0 = comp[0]
    ct = {t0: {t0[0]: 0, t0[1]: 1, t0[2]: 2}}
    queue, gens = deque([t0]), set()
    tree = set()
    while queue:
        t = queue.popleft()
        for e in _edges_of(t):
            for t2 in by_edge[e]:
                if t2 == t:
                    continue
                w = _third(t2, e)
                T = {e[0]: ct[t][e[0]], e[1]: ct[t][e[1]], w: (-ct[t][e[0]] - ct[t][e[1]]) % 3}
                if t2 not in ct:
                    ct[t2] = T
                    tree.add(frozenset((t, t2)))
                    queue.append(t2)
                elif frozenset((t, t2)) not in tree:
                    sigma = [None] * 3
                    for x in t2:
                        sigma[ct[t2][x]] = T[x]
                    gens.add(tuple(sigma))
    H = _closure(gens)
    pinches = 0
    if len(H) == 1:
        at = defaultdict(set)
        for t, c in ct.items():
            for x, v in c.items():
                at[x].add(v)
        pinches = sum(1 for x in at if len(at[x]) > 1)
    return H, pinches


def rref_mod(rows, ncols, p):
    """Reduced row echelon form mod p of a list of integer rows; returns (rows, pivot columns)."""
    M = [[x % p for x in r] for r in rows]
    piv, r = [], 0
    for c in range(ncols):
        k = next((i for i in range(r, len(M)) if M[i][c]), None)
        if k is None:
            continue
        M[r], M[k] = M[k], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [(x * inv) % p for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % p for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return M[:r], piv


def nullity_mod(rows, ncols, p):
    return ncols - len(rref_mod(rows, ncols, p)[1]) if rows else ncols


def f3_rows(G, triangles=None):
    rows = []
    for (a, b, c) in (G.triangles() if triangles is None else triangles):
        r = [0] * G.n
        r[a] = r[b] = r[c] = 1
        rows.append(r)
    return rows


def f3_component_dimension(G, comp):
    """dim_F3 of the solutions of {c_u + c_v + c_w = 0 : uvw in comp} restricted to V(comp): 2 iff colourable."""
    V = sorted({x for t in comp for x in t})
    pos = {x: i for i, x in enumerate(V)}
    rows = []
    for (a, b, c) in comp:
        r = [0] * len(V)
        r[pos[a]] = r[pos[b]] = r[pos[c]] = 1
        rows.append(r)
    return nullity_mod(rows, len(V), 3)


def hyperplane_count(G, max_k=18):
    """Theorem 3.2.  For triangle-saturated G with triangle components K_1..K_k:
         P(G,3) = sum_{S subset [k]} (-1)^|S| 3^{dim H_S},   H_S = {c in L : c constant on V(K_i), i in S},
    L = {c in F_3^V : c_u + c_v + c_w = 0 for every triangle}.  Returns (P(G,3), k, decision by the covering
    lemma when k <= 3 (else None))."""
    assert G.every_edge_in_triangle(), "Theorem 3.2 needs every edge in a triangle"
    comps = triangle_components(G)
    k = len(comps)
    if k > max_k:
        raise ValueError("k = %d components: inclusion-exclusion capped at %d" % (k, max_k))
    base = f3_rows(G)
    const_rows = []
    for comp in comps:
        V = sorted({x for t in comp for x in t})
        rs = []
        for x in V[1:]:
            r = [0] * G.n
            r[x], r[V[0]] = 1, 2  # c_x - c_{V0} = 0 mod 3
            rs.append(r)
        const_rows.append(rs)
    total = 0
    for S in itertools.product((0, 1), repeat=k):
        rows = base + [r for i in range(k) if S[i] for r in const_rows[i]]
        total += (-1) ** sum(S) * 3 ** nullity_mod(rows, G.n, 3)
    decision = None
    if k <= 3:
        dimL = nullity_mod(base, G.n, 3)
        decision = all(nullity_mod(base + const_rows[i], G.n, 3) < dimL for i in range(k))
    return total, k, decision


# =====================================================================================================
# Part D.  Nullstellensatz certificates
# =====================================================================================================

PRIMES = [2147483647, 2147483629, 2147483587, 2147483579, 2147483563, 2147483549, 2147483543, 2147483497]


def reduced_monomials(n, deg):
    """All x^e with e in {0,1,2}^n and |e| = deg (sparse tuples)."""
    out = []
    for e in itertools.product(range(3), repeat=n):
        if sum(e) == deg:
            out.append(tuple((v, k) for v, k in enumerate(e) if k))
    return out


def ns_columns(G, d, grade):
    """Columns x^e q_uv (in R) of nominal degree |e| + 2 == d, of the given Z/3-grade."""
    cols, labels = [], []
    if (d - grade) % 3:
        return cols, labels
    for e in reduced_monomials(G.n, d - 2):
        for (u, v) in G.edges:
            col = {}
            for m, c in q_poly(u, v).items():
                mm = m_mul(e, m)
                col[mm] = col.get(mm, 0) + c
            cols.append(col)
            labels.append(((u, v), e))
    return cols, labels


def _dense_solve(cols, target, p):
    """Over GF(p): decide target in span(cols).  Returns ('IN', coeffs on columns) or ('OUT', y on rows)
    with y.col = 0 for every column and y.target != 0 (normalised to 1)."""
    keys = sorted(set().union(*[c.keys() for c in cols], target.keys()))
    pos = {k: i for i, k in enumerate(keys)}
    Rn, Cn = len(keys), len(cols)
    M = np.zeros((Rn, Cn + 1 + Rn), dtype=np.int64)
    for j, col in enumerate(cols):
        for k, c in col.items():
            M[pos[k], j] = c % p
    for k, c in target.items():
        M[pos[k], Cn] = c % p
    M[np.arange(Rn), Cn + 1 + np.arange(Rn)] = 1
    r, piv = 0, []
    for c in range(Cn):
        if r == Rn:
            break
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        inv = pow(int(M[r, c]), p - 2, p) if p > 2 else 1
        M[r] = (M[r] * inv) % p
        colv = M[:, c].copy()
        colv[r] = 0
        nzr = np.nonzero(colv)[0]
        if nzr.size:
            M[nzr] = (M[nzr] - (colv[nzr, None] * M[r][None, :]) % p) % p
        piv.append(c)
        r += 1
    bad = np.nonzero(M[r:, Cn])[0]
    if bad.size:
        i = r + int(bad[0])
        inv = pow(int(M[i, Cn]), p - 2, p) if p > 2 else 1
        y = (M[i, Cn + 1:] * inv) % p
        return "OUT", {keys[t]: int(y[t]) for t in range(Rn) if y[t]}
    coeffs = {piv[j]: int(M[j, Cn]) for j in range(r) if M[j, Cn]}
    return "IN", coeffs


def _ratrecon(a, m):
    a %= m
    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a, 0, 1
    while r1 > bound:
        qq = r0 // r1
        r0, r1 = r1, r0 - qq * r1
        s0, s1 = s1, s0 - qq * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return Fraction(r1, s1)


def _crt_reconstruct(vectors, primes):
    keys = set().union(*[v.keys() for v in vectors])
    out = {}
    for k in keys:
        a, m = 0, 1
        for v, p in zip(vectors, primes):
            x = v.get(k, 0)
            t = ((x - a) * pow(m, -1, p)) % p
            a, m = a + m * t, m * p
        f = _ratrecon(a, m)
        if f is None:
            return None
        if f:
            out[k] = f
    return out


def _verify_in(cols, target, coeffs, p=None):
    acc = defaultdict(Fraction) if p is None else defaultdict(int)
    for j, c in coeffs.items():
        for k, v in cols[j].items():
            acc[k] += c * v
    for k, v in target.items():
        acc[k] -= v
    return all((x == 0) if p is None else (x % p == 0) for x in acc.values())


def _verify_out(cols, target, y, p=None):
    def dot(v):
        s = sum(y.get(k, 0) * c for k, c in v.items())
        return s if p is None else s % p
    return all(dot(c) == 0 for c in cols) and dot(target) == 1


def _solve_exact(cols, target, nprimes=8):
    """Exact rational answer by multi-prime solving, CRT, rational reconstruction and exact verification."""
    sols, used, kind = [], [], None
    for p in PRIMES[:nprimes]:
        k, v = _dense_solve(cols, target, p)
        if kind is None:
            kind = k
        if k != kind:
            continue  # a bad prime; skip it
        sols.append(v)
        used.append(p)
        rec = _crt_reconstruct(sols, used)
        if rec is None:
            continue
        if kind == "IN" and _verify_in(cols, target, rec):
            return "IN", rec
        if kind == "OUT" and _verify_out(cols, target, rec):
            return "OUT", rec
    return kind, None


def ns_degree(G, field="Q", target="one", dmax=14):
    """Exact Nullstellensatz degree of `target` (1, or s = sum of x_v over all vertices when target='sum')
    modulo I_G, over Q (field='Q') or GF(p) (field=p, p != 3).  Returns a dict with the degree d*, an exactly
    verified certificate at d* and an exactly verified dual design at d*-1 (the lower bound)."""
    if target == "one":
        tgt, grade = {(): 1}, 0
    else:
        tgt, grade = {mono((v, 1)): 1 for v in range(G.n)}, 1
    cols, labels, last_out = [], [], None
    p0 = PRIMES[0] if field == "Q" else field
    for d in range(2, dmax + 1):
        c, l = ns_columns(G, d, grade)
        if not c:
            continue
        cols += c
        labels += l
        kind, val = _dense_solve(cols, tgt, p0)          # cheap modular screening at every degree
        if kind == "OUT":
            last_out = (d, list(cols), val)
            continue
        if field == "Q":                                   # exact rational answer at the transition only
            kind, val = _solve_exact(cols, tgt)
            if kind != "IN" or val is None:
                return {"degree": None, "error": "modular/rational disagreement at d=%d" % d}
            design = None if last_out is None else _solve_exact(last_out[1], tgt)
            design_ok = design is not None and design[0] == "OUT" and design[1] is not None
            design_support = None if not design_ok else len(design[1])
        else:
            if not _verify_in(cols, tgt, val, field):
                return {"degree": None, "error": "certificate failed verification"}
            design_ok = last_out is not None and _verify_out(last_out[1], tgt, last_out[2], field)
            design_support = None if not design_ok else len(last_out[2])
        cert = defaultdict(dict)
        for j, cf in val.items():
            (edge, e) = labels[j]
            cert[edge][e] = cf
        return {"degree": d, "certificate_verified": True, "certificate": dict(cert), "columns": len(cols),
                "design_degree": None if last_out is None else last_out[0], "design_verified": design_ok,
                "design_support": design_support}
    return {"degree": None, "dmax": dmax}


def universal_degree3_design(G):
    """Theorem 4.1(a): L(1)=1, L(x_a^2 x_b)=-1/2, L(x_a x_b x_c)=1 annihilates every degree-3 column x_w q_uv,
    for every graph; hence NS_K(G) >= 6 whenever char K is not 2 or 3.  Returns True if verified for G."""
    def L(m):
        ex = sorted(e for _, e in m)
        return Fraction(1) if not ex else Fraction(-1, 2) if ex == [1, 2] else Fraction(1) if ex == [1, 1, 1] else None
    cols, _ = ns_columns(G, 3, 0)
    return all(sum(L(m) * c for m, c in col.items()) == 0 for col in cols)


def ns_gf2(G, d):
    """GF(2) bitset elimination: is 1 in the span of the columns of nominal degree <= d?  If not, an explicit
    design y (y.col = 0 for all columns, y.1 = 1) is built and verified.  Feasible for n around 11, d <= 6."""
    keys, cols = {}, []
    for dd in range(3, d + 1, 3):
        cs, _ = ns_columns(G, dd, 0)
        for col in cs:
            v = 0
            for m, c in col.items():
                if c % 2:
                    if m not in keys:
                        keys[m] = len(keys)
                    v ^= 1 << keys[m]
            cols.append(v)
    if () not in keys:
        keys[()] = len(keys)
    one = 1 << keys[()]
    piv = {}
    for v in cols:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                break
    t = one
    while t:
        h = t.bit_length() - 1
        if h in piv:
            t ^= piv[h]
        else:
            break
    if t == 0:
        return {"degree<=d": True, "columns": len(cols), "rank": len(piv)}
    for h in sorted(piv):                       # full reduction (RREF over GF(2))
        b = piv[h]
        x = b & ~(1 << h)
        while x:
            g = x.bit_length() - 1
            if g in piv and g != h:
                b ^= piv[g]
            x &= ~(1 << g)
        piv[h] = b
    t = one
    for h in piv:
        if t >> h & 1:
            t ^= piv[h]
    s0 = (t & -t).bit_length() - 1
    y = 1 << s0
    for h, b in piv.items():
        if b >> s0 & 1:
            y |= 1 << h
    ok = all(bin(y & v).count("1") % 2 == 0 for v in cols) and bin(y & one).count("1") % 2 == 1
    return {"degree<=d": False, "columns": len(cols), "rank": len(piv), "design_verified": ok,
            "design_support": bin(y).count("1")}


def ns_membership_via_kernel(G, d, nprimes=6):
    """Exact decision of '1 in span_Q(columns of nominal degree <= d)' without writing the certificate:
    rank_Q >= rank_p always; the left kernel (y with y.col = 0 for all columns) is computed modulo several
    primes (sparse elimination), lifted to Q by CRT + rational reconstruction and verified exactly.  If the
    exact kernel has dimension rows - rank_p, it is the whole left kernel over Q, and 1 is in the span iff
    every kernel vector vanishes at the monomial 1."""
    cols = []
    for dd in range(3, d + 1, 3):
        cols += ns_columns(G, dd, 0)[0]
    keys = {}
    eqs = []
    for col in cols:
        eqs.append({keys.setdefault(m, len(keys)): c for m, c in col.items()})
    one = keys.setdefault((), len(keys))
    nk = len(keys)

    def kernel_mod(p):
        piv = {}
        for e in eqs:
            v = {k: c % p for k, c in e.items() if c % p}
            while v:
                h = max(v)
                if h in piv:
                    f = v[h]
                    for k, c in piv[h].items():
                        x = (v.get(k, 0) - f * c) % p
                        if x:
                            v[k] = x
                        else:
                            v.pop(k, None)
                else:
                    inv = pow(v[h], p - 2, p)
                    piv[h] = {k: (c * inv) % p for k, c in v.items()}
                    break
        free = [k for k in range(nk) if k not in piv]
        basis = []
        for f0 in free:
            y = {f0: 1}
            for h in sorted(piv):
                s = sum(c * y.get(k, 0) for k, c in piv[h].items() if k != h) % p
                if s:
                    y[h] = (-s) % p
            basis.append(y)
        return len(piv), free, basis

    sols, used, free_ref, rank_p = [], [], None, None
    for p in PRIMES[:nprimes]:
        rk, free, basis = kernel_mod(p)
        if free_ref is None:
            free_ref, rank_p = free, rk
        if free != free_ref:
            continue
        sols.append(basis)
        used.append(p)
        exact = []
        for i in range(len(free_ref)):
            rec = _crt_reconstruct([b[i] for b in sols], used)
            if rec is None:
                exact = None
                break
            exact.append(rec)
        if exact is None:
            continue
        ok = all(sum(y.get(k, 0) * c for k, c in e.items()) == 0 for y in exact for e in eqs)
        if ok:
            return {"rows": nk, "rank_mod_p": rank_p, "corank": len(free_ref), "kernel_verified": True,
                    "one_in_span_over_Q": all(y.get(one, 0) == 0 for y in exact), "primes": len(used)}
    return {"rows": nk, "rank_mod_p": rank_p, "corank": len(free_ref), "kernel_verified": False}


# ------------------------------------------------------------------ the propagating regime (Thm 4.1(b))
class _Q:
    zero, one = Fraction(0), Fraction(1)

    @staticmethod
    def n(x):
        return Fraction(x)

    @staticmethod
    def inv(x):
        return 1 / Fraction(x)


class _P:
    def __init__(self, p):
        self.p, self.zero, self.one = p, 0, 1

    def n(self, x):
        if isinstance(x, Fraction):
            return x.numerator * pow(x.denominator, -1, self.p) % self.p
        return x % self.p

    def inv(self, x):
        return pow(x % self.p, -1, self.p)


def _padd(P, Q, F, s=1):
    out = dict(P)
    for m, c in Q.items():
        v = F.n(out.get(m, F.zero) + s * c)
        if v:
            out[m] = v
        else:
            out.pop(m, None)
    return out


def _pmul(P, Q, F):
    out = {}
    for m1, c1 in P.items():
        for m2, c2 in Q.items():
            m = m_mul(m1, m2)
            v = F.n(out.get(m, F.zero) + c1 * c2)
            if v:
                out[m] = v
            else:
                out.pop(m, None)
    return out


def _pscale(P, c, F):
    return {m: F.n(v * c) for m, v in P.items() if F.n(v * c)}


_D0_CACHE = {}


def triangle_vieta_certificate(field="Q"):
    """s = x0 + x1 + x2 in I_{K3}: minimal-degree certificate (Theorem 4.1(a)); cached per field."""
    if field not in _D0_CACHE:
        res = ns_degree(complete(3), field=field, target="sum")
        _D0_CACHE[field] = res
    return _D0_CACHE[field]


def propagation_relation(G, comp):
    """Linear tracking of the Vieta transport: x_w = L_w(x_v0, x_v1) + S_w with S_w in span{s_tau}.
    Returns (v0, v1, (alpha, beta), combo) with alpha x_v0 + beta x_v1 = sum combo[tau] s_tau, or None."""
    by_edge = defaultdict(list)
    for t in comp:
        for e in _edges_of(t):
            by_edge[e].append(t)
    t0 = comp[0]
    v0, v1, v2 = t0
    L = {v0: (1, 0), v1: (0, 1), v2: (-1, -1)}
    S = {v0: {}, v1: {}, v2: {t0: 1}}
    done, queue = {t0}, deque([t0])

    def lin(Su, Sv, t):
        out = defaultdict(int)
        out[t] += 1
        for d_ in (Su, Sv):
            for k, c in d_.items():
                out[k] -= c
        return {k: c for k, c in out.items() if c}

    while queue:
        t = queue.popleft()
        for e in _edges_of(t):
            for t2 in by_edge[e]:
                if t2 in done:
                    continue
                done.add(t2)
                w = _third(t2, e)
                Ln = (-L[e[0]][0] - L[e[1]][0], -L[e[0]][1] - L[e[1]][1])
                Sn = lin(S[e[0]], S[e[1]], t2)
                if w in L and L[w] != Ln:
                    a, b = L[w][0] - Ln[0], L[w][1] - Ln[1]
                    combo = defaultdict(int)
                    for k, c in Sn.items():
                        combo[k] += c
                    for k, c in S[w].items():
                        combo[k] -= c
                    return v0, v1, (a, b), {k: c for k, c in combo.items() if c}
                if w not in L:
                    L[w], S[w] = Ln, Sn
                queue.append(t2)
    return None


def propagation_certificate(G, field="Q"):
    """Constructive Theorem 4.1(b): if some triangle component of G conflicts, build an explicit
    certificate 1 = sum_e beta_e q_e in R of degree <= D0 + 2 and verify it exactly.  Returns a dict or None."""
    F = _Q if field == "Q" else _P(field)
    base = triangle_vieta_certificate(field)
    for comp in triangle_components(G):
        rel = propagation_relation(G, comp)
        if rel is None:
            continue
        v0, v1, (a, b), combo = rel
        beta = defaultdict(dict)

        def add_l_times(Mpoly):  # add Mpoly * l, l = sum combo s_tau
            for tri, c in combo.items():
                relab = dict(zip((0, 1, 2), tri))
                for (i, j), poly in base["certificate"].items():
                    u, v = relab[i], relab[j]
                    key = (min(u, v), max(u, v))
                    P = {tuple(sorted((relab[x], e) for x, e in m)): cc for m, cc in poly.items()}
                    beta[key] = _padd(beta[key], _pmul(_pscale(P, F.n(c), F), Mpoly, F), F)

        a_, b_ = F.n(a), F.n(b)
        x0, x1 = mono((v0, 1)), mono((v1, 1))
        if b_ == 0:  # l = a x0  ->  1 = x0^2 l / a
            add_l_times({mono((v0, 2)): F.inv(a_)})
        else:
            r = F.n(-a_ * F.inv(b_))  # l / b = x1 - r x0
            if F.n(r * r * r - 1) != 0:  # 1 = (l/b)(x1^2 + r x1 x0 + r^2 x0^2) / (1 - r^3)
                k = F.inv(F.n((1 - r * r * r)) * b_)
                add_l_times({mono((v1, 2)): k, mono((v1, 1), (v0, 1)): F.n(r * k), mono((v0, 2)): F.n(r * r * k)})
            else:  # r = 1:  3 = x0 q_{v0 v1} - x0 (x1 + 2 x0) (l/b)
                k3 = F.inv(F.n(3))
                key = (min(v0, v1), max(v0, v1))
                beta[key] = _padd(beta[key], {x0: k3}, F)
                add_l_times({mono((v0, 1), (v1, 1)): F.n(-k3 * F.inv(b_)), mono((v0, 2)): F.n(-2 * k3 * F.inv(b_))})
        total = {}
        for (u, v), P in beta.items():
            total = _padd(total, _pmul(P, q_poly(u, v), F), F)
        ok = total == {(): F.one}
        deg = max((m_deg(m) + 2 for P in beta.values() for m in P), default=None)
        return {"verified": ok, "degree": deg, "D0": base["degree"], "relation": (a, b),
                "triangles_used": len(combo), "component_triangles": len(comp)}
    return None


def gadget_substitution_check(G):
    """Theorem 4.1(c): the substitution phi (p -> w x_u, q -> w^2 x_u, s -> x_u, t -> -x_u - x_v, w = omega)
    sends every generator g of I_{G^nat} to 0, to a generator of I_G, or to
    2(x_u^3-1) - (x_v^3-1) - 3 x_u q_uv  -- an element of I_G written in degree <= deg g.
    Checked exactly (reduction modulo w^2 + w + 1)."""
    H = strip_gadget(G)
    w = sp.Symbol("w")
    xs = sp.symbols("x0:%d" % G.n)
    sub = {v: xs[v] for v in range(G.n)}
    for (u, v), (p, q, s, t) in H.gadget_labels.items():
        sub[p], sub[q], sub[s], sub[t] = w * xs[u], w ** 2 * xs[u], xs[u], -xs[u] - xs[v]
    allowed = {sp.Integer(0)}
    for v in range(G.n):
        allowed.add(sp.expand(xs[v] ** 3 - 1))
    for u, v in G.edges:
        quv = xs[u] ** 2 + xs[u] * xs[v] + xs[v] ** 2
        allowed.add(sp.expand(quv))
        for a_, b_ in ((u, v), (v, u)):
            allowed.add(sp.expand(2 * (xs[a_] ** 3 - 1) - (xs[b_] ** 3 - 1) - 3 * xs[a_] * quv))
    ys = sp.symbols("y0:%d" % H.n)
    ok = True
    gens = [y ** 3 - 1 for y in ys] + [ys[a] ** 2 + ys[a] * ys[b] + ys[b] ** 2 for a, b in H.edges]
    for g in gens:
        img = sp.expand(g.subs({ys[i]: sub[i] for i in range(H.n)}, simultaneous=True))
        img = sp.expand(sp.rem(sp.Poly(img, w), sp.Poly(w ** 2 + w + 1, w)).as_expr())
        ok &= img in allowed
    return ok


# =====================================================================================================
# Part E.  Rank-2 Gram matrices and Kneser witnesses
# =====================================================================================================


def kneser_vector_colouring(k):
    """KG(3k,k): v_S = 3*1_S - 1 gives a Gram matrix with diagonal 1 and -1/2 on every edge (exact)."""
    G, V = kneser(3 * k, k)
    vec = [[3 if i in S else 0 for i in range(3 * k)] for S in V]
    vec = [[x - 1 for x in v] for v in vec]
    norm = sum(x * x for x in vec[0])
    ok = all(sum(x * x for x in v) == norm for v in vec)
    ok &= all(Fraction(sum(a * b for a, b in zip(vec[i], vec[j])), norm) == Fraction(-1, 2) for i, j in G.edges)
    rank = sp.Matrix(vec).rank()
    return {"graph": G.name, "n": G.n, "gram_ok": ok, "rank": rank, "3-colourable": is_3colourable(G)}


def rank2_gram_check(G, colouring):
    """For a proper colouring the Gram matrix Re(a_u conj a_v) has rank 2, unit diagonal, -1/2 on edges."""
    pts = [(math.cos(2 * math.pi * c / 3), math.sin(2 * math.pi * c / 3)) for c in colouring]
    M = np.array([[p[0] * q[0] + p[1] * q[1] for q in pts] for p in pts])
    ok = all(abs(M[u, v] + 0.5) < 1e-12 for u, v in G.edges)
    return ok and np.linalg.matrix_rank(M, tol=1e-9) <= 2


# =====================================================================================================
# Part F.  Directed vertex geography, potentials
# =====================================================================================================

QZ, ZG = sp.field("z", sp.QQ)
_ZSYM = sp.Symbol("z")


class Digraph:
    def __init__(self, n, arcs, name=""):
        self.n = int(n)
        self.arcs = sorted({(a, b) for a, b in arcs if a != b})
        self.out = [sorted(b for a, b in self.arcs if a == v) for v in range(self.n)]
        self.name = name

    def positions(self):
        V = frozenset(range(self.n))
        seen, todo = set(), [(V, s) for s in range(self.n)]
        while todo:
            W, v = todo.pop()
            if (W, v) in seen:
                continue
            seen.add((W, v))
            for w in self.out[v]:
                if w in W and w != v:
                    todo.append((W - {v}, w))
        return seen

    def resolvent(self, zval=None, eta=None):
        """R(W,v) with 1/R = z - eta_v - sum_w R(W - v, w)  (the game recursion)."""
        zz, zero = (ZG, QZ(0)) if zval is None else (Fraction(zval), Fraction(0))
        et = {v: (eta.get(v, zero) if eta else zero) for v in range(self.n)}
        out = self.out

        @lru_cache(maxsize=None)
        def R(W, v):
            W2 = W - {v}
            s = zero
            for w in out[v]:
                if w in W2:
                    s = s + R(W2, w)
            return 1 / (zz - et[v] - s)
        return R

    def win(self):
        out = self.out

        @lru_cache(maxsize=None)
        def wn(W, v):
            W2 = W - {v}
            return any(not wn(W2, w) for w in out[v] if w in W2)
        return wn

    def scc_symmetric(self):
        reach = []
        for v in range(self.n):
            seen, st = {v}, [v]
            while st:
                x = st.pop()
                for y in self.out[x]:
                    if y not in seen:
                        seen.add(y)
                        st.append(y)
            reach.append(seen)
        A = set(self.arcs)
        return all((b, a) in A for a, b in A if a in reach[b])


def symmetric_digraph(G):
    return Digraph(G.n, [(a, b) for a, b in G.edges] + [(b, a) for a, b in G.edges], G.name + "<->")


def ord0(f):
    lo = lambda p: min(m[0] for m, c in p.terms())
    return lo(f.numer) - lo(f.denom)


def check_resolvent_outcome(D):
    """Resolvent outcome theorem: ord_0 R(W,v) = +1 at N-positions and -1 at P-positions."""
    R, wn = D.resolvent(), D.win()
    return all(ord0(R(W, v)) == (1 if wn(W, v) else -1) for W, v in D.positions())


def set_potential(D, eta=None):
    """Phi(W - v) = R(W,v) Phi(W) along all plays (field-independent).  Returns (exists, witness)."""
    R = D.resolvent(eta=eta)
    pos = sorted(D.positions(), key=lambda p: -len(p[0]))
    Phi = {frozenset(range(D.n)): QZ(1)}
    for W, v in pos:
        val = R(W, v) * Phi[W]
        W2 = W - {v}
        if W2 in Phi and Phi[W2] != val:
            return False, (sorted(W2), "two plays disagree")
        Phi[W2] = val
    return True, None


def back_arc_violation(D):
    """Theorem 10.7 (any field): directed paths with equal vertex sets must have equal back-arc counts."""
    A = set(D.arcs)
    seen = {}
    stack = [(v,) for v in range(D.n)]
    while stack:
        P = stack.pop()
        X = frozenset(P)
        b = sum(1 for i in range(len(P)) for j in range(i + 1, len(P)) if (P[j], P[i]) in A)
        if X in seen and seen[X][0] != b:
            return sorted(X), seen[X][1], P
        seen.setdefault(X, (b, P))
        for w in D.out[P[-1]]:
            if w not in X:
                stack.append(P + (w,))
    return None


def mu_weighted(S, adj, a, one):
    """mu_a(U[S]) = sum over matchings M of (-1)^|M| prod_{x unmatched} a_x  (vertex recurrence)."""
    memo = {}

    def rec(T):
        if T in memo:
            return memo[T]
        if not T:
            return one
        v = min(T)
        rest = T - {v}
        val = a[v] * rec(rest)
        for w in adj.get(v, ()):
            if w in rest:
                val = val - rec(rest - {w})
        memo[T] = val
        return val
    return rec(frozenset(S))


def symmetric_widget_potential_ok(G):
    """Undirected widgets read as symmetric digraphs: U = G, a = z is a potential (Godsil / Thm 6.6)."""
    D = symmetric_digraph(G)
    R = D.resolvent()
    adj = {v: sorted(G.adj[v]) for v in range(G.n)}
    a = {v: ZG for v in range(G.n)}
    return all(mu_weighted(W - {v}, adj, a, QZ(1)) == R(W, v) * mu_weighted(W, adj, a, QZ(1))
               for W, v in D.positions())


def _potential_job(D, U, nf, fixed, zval, extract):
    """Groebner/Rabinowitsch saturation test for a (frozen) local weighted matching potential."""
    from sympy.polys.rings import ring
    from sympy.polys.orderings import grevlex, lex
    from sympy.polys.groebnertools import groebner
    N = D.n + nf
    F = frozenset(range(D.n, N))
    fixed = fixed or {}
    unknown = [i for i in range(N) if i not in fixed]
    dom = sp.QQ.frac_field(_ZSYM) if zval is None else sp.QQ
    names = ["a%d" % i for i in unknown] + ["t_sat"]
    order = lex if extract else grevlex
    Rng, *gens = ring(",".join(["t_sat"] + names[:-1]) if extract else ",".join(names), dom, order)
    if extract:
        t = gens[0]
        av = dict(zip(unknown, gens[1:]))
    else:
        t = gens[-1]
        av = dict(zip(unknown, gens[:-1]))
    conv = (lambda r: dom.from_sympy(r.as_expr())) if zval is None else (lambda r: sp.QQ(r.numerator, r.denominator))
    a = {}
    for i in range(N):
        if i in fixed:
            fv = sp.sympify(fixed[i])
            if zval is None:
                a[i] = Rng(dom.from_sympy(fv))
            else:
                fz = sp.Rational(fv.subs(_ZSYM, sp.Rational(Fraction(zval).numerator, Fraction(zval).denominator)))
                a[i] = Rng(sp.QQ(int(fz.p), int(fz.q)))
        else:
            a[i] = av[i]
    adj = defaultdict(list)
    for i, j in U:
        adj[i].append(j)
        adj[j].append(i)
    R = D.resolvent(zval=zval)
    eqs, nonzero = [], {}
    for W, v in D.positions():
        den = mu_weighted(W | F, adj, a, Rng(1))
        num = mu_weighted((W - {v}) | F, adj, a, Rng(1))
        e = num - den * Rng(conv(R(W, v)))
        if e != 0:
            eqs.append(e)
        nonzero[W] = den
    prod = Rng(1)
    for f in nonzero.values():
        if f == 0:
            return {"status": "NO POTENTIAL (a denominator vanishes identically)"}
        if not f.is_ground:
            prod *= f
    if not eqs:
        return {"status": "POTENTIAL EXISTS (no equations)"}
    GB = groebner(eqs + [Rng(1) - t * prod], Rng)
    if len(GB) == 1 and GB[0].is_ground:
        return {"status": "NO POTENTIAL (saturated ideal is the unit ideal)"}
    info = {"status": "POTENTIAL EXISTS over the algebraic closure (proper saturated ideal)", "basis_size": len(GB)}
    if extract:
        last = gens[-1]
        uni = [g for g in GB if all(m[:-1] == (0,) * (len(gens) - 1) for m in g.monoms())]
        if uni:
            info["eliminant_degree"] = max(m[-1] for m in uni[0].monoms())
            info["eliminant"] = str(uni[0].as_expr())
    return info


def decide_potential(D, U, nf=0, fixed=None, zval=None, timeout=60, extract=False):
    """Time-limited decision: does (D, U) carry a (frozen, nf environment vertices) local weighted matching
    potential?  zval=None works over Q(z) (rigorous); zval in Q specialises z (a proper ideal at a random
    z_0 is evidence, not proof -- see the draft, Remark 2.4).  Never hangs: returns 'TIMEOUT' instead."""
    st, val = run_with_timeout(_potential_job, (D, U, nf, fixed, zval, extract), timeout)
    return val if st == "OK" else {"status": st}


def directed_triangle():
    return Digraph(3, [(0, 1), (1, 2), (2, 0)], "C3")


def triangle_plus_isolated(k):
    return Digraph(3 + k, [(0, 1), (1, 2), (2, 0)], "C3+%dK1" % k)


def bowtie():
    return Digraph(5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)], "bowtie")


def book(r, sinks_at_tail=0):
    """Spine x=0 -> y=1, pages p_i: y -> p_i -> x (two directed triangles share the arc x->y when r = 2)."""
    arcs = [(0, 1)] + [(1, 2 + i) for i in range(r)] + [(2 + i, 0) for i in range(r)]
    n = 2 + r
    arcs += [(0, n + j) for j in range(sinks_at_tail)]
    return Digraph(n + sinks_at_tail, arcs, "B%d+%dsink" % (r, sinks_at_tail))


def vieta_D3_check(m=3):
    """Theorem 11.6 of the companion paper (vanishing cycle weights, environment = roots of Y_m), verified
    exactly through elementary symmetric functions, against resolvents from the game recursion."""
    D = Digraph(2 * m + 1, [(i, (i + 1) % m) for i in range(m)], "D%d" % m)
    C, F = list(range(m)), list(range(m, 2 * m + 1))
    U = [(c, f) for c in C for f in F]
    b = sp.symbols("b0:%d" % (m + 1))
    a = {c: sp.Integer(0) for c in C}
    a.update({f: b[i] for i, f in enumerate(F)})
    adj = defaultdict(list)
    for i, j in U:
        adj[i].append(j)
        adj[j].append(i)
    z = _ZSYM

    def mu_path(l, x):
        p0, p1 = sp.Integer(1), x
        if l == 0:
            return p0
        for _ in range(l - 1):
            p0, p1 = p1, sp.expand(x * p1 - p0)
        return p1
    c = (-1) ** m * sp.factorial(m) * z / mu_path(m, z)
    e = {m + 1 - l: (-1) ** l * c * mu_path(l, z) / sp.factorial(l) for l in range(m + 1)}
    R = D.resolvent()
    cache = {}

    def val(S):
        if S not in cache:
            poly = sp.expand(mu_weighted(S, adj, a, sp.Integer(1)))
            if poly.free_symbols & set(b):
                sym, rem, defs = sp.polys.polyfuncs.symmetrize(poly, *b, formal=True)
                assert rem == 0
                cache[S] = sp.simplify(sym.subs({s: e[int(str(s)[1:])] for s, _ in defs}))
            else:
                cache[S] = poly
        return cache[S]
    ok = all(sp.simplify(val(W) ) != 0 and sp.simplify(val(W - {v}) - R(W, v).as_expr() * val(W)) == 0
             for W, v in D.positions())
    Y = sp.expand(sp.Symbol("y") ** (m + 1) + sum((-1) ** j * e[j] * sp.Symbol("y") ** (m + 1 - j) for j in range(1, m + 2)))
    return ok, sp.factor(sp.together(Y))


def rescued_triangle_check():
    """Theorem 11.3 of the companion paper: cycle weights z + t (t^3 - 3t = z), environment weights the
    roots of (t^2-1)x^2 - 2tx + 2, verified modulo t^3 - 3t - z through e1, e2 of the environment."""
    z, t = _ZSYM, sp.Symbol("t")
    x1, x2 = sp.symbols("x1 x2")
    alpha = z + t
    a = {0: alpha, 1: alpha, 2: alpha, 3: x1, 4: x2}
    U = [(c, f) for c in range(3) for f in (3, 4)]
    adj = defaultdict(list)
    for i, j in U:
        adj[i].append(j)
        adj[j].append(i)
    s_val, p_val = 2 * t / (t ** 2 - 1), 2 / (t ** 2 - 1)
    cheb = t ** 3 - 3 * t - z
    mu = lambda k: [sp.Integer(1), z, z ** 2 - 1, z ** 3 - 2 * z][k]
    ok = True
    for k in range(4):
        poly = sp.expand(mu_weighted(frozenset(range(k)) | {3, 4}, adj, a, sp.Integer(1)))
        sym, rem, defs = sp.polys.polyfuncs.symmetrize(poly, x1, x2, formal=True)
        expr = sym.subs({defs[0][0]: s_val, defs[1][0]: p_val}) - p_val * mu(k)
        num = sp.numer(sp.together(expr))
        ok &= sp.rem(sp.expand(num), cheb, t) == 0
    return ok


# =====================================================================================================
# Part G.  Monodromy of the triangle rescues
# =====================================================================================================


def monodromy_report(z_values=(3, 5, 7)):
    z, t, x, y = sp.symbols("z t x y")
    cheb = t ** 3 - 3 * t - z
    quad = (t ** 2 - 1) * x ** 2 - 2 * t * x + 2
    sextic = sp.factor(sp.resultant(cheb, quad, t))
    paper = (z ** 2 - 4) * x ** 6 + 12 * x ** 4 + 4 * z * x ** 3 - 12 * x ** 2 + 8
    same = sp.simplify(sp.resultant(cheb, quad, t) / paper)
    kummer = sp.expand(sp.resultant(cheb, 2 - t ** 2, t))
    Y3 = (z ** 2 - 2) * y ** 4 - z * (z ** 2 - 2) * y ** 3 - 3 * (z ** 2 - 1) * y ** 2 - 6 * z * y - 6
    from sympy.polys.numberfields.galoisgroups import galois_group
    rows = []
    for z0 in z_values:
        out = {"z0": z0}
        for name, f, var in (("chebyshev", cheb, t), ("sextic", paper, x), ("Y3", Y3, y)):
            P = sp.Poly(f.subs(z, z0), var)
            if not P.is_irreducible:
                out[name] = "reducible"
                continue
            G, _ = galois_group(P)
            out[name] = (G.order(), str(galois_group(P, by_name=True)[0]))
        rows.append(out)
    return {"sextic_matches_paper": same, "sextic": str(sextic), "prod(2 - t_k^2)": str(kummer), "galois": rows}


# =====================================================================================================
# Part H.  Report driver
# =====================================================================================================


def _hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100, flush=True)


def report(quick=False, seed=2026):
    rng = random.Random(seed)
    T0 = time.time()

    _hdr("B. Root-3 encoding")
    print("Vieta lemma on Omega_3^3 (proper <=> e1=0 <=> (e1,e2,e3)=(0,0,1) <=> q's vanish):", vieta_lemma_check())
    print("I_{K3} = <e1, e2, e3 - 1> (reduced Groebner bases coincide):", vieta_ideal_equals_triangle_ideal())
    for G in (complete(3), complete(4), cycle_graph(5), octahedron(), wheel(5), moser_spindle()):
        r = variety_report(G, timeout=120)
        print("  %-11s Groebner %-7s %-55s P(G,3)=%-4d point check %s" % (
            G.name, r["status"], r["groebner"], r["P(G,3)"], r["sample_point_ok"]))

    _hdr("C. Triangle propagation, holonomy in Sym(Omega_3) = D3, GF(3) collapse")
    examples = [complete(4), octahedron(), icosahedron(), wheel(5), wheel(6), triangular_torus(4, 4),
                triangular_torus(6, 6), triangular_torus(4, 6), triangular_torus(6, 9), moser_spindle()]
    for G in examples:
        comps = triangle_components(G)
        info = []
        for comp in comps:
            H, pin = holonomy(G, comp)
            col, conf = propagate(G, comp)
            info.append("[%d tri: hol %s, pinches %d, F3-dim %d, colourable %s]" % (
                len(comp), group_name(H), pin, f3_component_dimension(G, comp), col is not None))
        print("  %-13s chi<=3: %-5s components %d  %s" % (G.name, is_3colourable(G), len(comps), " ".join(info)))
    # random cross-check of Theorem 3.1 and 3.2
    agree, agree_cnt, trials = 0, 0, (60 if quick else 300)
    for _ in range(trials):
        G = random_triangle_graph(rng.randint(5, 9), rng.randint(2, 7), rng)
        comps = triangle_components(G)
        thm31 = all(propagate(G, c)[0] is not None for c in comps)
        decided = all(holonomy(G, c)[0] == {(0, 1, 2)} and holonomy(G, c)[1] == 0 for c in comps)
        agree += (thm31 == decided)
        cnt, k, dec = hyperplane_count(G)
        bf = colour_search(G)[0]
        agree_cnt += (cnt == bf) and (dec is None or dec == (bf > 0))
    print("  Random triangle-saturated graphs: holonomy+pinch test == propagation in %d/%d;"
          " inclusion-exclusion count == brute force (and k<=3 covering decision) in %d/%d" % (agree, trials, agree_cnt, trials))
    for G in (complete(4), cycle_graph(5), groetzsch(), kneser(5, 2)[0]):
        H = strip_gadget(G)
        comps = triangle_components(H)
        hol_trivial = all(holonomy(H, c)[0] == {(0, 1, 2)} for c in comps)
        same = is_3colourable(H) == is_3colourable(G)
        print("  G^nat for %-10s n=%-4d components k=%-4d all holonomy trivial: %s; chi<=3 preserved: %s" % (
            G.name, H.n, len(comps), hol_trivial, same))

    _hdr("D. Nullstellensatz degrees (exact over Q: certificate at d*, design at d*-1; and over GF(2))")
    d0 = triangle_vieta_certificate("Q")
    print("  D0 = deg of s=x0+x1+x2 in I_K3 over Q:", d0["degree"], "certificate verified:", d0["certificate_verified"],
          "design at", d0["design_degree"], "verified:", d0["design_verified"])
    d02 = triangle_vieta_certificate(2)
    print("  D0 over GF(2):", d02["degree"], d02["certificate_verified"], d02["design_verified"])
    print("  universal degree-3 design (=> NS >= 6 when char != 2,3) annihilates all degree-3 columns:",
          {G.name: universal_degree3_design(G) for G in (complete(4), wheel(5), moser_spindle(), groetzsch(), kneser(5, 2)[0])})
    graphs = [complete(4), wheel(5), moser_spindle()] if not quick else [complete(4), wheel(5)]
    for G in graphs:
        for fld in ("Q", 2):
            t1 = time.time()
            r = ns_degree(G, field=fld, dmax=12)
            print("  NS_%s(%s) = %s  [certificate verified %s, design at d*-3 verified %s, %d columns, %.1fs]" % (
                fld if fld == "Q" else "GF(2)", G.name, r.get("degree"), r.get("certificate_verified"),
                r.get("design_verified"), r.get("columns", 0), time.time() - t1))
    if not quick:
        Gz = groetzsch()
        print("  Groetzsch (triangle-free) over GF(2): d=3 ->", ns_gf2(Gz, 3))
        print("  Groetzsch over Q: d=3 ->", ns_membership_via_kernel(Gz, 3))
        print("  Groetzsch over Q: d=6 ->", ns_membership_via_kernel(Gz, 6))
    dense = [g for g in (random_triangle_graph(24, 70, rng) for _ in range(30)) if not is_3colourable(g)][:3]
    for G in [complete(4), wheel(5), wheel(7), icosahedron(), triangular_torus(4, 4), triangular_torus(5, 7)] + dense:
        for fld in ("Q", 2, 7):
            c = propagation_certificate(G, field=fld)
            if c is None:
                print("  %-15s no conflicting triangle component" % G.name)
                break
            print("  propagating regime %-15s n=%-3d field %-3s certificate degree %s <= D0+2=%s verified %s (relation %s, %d triangles)" % (
                G.name, G.n, fld, c["degree"], c["D0"] + 2, c["verified"], c["relation"], c["triangles_used"]))
    census = defaultdict(int)
    for _ in range(40 if quick else 200):
        G = random_triangle_graph(30, rng.randint(10, 45), rng)
        col = is_3colourable(G)
        conflict = any(propagate(G, c)[1] is not None for c in triangle_components(G))
        census[("3-colourable" if col else "not 3-colourable",
                "conflicting component" if conflict else "no conflicting component")] += 1
    print("  random triangle-saturated graphs, n=30 (regime census):", dict(census))
    for G in (complete(4), cycle_graph(5), moser_spindle()):
        print("  gadget substitution phi maps I_{G^nat} into I_G without raising degree (%s): %s" % (G.name, gadget_substitution_check(G)))

    _hdr("E. Rank-2 Gram matrices; Kneser witnesses for the rank relaxation")
    print("  rank-2 Gram of a colouring of the octahedron:", rank2_gram_check(octahedron(), colour_search(octahedron(), False)[1]))
    for k in ((2,) if quick else (2, 3)):
        print("  ", kneser_vector_colouring(k))

    _hdr("F. Directed vertex geography: game recursion and complex potentials on colouring widgets")
    widgets = [complete(3), complete(4), octahedron(), wheel(5), moser_spindle(), Graph(5, [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)], "bowtie_u")]
    for G in widgets:
        print("  symmetric widget %-11s resolvent outcome theorem %s; matching potential (U=G, a=z) %s" % (
            G.name, check_resolvent_outcome(symmetric_digraph(G)), symmetric_widget_potential_ok(G)))
    for D in (directed_triangle(), triangle_plus_isolated(1), triangle_plus_isolated(4), bowtie(), book(2), book(2, 1)):
        sp_ok, _ = set_potential(D)
        print("  %-12s SCC-symmetric %-5s outcome thm %-5s back-arc violation %-5s set potential %s" % (
            D.name, D.scc_symmetric(), check_resolvent_outcome(D), back_arc_violation(D) is not None, sp_ok))
    print("  rescued triangle (Thm 11.3) identities modulo t^3-3t-z:", rescued_triangle_check())
    ok, Y = vieta_D3_check(3)
    print("  D3 (Thm 11.6) every identity exact via e_j:", ok, "  Y_3 =", Y)
    print("  Groebner/Rabinowitsch saturation over Q(z) (time-limited):")
    C3 = directed_triangle()
    for U in ([], [(0, 1), (1, 2), (0, 2)]):
        print("    C3, U=%s:" % U, decide_potential(C3, U, timeout=60))
    D3 = triangle_plus_isolated(4)
    K34 = [(c, f) for c in range(3) for f in range(3, 7)]
    print("    D3, U=K_{3,4}, cycle weights fixed 0 (extract eliminant):",
          decide_potential(D3, K34, fixed={0: 0, 1: 0, 2: 0}, timeout=240, extract=True))
    print("    frozen C3 with one frozen vertex, U = all edges to it:",
          decide_potential(C3, [(0, 3), (1, 3), (2, 3)], nf=1, timeout=120))
    nU = 0
    allU = [list(c) for r in range(7) for c in itertools.combinations(list(itertools.combinations(range(4), 2)), r)]
    sample = allU if not quick else allU[:16]
    verdicts = defaultdict(int)
    for U in sample:
        res = decide_potential(triangle_plus_isolated(1), U, zval=Fraction(7, 3), timeout=30)
        verdicts[res["status"].split(" (")[0]] += 1
        nU += 1
    print("    C3+K1, all %d graphs U, z specialised to 7/3:" % nU, dict(verdicts))
    rnd = defaultdict(int)
    for _ in range(20 if quick else 80):
        n = rng.randint(4, 6)
        arcs = [(0, 1), (1, 2), (2, 0)]
        if rng.random() < 0.5:
            arcs += [(0, 3), (3, 4 % n), (4 % n, 0)] if n >= 5 else [(1, 3), (3, 0)]
        arcs += [(a, b) for a in range(n) for b in range(n) if a != b and rng.random() < 0.15]
        D = Digraph(n, arcs)
        rnd[(D.scc_symmetric(), back_arc_violation(D) is None, set_potential(D)[0], check_resolvent_outcome(D))] += 1
    print("  random triangle widgets (SCC-sym, back-arc ok, set potential, outcome thm) -> count:", dict(rnd))

    _hdr("G. Monodromy ladder of the triangle rescues")
    m = monodromy_report()
    print("  sextic of Prop 11.2 = Res_t(t^3-3t-z, (t^2-1)x^2-2tx+2) up to constant:", m["sextic_matches_paper"])
    print("  prod_k (2 - t_k^2) = Res_t(t^3-3t-z, 2-t^2) =", m["prod(2 - t_k^2)"])
    for row in m["galois"]:
        print("  ", row)
    print("\nTotal time %.1fs" % (time.time() - T0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    report(quick=args.quick)
