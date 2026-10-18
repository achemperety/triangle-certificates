#!/usr/bin/env python3
"""
hypercomplex_assault.py -- quaternionic, octonionic and Clifford extensions of Root-3 geography
==============================================================================================

Standalone (numpy + sympy only).  Every claim marked EXACT is checked in exact arithmetic
(integers, rationals, Q(sqrt 2), Q(sqrt 5), Gaussian rationals); NUMERIC checks use floats.

    python hypercomplex_assault.py           # full report (about 1 minute)
    python -m pytest -q test_hypercomplex_assault.py

Sections (numbering follows the draft)
  1  exact scalars Q(sqrt d), quaternions, octonions (Cayley-Dickson), Q_8
  2  Frontier A: spherical roots; the Q_8 colour system Q_G and the phase system A_G; Clifford points
     (J(Q_G), J(A_G) proper for every graph); Peres 33 rays; Pauli transfer (local radical ideals)
  3  Frontier B: nucleus and Artin; ladder resolvent degrees and the algebra-independent bound;
     cost of simulating octonion arithmetic
  4  Frontier C: hypercomplex Root-3 = vector colourings; Re-projection and the complex representation;
     the binary icosahedral witness (rank 4, chi > 3); KG(6,2) rigidity (unique Gram, rank 5);
     quaternionic Hermitian squares are real sums of squares; Albert algebra (Jordan only for n <= 3)
"""
from __future__ import annotations

import itertools
import math
import random
import sys
import time
from fractions import Fraction

import numpy as np
import sympy as sp

# =====================================================================================================
# 1.  Exact scalars, quaternions, octonions, Q_8
# =====================================================================================================


class QR:
    """a + b sqrt(d), a, b rational (exact).  d = 0 gives plain rationals."""
    __slots__ = ("a", "b", "d")

    def __init__(self, a, b=0, d=0):
        self.a, self.b, self.d = Fraction(a), Fraction(b), d

    def _c(self, o):
        return o if isinstance(o, QR) else QR(o, 0, self.d)

    def __add__(self, o):
        o = self._c(o)
        return QR(self.a + o.a, self.b + o.b, self.d or o.d)

    __radd__ = __add__

    def __neg__(self):
        return QR(-self.a, -self.b, self.d)

    def __sub__(self, o):
        return self + (-self._c(o))

    def __rsub__(self, o):
        return self._c(o) - self

    def __mul__(self, o):
        o = self._c(o)
        d = self.d or o.d
        return QR(self.a * o.a + d * self.b * o.b, self.a * o.b + self.b * o.a, d)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._c(o)
        d = self.d or o.d
        den = o.a * o.a - d * o.b * o.b
        return self * QR(o.a / den, -o.b / den, d)

    def __eq__(self, o):
        o = self._c(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def is_zero(self):
        return self.a == 0 and self.b == 0

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(self.d)

    def __repr__(self):
        return "%s%+s*sqrt(%d)" % (self.a, self.b, self.d) if self.b else str(self.a)


class Quat:
    """Quaternion w + x i + y j + z k over any commutative scalar ring (exact if the scalars are)."""
    __slots__ = ("c",)

    def __init__(self, w, x=0, y=0, z=0):
        self.c = (w, x, y, z)

    def __add__(self, o):
        return Quat(*[p + q for p, q in zip(self.c, o.c)])

    def __sub__(self, o):
        return Quat(*[p - q for p, q in zip(self.c, o.c)])

    def __neg__(self):
        return Quat(*[-p for p in self.c])

    def __mul__(self, o):
        if not isinstance(o, Quat):
            return Quat(*[p * o for p in self.c])
        a1, b1, c1, d1 = self.c
        a2, b2, c2, d2 = o.c
        return Quat(a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                    a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)

    def conj(self):
        return Quat(self.c[0], -self.c[1], -self.c[2], -self.c[3])

    def re(self):
        return self.c[0]

    def is_zero(self):
        return all((p.is_zero() if hasattr(p, "is_zero") else p == 0) for p in self.c)

    def __eq__(self, o):
        return (self - o).is_zero()

    def key(self):
        return tuple((p.a, p.b) if isinstance(p, QR) else p for p in self.c)

    def __repr__(self):
        return "Quat%s" % (self.c,)


ONE, I_, J_, K_ = Quat(1), Quat(0, 1), Quat(0, 0, 1), Quat(0, 0, 0, 1)
Q8 = [ONE, -ONE, I_, -I_, J_, -J_, K_, -K_]


def oct_mul(x, y):
    """Octonion product by Cayley-Dickson: (a,b)(c,d) = (ac - conj(d) b, d a + b conj(c)); x, y in R^8."""
    def qm(p, q):
        a1, b1, c1, d1 = p
        a2, b2, c2, d2 = q
        return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                         a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])
    qc = lambda p: np.array([p[0], -p[1], -p[2], -p[3]])
    a, b, c, d = x[:4], x[4:], y[:4], y[4:]
    return np.concatenate([qm(a, c) - qm(qc(d), b), qm(d, a) + qm(b, qc(c))])


def oct_conj(x):
    return np.concatenate([[x[0]], -x[1:]])


# --------------------------------------------------------------------------------- graphs (self-contained)
class Graph:
    def __init__(self, n, edges, name=""):
        self.n = n
        self.edges = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
        self.adj = [set() for _ in range(n)]
        for a, b in self.edges:
            self.adj[a].add(b)
            self.adj[b].add(a)
        self.name = name

    def triangles(self):
        return [(a, b, c) for a, b in self.edges for c in self.adj[a] & self.adj[b] if c > b]


def colour_count(G, stop_at_one=False):
    """Number of proper 3-colourings (backtracking with forward checking)."""
    n, adj = G.n, [sorted(s) for s in G.adj]
    dom, col, st = [7] * n, [-1] * n, {"c": 0}

    def pick():
        best, bd = -1, 9
        for v in range(n):
            if col[v] < 0:
                d = bin(dom[v]).count("1")
                if d < bd or (d == bd and len(adj[v]) > len(adj[best])):
                    best, bd = v, d
        return best

    def rec(k):
        if stop_at_one and st["c"]:
            return
        if k == n:
            st["c"] += 1
            return
        v = pick()
        for c in range(3):
            if not dom[v] >> c & 1:
                continue
            ch, ok = [], True
            for w in adj[v]:
                if col[w] < 0 and dom[w] >> c & 1:
                    dom[w] &= ~(1 << c)
                    ch.append(w)
                    ok &= dom[w] != 0
            if ok:
                col[v] = c
                rec(k + 1)
                col[v] = -1
            for w in ch:
                dom[w] |= 1 << c
    rec(0)
    return st["c"]


def triangle_components(G):
    T = G.triangles()
    by_edge = {}
    for i, (a, b, c) in enumerate(T):
        for e in ((a, b), (a, c), (b, c)):
            by_edge.setdefault(e, []).append(i)
    parent = list(range(len(T)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for lst in by_edge.values():
        for j in lst[1:]:
            parent[find(j)] = find(lst[0])
    return len({find(i) for i in range(len(T))}), len(T)


def complete(n):
    return Graph(n, itertools.combinations(range(n), 2), "K%d" % n)


# =====================================================================================================
# 2.  Frontier A: noncommutative (quaternionic) encodings
# =====================================================================================================


def spherical_roots_check(trials=200, seed=1):
    """Lemma 2.1 (Niven): every s in S^2 gives roots s of t^2+1 and r = -1/2 + (sqrt3/2)s of t^2+t+1,
    so {q : q^3 = 1} = {1} u (a 2-sphere).  NUMERIC on random axes."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        s = rng.normal(size=3)
        s /= np.linalg.norm(s)
        S = Quat(0.0, *s)
        r = Quat(-0.5, *(math.sqrt(3) / 2 * s))
        e1 = S * S + Quat(1.0)
        e2 = r * r + r + Quat(1.0)
        e3 = r * r * r - Quat(1.0)
        worst = max(worst, max(abs(c) for q in (e1, e2, e3) for c in q.c))
    return worst


def axis_product(x):
    """[x,i][x,j][x,k]; for x in the unit sphere of pure quaternions it vanishes iff x is on an axis."""
    return (x * I_ - I_ * x) * (x * J_ - J_ * x) * (x * K_ - K_ * x)


def q8_colour_system_ok(x, G):
    """All relations of the Q_8 colour system Q_G at a point x in H^V (exact for exact scalars)."""
    for q in x:
        if not (q * q + ONE).is_zero() or not axis_product(q).is_zero():
            return False
    return all((x[u] * x[v] + x[v] * x[u]).is_zero() for u, v in G.edges)


def q8_points_count(G):
    """EXACT: H-points of Q_G with coordinates in Q_8 (Prop. 2.2 shows there are no others):
    their number is 2^n P(G,3)."""
    return sum(1 for x in itertools.product(Q8, repeat=G.n) if q8_colour_system_ok(x, G))


def off_axis_is_rejected(trials=200, seed=2):
    """NUMERIC: random unit pure quaternions off the axes have a nonzero axis product."""
    rng = np.random.default_rng(seed)
    mn = 1e9
    for _ in range(trials):
        s = rng.normal(size=3)
        s /= np.linalg.norm(s)
        mn = min(mn, max(abs(c) for c in axis_product(Quat(0.0, *s)).c))
    return mn


def clifford_generators(N):
    """N pairwise anticommuting complex matrices squaring to -I (Jordan-Wigner), size 2^ceil(N/2)."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Id = np.eye(2, dtype=complex)
    m = (N + 1) // 2
    gens = []
    for k in range(m):
        for P in (X, Y):
            ops = [Z] * k + [P] + [Id] * (m - k - 1)
            M = ops[0]
            for o in ops[1:]:
                M = np.kron(M, o)
            gens.append(1j * M)
    return gens[:N]


def clifford_point(n):
    """Theorem 2.3.  In Cl_{0,n+2} (complex matrices): x_v -> g_v (v < n), i -> g_n, j -> g_{n+1}, k -> g_n g_{n+1}.
    This is a ring homomorphism from the free H-ring that kills every relation of Q_{K_n}, hence of Q_G and A_G
    for every graph G on n vertices.  EXACT (Gaussian-integer matrices)."""
    g = clifford_generators(n + 2)
    I2 = np.eye(g[0].shape[0])
    qi, qj = g[n], g[n + 1]
    qk = qi @ qj
    emb = all(np.allclose(M @ M, -I2) for M in (qi, qj, qk)) and np.allclose(qi @ qj, qk) \
        and np.allclose(qj @ qk, qi) and np.allclose(qk @ qi, qj)
    ok = emb
    for v in range(n):
        x = g[v]
        ok &= np.allclose(x @ x, -I2)
        prod = (x @ qi - qi @ x) @ (x @ qj - qj @ x) @ (x @ qk - qk @ x)
        ok &= np.allclose(prod, 0)
        for u in range(v):
            ok &= np.allclose(g[u] @ x + x @ g[u], 0)
    return {"n": n, "matrix_size": g[0].shape[0], "H_embedding": emb, "all_relations_vanish": ok}


def peres_rays():
    """Peres's 33 rays in R^3 over Z[sqrt 2] (EXACT), one representative per ray."""
    s2 = QR(0, 1, 2)
    z, o = QR(0, 0, 2), QR(1, 0, 2)
    rays = {}
    for base in [(z, z, o), (z, o, o), (z, o, s2), (o, o, s2)]:
        for perm in itertools.permutations(range(3)):
            for sg in itertools.product((1, -1), repeat=3):
                v = [base[perm[t]] * sg[t] for t in range(3)]
                k = next(t for t in range(3) if not v[t].is_zero())
                if float(v[k]) < 0:
                    v = [-c for c in v]
                rays[tuple((c.a, c.b) for c in v)] = v
    return list(rays.values())


def peres_witness():
    """Proposition 2.4: the orthogonality graph of the 33 Peres rays is not 3-colourable, yet the phase
    system A_G has an H-point (pure quaternions along the rays: anticommuting, squares negative real)."""
    R = peres_rays()
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    E = [(a, b) for a in range(len(R)) for b in range(a + 1, len(R)) if dot(R[a], R[b]).is_zero()]
    G = Graph(len(R), E, "Peres33")
    X = [Quat(QR(0, 0, 2), *v) for v in R]
    anti = all((X[u] * X[v] + X[v] * X[u]).is_zero() for u, v in E)
    neg_real_squares = all(all(c.is_zero() for c in (X[v] * X[v]).c[1:]) and float((X[v] * X[v]).c[0]) < 0
                           for v in range(len(R)))
    return {"rays": len(R), "edges": len(E), "triangles": len(G.triangles()),
            "3-colourable": colour_count(G, True) > 0, "anticommute_on_edges": anti,
            "squares_negative_real": neg_real_squares}


def pauli_transfer_check():
    """Theorem 2.5, local hypotheses (EXACT, sympy):
    rho(i), rho(j), rho(k) satisfy the quaternion relations; every entry of rho(Q_v) lies in the radical
    ideal J_v (6 points) and every entry of rho(Q_uv) in J_uv (24 points); the sign-forgetting map sends
    J_sgn into J_01 in degree <= 2."""
    im = sp.Symbol("im")  # imaginary unit, im^2 = -1 added to the ideals
    E1 = sp.Matrix([[im, 0], [0, -im]])
    E2 = sp.Matrix([[0, 1], [-1, 0]])
    E3 = sp.Matrix([[0, im], [im, 0]])
    red = lambda M: M.applyfunc(lambda e: sp.expand(e).subs(im ** 2, -1))
    I2 = sp.eye(2)
    quat_ok = all(red(A * B - C) == sp.zeros(2) for A, B, C in ((E1, E2, E3), (E2, E3, E1), (E3, E1, E2))) \
        and all(red(A * A + I2) == sp.zeros(2) for A in (E1, E2, E3))
    y = sp.symbols("y1:4")
    w = sp.symbols("w1:4")
    X = y[0] * E1 + y[1] * E2 + y[2] * E3
    Yv = w[0] * E1 + w[1] * E2 + w[2] * E3
    Jv = [y[0] * y[1], y[0] * y[2], y[1] * y[2], sum(t ** 2 for t in y) - 1, im ** 2 + 1]
    Jw = [w[0] * w[1], w[0] * w[2], w[1] * w[2], sum(t ** 2 for t in w) - 1]
    Juv = Jv + Jw + [y[a] * w[a] for a in range(3)]
    GBv = sp.groebner(Jv, im, *y, order="grevlex")
    GBuv = sp.groebner(Juv, im, *y, *w, order="grevlex")
    comm = lambda A, B: A * B - B * A
    vert = [red(X * X + I2), red(comm(X, E1) * comm(X, E2) * comm(X, E3))]
    edge = [red(X * Yv + Yv * X)]
    in_v = all(GBv.reduce(sp.expand(e))[1] == 0 for M in vert for e in M)
    in_uv = all(GBuv.reduce(sp.expand(e))[1] == 0 for M in edge for e in M)

    def qdim(GB, gens):
        leads = [sp.Poly(g, *gens).monoms(order="grevlex")[0] for g in GB.exprs]
        return sum(1 for e in itertools.product(range(4), repeat=len(gens))
                   if not any(all(e[t] >= L[t] for t in range(len(gens))) for L in leads))
    rad_v = qdim(sp.groebner(Jv[:-1], *y, order="grevlex"), y)
    rad_uv = qdim(sp.groebner(Juv[:4] + Jw + [y[a] * w[a] for a in range(3)], *y, *w, order="grevlex"), y + w)
    z = sp.symbols("z1:4")
    J01 = [t ** 2 - t for t in z] + [z[0] * z[1], z[0] * z[2], z[1] * z[2], sum(z) - 1]
    GB01 = sp.groebner(J01, *z, order="grevlex")
    psi = all(GB01.reduce(sp.expand(g.subs(dict(zip(y, z)))))[1] == 0 for g in Jv[:4])
    return {"rho_is_quaternion_embedding": quat_ok, "entries_of_rho(Q_v)_in_J_v": in_v,
            "entries_of_rho(Q_uv)_in_J_uv": in_uv, "dim C[y]/J_v": rad_v, "dim C[y,w]/J_uv": rad_uv,
            "psi(J_v) in J_01": psi}


# =====================================================================================================
# 3.  Frontier B: octonions
# =====================================================================================================


def octonion_checks(trials=300, seed=3):
    """NUMERIC: nonassociative in general; alternative; scalars in the nucleus; Artin (2-generated
    subalgebras are associative); Moufang identity."""
    rng = np.random.default_rng(seed)
    assoc = lambda a, b, c: np.linalg.norm(oct_mul(oct_mul(a, b), c) - oct_mul(a, oct_mul(b, c)))
    out = {"max_assoc_generic": 0.0, "max_assoc_scalar": 0.0, "max_artin": 0.0, "max_moufang": 0.0,
           "max_alternative": 0.0}
    for _ in range(trials):
        x, y, u = rng.normal(size=8), rng.normal(size=8), rng.normal(size=8)
        s = np.zeros(8)
        s[0] = rng.normal()
        out["max_assoc_generic"] = max(out["max_assoc_generic"], assoc(x, y, u))
        out["max_assoc_scalar"] = max(out["max_assoc_scalar"], assoc(s, x, y), assoc(x, s, y), assoc(x, y, s))
        words = [x, y, oct_mul(x, y), oct_mul(y, x), oct_mul(oct_mul(x, y), x), oct_mul(y, oct_mul(x, y))]
        a, b, c = (words[t] for t in rng.choice(len(words), 3))
        out["max_artin"] = max(out["max_artin"], assoc(a, b, c))
        out["max_alternative"] = max(out["max_alternative"], assoc(x, x, y), assoc(y, x, x))
        lhs = oct_mul(oct_mul(x, y), oct_mul(u, x))
        rhs = oct_mul(oct_mul(x, oct_mul(y, u)), x)
        out["max_moufang"] = max(out["max_moufang"], np.linalg.norm(lhs - rhs))
    return out


def ladder_degrees(mmax=7):
    """EXACT: denominator degree of R(V, t_0) on the ladder L_m (companion paper, Thm 14.1) is 2^{m+1}-2,
    via F_i = (z - F_{i+1}) / ((z - F_{i+1})^2 - 1), F_{m-1} = z/(z^2-1)."""
    z = sp.Symbol("z")
    rows = []
    for m in range(1, mmax + 1):
        F = z / (z ** 2 - 1)
        for _ in range(m - 1):
            F = sp.cancel((z - F) / ((z - F) ** 2 - 1))
        den = sp.Poly(sp.denom(sp.cancel(F)), z).degree()
        rows.append((m, den, 2 ** (m + 1) - 2))
    return rows


def algebra_degree_bound(m, N=8, r=2, Delta=2):
    """Theorem 3.2: coordinate degree d >= (2^{m+1}-2) / (N r Delta) for any N-dimensional algebra."""
    return math.ceil((2 ** (m + 1) - 2) / (N * r * Delta))


def octonion_simulation_cost():
    """Theorem 3.3: one octonion product = 64 scalar products and 56 additions (structure constants +-1)."""
    E = np.eye(8)
    nz = sum(int(np.count_nonzero(oct_mul(E[a], E[b]))) for a in range(8) for b in range(8))
    return {"scalar_products_per_octonion_product": nz, "additions": nz - 8}


# =====================================================================================================
# 4.  Frontier C: quaternionic Gram matrices, SDP and SOS
# =====================================================================================================


def binary_icosahedral():
    """The 120 unit quaternions of 2I with coordinates in Q(sqrt 5) (EXACT)."""
    h = QR(Fraction(1, 2), 0, 5)
    phi2 = QR(Fraction(1, 4), Fraction(1, 4), 5)    # phi/2
    iphi2 = QR(Fraction(-1, 4), Fraction(1, 4), 5)  # 1/(2 phi)
    z, o = QR(0, 0, 5), QR(1, 0, 5)
    els = {}

    def add(c):
        q = Quat(*c)
        els[q.key()] = q
    for t in range(4):
        for s in (1, -1):
            c = [z] * 4
            c[t] = o * s
            add(c)
    for sg in itertools.product((1, -1), repeat=4):
        add([h * s for s in sg])
    base = (z, h, phi2, iphi2)
    for p in itertools.permutations(range(4)):
        if sum(1 for a in range(4) for b in range(a + 1, 4) if p[a] > p[b]) % 2:
            continue
        for sg in itertools.product((1, -1), repeat=4):
            add([base[p[t]] * sg[t] for t in range(4)])
    return list(els.values())


def icosahedral_witness():
    """Theorem 4.4(b): Cay(2I, C_3): p ~ q iff Re(p conj q) = -1/2 iff (p conj q)^2 + p conj q + 1 = 0.
    The identity map is a quaternionic Root-3 colouring (Gram rank 4); the graph is not 3-colourable;
    its 1200 edges form 400 edge-disjoint triangles (breaking regime).  EXACT over Q(sqrt 5)."""
    P = binary_icosahedral()
    keys = {q.key() for q in P}
    closed = all((p * q).key() in keys for p in P for q in P)
    E, cube_root = [], True
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            r = P[a] * P[b].conj()
            if r.re() == QR(Fraction(-1, 2), 0, 5):
                E.append((a, b))
                cube_root &= (r * r + r + Quat(QR(1, 0, 5))).is_zero()
    G = Graph(len(P), E, "Cay(2I,C3)")
    comps, ntri = triangle_components(G)
    Re = np.array([[float((p * q.conj()).re()) for q in P] for p in P])
    return {"order": len(P), "closed_under_product": closed, "edges": len(E), "edge_ratio_is_cube_root": cube_root,
            "triangles": ntri, "triangle_components": comps, "3-colourable": colour_count(G, True) > 0,
            "real_Gram_rank": int(np.linalg.matrix_rank(Re, tol=1e-9))}


def tetrahedral_control():
    """2T (24-cell) with the same rule is 3-colourable (through 2T -> Z_3): the relaxation is not always loose."""
    P = [q for q in binary_icosahedral() if all(float(c) in (0.0, 1.0, -1.0, 0.5, -0.5) for c in q.c)]
    E = [(a, b) for a in range(len(P)) for b in range(a + 1, len(P))
         if (P[a] * P[b].conj()).re() == QR(Fraction(-1, 2), 0, 5)]
    return {"order": len(P), "edges": len(E), "3-colourable": colour_count(Graph(len(P), E), True) > 0}


def kneser_rigidity():
    """Theorem 4.4(c): every strict vector 3-colouring of KG(6,2) has the same Gram matrix, of rank 5
    (EXACT over Q).  Triangles = perfect matchings of K_6 force sum-zero triples; the matching system has
    rank 10 (solution space of dimension 5) and the 15 norm forms span Sym^2 of the 5-space."""
    V = list(itertools.combinations(range(6), 2))
    Ms = [M for M in itertools.combinations(range(15), 3)
          if all(not set(V[a]) & set(V[b]) for a, b in itertools.combinations(M, 2))]
    A = sp.Matrix([[1 if e in M else 0 for e in range(15)] for M in Ms])
    rank_match = A.rank()
    basis = sp.Matrix([[1 if i == j else 0 for j in range(6)] for i in range(5)])
    for i in range(5):
        basis[i, 5] = -1  # basis of the sum-zero space
    U = sp.Matrix([[3 * (i in S) - 1 for i in range(6)] for S in V])
    coords = U * basis.T  # coordinates in a basis of the 5-dimensional space
    forms = sp.Matrix([[coords[s, a] * coords[s, b] * (1 if a == b else 2) for a in range(5) for b in range(a, 5)]
                       for s in range(15)])
    G = sp.Matrix(15, 15, lambda s, t: sp.Rational(sum(U[s, i] * U[t, i] for i in range(6)), 12))
    adj_ok = all(G[s, t] == sp.Rational(-1, 2) for s in range(15) for t in range(15) if not set(V[s]) & set(V[t]))
    return {"perfect_matchings": len(Ms), "rank_of_matching_system": rank_match,
            "rank_of_norm_forms": forms.rank(), "unique_Gram_rank": G.rank(), "edges_at_-1/2": adj_ok,
            "3-colourable": colour_count(Graph(15, [(s, t) for s in range(15) for t in range(s + 1, 15)
                                                    if not set(V[s]) & set(V[t])]), True) > 0}


def icosahedral_root3_dimension():
    """Theorem 4.4(b), sharp form (NUMERIC rank computations on exact data): the triangle-sum-free space W of
    Cay(2I,C_3) is spanned by the 8 coordinates of q -> (q, sigma(q)) (sigma: sqrt5 -> -sqrt5); the unit-norm
    conditions leave the one-parameter family B = (1-t) I_4 + t I_4' (t in [0,1]); so every strict vector
    3-colouring has rank 4 (t = 0, 1) or 8, and the Root-3 dimension is exactly 4."""
    P = binary_icosahedral()
    x = np.array([[float(c) for c in q.c] + [float(QR(c.a, -c.b, 5)) for c in q.c] for q in P])
    E = [(a, b) for a in range(120) for b in range(a + 1, 120)
         if (P[a] * P[b].conj()).re() == QR(Fraction(-1, 2), 0, 5)]
    T = Graph(120, E).triangles()
    A = np.zeros((len(T), 120))
    for i, t in enumerate(T):
        A[i, list(t)] = 1
    dimW = 120 - np.linalg.matrix_rank(A)
    iu = np.triu_indices(8)
    F = np.array([np.outer(v, v)[iu] * np.where(iu[0] == iu[1], 1, 2) for v in x])
    rk = np.linalg.matrix_rank(F, tol=1e-9)
    kern = np.linalg.svd(F)[2][rk:]
    K = np.zeros((8, 8))
    K[iu] = kern[0]
    K = (K + K.T) / 2
    K = K / np.abs(np.diag(K)).max()
    diag_family = np.allclose(np.abs(np.diag(K)), 1) and np.allclose(K - np.diag(np.diag(K)), 0) \
        and np.allclose(np.diag(K)[:4], -np.diag(K)[4:])
    return {"dim_W": int(dimW), "coordinates_span_W": int(np.linalg.matrix_rank(x)) == dimW and np.allclose(A @ x, 0),
            "affine_family_dim": 36 - int(rk), "family_is_(1-t)I4+tI4'": bool(diag_family), "root3_dimension": 4}


def kneser_root3_dimension(k):
    """Root-3 dimension of KG(3k,k) (NUMERIC ranks): dim W = 3k-1 and the norm forms span Sym^2(R^{3k-1}),
    so the Gram matrix is unique and the Root-3 dimension is exactly 3k-1."""
    n = 3 * k
    V = list(itertools.combinations(range(n), k))
    idx = {S: i for i, S in enumerate(V)}
    T = []
    for S in V:
        rest = [i for i in range(n) if i not in S]
        for Tt in itertools.combinations(rest, k):
            U = tuple(i for i in rest if i not in Tt)
            if S < Tt < U:
                T.append((idx[S], idx[Tt], idx[U]))
    A = np.zeros((len(T), len(V)))
    for i, t in enumerate(T):
        A[i, list(t)] = 1
    dimW = len(V) - np.linalg.matrix_rank(A)
    U = np.array([[3 * (i in S) - 1 for i in range(n)] for S in V], float)
    Bz = np.linalg.qr(np.vstack([np.ones(n), np.eye(n)]).T)[0][:, 1:n]
    u = U @ Bz
    iu = np.triu_indices(n - 1)
    rk = np.linalg.matrix_rank(np.array([np.outer(a, a)[iu] for a in u]), tol=1e-8)
    return {"graph": "KG(%d,%d)" % (n, k), "chi": k + 2, "vertices": len(V), "triangles": len(T), "dim_W": int(dimW),
            "norm_form_rank": int(rk), "dim_Sym2": len(iu[0]),
            "root3_dimension": n - 1 if (dimW == n - 1 and rk == len(iu[0])) else None}


def quaternion_matrix(Ms):
    """(A,B,C,D) real n x n -> complex 2n x 2n representation of A + Bi + Cj + Dk."""
    A, B, C, D = Ms
    P = A + 1j * B
    Q = C + 1j * D
    return np.block([[P, Q], [-Q.conj(), P.conj()]])


def reprojection_checks(trials=50, n=6, seed=4):
    """Lemma 4.1 (NUMERIC): for random quaternionic Hermitian PSD M = X X*, the complex representation is
    PSD and Re(M) is PSD; quaternionic rank 1 has real-part rank <= 4."""
    rng = np.random.default_rng(seed)
    worst_c, worst_r, max_rank1 = 0.0, 0.0, 0
    for _ in range(trials):
        k = rng.integers(1, n + 1)
        Xq = [rng.normal(size=(n, k)) for _ in range(4)]
        Xc = np.block([[Xq[0] + 1j * Xq[1], Xq[2] + 1j * Xq[3]], [-(Xq[2] - 1j * Xq[3]), Xq[0] - 1j * Xq[1]]])
        Mc = Xc @ Xc.conj().T
        worst_c = min(worst_c, np.linalg.eigvalsh(Mc).min())
        ReM = Mc[:n, :n].real
        worst_r = min(worst_r, np.linalg.eigvalsh((ReM + ReM.T) / 2).min())
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    max_rank1 = int(np.linalg.matrix_rank(q @ q.T, tol=1e-9))
    return {"min_eig_complex_rep": worst_c, "min_eig_real_part": worst_r, "rank_of_Re(qq*)": max_rank1}


def hermitian_square_is_real_sos():
    """Theorem 4.5 (EXACT, sympy): for p = a + b i + c j + d k with central polynomial coefficients,
    conj(p) p = a^2 + b^2 + c^2 + d^2."""
    a, b, c, d = sp.symbols("a b c d")
    p = Quat(a, b, c, d)
    s = p.conj() * p
    return all(sp.expand(t) == 0 for t in s.c[1:]) and sp.expand(s.c[0] - (a ** 2 + b ** 2 + c ** 2 + d ** 2)) == 0


def albert_jordan_check(trials=5, seed=5):
    """Proposition 4.6 (NUMERIC): the Jordan identity (A o B) o (A o A) = A o (B o (A o A)) holds on H_3(O)
    and fails on H_4(O): there is no octonionic Hermitian (Jordan) cone of size n >= 4."""
    rng = np.random.default_rng(seed)

    def herm(n):
        M = np.zeros((n, n, 8))
        for i in range(n):
            M[i, i, 0] = rng.normal()
            for j in range(i + 1, n):
                x = rng.normal(size=8)
                M[i, j], M[j, i] = x, oct_conj(x)
        return M

    def mm(A, B):
        n = A.shape[0]
        C = np.zeros((n, n, 8))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i, j] += oct_mul(A[i, k], B[k, j])
        return C
    jp = lambda A, B: (mm(A, B) + mm(B, A)) / 2
    res = {}
    for n in (3, 4):
        worst = 0.0
        for _ in range(trials):
            A, B = herm(n), herm(n)
            A2 = jp(A, A)
            worst = max(worst, np.abs(jp(jp(A, B), A2) - jp(A, jp(B, A2))).max())
        res["H_%d(O) Jordan defect" % n] = worst
    return res


def cube_root_ratio_identity(trials=200, seed=6):
    """Prop. 4.2 (NUMERIC): for unit r in H or O, r^2 + r + 1 = 0 iff Re r = -1/2."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        s = rng.normal(size=7)
        s /= np.linalg.norm(s)
        r = np.concatenate([[-0.5], math.sqrt(3) / 2 * s])
        worst = max(worst, np.linalg.norm(oct_mul(r, r) + r + np.eye(8)[0]))
    return worst


# =====================================================================================================
# Report
# =====================================================================================================


def report():
    t0 = time.time()
    hdr = lambda s: print("\n" + "=" * 96 + "\n" + s + "\n" + "=" * 96, flush=True)
    hdr("2. Frontier A: quaternionic Q_8 encodings")
    print("spherical roots (t^2+1, t^2+t+1, t^3-1 vanish on 2-spheres), max residual:", spherical_roots_check())
    print("off-axis unit pure quaternions: min |axis product| over random samples:", off_axis_is_rejected())
    for G in (complete(3), complete(4), Graph(4, [(0, 1), (1, 2), (2, 3)], "P4"), Graph(5, [(i, (i + 1) % 5) for i in range(5)], "C5")):
        print("  Q_8 points of Q_G for %-3s: %6d   2^n P(G,3) = %6d" % (G.name, q8_points_count(G), 2 ** G.n * colour_count(G)))
    for n in (4, 5, 6):
        print("  Clifford point (Thm 2.3):", clifford_point(n))
    print("  Peres witness (Prop 2.4):", peres_witness())
    print("  Pauli transfer, local hypotheses (Thm 2.5):", pauli_transfer_check())

    hdr("3. Frontier B: octonions")
    print("  octonion identities:", octonion_checks())
    for m, den, pred in ladder_degrees():
        print("  ladder L_%d: denominator degree %4d (= 2^{m+1}-2: %s); octonionic coordinate degree bound "
              "(r=2, Delta=2): d >= %d" % (m, den, den == pred, algebra_degree_bound(m)))
    print("  simulation cost:", octonion_simulation_cost())

    hdr("4. Frontier C: quaternionic Gram matrices, SDP and SOS")
    print("  unit r with Re r = -1/2 satisfies r^2+r+1 = 0 in O (max residual):", cube_root_ratio_identity())
    print("  re-projection:", reprojection_checks())
    print("  quaternionic Hermitian square = real sum of four squares:", hermitian_square_is_real_sos())
    print("  binary icosahedral witness:", icosahedral_witness())
    print("  binary tetrahedral control:", tetrahedral_control())
    print("  KG(6,2) rigidity:", kneser_rigidity())
    print("  Root-3 dimension of Cay(2I,C3):", icosahedral_root3_dimension())
    for k in (2, 3, 4):
        print("  Root-3 dimension:", kneser_root3_dimension(k))
    print("  Albert algebra:", albert_jordan_check())
    print("\nTotal %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    report()
