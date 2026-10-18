#!/usr/bin/env python3
"""
quantum_nonlocality_assault.py

Computations for the draft "Non-Tracial Operator Fields, Fibonacci Braids, and the
Non-Natural Obstructions of Hopf Deformations" (Sections 12-16, continuing the drafts
"Nonlocal Operator Fields, Anyonic Braids, ..." (Sections 7-11) and the hypercomplex
addendum (Sections 1-6)).  Standalone: numpy + the standard library; networkx is used
only by the optional census in B.6 and is skipped if absent.

  Part A  non-tracial operator fields
          A.1 unitarity forcing: x*x = 1 and x^3 = 1 imply x x* = 1 (exact NC identity)
          A.2 the Cuntz algebra O_n on l^2(N): exact word calculus s_mu s_nu^*
          A.3 Cuntz colourings from 3-colourings (S_3-orbit (x) l^2), proper isometries
          A.4 non-tracial Positivstellensatz certificate for K_4 in Cuntz variables (degree 4)
          A.5 the strip gadget is operator-faithful (exact NC identities)
  Part B  Fibonacci braids
          B.1 Fibonacci fusion trees, F and R, braid relations, unitarity, dimensions
          B.2 Fibonacci quantum trace = (-1)^{|beta|} x Kauffman bracket at A = e^{-2 pi i/5}
          B.3 density certificate for B_3 (Galois conjugates in the rational gauge)
          B.4 medial links: identical Fibonacci amplitudes for W_6 and H_7
          B.5 Alexander invariant (HOMFLY-PT at a = 1) of the oriented medial links,
              all orientations: it separates W_6 from H_7
          B.6 census of exact and absolute Fibonacci amplitudes (needs networkx)
          B.7 Fibonacci Reshetikhin-Turaev invariants of lens spaces L(p,1)
  Part C  Hopf deformations and Kronecker multiplicities
          C.1 characters of S_n (Murnaghan-Nakayama on beta-sets)
          C.2 Kronecker coefficients: identities, dimension check, saturation holes
          C.3 rectangular Kronecker coefficients: census of zero weights
          C.4 plethysm coefficients h_m[h_d] and a Foulkes difference
          C.5 the group-like coproduct of S_n deforms to the Hecke algebra only at q = 1
          C.6 S_3-isotypic multiplicities of C[Col(G)]: occurrence <=> 3-colourable

Run:  python quantum_nonlocality_assault.py            full report
      python quantum_nonlocality_assault.py --quick    skips the census and the larger tables
      python quantum_nonlocality_assault.py --snappy   full report plus an independent SnapPy/spherogram
                                                       recomputation of the census invariants (optional packages)
"""
import argparse
import cmath
import itertools
import math
import random
import time
from fractions import Fraction as Fr
from functools import lru_cache

import numpy as np

W3 = cmath.exp(2j * math.pi / 3)
PHI = (1 + 5 ** 0.5) / 2


# =============================================================================
# Part 0. Exact arithmetic
# =============================================================================
def ladd(p, q, s=1):
    r = dict(p)
    for e, c in q.items():
        v = r.get(e, 0) + s * c
        if v:
            r[e] = v
        else:
            r.pop(e, None)
    return r


def lmul(p, q):
    r = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            r[e1 + e2] = r.get(e1 + e2, 0) + c1 * c2
    return {e: c for e, c in r.items() if c}


def lscale(p, c=1, shift=0):
    return {e + shift: c * v for e, v in p.items() if c * v}


def lpow(p, k):
    r = {0: 1}
    for _ in range(k):
        r = lmul(r, p)
    return r


def leval(p, z):
    return sum(c * z ** e for e, c in p.items())


def lunit_ratio(p, q):
    """(sign, shift) with p = sign * A^shift * q, or None."""
    if not p or not q or len(p) != len(q):
        return None
    ep, eq = min(p), min(q)
    shift = ep - eq
    if p[ep] == q[eq]:
        s = 1
    elif p[ep] == -q[eq]:
        s = -1
    else:
        return None
    for e, c in q.items():
        if p.get(e + shift) != s * c:
            return None
    return s, shift


D_LP = {2: -1, -2: -1}          # delta = -A^2 - A^{-2}


class NC:
    """Element of a free algebra: dict word (tuple of letters) -> rational coefficient."""
    __slots__ = ("d",)

    def __init__(self, d=None):
        self.d = {w: c for w, c in (d or {}).items() if c}

    @staticmethod
    def var(x):
        return NC({(x,): 1})

    @staticmethod
    def const(c):
        return NC({(): c})

    def __add__(self, o):
        o = _nc(o)
        r = dict(self.d)
        for w, c in o.d.items():
            v = r.get(w, 0) + c
            if v:
                r[w] = v
            else:
                r.pop(w, None)
        return NC(r)

    __radd__ = __add__

    def __neg__(self):
        return NC({w: -c for w, c in self.d.items()})

    def __sub__(self, o):
        return self + (-_nc(o))

    def __rsub__(self, o):
        return _nc(o) - self

    def __mul__(self, o):
        if not isinstance(o, NC):
            return NC({w: c * o for w, c in self.d.items()})
        r = {}
        for w1, c1 in self.d.items():
            for w2, c2 in o.d.items():
                w = w1 + w2
                r[w] = r.get(w, 0) + c1 * c2
        return NC(r)

    def __rmul__(self, o):
        return NC({w: o * c for w, c in self.d.items()})

    def __pow__(self, k):
        r = NC.const(1)
        for _ in range(k):
            r = r * self
        return r

    def is_zero(self):
        return not self.d

    def degree(self):
        return max((len(w) for w in self.d), default=-1)

    def star(self, swap):
        """Involution: reverse words, apply the letter involution `swap`, real coefficients."""
        return NC({tuple(swap(x) for x in reversed(w)): c for w, c in self.d.items()})


def _nc(x):
    return x if isinstance(x, NC) else NC.const(x)


def nsum(it):
    s = NC()
    for x in it:
        s = s + x
    return s


def rank_mod_p(M, p=3):
    A = [[int(x) % p for x in row] for row in M]
    m = len(A)
    n = len(A[0]) if m else 0
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = pow(A[r][c], p - 2, p)
        A[r] = [(x * inv) % p for x in A[r]]
        for i in range(m):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[r])]
        r += 1
        if r == m:
            break
    return r


def det_fraction(M):
    M = [row[:] for row in M]
    n = len(M)
    det = Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return Fr(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            det = -det
        det *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return det


def interpolate(points, values):
    """Exact Lagrange interpolation; returns integer coefficient list (constant first)."""
    deg = len(points) - 1
    coeffs = [Fr(0)] * (deg + 1)
    for k, (xk, yk) in enumerate(zip(points, values)):
        num, den = [Fr(1)], Fr(1)
        for j, xj in enumerate(points):
            if j != k:
                num = [a - xj * b for a, b in zip([Fr(0)] + num, num + [Fr(0)])]
                den *= (xk - xj)
        for a in range(len(num)):
            coeffs[a] += yk * num[a] / den
    assert all(c.denominator == 1 for c in coeffs)
    return [int(c) for c in coeffs]


def normalize_poly(c):
    """Strip zeros at both ends and fix the sign: the class of c up to units +-t^k."""
    c = list(c)
    while c and c[-1] == 0:
        c.pop()
    while c and c[0] == 0:
        c.pop(0)
    if c and c[-1] < 0:
        c = [-x for x in c]
    return tuple(c)


# =============================================================================
# Part A. Non-tracial operator fields, Cuntz isometries
# =============================================================================
def unitarity_forcing():
    """x*x = 1 and x^3 = 1 force x x* = 1:
       x* - x^2 = x*(1 - x^3) + (x*x - 1) x^2,   x x* - 1 = x (x* - x^2) + (x^3 - 1)."""
    x, s = NC.var("x"), NC.var("s")                  # s stands for x^*
    iso = s * x - 1
    cube = x ** 3 - 1
    terms1 = [-(s * cube), iso * x * x]               # each term is (word) * relation * (word)
    id1 = (s - x * x - nsum(terms1)).is_zero()
    terms2 = [x * t for t in terms1] + [cube]
    id2 = (x * s - 1 - nsum(terms2)).is_zero()
    d1, d2 = max(t.degree() for t in terms1), max(t.degree() for t in terms2)
    # general p(x) with p(0) != 0: x g(x) = 1 for the polynomial g = -(p(x) - p(0))/(p(0) x),
    # so x* = x*(x g(x)) = g(x) and x x* = x g(x) = 1.  Check for p = x^2 + x + 1 (the Bayer factor):
    g = -(x + 1)                                      # x (-(x+1)) = -x^2 - x = 1 modulo x^2 + x + 1
    p = x * x + x + 1
    id3 = ((x * g - 1) + p).is_zero()
    return {"x* - x^2 in <x*x-1, x^3-1>: identity, degree": (id1, d1),
            "x x* - 1 in <x*x-1, x^3-1>: identity, degree": (id2, d2),
            "x g(x) = 1 mod p for p = x^2+x+1, g = -(x+1)": id3}


class CuntzRep:
    """Canonical representation of O_n on l^2(N): s_i e_k = e_{n k + i}."""

    def __init__(self, n):
        self.n = n

    def s(self, i, k):
        return self.n * k + i

    def s_star(self, i, m):
        return (m - i) // self.n if m % self.n == i else None

    def apply(self, op, k):
        """op: list of (letter, starred) read as an operator product left to right."""
        for i, star in reversed(op):
            if k is None:
                return None
            k = self.s_star(i, k) if star else self.s(i, k)
        return k


def cuntz_word_product(a, b):
    """(s_mu s_nu^*)(s_al s_be^*) in normal form (mu, nu) or None for 0."""
    (mu, nu), (al, be) = a, b
    if al[:len(nu)] == nu:
        return (mu + al[len(nu):], be)
    if nu[:len(al)] == al:
        return (mu, be + nu[len(al):])
    return None


def _as_op(mu, nu):
    return [(i, False) for i in mu] + [(i, True) for i in reversed(nu)]


def cuntz_checks(n=3, window=2000, trials=400, seed=1):
    rep = CuntzRep(n)
    rel = all(rep.s_star(i, rep.s(j, k)) == (k if i == j else None) for i in range(n) for j in range(n)
              for k in range(window))
    partition = all(sum(1 for i in range(n) if rep.s_star(i, m) is not None) == 1 for m in range(window))
    proper = all(any(rep.s_star(i, m) is None for m in range(n)) for i in range(n))
    rng = random.Random(seed)
    ok = True
    for _ in range(trials):
        w = [tuple(rng.randrange(n) for _ in range(rng.randint(0, 3))) for _ in range(4)]
        a, b = (w[0], w[1]), (w[2], w[3])
        prod = cuntz_word_product(a, b)
        opab = _as_op(*a) + _as_op(*b)
        for k in range(200):
            lhs = rep.apply(opab, k)
            rhs = None if prod is None else rep.apply(_as_op(*prod), k)
            ok &= (lhs == rhs)
    return {"s_i* s_j = delta_ij (window)": rel, "sum_i s_i s_i* = 1 (window)": partition,
            "s_i s_i* != 1 (proper isometries)": proper,
            "word calculus = representation (random words)": ok,
            "K_0 relation [1] = n[1]: K_0(O_n) = Z/(n-1)": f"Z/{n - 1}" if n > 2 else "0"}


S3 = list(itertools.permutations(range(3)))


def cuntz_colouring(nv, edges, col, window=600):
    """Isometries S_{v,c} on l^2(N) = C^6 (x) l^2(N) from a proper colouring col, via the
    S_3-orbit: S_{v,c} e_m = e_{6 floor(m/2) + t}, t the (m mod 2)-th g in S_3 with g(col v) = c."""
    T = {(v, c): [gi for gi, g in enumerate(S3) if g[col[v]] == c] for v in range(nv) for c in range(3)}

    def S(v, c, m):
        return 6 * (m // 2) + T[v, c][m % 2]

    def S_star(v, c, k):
        q, r = divmod(k, 6)
        return 2 * q + T[v, c].index(r) if r in T[v, c] else None
    iso = all(S_star(v, c, S(v, c, m)) == m for v in range(nv) for c in range(3) for m in range(window))
    orth = all(S_star(v, c, S(v, c2, m)) is None for v in range(nv) for c in range(3) for c2 in range(3)
               if c != c2 for m in range(window))
    cuntz_sum = all(sum(1 for c in range(3) if S_star(v, c, k) is not None) == 1 for v in range(nv)
                    for k in range(6 * window))
    edge_orth = all(S_star(u, c, S(v, c, m)) is None and S_star(v, c, S(u, c, m)) is None
                    for u, v in edges for c in range(3) for m in range(window))
    proper = all(any(S_star(v, c, k) is None for k in range(6)) for v in range(nv) for c in range(3))
    # the induced Bayer unitaries X_v = sum_c w^c S S* are diagonal: X_v e_k = w^{c_v(k)} e_k
    cv = {v: [next(c for c in range(3) if S_star(v, c, k) is not None) for k in range(6 * 50)] for v in range(nv)}
    bayer = all(abs(W3 ** (2 * cv[u][k]) + W3 ** (cv[u][k] + cv[v][k]) + W3 ** (2 * cv[v][k])) < 1e-12
                for u, v in edges for k in range(6 * 50))
    return {"isometries S*S = 1": iso, "S_{v,c}*S_{v,c'} = 0": orth, "sum_c S S* = 1": cuntz_sum,
            "edges: S_{u,c}*S_{v,c} = 0": edge_orth, "proper (SS* != 1)": proper,
            "Bayer relations for X_v = sum w^c S S*": bayer}


def proper_colouring(nv, edges):
    col = [-1] * nv
    adj = [[] for _ in range(nv)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def rec(k):
        if k == nv:
            return True
        for c in range(3):
            if all(col[w] != c for w in adj[k]):
                col[k] = c
                if rec(k + 1):
                    return True
        col[k] = -1
        return False
    return list(col) if rec(0) else None


CUNTZ_GRAPHS = {
    "K3": (3, [(0, 1), (1, 2), (0, 2)]),
    "C5": (5, [(i, (i + 1) % 5) for i in range(5)]),
    "Petersen": (10, [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
                 + [(i, i + 5) for i in range(5)]),
    "W6": (7, [(k, (k + 1) % 6) for k in range(6)] + [(k, 6) for k in range(6)]),
    "prism": (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
}


def cuntz_colourings_report(window=400):
    out = {}
    for name, (nv, edges) in CUNTZ_GRAPHS.items():
        col = proper_colouring(nv, edges)
        out[name] = all(cuntz_colouring(nv, edges, col, window).values())
    return out


def k4_cuntz_certificate():
    """-1 = sum_c (1-Q_c)^*(1-Q_c) - [sum_c (Q_c^2 - Q_c) - eps], Q_c = sum_v S_{v,c} S_{v,c}^*,
    with every bracket written in the Cuntz relations of the K_4 colouring system."""
    S = {(v, c): NC.var(("S", v, c)) for v in range(4) for c in range(3)}
    T = {(v, c): NC.var(("T", v, c)) for v in range(4) for c in range(3)}     # T = S^*
    swap = (lambda x: ("T",) + x[1:] if x[0] == "S" else ("S",) + x[1:])
    iso = {(v, c): T[v, c] * S[v, c] - 1 for v in range(4) for c in range(3)}
    edge = {(u, v, c): T[u, c] * S[v, c] for u in range(4) for v in range(4) if u != v for c in range(3)}
    ssum = {v: nsum(S[v, c] * T[v, c] for c in range(3)) - 1 for v in range(4)}
    Q = [nsum(S[v, c] * T[v, c] for v in range(4)) for c in range(3)]
    selfadj = all(((NC.const(1) - Q[c]).star(swap) - (NC.const(1) - Q[c])).is_zero() for c in range(3))
    qq = [nsum(S[v, c] * iso[v, c] * T[v, c] for v in range(4))
          + nsum(S[u, c] * edge[u, v, c] * T[v, c] for u in range(4) for v in range(4) if u != v) for c in range(3)]
    q_ok = all((Q[c] * Q[c] - Q[c] - qq[c]).is_zero() for c in range(3))
    eps = nsum(ssum.values())
    eps_ok = (Q[0] + Q[1] + Q[2] - 4 - eps).is_zero()
    sos = nsum((NC.const(1) - Q[c]).star(swap) * (NC.const(1) - Q[c]) for c in range(3))
    ideal = nsum(qq) - eps
    cert = (sos + 1 - ideal).is_zero()
    deg = max(sos.degree(), max(q.degree() for q in qq), eps.degree())
    return {"(1-Q_c)^* = 1-Q_c": selfadj, "Q_c^2 - Q_c in Cuntz ideal (explicit)": q_ok,
            "sum Q_c - 4 in Cuntz ideal (explicit)": eps_ok, "certificate identity": cert, "degree": deg}


def strip_gadget_identities():
    """Operator faithfulness of the strip gadget {u,p,q},{p,q,s},{s,v,t} for an edge uv:
    e_s = e_u, e_t = 1 - e_u - e_v, e_p = e_u shifted by 1, e_q by 2; every strip relation is
    an explicit combination of the relations of the edge uv (exact)."""
    e = {(x, c): NC.var((x, c)) for x in "uv" for c in range(3)}
    idem = {(x, c): e[x, c] * e[x, c] - e[x, c] for x in "uv" for c in range(3)}
    vsum = {x: nsum(e[x, c] for c in range(3)) - 1 for x in "uv"}
    orth = {(x, c, d): e[x, c] * e[x, d] for x in "uv" for c in range(3) for d in range(3) if c != d}
    edge = {(x, y, c): e[x, c] * e[y, c] for (x, y) in (("u", "v"), ("v", "u")) for c in range(3)}
    ext = {}
    for c in range(3):
        ext["u", c], ext["v", c] = e["u", c], e["v", c]
        ext["s", c] = e["u", c]
        ext["t", c] = NC.const(1) - e["u", c] - e["v", c]
        ext["p", c] = e["u", (c + 1) % 3]
        ext["q", c] = e["u", (c + 2) % 3]
    ok = True
    # idempotents and sums
    for c in range(3):
        ok &= ((ext["t", c] * ext["t", c] - ext["t", c])
               - (idem["u", c] + idem["v", c] + edge["u", "v", c] + edge["v", "u", c])).is_zero()
    ok &= (nsum(ext["t", c] for c in range(3)) - 1 - (-vsum["u"] - vsum["v"])).is_zero()
    ok &= all((nsum(ext[x, c] for c in range(3)) - 1 - vsum["u"]).is_zero() for x in "pqs")
    # strip edges, both orders, as explicit combinations
    strip_edges = [("u", "p"), ("u", "q"), ("p", "q"), ("p", "s"), ("q", "s"), ("s", "v"), ("s", "t"), ("v", "t")]
    for x, y in strip_edges:
        for a, b in ((x, y), (y, x)):
            for c in range(3):
                prod = ext[a, c] * ext[b, c]
                # reduce: every product is a sum of orth/edge/idem relations (with signs) -> check membership
                # by expressing in the relation basis via exact rational solve on the finite list of relations
                rels = list(idem.values()) + list(orth.values()) + list(edge.values())
                ok &= _in_span(prod, rels)
    return {"strip extension satisfies all strip relations modulo the relations of uv": ok}


class Cert:
    """Formal element sum_k coef_k * L_k * r_k * R_k of a two-sided ideal, r_k a generator relation.
    value() expands it in the free algebra; degree() is max(deg L + deg r + deg R)."""

    def __init__(self, terms=None):
        self.terms = list(terms or [])

    @staticmethod
    def rel(r):
        return Cert([(1, NC.const(1), r, NC.const(1))])

    def __add__(self, o):
        return Cert(self.terms + o.terms)

    def __sub__(self, o):
        return Cert(self.terms + [(-c, L, r, R) for c, L, r, R in o.terms])

    def __neg__(self):
        return Cert([(-c, L, r, R) for c, L, r, R in self.terms])

    def scale(self, k):
        return Cert([(k * c, L, r, R) for c, L, r, R in self.terms])

    def lmul(self, p):
        return Cert([(c, p * L, r, R) for c, L, r, R in self.terms])

    def rmul(self, p):
        return Cert([(c, L, r, R * p) for c, L, r, R in self.terms])

    def value(self):
        return nsum((L * r * R) * c for c, L, r, R in self.terms)

    def degree(self):
        return max(L.degree() + r.degree() + R.degree() for c, L, r, R in self.terms)


def triangle_certificate(e, tri, c0):
    """Q_{c0} - 1 in the ideal of the triangle tri, Q_c = sum_{w in tri} e_{w,c}, from the defining relations
    e^2 - e, sum_c e_{w,c} - 1 and e_{w,c} e_{w',c} (w != w') only.  With a = Q_{c0}, b = Q_{c0+1}:
      s     = ab + ba - 4a - 4b + 6 = R_2 - R_0 - R_1 - (u eps + eps u + eps^2 - eps),  u = 3 - a - b,
      kappa = ab - ba = -(a s - s a - R_0 b + b R_0)/3,
      ab - a = -(a s - 2 R_0 b + a kappa + 4 R_0)/2,   ab - b = -(s b - 2 a R_1 + kappa b + 4 R_1)/2,
      a - 1 = (-s + 2(ab - a) - kappa + 4((ab - b) - (ab - a)))/6."""
    cols = [c0 % 3, (c0 + 1) % 3, (c0 + 2) % 3]
    Q = {c: nsum(e[w, c] for w in tri) for c in range(3)}

    def rcert(c):
        out = Cert()
        for w in tri:
            out = out + Cert.rel(e[w, c] * e[w, c] - e[w, c])
        for w in tri:
            for w2 in tri:
                if w != w2:
                    out = out + Cert.rel(e[w, c] * e[w2, c])
        return out
    eps = Cert()
    for w in tri:
        eps = eps + Cert.rel(nsum(e[w, c] for c in range(3)) - 1)
    a, b = Q[cols[0]], Q[cols[1]]
    R0, R1, R2 = rcert(cols[0]), rcert(cols[1]), rcert(cols[2])
    u = NC.const(3) - a - b
    s = R2 - R0 - R1 - eps.lmul(u) - eps.rmul(u) - eps.lmul(eps.value()) + eps
    kappa = (s.lmul(a) - s.rmul(a) - R0.rmul(b) + R0.lmul(b)).scale(Fr(-1, 3))
    X = (s.lmul(a) - R0.rmul(b).scale(2) + kappa.lmul(a) + R0.scale(4)).scale(Fr(-1, 2))
    Y = (s.rmul(b) - R1.lmul(a).scale(2) + kappa.rmul(b) + R1.scale(4)).scale(Fr(-1, 2))
    cert = (-s + X.scale(2) - kappa + (Y - X).scale(4)).scale(Fr(1, 6))
    return cert, a - 1


def triangle_and_strip_restriction():
    """(i) Q_c - 1 in I(K_3) for every c, explicitly, degree 4 (triangle rigidity).
    (ii) In the strip {u,p,q},{p,q,s},{s,v,t}: e_{u,c} - e_{s,c} = (Q^{upq}_c - 1) - (Q^{spq}_c - 1) and
    e_{u,c} e_{v,c} = (e_{u,c} - e_{s,c}) e_{v,c} + e_{s,c} e_{v,c} lie in I(strip): the edge relations of uv hold
    in A(G^nat), so e_{v,c} -> e_{v,c} is a unital *-homomorphism A(G) -> A(G^nat)."""
    e = {(x, c): NC.var((x, c)) for x in "uvpqst" for c in range(3)}
    tri_ok, tri_deg = True, 0
    for c in range(3):
        cert, target = triangle_certificate(e, ("u", "p", "q"), c)
        tri_ok &= (cert.value() - target).is_zero()
        tri_deg = max(tri_deg, cert.degree())
    ok, degs = True, []
    for c in range(3):
        c1, _ = triangle_certificate(e, ("u", "p", "q"), c)
        c2, _ = triangle_certificate(e, ("s", "p", "q"), c)
        diff = c1 - c2
        ok &= (diff.value() - (e["u", c] - e["s", c])).is_zero()
        cert = diff.rmul(e["v", c]) + Cert.rel(e["s", c] * e["v", c])
        ok &= (cert.value() - e["u", c] * e["v", c]).is_zero()
        degs.append(cert.degree())
        cert = diff.lmul(e["v", c]) + Cert.rel(e["v", c] * e["s", c])
        ok &= (cert.value() - e["v", c] * e["u", c]).is_zero()
        degs.append(cert.degree())
    return {"triangle: Q_c - 1 in I(K_3), all c (explicit)": tri_ok, "triangle certificate degree": tri_deg,
            "strip: e_uc e_vc and e_vc e_uc in I(strip) (explicit)": ok, "strip certificate degree": max(degs)}


def fphp_certificate(pigeons, holes, adj):
    """Degree-2 noncommutative SOS certificate for the graph pigeonhole principle (pigeon i -> holes adj[i]):
       (n - m) = sum_j (1 - h_j)^*(1 - h_j) + sum_i (sum_{j in adj i} x_ij - 1) - sum_j (h_j^2 - h_j),
    h_j = sum_{i: j in adj i} x_ij, with h_j^2 - h_j = sum_i (x_ij^2 - x_ij) + sum_{i != k} x_ij x_kj."""
    x = {(i, j): NC.var(("x", i, j)) for i in range(pigeons) for j in adj[i]}
    h = {j: nsum(x[i, j] for i in range(pigeons) if j in adj[i]) for j in range(holes)}
    hole = Cert()
    for j in range(holes):
        ps = [i for i in range(pigeons) if j in adj[i]]
        for i in ps:
            hole = hole + Cert.rel(x[i, j] * x[i, j] - x[i, j])
            for k in ps:
                if i != k:
                    hole = hole + Cert.rel(x[i, j] * x[k, j])
    pig = Cert()
    for i in range(pigeons):
        pig = pig + Cert.rel(nsum(x[i, j] for j in adj[i]) - 1)
    hole_ok = (hole.value() - nsum(h[j] * h[j] - h[j] for j in range(holes))).is_zero()
    sos = nsum((NC.const(1) - h[j]) * (NC.const(1) - h[j]) for j in range(holes))
    ident = (NC.const(holes - pigeons) - (sos + pig.value() - hole.value())).is_zero()
    return {"m": pigeons, "n": holes, "identity": ident and hole_ok,
            "degree": max(sos.degree(), pig.degree(), hole.degree())}


def fphp_report():
    rows = [fphp_certificate(n + 1, n, [list(range(n))] * (n + 1)) for n in range(2, 6)]
    rng = random.Random(3)
    m, n = 9, 7
    adj = [sorted(rng.sample(range(n), 3)) for _ in range(m)]
    rows.append(dict(fphp_certificate(m, n, adj), graph="random 3-regular-left bipartite"))
    return rows


def cuntz_transfer_and_archimedean():
    """P = S S^* maps the relations of A(G) into the Cuntz ideal (degrees double), and the quadratic
    modules are archimedean: 1 - e^*e = (1-e)^*(1-e) - 2(e^2-e); 1 - S S^* = sum_{c' != c} S_c' S_c'^* - (sum S S^* - 1)."""
    S = {(v, c): NC.var(("S", v, c)) for v in "uv" for c in range(3)}
    T = {(v, c): NC.var(("T", v, c)) for v in "uv" for c in range(3)}
    swap = (lambda x: ("T",) + x[1:] if x[0] == "S" else ("S",) + x[1:])
    P = {k: S[k] * T[k] for k in S}
    iso = {k: T[k] * S[k] - 1 for k in S}
    idem = all(((P[k] * P[k] - P[k]) - S[k] * iso[k] * T[k]).is_zero() for k in S)
    selfadj = all((P[k].star(swap) - P[k]).is_zero() for k in P)
    edge = all((P["u", c] * P["v", c] - S["u", c] * (T["u", c] * S["v", c]) * T["v", c]).is_zero() for c in range(3))
    e = NC.var("e")
    arch_proj = ((NC.const(1) - e * e) - ((NC.const(1) - e) * (NC.const(1) - e) - (e * e - e) * 2)).is_zero()
    csum = nsum(P["u", c] for c in range(3)) - 1
    arch_cuntz = all(((NC.const(1) - P["u", c])
                      - (nsum(T["u", d].star(swap) * T["u", d] for d in range(3) if d != c) - csum)).is_zero()
                     for c in range(3))
    return {"P^2 - P = S (S^*S - 1) S^*": idem, "P^* = P": selfadj,
            "P_uc P_vc = S_uc (S_uc^* S_vc) S_vc^*": edge,
            "archimedean: 1 - e^*e in Sigma^2 + I": arch_proj, "archimedean: 1 - S S^* in Sigma^2 + I": arch_cuntz}


def o3_shift_colouring(nv, edges, col):
    """A 3-colouring gives a unital *-homomorphism C(G) -> O_3, S_{v,c} = s_{(c - col v) mod 3}; checked in the
    Cuntz word calculus.  The Bayer unitaries are X_v = w^{col v} U with U = sum_k w^k s_k s_k^*."""
    S = {(v, c): (((c - col[v]) % 3,), ()) for v in range(nv) for c in range(3)}

    def star(a):
        return (a[1], a[0])
    one = ((), ())
    iso = all(cuntz_word_product(star(S[k]), S[k]) == one for k in S)
    orth = all(cuntz_word_product(star(S[v, c]), S[v, d]) is None
               for v in range(nv) for c in range(3) for d in range(3) if c != d)
    edge = all(cuntz_word_product(star(S[u, c]), S[v, c]) is None and cuntz_word_product(star(S[v, c]), S[u, c]) is None
               for u, v in edges for c in range(3))
    csum = all(sorted(S[v, c][0] for c in range(3)) == [(0,), (1,), (2,)] for v in range(nv))
    # operator check in the canonical representation of O_3 on l^2(N): X_v = sum_c w^c S_vc S_vc^* is diagonal;
    # compute it on a window from the representation and evaluate q_uv(X) and X_v = w^{col v} U there
    rep, window = CuntzRep(3), 300

    def X(v, m):
        """(X_v e_m) / e_m, where S_vc S_vc^* = s_k s_k^* (k = c - col v) is applied in the representation."""
        val = 0
        for c in range(3):
            k = S[v, c][0][0]
            if rep.apply([(k, False), (k, True)], m) == m:
                val += W3 ** c
        return val
    Xd = {v: [X(v, m) for m in range(window)] for v in range(nv)}
    U = [sum(W3 ** k for k in range(3) if rep.apply([(k, False), (k, True)], m) == m) for m in range(window)]
    bayer = all(abs(Xd[u][m] ** 2 + Xd[u][m] * Xd[v][m] + Xd[v][m] ** 2) < 1e-12 for u, v in edges for m in range(window))
    shift = all(abs(Xd[v][m] - W3 ** col[v] * U[m]) < 1e-12 for v in range(nv) for m in range(window))
    return {"S^*S = 1": iso, "S_vc^* S_vd = 0 (c != d)": orth, "edges: S_uc^* S_vc = 0": edge,
            "sum_c S S^* = sum_k s_k s_k^* = 1": csum, "q_uv(X) = 0 on l^2 (window)": bayer,
            "X_v = w^{col v} U": shift}


def magic_square():
    """Mermin-Peres square: a 4-dimensional operator solution with J -> -1.  So J != 1 in the solution group,
    C Gamma/<J+1> != 0 and tr/4 is a tracial state: the alg, C* and tracial levels are all nonempty."""
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    k = np.kron
    A = [[k(X, I2), k(I2, X), k(X, X)], [k(I2, Y), k(Y, I2), k(Y, Y)], [k(X, Y), k(Y, X), k(Z, Z)]]
    E = np.eye(4)
    inv = all(np.allclose(A[i][j] @ A[i][j], E) and np.allclose(A[i][j], A[i][j].conj().T) for i in range(3) for j in range(3))
    comm = all(np.allclose(A[i][a] @ A[i][b], A[i][b] @ A[i][a]) and np.allclose(A[a][i] @ A[b][i], A[b][i] @ A[a][i])
               for i in range(3) for a in range(3) for b in range(3))
    rows = [A[i][0] @ A[i][1] @ A[i][2] for i in range(3)]
    cols = [A[0][j] @ A[1][j] @ A[2][j] for j in range(3)]
    signs = ([int(round(np.trace(r).real / 4)) for r in rows], [int(round(np.trace(c).real / 4)) for c in cols])
    scal = all(np.allclose(r, np.trace(r) / 4 * E) for r in rows + cols)
    return {"A_ij Hermitian involutions": inv, "rows and columns commute": comm, "products are scalars": scal,
            "row signs, column signs": signs, "J -> -1": signs == ([1, 1, 1], [1, 1, -1])}


def _in_span(target, rels):
    """Is target an exact Q-linear combination of rels? (small dense elimination)"""
    keys = sorted(set(k for r in rels for k in r.d) | set(target.d), key=repr)
    idx = {k: i for i, k in enumerate(keys)}
    m, n = len(keys), len(rels)
    M = [[Fr(0)] * (n + 1) for _ in range(m)]
    for j, r in enumerate(rels):
        for k, v in r.d.items():
            M[idx[k]][j] = Fr(v)
    for k, v in target.d.items():
        M[idx[k]][n] = Fr(v)
    r = 0
    for j in range(n):
        p = next((i for i in range(r, m) if M[i][j] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][j]
        M[r] = [x * inv for x in M[r]]
        for i in range(m):
            if i != r and M[i][j] != 0:
                f = M[i][j]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
    return all(M[i][n] == 0 for i in range(r, m))


# =============================================================================
# Part B. Fibonacci braids, Kauffman bracket, Alexander invariant, WRT
# =============================================================================
FIB_F = np.array([[1 / PHI, PHI ** -0.5], [PHI ** -0.5, -1 / PHI]])
FIB_R = (cmath.exp(-4j * math.pi / 5), cmath.exp(3j * math.pi / 5))     # R^{tau tau}_1, R^{tau tau}_tau


def fusion_basis(n, total):
    """Fusion trees of n tau-anyons: (x_1..x_{n-1}), x_k = charge of the first k+1 anyons (0 = 1, 1 = tau)."""
    out = []

    def rec(k, prev, seq):
        if k == n:
            if prev == total:
                out.append(tuple(seq))
            return
        for x in ([1] if prev == 0 else [0, 1]):
            rec(k + 1, x, seq + [x])
    rec(1, 1, [])
    return out


def fib_generator(n, total, i, F=FIB_F, R=FIB_R):
    B = fusion_basis(n, total)
    idx = {b: j for j, b in enumerate(B)}
    M = np.zeros((len(B), len(B)), complex)
    Bmat = F @ np.diag(R) @ F
    for col, b in enumerate(B):
        x = (1,) + b
        if i == 1:
            M[col, col] = R[x[1]]
            continue
        a, mid, c = x[i - 2], x[i - 1], x[i]
        if a == 0:
            M[col, col] = R[c]
        elif c == 0:
            M[col, col] = R[1]
        else:
            for new in (0, 1):
                y = list(x)
                y[i - 1] = new
                key = tuple(y[1:])
                if key in idx:
                    M[idx[key], col] += Bmat[new, mid]
    return M


def fib_rep(n, total):
    return [fib_generator(n, total, i) for i in range(1, n)]


def fibonacci_checks(nmax=8):
    rows = []
    fib = [0, 1]
    while len(fib) < nmax + 3:
        fib.append(fib[-1] + fib[-2])
    for n in range(3, nmax + 1):
        for total in (0, 1):
            g = fib_rep(n, total)
            d = g[0].shape[0]
            br = max([np.abs(g[i] @ g[i + 1] @ g[i] - g[i + 1] @ g[i] @ g[i + 1]).max() for i in range(n - 2)]
                     + [np.abs(g[i] @ g[j] - g[j] @ g[i]).max() for i in range(n - 1) for j in range(i + 2, n - 1)])
            un = max(np.abs(x @ x.conj().T - np.eye(d)).max() for x in g)
            rows.append({"n": n, "total": "tau" if total else "1", "dim": d,
                         "dim = Fibonacci": d == (fib[n] if total else fib[n - 1]),
                         "braid relations": br < 1e-12, "unitary": un < 1e-12})
    F = FIB_F
    hexa = np.abs(F @ F - np.eye(2)).max()
    return {"F^2 = 1": hexa < 1e-14, "rows": rows}


# ---- Temperley-Lieb bracket of braid closures (for B.2) ---------------------
def tl_id(n):
    return tuple([n + i for i in range(n)] + [i for i in range(n)])


def tl_e(n, i):
    p = list(tl_id(n))
    p[i], p[i + 1] = i + 1, i
    p[n + i], p[n + i + 1] = n + i + 1, n + i
    return tuple(p)


def tl_compose(p, q, n):
    res = [None] * (2 * n)
    seen = [False] * n

    def walk(side, k):
        while True:
            if side == "p":
                j = p[k]
                if j < n:
                    return j
                seen[j - n] = True
                side, k = "q", j - n
            else:
                j = q[k]
                if j >= n:
                    return j
                seen[j] = True
                side, k = "p", n + j
    for k in range(n):
        if res[k] is None:
            t = walk("p", k)
            res[k], res[t] = t, k
    for k in range(n, 2 * n):
        if res[k] is None:
            t = walk("q", k)
            res[k], res[t] = t, k
    loops = 0
    for m in range(n):
        if not seen[m]:
            loops += 1
            k = m
            seen[m] = True
            while True:
                j = q[k]
                seen[j] = True
                j2 = p[n + j] - n
                seen[j2] = True
                if j2 == m:
                    break
                k = j2
    return tuple(res), loops


def closure_loops(p, n):
    parent = list(range(2 * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in range(2 * n):
        a, b = find(k), find(p[k])
        if a != b:
            parent[a] = b
    for i in range(n):
        a, b = find(i), find(n + i)
        if a != b:
            parent[a] = b
    return len({find(k) for k in range(2 * n)})


def bracket_closure(word, n):
    """Kauffman bracket of the closure, normalised <O> = 1; sigma = A + A^{-1} e."""
    elem = {tl_id(n): {0: 1}}
    E = [tl_e(n, i) for i in range(n - 1)]
    for g in word:
        i = abs(g) - 1
        a_id, a_e = (1, -1) if g > 0 else (-1, 1)
        out = {}
        for D, c in elem.items():
            out[D] = ladd(out.get(D, {}), lscale(c, 1, a_id))
            R, loops = tl_compose(D, E[i], n)
            out[R] = ladd(out.get(R, {}), lmul(lscale(c, 1, a_e), lpow(D_LP, loops)))
        elem = {D: c for D, c in out.items() if c}
    total = {}
    for D, c in elem.items():
        total = ladd(total, lmul(c, lpow(D_LP, closure_loops(D, n) - 1)))
    return total


BRAID_PANEL = [("trefoil", [1, 1, 1], 2), ("figure-eight", [1, -2, 1, -2], 3), ("5_1", [1] * 5, 2),
               ("5_2", [1, 1, 1, 2, -1, 2], 3), ("6_1", [1, 1, 2, -1, -3, 2, -3], 4),
               ("T(3,4)", [1, 2] * 4, 3), ("Hopf", [1, 1], 2), ("Borromean", [1, -2] * 3, 3),
               ("T(3,3)", [1, 2] * 3, 3), ("granny", [1, 1, 1, 2, 2, 2], 3)]


def fib_quantum_trace(word, n):
    tot = 0
    for total, dq in ((0, 1.0), (1, PHI)):
        g = fib_rep(n, total)
        if g[0].shape[0] == 0:
            continue
        M = np.eye(g[0].shape[0], dtype=complex)
        for w in word:
            G = g[abs(w) - 1]
            M = M @ (G if w > 0 else G.conj().T)
        tot += dq * np.trace(M)
    return tot


def fibonacci_vs_kauffman():
    """Z_Fib(closure) = (-1)^{|beta|} <closure>_unnormalised at A = e^{-2 pi i/5} (loop value phi)."""
    A = cmath.exp(-2j * math.pi / 5)
    delta = -A ** 2 - A ** -2
    err = 0.0
    for name, word, n in BRAID_PANEL:
        zf = fib_quantum_trace(word, n)
        br = leval(bracket_closure(word, n), A) * delta
        err = max(err, abs(zf - (-1) ** len(word) * br))
    return {"loop value -A^2-A^-2 = phi": abs(delta - PHI) < 1e-12, "max error over panel": err,
            "t = A^{-4}": "e^{-2 pi i/5}"}


def legendre5(k):
    r = k % 5
    return 0 if r == 0 else (1 if r in (1, 4) else -1)


def fib_galois_certificate():
    """B_3, total charge tau, rational gauge F~ = [[1/phi, 1/phi], [1, -1/phi]] (entries in Q(sqrt5)).
    Galois conjugates zeta_20 -> zeta_20^k, sqrt5 -> (k|5) sqrt5.  Finite order would force
    |tr|/|det|^{1/2} <= 2 for every conjugate."""
    rows = []
    for k in range(1, 20):
        if math.gcd(k, 20) != 1:
            continue
        z = cmath.exp(2j * math.pi * k / 20)
        R = (z ** -8, z ** 6)
        s5 = legendre5(k) * 5 ** 0.5
        ph = (1 + s5) / 2
        Ft = np.array([[1 / ph, 1 / ph], [1, -1 / ph]])
        s1 = np.diag(R).astype(complex)
        s2 = Ft @ np.diag(R) @ Ft
        g = s1 @ np.linalg.inv(s2)
        ratio = abs(np.trace(g)) / abs(np.linalg.det(g)) ** 0.5
        rows.append((k, round(ratio, 6)))
    # the rational gauge is a diagonal conjugate of the unitary one: Delta F Delta^{-1} = F~, Delta = diag(1, phi^{1/2});
    # so at k = 1 the element s1 s2^{-1} has the same characteristic polynomial in both gauges
    Ft1 = np.array([[1 / PHI, 1 / PHI], [1, -1 / PHI]])
    Dg = np.diag([1, PHI ** 0.5])
    gauge = np.allclose(Dg @ FIB_F @ np.linalg.inv(Dg), Ft1)
    gu = np.diag(FIB_R) @ np.linalg.inv(FIB_F @ np.diag(FIB_R) @ FIB_F)
    gt = np.diag(FIB_R) @ np.linalg.inv(Ft1 @ np.diag(FIB_R) @ Ft1)
    same_charpoly = np.allclose(np.poly(gu), np.poly(gt))
    return {"conjugates (k, |tr|/|det|^1/2) for s1 s2^-1": rows,
            "infinite order": max(r for _, r in rows) > 2 + 1e-9,
            "F~^2 = 1": bool(np.allclose(Ft1 @ Ft1, np.eye(2))),
            "Delta F Delta^-1 = F~": bool(gauge),
            "same characteristic polynomial in both gauges (k = 1)": bool(same_charpoly)}


# ---- plane graphs, medial links -----------------------------------------------
PLANE_GRAPHS = {
    "C3": ([(0, 1), (-0.866, -0.5), (0.866, -0.5)], [(0, 1), (1, 2), (2, 0)]),
    "C5": ([(math.cos(2 * math.pi * k / 5 + math.pi / 2), math.sin(2 * math.pi * k / 5 + math.pi / 2)) for k in range(5)],
           [(k, (k + 1) % 5) for k in range(5)]),
    "K4": ([(0, 0), (0, 1), (-0.866, -0.5), (0.866, -0.5)], [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]),
    "prism": ([(3 * math.cos(math.radians(90 + 120 * i)), 3 * math.sin(math.radians(90 + 120 * i))) for i in range(3)]
              + [(math.cos(math.radians(90 + 120 * i)), math.sin(math.radians(90 + 120 * i))) for i in range(3)],
              [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
    "bipyramid": ([(0, 3), (-3, -2), (3, -2), (0, 0), (0, -1)],
                  [(4, 1), (4, 2), (4, 3), (0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]),
    "W5": ([(0, 0)] + [(math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5)) for k in range(5)],
           [(0, k) for k in range(1, 6)] + [(k, k % 5 + 1) for k in range(1, 6)]),
    "W6": ([(math.cos(math.pi * k / 3), math.sin(math.pi * k / 3)) for k in range(6)] + [(0, 0)],
           [(k, (k + 1) % 6) for k in range(6)] + [(k, 6) for k in range(6)]),
    "H7": ([(0, 4), (-0.6, -1.0), (0.6, -1.0), (1.5, -2.2), (-4, -3), (4, -3), (0, 0)],
           [(0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 6), (2, 3), (2, 6), (3, 5), (4, 5), (4, 6), (5, 6)]),
}


def rotation_system(coords, edges):
    out = [[] for _ in range(len(coords))]
    for j, (u, v) in enumerate(edges):
        out[u].append(2 * j)
        out[v].append(2 * j + 1)

    def ang(d):
        u, v = edges[d // 2]
        t, h = (u, v) if d % 2 == 0 else (v, u)
        return math.atan2(coords[h][1] - coords[t][1], coords[h][0] - coords[t][0])
    return [sorted(o, key=ang) for o in out]


def ribbon_loops(nv, edges, rot, S):
    inS = [False] * len(edges)
    for j in S:
        inS[j] = True
    nxt = {}
    for x in range(nv):
        ds = [d for d in rot[x] if inS[d // 2]]
        for i, d in enumerate(ds):
            nxt[d] = ds[(i + 1) % len(ds)]
    isolated = sum(1 for x in range(nv) if not any(inS[d // 2] for d in rot[x]))
    seen, orbits = set(), 0
    for d in nxt:
        if d in seen:
            continue
        orbits += 1
        k = d
        while k not in seen:
            seen.add(k)
            k = nxt[k ^ 1]
    return orbits + isolated


def medial_bracket(coords, edges):
    nv, m = len(coords), len(edges)
    rot = rotation_system(coords, edges)
    br = {}
    for mask in range(1 << m):
        S = [j for j in range(m) if mask >> j & 1]
        l = ribbon_loops(nv, edges, rot, S)
        br = ladd(br, lscale(lpow(D_LP, l - 1), 1, 2 * len(S) - m))
    return br


def _canon(edges):
    lab, out = {}, []
    for (u, v) in sorted(tuple(sorted(e)) for e in edges):
        for w in (u, v):
            if w not in lab:
                lab[w] = len(lab)
        a, b = lab[u], lab[v]
        out.append((min(a, b), max(a, b)))
    return tuple(sorted(out))


def _is_bridge(edges, idx):
    u, v = edges[idx]
    if u == v:
        return False
    adj = {}
    for j, (a, b) in enumerate(edges):
        if j != idx:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    seen, st = {u}, [u]
    while st:
        w = st.pop()
        for z in adj.get(w, []):
            if z not in seen:
                seen.add(z)
                st.append(z)
    return v not in seen


@lru_cache(maxsize=None)
def tutte(edges):
    if not edges:
        return {(0, 0): 1}
    E = list(edges)
    for i, (u, v) in enumerate(E):
        if u == v:
            return {(a, b + 1): c for (a, b), c in tutte(_canon(E[:i] + E[i + 1:])).items()}
    for i, (u, v) in enumerate(E):
        if _is_bridge(E, i):
            rest = E[:i] + E[i + 1:]
            con = [((v if a == u else a), (v if b == u else b)) for (a, b) in rest]
            return {(a + 1, b): c for (a, b), c in tutte(_canon(con)).items()}
    u, v = E[0]
    rest = E[1:]
    con = [((v if a == u else a), (v if b == u else b)) for (a, b) in rest]
    out = dict(tutte(_canon(rest)))
    for k, c in tutte(_canon(con)).items():
        out[k] = out.get(k, 0) + c
        if out[k] == 0:
            del out[k]
    return out


def count_3_colourings(nv, edges):
    adj = [[] for _ in range(nv)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    col = [-1] * nv

    def rec(k):
        if k == nv:
            return 1
        tot = 0
        for c in range(3):
            if all(col[w] != c for w in adj[k]):
                col[k] = c
                tot += rec(k + 1)
                col[k] = -1
        return tot
    return rec(0)


def medial_fibonacci(names=("W6", "H7", "prism", "bipyramid", "K4")):
    A = cmath.exp(-2j * math.pi / 5)
    out = {}
    for name in names:
        coords, edges = PLANE_GRAPHS[name]
        br = medial_bracket(coords, edges)
        out[name] = {"P(G,3)": count_3_colourings(len(coords), edges), "|<L(G)>(A_Fib)|": abs(leval(br, A)),
                     "bracket": br}
    w, h = out["W6"]["bracket"], out["H7"]["bracket"]
    return {"W6 and H7 brackets identical": w == h,
            "Fibonacci amplitudes |<L(W6)>| = |<L(H7)>|": abs(out["W6"]["|<L(G)>(A_Fib)|"] - out["H7"]["|<L(G)>(A_Fib)|"]) < 1e-12,
            "P(W6,3), P(H7,3)": (out["W6"]["P(G,3)"], out["H7"]["P(G,3)"]),
            "values": {k: round(v["|<L(G)>(A_Fib)|"], 6) for k, v in out.items()}}


# ---- oriented medial diagrams and the Alexander invariant --------------------------
def medial_crossings(coords, edges):
    """Traverse the alternating medial diagram.  Returns, per crossing x (= edge x of G):
    under-strand (direction, in-arc, out-arc, component) and over-strand (direction, component).
    Arcs correspond to edges (each arc passes over exactly one crossing)."""
    rot = rotation_system(coords, edges)
    nxt, prv, tail = {}, {}, {}
    for x in range(len(coords)):
        k = len(rot[x])
        for i, d in enumerate(rot[x]):
            nxt[d], prv[d], tail[d] = rot[x][(i + 1) % k], rot[x][(i - 1) % k], x

    def frame(d):
        u, v = tail[d], tail[d ^ 1]
        e = np.array(coords[v], float) - np.array(coords[u], float)
        e /= np.linalg.norm(e)
        return e, np.array([-e[1], e[0]])
    under, over, visited, comp = {}, {}, set(), 0
    for start in sorted(nxt):
        if start in visited:
            continue
        d = start
        while True:
            s = nxt[d]                       # U-step at crossing edge(sigma d)
            d2 = prv[nxt[d] ^ 1]
            e, n = frame(s)
            under[s // 2] = (2 * e + 2 * n, d // 2, d2 // 2, comp)
            visited.add(d)
            e2, n2 = frame(d2)               # O-step at crossing edge(d2)
            over[d2 // 2] = (2 * e2 - 2 * n2, comp)
            visited.add(d2)
            d = d2 ^ 1
            if d == start:
                break
        comp += 1
    return under, over, comp


def alexander_minor(coords, edges, flips=()):
    """(m-1)-minor of the Alexander matrix of the oriented medial link (components in `flips`
    reversed), as an integer polynomial up to units: an invariant of the oriented link."""
    under, over, ncomp = medial_crossings(coords, edges)
    m = len(edges)
    rows = []
    for x in range(m):
        ud, i_arc, j_arc, cu = under[x]
        od, co = over[x]
        if cu in flips:
            ud, i_arc, j_arc = -ud, j_arc, i_arc
        if co in flips:
            od = -od
        sign = 1 if od[0] * ud[1] - od[1] * ud[0] > 0 else -1
        rows.append((sign, x, i_arc, j_arc))

    def minor_at(t):
        M = [[Fr(0)] * m for _ in range(m)]
        for r, (sg, o, i, j) in enumerate(rows):
            if sg > 0:
                M[r][o] += 1 - t
                M[r][i] += t
                M[r][j] -= 1
            else:
                M[r][o] += t - 1
                M[r][i] += 1
                M[r][j] -= t
        return det_fraction([row[1:] for row in M[1:]])
    pts = list(range(2, m + 2))
    coeffs = interpolate([Fr(p) for p in pts], [minor_at(Fr(p)) for p in pts])
    return normalize_poly(coeffs), sum(r[0] for r in rows), ncomp


def alexander_report():
    out = {}
    for name in ("C3", "C5", "W5", "prism", "bipyramid", "K4", "W6", "H7"):
        coords, edges = PLANE_GRAPHS[name]
        poly, writhe, ncomp = alexander_minor(coords, edges)
        T = tutte(_canon(edges))
        det = sum(c for c in T.values())               # T(1,1) = spanning trees = det
        at_m1 = abs(sum(c * (-1) ** k for k, c in enumerate(poly)))
        classes = sorted({alexander_minor(coords, edges, flips)[0]
                          for r in range(ncomp) for flips in itertools.combinations(range(1, ncomp), r)})
        out[name] = {"components": ncomp, "minor": poly, "|minor(-1)| = det": at_m1 == det,
                     "minors over all orientations": classes}
    w = set(out["W6"]["minors over all orientations"])
    h = set(out["H7"]["minors over all orientations"])
    return {"rows": out, "W6 and H7 separated by the Alexander invariant (all orientations)": not (w & h),
            "trefoil minor": out["C3"]["minor"], "5_1 minor": out["C5"]["minor"]}


def thistlethwaite_bracket(edges, nv):
    """<L(G)> = A^{2|V|-2-|E|} T_G(-A^{-4}, -A^4) for a connected plane graph (same chirality convention as
    medial_bracket, whose state sum it equals by Euler's formula)."""
    T = tutte(_canon(edges))
    out = {}
    for (i, j), c in T.items():
        term = lmul(lpow({-4: -1}, i), lpow({4: -1}, j))
        out = ladd(out, lscale(term, c, 2 * nv - 2 - len(edges)))
    return out


def _unit_class(br):
    lo = min(br)
    items = sorted((e - lo, c) for e, c in br.items())
    if items[0][1] < 0:
        items = [(e, -c) for e, c in items]
    return tuple(items)


def _seg_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


def topological_census():
    """All bridgeless connected simple planar graphs with <= 7 vertices (networkx atlas), with straight-line
    planar drawings.  For each: 3-colourability, the bracket of L(G) up to units (Thistlethwaite; checked
    against the ribbon state sum on a subsample), the bracket up to units and mirror image, and the set of
    Alexander minors over all orientations.  Reports the number of classes and of 'mixed' classes, i.e.
    classes containing both 3-colourable and non-3-colourable graphs.  Needs networkx."""
    try:
        import networkx as nx
    except ImportError:
        return None
    rows, drawings_ok, det_ok, state_sum_ok = [], True, True, True
    for G in nx.graph_atlas_g()[1:]:
        if G.number_of_edges() == 0 or not nx.is_connected(G) or not nx.check_planarity(G)[0]:
            continue
        if any(True for _ in nx.bridges(G)):
            continue
        nodes = sorted(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        pos = nx.planar_layout(G)
        coords = [tuple(float(t) for t in pos[v]) for v in nodes]
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        nv = len(coords)
        drawings_ok &= not any(len({a, b, c, d}) == 4 and _seg_cross(coords[a], coords[b], coords[c], coords[d])
                               for (a, b), (c, d) in itertools.combinations(edges, 2))
        rot = rotation_system(coords, edges)
        drawings_ok &= nv - len(edges) + ribbon_loops(nv, edges, rot, list(range(len(edges)))) == 2
        br = thistlethwaite_bracket(edges, nv)
        if len(rows) % 13 == 0:
            state_sum_ok &= medial_bracket(coords, edges) == br
        poly, _, ncomp = alexander_minor(coords, edges)
        polys = [alexander_minor(coords, edges, flips)[0]
                 for r in range(ncomp) for flips in itertools.combinations(range(1, ncomp), r)]
        alex = frozenset(polys)
        t11 = sum(tutte(_canon(edges)).values())
        det_ok &= all(abs(sum(c * (-1) ** k for k, c in enumerate(pl))) == t11 for pl in polys)
        rows.append({"c3": count_3_colourings(nv, edges) > 0, "br": _unit_class(br),
                     "brm": _unit_class({-e: c for e, c in br.items()}), "alex": alex, "edges": len(edges), "nv": nv})

    def classes(key):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r["c3"])
        mixed = [g for g in groups.values() if len(set(g)) == 2]
        return len(groups), len(mixed), sum(len(g) for g in mixed)
    return {"graphs": len(rows), "3-colourable": sum(r["c3"] for r in rows),
            "straight-line drawings planar (Euler)": drawings_ok, "Thistlethwaite = state sum (sample)": state_sum_ok,
            "|Alexander minor(-1)| = T(1,1), every orientation": det_ok,
            "bracket up to units: classes, mixed, graphs in mixed": classes(lambda r: r["br"]),
            "bracket up to units and mirror": classes(lambda r: min(r["br"], r["brm"])),
            "Alexander, all orientations": classes(lambda r: r["alex"]),
            "bracket + Alexander": classes(lambda r: (r["br"], r["alex"])),
            "bracket up to mirror + Alexander": classes(lambda r: (min(r["br"], r["brm"]), r["alex"]))}


def _medial_passes(coords, edges):
    """The traversal of medial_crossings, recorded in order: per component, alternating under/over passes
    (kind, crossing, direction vector)."""
    rot = rotation_system(coords, edges)
    nxt, prv, tail = {}, {}, {}
    for x in range(len(coords)):
        k = len(rot[x])
        for i, d in enumerate(rot[x]):
            nxt[d], prv[d], tail[d] = rot[x][(i + 1) % k], rot[x][(i - 1) % k], x

    def frame(d):
        u, v = tail[d], tail[d ^ 1]
        e = np.array(coords[v], float) - np.array(coords[u], float)
        e /= np.linalg.norm(e)
        return e, np.array([-e[1], e[0]])
    comps, visited = [], set()
    for start in sorted(nxt):
        if start in visited:
            continue
        seq, d = [], start
        while True:
            s = nxt[d]
            d2 = prv[nxt[d] ^ 1]
            e, n = frame(s)
            seq.append(("U", s // 2, 2 * e + 2 * n))
            visited.add(d)
            e2, n2 = frame(d2)
            seq.append(("O", d2 // 2, 2 * e2 - 2 * n2))
            visited.add(d2)
            d = d2 ^ 1
            if d == start:
                break
        comps.append(seq)
    return comps


def medial_pd_code(coords, edges, flips=()):
    """PD code (KnotTheory convention: incoming under-strand first, then counterclockwise) of the oriented medial
    link, components in `flips` reversed.  Segments are labelled consecutively along each component."""
    comps = _medial_passes(coords, edges)
    ends, label = {}, 0
    for ci, seq in enumerate(comps):
        s, sign = (seq[::-1], -1) if ci in flips else (list(seq), 1)
        m = len(s)
        for k, (kind, x, dvec) in enumerate(s):
            ends.setdefault(x, []).append((kind, sign * np.asarray(dvec), label + (k - 1) % m, label + k))
        label += m
    pd = []
    for x, lst in ends.items():
        U = next(t for t in lst if t[0] == "U")
        O = next(t for t in lst if t[0] == "O")
        items = [(math.atan2(-U[1][1], -U[1][0]), U[2]), (math.atan2(U[1][1], U[1][0]), U[3]),
                 (math.atan2(-O[1][1], -O[1][0]), O[2]), (math.atan2(O[1][1], O[1][0]), O[3])]
        a0 = items[0][0]
        items.sort(key=lambda t: (t[0] - a0) % (2 * math.pi))
        pd.append(tuple(t[1] for t in items))
    return pd, len(comps)


def pd_bracket(pd):
    """Kauffman bracket of a PD code: the A-smoothing at (a,b,c,d) joins a-b and c-d, the B-smoothing a-d and b-c."""
    n, out = len(pd), {}
    for mask in range(1 << n):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        na = 0
        for k, (a, b, c, d) in enumerate(pd):
            pairs = ((a, b), (c, d)) if mask >> k & 1 else ((a, d), (b, c))
            na += mask >> k & 1
            for x, y in pairs:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry
        loops = len({find(x) for t in pd for x in t})
        out = ladd(out, lscale(lpow(D_LP, loops - 1), 1, 2 * na - n))
    return out


def snappy_crosscheck():
    """Optional (--snappy): independent recomputation of the census invariants with SnapPy/spherogram.
    Alexander polynomials from Seifert matrices (Seifert's algorithm), det(V - t V^T), for every orientation;
    for knots also the Euler characteristic of knot Floer homology; and the bracket of the PD code against the
    ribbon state sum.  Returns None if spherogram or knot_floer_homology is missing."""
    try:
        import networkx as nx
        import spherogram
        from knot_floer_homology import pd_to_hfk
    except ImportError:
        return None
    st = {"graphs": 0, "oriented links": 0, "Seifert = Fox": 0, "knots": 0, "HFK Euler char = Fox": 0,
          "PD bracket = ribbon bracket": 0, "components agree": 0}
    for G in nx.graph_atlas_g()[1:]:
        if G.number_of_edges() == 0 or not nx.is_connected(G) or not nx.check_planarity(G)[0]:
            continue
        if any(True for _ in nx.bridges(G)):
            continue
        st["graphs"] += 1
        nodes = sorted(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        pos = nx.planar_layout(G)
        coords = [tuple(float(t) for t in pos[v]) for v in nodes]
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        pd0, ncomp = medial_pd_code(coords, edges)
        st["PD bracket = ribbon bracket"] += lunit_ratio(pd_bracket(pd0), medial_bracket(coords, edges)) is not None
        for r in range(ncomp):
            for flips in itertools.combinations(range(1, ncomp), r):
                st["oriented links"] += 1
                pd, _ = medial_pd_code(coords, edges, flips)
                L = spherogram.Link(pd)
                st["components agree"] += len(L.link_components) == ncomp
                V = [[int(x) for x in row] for row in L.seifert_matrix()]
                k = len(V)
                pts = list(range(2, k + 3))
                vals = [det_fraction([[Fr(V[i][j] - t * V[j][i]) for j in range(k)] for i in range(k)]) if k else Fr(1)
                        for t in pts]
                seif = normalize_poly(interpolate([Fr(t) for t in pts], vals))
                fox = alexander_minor(coords, edges, flips)[0]
                st["Seifert = Fox"] += seif == fox
                if ncomp == 1:
                    st["knots"] += 1
                    chi = {}
                    for (a, mg), rk in pd_to_hfk(pd)["ranks"].items():
                        chi[a] = chi.get(a, 0) + (-1) ** mg * rk
                    lo = min(chi)
                    st["HFK Euler char = Fox"] += normalize_poly([chi.get(lo + i, 0) for i in range(max(chi) - lo + 1)]) == fox
    return st


def majorana_braids(m):
    """2m Majorana modes on m qubits (Jordan-Wigner), braids B_i = (1 + g_i g_{i+1})/sqrt2.  Checks the braid
    relations, that each B_i normalizes the Pauli group (Clifford), and the order of the image mod phases."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)

    def kr(ops):
        out = np.eye(1, dtype=complex)
        for o in ops:
            out = np.kron(out, o)
        return out
    gam = [kr([Z] * k + [P] + [I2] * (m - k - 1)) for k in range(m) for P in (X, Y)]
    d = 2 ** m
    B = [(np.eye(d) + gam[i] @ gam[i + 1]) / math.sqrt(2) for i in range(2 * m - 1)]
    braid = all(np.allclose(B[i] @ B[i + 1] @ B[i], B[i + 1] @ B[i] @ B[i + 1]) for i in range(2 * m - 2)) and \
        all(np.allclose(B[i] @ B[j], B[j] @ B[i]) for i in range(2 * m - 1) for j in range(i + 2, 2 * m - 1))
    paulis = [kr(ops) for ops in itertools.product((I2, X, Y, Z), repeat=m)]

    def is_pauli(M):
        return any(any(np.allclose(M, ph * P) for ph in (1, -1, 1j, -1j)) for P in paulis)
    clifford = all(is_pauli(b @ P @ b.conj().T) for b in B for P in paulis)

    def key(M):
        flat = M.flatten()
        k = next(i for i in range(len(flat)) if abs(flat[i]) > 1e-9)
        ph = flat[k] / abs(flat[k])
        return tuple(np.round((M / ph).flatten(), 6))
    seen, frontier = {key(np.eye(d)): np.eye(d)}, [np.eye(d)]
    while frontier:
        new = []
        for M in frontier:
            for b in B:
                N = b @ M
                kk = key(N)
                if kk not in seen:
                    seen[kk] = N
                    new.append(N)
        frontier = new
    # conjugation action on the Majorana operators: B g_k B^dag = sum_j M[j,k] g_j
    def action(b):
        return np.array([[np.trace(gam[j].conj().T @ b @ gam[k] @ b.conj().T).real / d for k in range(2 * m)]
                         for j in range(2 * m)])
    gens = [np.rint(action(b)).astype(int) for b in B]

    def is_signed_perm(g):
        return (set(np.unique(g)) <= {-1, 0, 1} and all((np.abs(g).sum(axis=0) == 1))
                and all((np.abs(g).sum(axis=1) == 1)))
    signed_perm = all(np.allclose(action(b), g) and is_signed_perm(g) and round(np.linalg.det(g)) == 1
                      for b, g in zip(B, gens))
    seen_m, frontier_m = {np.eye(2 * m, dtype=int).tobytes()}, [np.eye(2 * m, dtype=int)]
    while frontier_m:
        new = []
        for M in frontier_m:
            for g in gens:
                N = g @ M
                kk = N.tobytes()
                if kk not in seen_m:
                    seen_m.add(kk)
                    new.append(N)
        frontier_m = new
    expected = 2 ** (2 * m - 1) * math.factorial(2 * m)
    return {"Majoranas": 2 * m, "braid relations": braid, "Clifford (normalizes Paulis)": clifford,
            "B_i^8 = 1": all(np.allclose(np.linalg.matrix_power(b, 8), np.eye(d)) for b in B),
            "|image mod phases|": len(seen),
            "generators act as det-1 signed permutations of the g_j": signed_perm,
            "|group of the action| = |image mod phases| = 2^(2m-1) (2m)!": len(seen_m) == len(seen) == expected}


def encoding_sizes():
    """Crossings of the two encodings of a graph by links: medial (|E| crossings, plane graphs) and the
    vertex-strand clasp braid prod A_uv with A_ij = s_{j-1}..s_{i+1} s_i^2 s_{i+1}^-1..s_{j-1}^-1 (2(j-i) crossings)."""
    out = {}
    for name, (nv, edges) in CUNTZ_GRAPHS.items():
        clasp = sum(2 * abs(u - v) for u, v in edges)
        out[name] = {"n": nv, "|E|": len(edges), "clasp crossings": clasp, "bound 2|E|(n-1)": 2 * len(edges) * (nv - 1)}
    return out


def fibonacci_census():
    """Exact Fibonacci-point bracket values (in Z[zeta_5], up to units) and absolute values over all
    bridgeless connected planar graphs with <= 7 vertices.  Needs networkx."""
    try:
        import networkx as nx
    except ImportError:
        return None

    def zmul_zeta(v):
        a, b, c, d = v
        return (-d, a - d, b - d, c - d)

    def zpow(k):
        v = (1, 0, 0, 0)
        for _ in range(k % 5):
            v = zmul_zeta(v)
        return v

    def zadd(u, v, s=1):
        return tuple(x + s * y for x, y in zip(u, v))

    def zmul(u, v):
        out = (0, 0, 0, 0)
        for k, c in enumerate(u):
            if c:
                w = v
                for _ in range(k):
                    w = zmul_zeta(w)
                out = zadd(out, w, c)
        return out

    def conj(u):
        out = (0, 0, 0, 0)
        for k, c in enumerate(u):
            out = zadd(out, zpow(-k), c)
        return out

    def canon(v):
        cands, w = [], v
        for _ in range(5):
            cands += [w, tuple(-x for x in w)]
            w = zmul_zeta(w)
        return min(cands)
    exact, absval, count, ncol = {}, {}, 0, 0
    for G in nx.graph_atlas_g()[1:]:
        if G.number_of_edges() == 0 or not nx.is_connected(G) or not nx.check_planarity(G)[0]:
            continue
        if any(True for _ in nx.bridges(G)):
            continue
        count += 1
        T = tutte(_canon(list(G.edges())))
        v = (0, 0, 0, 0)
        for (i, j), c in T.items():                 # T(-t, -1/t), t = zeta^4
            v = zadd(v, zpow(4 * (i - j)), c * (-1) ** (i + j))
        c3 = count_3_colourings(G.number_of_nodes(), list(G.edges())) > 0
        ncol += c3
        exact.setdefault(canon(v), []).append(c3)
        absval.setdefault(zmul(v, conj(v)), []).append(c3)

    def mixed(groups):
        return sum(1 for g in groups.values() if len(set(g)) == 2), sum(len(g) for g in groups.values() if len(set(g)) == 2)
    return {"graphs": count, "3-colourable": ncol, "exact classes": len(exact),
            "exact: mixed classes, graphs in them": mixed(exact),
            "|amplitude|^2 classes": len(absval), "|amplitude|^2: mixed classes, graphs in them": mixed(absval)}


def fibonacci_lens_spaces(pmax=10):
    """Reshetikhin-Turaev invariant of L(p,1) for the Fibonacci category (labels 1, tau; d_tau = phi,
    theta_tau = e^{4 pi i/5}), normalised Z(S^3) = 1/D, Z(S^2 x S^1) = 1:
    Z(L(p,1)) = D^{-2} (D / Delta_sign(p)) sum_c d_c^2 theta_c^p."""
    th = cmath.exp(4j * math.pi / 5)
    D = math.sqrt(1 + PHI ** 2)
    gauss = {s: 1 + PHI ** 2 * th ** s for s in (1, -1)}
    vals = {}
    for p in range(-pmax, pmax + 1):
        if p == 0:
            vals[p] = (1 + PHI ** 2) / D ** 2 * D / D            # = 1
            continue
        s = 1 if p > 0 else -1
        vals[p] = (1 + PHI ** 2 * th ** p) * D / (gauss[s] * D ** 2)
    c_central = (cmath.phase(gauss[1] / D) / (2 * math.pi) * 8) % 8
    checks = {"Z(L(1,1)) = Z(S^3) = 1/D": abs(vals[1] - 1 / D) < 1e-12,
              "Z(L(-1,1)) = 1/D": abs(vals[-1] - 1 / D) < 1e-12,
              "Delta_+/D = e^{2 pi i c/8} with c = 14/5": abs(c_central - 14 / 5) < 1e-12,
              "|Z(L(p,1))| depends on p mod 5": all(abs(abs(vals[p]) - abs(vals[p + 5])) < 1e-12
                                                     for p in range(1, pmax - 4)),
              "|Delta_+| = D (Gauss sum)": abs(abs(gauss[1]) - D) < 1e-12}
    # modular data: S = D^{-1}[[1, phi], [phi, -1]], T = diag(1, theta_tau); (ST)^3 = (Delta_+/D) S^2 and
    # |Z(L(p,1))| = |(S T^p S)_{00}|, an independent route through the modular representation
    Sm = np.array([[1, PHI], [PHI, -1]]) / D
    Tm = np.diag([1, th])
    st3 = np.linalg.matrix_power(Sm @ Tm, 3)
    checks["S^2 = 1 and (ST)^3 = (Delta_+/D) S^2"] = bool(np.allclose(Sm @ Sm, np.eye(2))
                                                         and np.allclose(st3, gauss[1] / D * Sm @ Sm))
    checks["|Z(L(p,1))| = |(S T^p S)_00|, |p| <= %d" % pmax] = all(
        abs(abs(vals[p]) - abs((Sm @ np.linalg.matrix_power(Tm, p) @ Sm)[0, 0] if p >= 0 else
                               (Sm @ np.linalg.matrix_power(np.linalg.inv(Tm), -p) @ Sm)[0, 0])) < 1e-12
        for p in range(-pmax, pmax + 1))
    return {"|Z(L(p,1))|, p = 1..5": [round(abs(vals[p]), 6) for p in range(1, 6)], **checks}


# =============================================================================
# Part C. Kronecker and plethysm coefficients, Hecke coproduct
# =============================================================================
def partitions(n, maxpart=None):
    maxpart = n if maxpart is None else maxpart
    if n == 0:
        yield ()
        return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def z_rho(rho):
    out = 1
    for k in set(rho):
        m = rho.count(k)
        out *= k ** m * math.factorial(m)
    return out


@lru_cache(maxsize=None)
def character(lam, rho):
    """chi^lam(rho) by Murnaghan-Nakayama on beta-sets."""
    if not rho:
        return 1 if sum(lam) == 0 else 0
    k, rest = rho[0], rho[1:]
    L = len(lam)
    beta = [lam[i] + (L - 1 - i) for i in range(L)]
    bset = set(beta)
    total = 0
    for b in beta:
        nb = b - k
        if nb < 0 or nb in bset:
            continue
        sign = (-1) ** sum(1 for x in beta if nb < x < b)
        newbeta = sorted([x for x in beta if x != b] + [nb], reverse=True)
        newlam = tuple(newbeta[i] - (L - 1 - i) for i in range(L))
        newlam = tuple(x for x in newlam if x > 0)
        total += sign * character(newlam, rest)
    return total


def kronecker(lam, mu, nu):
    n = sum(lam)
    s = Fr(0)
    for rho in partitions(n):
        s += Fr(character(lam, rho) * character(mu, rho) * character(nu, rho), z_rho(rho))
    assert s.denominator == 1
    return int(s)


def conjugate(lam):
    return tuple(sum(1 for x in lam if x > j) for j in range(lam[0])) if lam else ()


def kronecker_checks(nmax=6):
    ok_triv, ok_sign, ok_sym, ok_dim = True, True, True, True
    for n in range(1, nmax + 1):
        P = list(partitions(n))
        dims = {l: character(l, (1,) * n) for l in P}
        for l in P:
            for m in P:
                ok_triv &= kronecker(l, m, (n,)) == (1 if l == m else 0)
                ok_sign &= kronecker(l, m, (1,) * n) == (1 if m == conjugate(l) else 0)
                ok_dim &= sum(kronecker(l, m, nu) * dims[nu] for nu in P) == dims[l] * dims[m]
        for l, m, nu in itertools.combinations_with_replacement(P, 3):
            g = kronecker(l, m, nu)
            ok_sym &= all(kronecker(*perm) == g for perm in itertools.permutations((l, m, nu)))
    return {"g(l,m,(n)) = delta": ok_triv, "g(l,m,(1^n)) = delta_{m,l'}": ok_sign,
            "symmetric in (l,m,n)": ok_sym, "sum_nu g f^nu = f^l f^m": ok_dim}


def saturation_holes(nmax=4):
    """Triples with g(l,m,n) = 0 < g(2l,2m,2n) (Kronecker coefficients are not saturated)."""
    out = []
    for n in range(1, nmax + 1):
        P = list(partitions(n))
        for l, m, nu in itertools.combinations_with_replacement(P, 3):
            if kronecker(l, m, nu) == 0:
                g2 = kronecker(tuple(2 * x for x in l), tuple(2 * x for x in m), tuple(2 * x for x in nu))
                if g2 > 0:
                    out.append((l, m, nu, g2))
    return out


def rectangular_census(shapes=((2, 2), (2, 3), (3, 2), (2, 4), (3, 3))):
    """Zero weights of g(lam, (d^m), (d^m)) for lam |- md with at most m^2 rows."""
    rows = []
    for m, d in shapes:
        rect = (d,) * m
        lams = [l for l in partitions(m * d) if len(l) <= m * m]
        zeros = [l for l in lams if kronecker(l, rect, rect) == 0]
        rows.append({"rectangle": f"{m}x{d}", "#lambda": len(lams), "#zero": len(zeros), "zero weights": zeros})
    return rows


def plethysm(m, d):
    """Schur expansion of h_m[h_d] via power sums: p_k[h_d] = sum_{s|-d} p_{k s} / z_s."""
    f = {}
    for rho in partitions(m):
        terms = {(): Fr(1, z_rho(rho))}
        for k in rho:
            new = {}
            for tau, c in terms.items():
                for s in partitions(d):
                    key = tuple(sorted(tau + tuple(k * x for x in s), reverse=True))
                    new[key] = new.get(key, 0) + c * Fr(1, z_rho(s))
            terms = new
        for tau, c in terms.items():
            f[tau] = f.get(tau, 0) + c
    out = {}
    for lam in partitions(m * d):
        coef = sum(c * character(lam, tau) for tau, c in f.items())
        assert coef.denominator == 1
        if coef:
            out[lam] = int(coef)
    return out


def plethysm_checks():
    known = {(2, 2): {(4,): 1, (2, 2): 1}, (3, 2): {(6,): 1, (4, 2): 1, (2, 2, 2): 1}, (2, 3): {(6,): 1, (4, 2): 1},
             (3, 3): {(9,): 1, (7, 2): 1, (6, 3): 1, (5, 2, 2): 1, (4, 4, 1): 1}}
    res = {f"h_{m}[h_{d}]": plethysm(m, d) for (m, d) in known}
    ok = all(res[f"h_{m}[h_{d}]"] == v for (m, d), v in known.items())
    a, b = plethysm(3, 2), plethysm(2, 3)
    foulkes = {lam: a.get(lam, 0) - b.get(lam, 0) for lam in set(a) | set(b)}
    return {"matches known expansions": ok, "h_3[h_2] - h_2[h_3]": {k: v for k, v in foulkes.items() if v},
            "Foulkes difference nonnegative": all(v >= 0 for v in foulkes.values()),
            "h_4[h_2]": plethysm(4, 2), "h_2[h_4]": plethysm(2, 4)}


def hecke_grouplike_obstruction():
    """In H_2(q) = <1, T | T^2 = (q-1)T + q>, Delta(T) = T (x) T is compatible with the Hecke
    relation only at q = 1: (T(x)T)^2 - (q-1) T(x)T - q = (q-1)(q-2) T(x)T + q(q-1)(T(x)1 + 1(x)T) + q(q-1)."""
    # polynomials in q: dict power -> int; elements of H (x) H: dict basis -> poly, basis in {11, T1, 1T, TT}
    def padd(a, b, s=1):
        r = dict(a)
        for k, v in b.items():
            r[k] = r.get(k, 0) + s * v
            if r[k] == 0:
                del r[k]
        return r

    def pmul(a, b):
        r = {}
        for i, x in a.items():
            for j, y in b.items():
                r[i + j] = r.get(i + j, 0) + x * y
        return {k: v for k, v in r.items() if v}
    q = {1: 1}
    one = {0: 1}
    T2 = {"T": padd(q, one, -1), "1": q}                     # T^2 = (q-1) T + q
    # (T(x)T)^2 = T^2 (x) T^2
    sq = {}
    for a, ca in T2.items():
        for b, cb in T2.items():
            key = ("1" if a == "1" else "T") + ("1" if b == "1" else "T")
            sq[key] = padd(sq.get(key, {}), pmul(ca, cb))
    expr = dict(sq)
    expr["TT"] = padd(expr.get("TT", {}), padd(q, one, -1), -1)
    expr["11"] = padd(expr.get("11", {}), q, -1)
    target = {"TT": pmul(padd(q, one, -1), padd(q, {0: 2}, -1)),
              "T1": pmul(q, padd(q, one, -1)), "1T": pmul(q, padd(q, one, -1)), "11": pmul(q, padd(q, one, -1))}
    ok = all(expr.get(k, {}) == target[k] for k in target)
    roots = [qq for qq in range(-3, 4) if all(sum(c * qq ** i for i, c in expr.get(k, {}).items()) == 0 for k in expr)]
    return {"expansion verified": ok, "integer q in [-3,3] where Delta(T) = T(x)T is compatible": roots}


def coloring_isotypic(names=("C5", "K4", "prism", "W6", "H7", "W5")):
    """C[Col(G)] as S_3-module: S_3 acts freely, so it is (P/6) copies of the regular representation."""
    out = {}
    graphs = {k: (len(v[0]), v[1]) for k, v in PLANE_GRAPHS.items()}
    for name in names:
        nv, edges = graphs[name]
        cols = [c for c in itertools.product(range(3), repeat=nv) if all(c[u] != c[v] for u, v in edges)]
        # permutation character on conjugacy classes: identity, transposition, 3-cycle
        reps = {"e": (0, 1, 2), "t": (1, 0, 2), "c": (1, 2, 0)}
        fix = {k: sum(1 for c in cols if tuple(g[x] for x in c) == c) for k, g in reps.items()}
        chars = {"triv": {"e": 1, "t": 1, "c": 1}, "sign": {"e": 1, "t": -1, "c": 1}, "std": {"e": 2, "t": 0, "c": -1}}
        sizes = {"e": 1, "t": 3, "c": 2}
        mult = {lab: sum(sizes[k] * fix[k] * ch[k] for k in sizes) // 6 for lab, ch in chars.items()}
        P = len(cols)
        out[name] = {"P(G,3)": P, "multiplicities": mult,
                     "= f^lam P/6": mult == {"triv": P // 6, "sign": P // 6, "std": 2 * P // 6}}
    return out


def standard_zero_weight(nmax=8):
    """g((n-1,1), mu, mu) = (number of distinct parts of mu) - 1.  For every rectangle mu = (d^m) it is 0:
    a systematic zero weight of rectangular Kronecker coefficients."""
    ok, rect_zero = True, True
    for n in range(2, nmax + 1):
        for mu in partitions(n):
            g = kronecker((n - 1, 1), mu, mu)
            ok &= g == len(set(mu)) - 1
            if len(set(mu)) == 1:
                rect_zero &= g == 0
    return {"g((n-1,1),mu,mu) = #distinct parts - 1 (n <= %d)" % nmax: ok, "zero for every rectangle": rect_zero}


# ---- orbit closures of the triangle colouring tensor ------------------------------
PRIME = 2147483629


def _parts3(d):
    return [l for l in partitions(d) if len(l) <= 3]


def _pad3(l):
    return tuple(list(l) + [0] * (3 - len(l)))


def _tables(alpha, beta, gamma):
    """Exponent vectors (3x3x3 contingency tables) of the monomials x^n of torus weight (alpha, beta, gamma)."""
    out, n = [], [0] * 27
    a, b, c = list(alpha), list(beta), list(gamma)

    def rec(t):
        if t == 27:
            if not any(a) and not any(b) and not any(c):
                out.append(tuple(n))
            return
        i, j, k = t // 9, (t // 3) % 3, t % 3
        for v in range(min(a[i], b[j], c[k]), -1, -1):
            n[t] = v
            a[i] -= v
            b[j] -= v
            c[k] -= v
            rec(t + 1)
            a[i] += v
            b[j] += v
            c[k] += v
        n[t] = 0
    rec(0)
    return out


def _rank_mod_p(M, p=PRIME):
    M = [list(r) for r in M]
    rows = len(M)
    cols = len(M[0]) if rows else 0
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] % p), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [(x * inv) % p for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] % p:
                f = M[i][c]
                M[i] = [(x - f * y) % p for x, y in zip(M[i], M[r])]
        r += 1
        if r == rows:
            break
    return r


def _orbit_points(T, N, rng, p=PRIME):
    pts = []
    for _ in range(N):
        mats = []
        while len(mats) < 3:
            A = [[rng.randrange(p) for _ in range(3)] for _ in range(3)]
            det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                   + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) % p
            if det:
                mats.append(A)
        A, B, C = mats
        pt = [0] * 27
        for (i, j, k), v in T.items():
            for i2 in range(3):
                for j2 in range(3):
                    for k2 in range(3):
                        pt[9 * i2 + 3 * j2 + k2] = (pt[9 * i2 + 3 * j2 + k2] + v * A[i2][i] * B[j2][j] * C[k2][k]) % p
        pts.append(pt)
    return pts


def _weight_dims(points, d, cache, p=PRIME):
    out = {}
    for al in _parts3(d):
        for be in _parts3(d):
            for ga in _parts3(d):
                key = (_pad3(al), _pad3(be), _pad3(ga))
                monos = cache.setdefault(key, _tables(*key))
                if points is None:
                    out[key] = len(monos)
                    continue
                M = []
                for mono in monos:
                    row = []
                    for pt in points:
                        v = 1
                        for t, ex in enumerate(mono):
                            if ex:
                                v = (v * pow(pt[t], ex, p)) % p
                        row.append(v)
                    M.append(row)
                out[key] = _rank_mod_p(M) if monos else 0
    return out


def _sgn(perm):
    s = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                s = -s
    return s


def _multiplicities(wdims, d):
    """Brauer-Klimyk: m(lam) = sum_{w in S_3^3} sgn(w) dim M_{lam + rho - w rho}, rho = (2,1,0)."""
    rho = (2, 1, 0)

    def mult(w):
        if any(x < 0 for part in w for x in part):
            return 0
        return wdims.get(tuple(tuple(sorted(part, reverse=True)) for part in w), 0)
    res = {}
    for al in _parts3(d):
        for be in _parts3(d):
            for ga in _parts3(d):
                lam = (_pad3(al), _pad3(be), _pad3(ga))
                tot = 0
                for ws in itertools.product(S3, repeat=3):
                    shifted = tuple(tuple(part[i] + rho[i] - rho[w[i]] for i in range(3)) for part, w in zip(lam, ws))
                    tot += _sgn(ws[0]) * _sgn(ws[1]) * _sgn(ws[2]) * mult(shifted)
                if tot:
                    res[(al, be, ga)] = tot
    return res


TRIANGLE_TENSORS = {
    "perm3 = colouring tensor of K3": {p: 1 for p in itertools.permutations(range(3))},
    "det3 = signed colouring tensor": {p: _sgn(p) for p in itertools.permutations(range(3))},
    "W-state": {(0, 0, 1): 1, (0, 1, 0): 1, (1, 0, 0): 1},
}


def orbit_dimension(T):
    """dim GL_3^3 . T = rank of the Lie algebra action X.T (exact, mod a large prime)."""
    cols = []
    for f in range(3):
        for a in range(3):
            for b in range(3):
                v = [0] * 27
                for (i, j, k), val in T.items():
                    idx = [i, j, k]
                    if idx[f] == b:
                        idx2 = list(idx)
                        idx2[f] = a
                        v[9 * idx2[0] + 3 * idx2[1] + idx2[2]] += val
                cols.append(v)
    return rank_exact(cols)


def rank_exact(M):
    """Rank over Q by fraction-exact elimination."""
    M = [[Fr(x) for x in row] for row in M]
    rows = len(M)
    cols = len(M[0]) if rows else 0
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        r += 1
    return r


def slice_determinants():
    """det(sum_k z_k T[:,:,k]) as a polynomial in z: the highest weight vectors of weight ((1^3),(1^3),(3)).
    perm3 gives 2 z0 z1 z2; det3 gives 0 (its slices are skew-symmetric)."""
    out = {}
    for name, T in list(TRIANGLE_TENSORS.items())[:2]:
        M = [[{} for _ in range(3)] for _ in range(3)]
        for (i, j, k), v in T.items():
            key = tuple(1 if t == k else 0 for t in range(3))
            M[i][j][key] = M[i][j].get(key, 0) + v

        def pm(a, b):
            r = {}
            for k1, v1 in a.items():
                for k2, v2 in b.items():
                    kk = tuple(x + y for x, y in zip(k1, k2))
                    r[kk] = r.get(kk, 0) + v1 * v2
            return r
        det = {}
        for perm in itertools.permutations(range(3)):
            term = pm(pm(M[0][perm[0]], M[1][perm[1]]), M[2][perm[2]])
            for kk, v in term.items():
                det[kk] = det.get(kk, 0) + _sgn(perm) * v
        out[name] = {k: v for k, v in det.items() if v}
    return out


def orbit_closure_multiplicities(dmax=4, seed=11):
    """Isotypic multiplicities of C[orbit closure]_d for GL_3^3 acting on C^3 (x) C^3 (x) C^3, d <= dmax:
    weight-space dimensions = ranks of monomial evaluation matrices at random orbit points (exact over F_p,
    Monte Carlo in the choice of points), then Brauer-Klimyk.  The same code applied to all of Sym^d returns
    the Kronecker coefficients g(lam, mu, nu)."""
    rng = random.Random(seed)
    cache, rows = {}, []
    tensors = dict(TRIANGLE_TENSORS)
    tensors["generic"] = {(i, j, k): rng.randrange(1, 1000) for i in range(3) for j in range(3) for k in range(3)}
    for d in range(1, dmax + 1):
        full = _weight_dims(None, d, cache)
        mfull = _multiplicities(full, d)
        sym_is_kron = all(mfull.get((a, b, c), 0) == kronecker(a, b, c)
                          for a in _parts3(d) for b in _parts3(d) for c in _parts3(d))
        N = max(full.values()) + 8
        mult = {name: _multiplicities(_weight_dims(_orbit_points(T, N, rng), d, cache), d) for name, T in tensors.items()}
        diff = {}
        for lam in sorted(mfull):
            vals = tuple(mult[nm].get(lam, 0) for nm in tensors)
            if any(v != mfull[lam] for v in vals):
                diff[lam] = (mfull[lam],) + vals
        rows.append({"d": d, "Sym^d multiplicities = Kronecker coefficients": sym_is_kron,
                     "#(lam,mu,nu) with g > 0": len(mfull),
                     "triples where some orbit closure differs (g, perm3, det3, W, generic)": diff})
    return rows


# =============================================================================
# Report
# =============================================================================
def _print(title, obj):
    print(f"\n--- {title}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"    {k}: {v}")
    elif isinstance(obj, list):
        for r in obj:
            print(f"    {r}")
    else:
        print(f"    {obj}")


def report(quick=False, snappy=False):
    t0 = time.time()
    print("quantum_nonlocality_assault.py  --  Non-Tracial Operator Fields, Fibonacci Braids, Hopf Deformations")
    print("=" * 100)
    print("PART A  non-tracial operator fields")
    _print("A.1 unitarity forcing (exact NC identities)", unitarity_forcing())
    _print("A.2 the Cuntz algebra O_3 on l^2(N)", cuntz_checks())
    _print("    the Cuntz algebra O_2 on l^2(N)", cuntz_checks(n=2))
    _print("A.3 Cuntz colourings: S_3-orbit (x) l^2, the 2:1 fibres absorbed by O_2 (all checks pass)",
           cuntz_colourings_report())
    _print("    detail for the Petersen graph", cuntz_colouring(*CUNTZ_GRAPHS["Petersen"],
                                                               proper_colouring(*CUNTZ_GRAPHS["Petersen"])))
    _print("    colour-shifted generators in O_3 itself (Petersen graph, word calculus)",
           o3_shift_colouring(*CUNTZ_GRAPHS["Petersen"], proper_colouring(*CUNTZ_GRAPHS["Petersen"])))
    _print("A.4 non-tracial Positivstellensatz certificate for K_4 in Cuntz variables", k4_cuntz_certificate())
    _print("    transfer P = S S^* and archimedean identities", cuntz_transfer_and_archimedean())
    _print("    pigeonhole principle: degree-2 noncommutative SOS certificates", fphp_report())
    _print("A.5 strip gadget: extension (G -> G^nat) and restriction (G^nat -> G)",
           {**strip_gadget_identities(), **triangle_and_strip_restriction()})
    _print("A.6 magic square: alg = C* = tracial levels for linear systems (J -> -1)", magic_square())
    print("\nPART B  Fibonacci braids")
    fc = fibonacci_checks(nmax=6 if quick else 8)
    _print("B.1 Fibonacci representations", {"F^2 = 1": fc["F^2 = 1"]})
    for r in fc["rows"]:
        print(f"    {r}")
    _print("B.2 Fibonacci quantum trace vs Kauffman bracket", fibonacci_vs_kauffman())
    _print("B.3 density certificate (B_3, Galois conjugates)", fib_galois_certificate())
    mf = medial_fibonacci()
    _print("B.4 medial links at the Fibonacci point", mf)
    ar = alexander_report()
    print("\n--- B.5 Alexander invariant of oriented medial links (HOMFLY-PT at a = 1)")
    for k, v in ar["rows"].items():
        print(f"    {k}: {v}")
    print(f"    W6 and H7 separated (all orientations): {ar['W6 and H7 separated by the Alexander invariant (all orientations)']}")
    print(f"    prism and bipyramid (mirror pair) equal: {ar['rows']['prism']['minor'] == ar['rows']['bipyramid']['minor']}")
    if not quick:
        cs = fibonacci_census()
        _print("B.6 census over bridgeless connected planar graphs (<= 7 vertices): Fibonacci point",
               cs if cs is not None else "networkx not available; skipped")
        ts = topological_census()
        _print("    the same graphs: chirality, Alexander and the bracket",
               ts if ts is not None else "networkx not available; skipped")
    if snappy:
        sc = snappy_crosscheck()
        _print("    independent cross-check with SnapPy/spherogram (Seifert matrices, knot Floer homology)",
               sc if sc is not None else "spherogram / knot_floer_homology not available; skipped")
    _print("B.7 Fibonacci Reshetikhin-Turaev invariants of lens spaces", fibonacci_lens_spaces())
    _print("B.8 Majorana (Ising) braiding is Clifford: finite image", [majorana_braids(2)] + ([] if quick else [majorana_braids(3)]))
    _print("B.9 crossings of the encodings", encoding_sizes())
    print("\nPART C  Hopf deformations and Kronecker multiplicities")
    _print("C.2 Kronecker coefficients (n <= 6)", kronecker_checks(nmax=5 if quick else 6))
    _print("    saturation holes g(l,m,n) = 0 < g(2l,2m,2n), n <= 4", saturation_holes(nmax=3 if quick else 4))
    _print("C.3 rectangular Kronecker coefficients: zero weights", rectangular_census(((2, 2), (2, 3), (3, 2)) if quick
                                                                                     else ((2, 2), (2, 3), (3, 2), (2, 4), (3, 3))))
    _print("    the systematic zero weight (n-1,1)", standard_zero_weight(nmax=6 if quick else 8))
    _print("C.4 plethysm coefficients", plethysm_checks())
    _print("C.5 Hecke deformation of the group-like coproduct", hecke_grouplike_obstruction())
    _print("C.6 S_3-isotypic multiplicities of C[Col(G)]", coloring_isotypic())
    _print("C.7 triangle colouring tensors: orbit dimensions (exact)",
           {name: orbit_dimension(T) for name, T in TRIANGLE_TENSORS.items()})
    _print("    slice determinants det(sum_k z_k T_k)", slice_determinants())
    for row in orbit_closure_multiplicities(dmax=3 if quick else 4):
        _print(f"    orbit-closure multiplicities, degree {row['d']}", row)
    print(f"\nTotal time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--snappy", action="store_true", help="also run the SnapPy cross-check of the census (about 75 s)")
    args = ap.parse_args()
    report(quick=args.quick, snappy=args.snappy and not args.quick)
