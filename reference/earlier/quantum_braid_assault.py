#!/usr/bin/env python3
"""
quantum_braid_assault.py

Computations for the draft "Nonlocal Operator Fields, Anyonic Braids, and the
Global Quantum Minimax Collapse" (Sections 7-11, continuing the hypercomplex
addendum).  Standalone: numpy + the Python standard library.  networkx is used
only by the optional exhaustive search in B.10 and is skipped if absent.

  Part 0  exact arithmetic: integer Laurent polynomials, Q(omega), free algebras
  Part A  operator Bayer fields (Mandate A)
          A.1 spectral-idempotent identity  x_u^2+x_u x_v+x_v^2 = 3 sum w^{2c} P_uc P_vc
          A.2 numerical check on random order-3 unitaries
          A.3 three idempotents with scalar sum lambda (degree-bounded NC Nullstellensatz)
          A.4 K_4: algebraic certificate (no trace, no positivity) and C*-SOS certificates
          A.5 triangle rigidity and linear Vieta transport: explicit certificate for W_5
  Part B  anyonic braids and knots (Mandate B)
          B.1 Temperley-Lieb diagram algebra, Kauffman bracket, Jones polynomial
          B.2 Fox 3-colourings of braid closures, determinants (reduced Burau at t=-1)
          B.3 knot panel: Lickorish-Millett, determinant, Arf (t=i), t=exp(2 pi i/3)
          B.4 Fox = Vieta; the affine wall (the relation "!=" is not affine)
          B.5 Reidemeister I destroys properness but not the Fox count
          B.6 vertex-strand / edge-clasp encoding of graphs
          B.7 B_3 dichotomy: finite image iff r in {1,4,6,10}; Kronecker certificates
          B.8 B_4, B_5 standard modules at r = 6, 10
          B.9 medial links of plane graphs: ribbon bracket, Thistlethwaite, Fox,
              Gauss sum at (w^2, w) (also for non-planar graphs), one algebra / two coefficients
          B.10 Jones blindness: W_6 versus H_7, prism versus bipyramid
  Part C  quantum Hopf algebras (Mandate C)
          C.1 U_q(sl_2) on V^{(x)n}: relations, antipode, S^2 = Ad K
          C.2 Temperley-Lieb commutant; quantum trace with the pivotal K = Kauffman bracket
          C.3 Lusztig-Rosso rigidity: highest-weight multiplicities at generic q
          C.4 S_3^+ = S_3: explicit degree-3 certificate; S_4^+ is not classical
          C.5 Kneser graphs: Johnson-scheme spectra, theta = n/k, chi = n-2k+2

Run:  python quantum_braid_assault.py           full report
      python quantum_braid_assault.py --quick   skips the B_5 group enumeration and the search
"""
import argparse
import cmath
import itertools
import math
import random
import sys
import time
from fractions import Fraction as Fr
from functools import lru_cache, reduce

import numpy as np

W3 = cmath.exp(2j * math.pi / 3)          # numerical omega


# =============================================================================
# Part 0. Exact arithmetic
# =============================================================================
# ---- integer Laurent polynomials in A: dict exponent -> int -----------------
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


def lmirror(p):
    return {-e: c for e, c in p.items()}


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


def lstr(p, var="A", div=1):
    if not p:
        return "0"
    out = []
    for e in sorted(p):
        c = p[e]
        ee = Fr(e, div)
        mon = "" if ee == 0 else (var if ee == 1 else f"{var}^{{{ee}}}")
        if mon == "":
            term = str(abs(c))
        else:
            term = mon if abs(c) == 1 else f"{abs(c)}{mon}"
        out.append(("-" if c < 0 else "+") + term)
    s = " ".join(out)
    return s[1:] if s.startswith("+") else s


D_LP = {2: -1, -2: -1}          # d = -A^2 - A^{-2}


# ---- Eisenstein rationals a + b*omega -----------------------------------------
class Eis:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fr(a)
        self.b = Fr(b)

    def __add__(self, o):
        o = _eis(o)
        return Eis(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return Eis(-self.a, -self.b)

    def __sub__(self, o):
        return self + (-_eis(o))

    def __rsub__(self, o):
        return _eis(o) - self

    def __mul__(self, o):
        o = _eis(o)
        return Eis(self.a * o.a - self.b * o.b, self.a * o.b + self.b * o.a - self.b * o.b)

    __rmul__ = __mul__

    def __eq__(self, o):
        o = _eis(o)
        return self.a == o.a and self.b == o.b

    def __bool__(self):
        return bool(self.a) or bool(self.b)

    def __hash__(self):
        return hash((self.a, self.b))

    def conj(self):
        return Eis(self.a - self.b, -self.b)

    def __repr__(self):
        return f"({self.a}{'+' if self.b >= 0 else '-'}{abs(self.b)}w)"


def _eis(x):
    return x if isinstance(x, Eis) else Eis(x, 0)


def wpow(k):
    return [Eis(1), Eis(0, 1), Eis(-1, -1)][k % 3]


# ---- free (noncommutative) polynomials --------------------------------------
class NC:
    """Element of the free algebra: dict word(tuple of letters) -> coefficient."""
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

    def reverse(self):
        return NC({tuple(reversed(w)): c for w, c in self.d.items()})

    def abelianize(self):
        r = {}
        for w, c in self.d.items():
            k = tuple(sorted(w, key=repr))
            r[k] = r.get(k, 0) + c
        return NC(r)

    def substitute(self, images):
        out = NC()
        for w, c in self.d.items():
            term = NC.const(c)
            for x in w:
                term = term * images[x]
            out = out + term
        return out


def _nc(x):
    return x if isinstance(x, NC) else NC.const(x)


def nsum(it):
    s = NC()
    for x in it:
        s = s + x
    return s


def reduce_cubes(p, letters):
    """Normal form modulo x^3 = 1 for the given letters (complete rewriting system)."""
    acc = {}
    for w, c in p.d.items():
        stack = []
        for x in w:
            if stack and stack[-1][0] == x:
                stack[-1][1] += 1
            else:
                stack.append([x, 1])
            if x in letters and stack[-1][1] == 3:
                stack.pop()
        w2 = tuple(x for x, k in stack for _ in range(k))
        acc[w2] = acc.get(w2, 0) + c
    return NC(acc)


# ---- exact linear algebra ------------------------------------------------------
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


def det_int(M):
    """Bareiss fraction-free determinant of an integer matrix."""
    A = [list(map(int, row)) for row in M]
    n = len(A)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def solve_rational(cols, target):
    """Exact solution of sum_j x_j cols[j] = target (dicts key -> rational), or None."""
    keys = sorted(set(k for c in cols for k in c) | set(target), key=repr)
    idx = {k: i for i, k in enumerate(keys)}
    m, n = len(keys), len(cols)
    M = [[Fr(0)] * (n + 1) for _ in range(m)]
    for j, c in enumerate(cols):
        for k, v in c.items():
            M[idx[k]][j] = Fr(v)
    for k, v in target.items():
        M[idx[k]][n] = Fr(v)
    piv, r = [], 0
    for j in range(n):
        p = next((i for i in range(r, m) if M[i][j] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][j]
        M[r] = [v * inv for v in M[r]]
        for i in range(m):
            if i != r and M[i][j] != 0:
                f = M[i][j]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(j)
        r += 1
        if r == m:
            break
    if any(M[i][n] != 0 for i in range(r, m)):
        return None
    sol = [Fr(0)] * n
    for i, j in enumerate(piv):
        sol[j] = M[i][n]
    return sol


# =============================================================================
# Part A. Operator Bayer fields
# =============================================================================
def P_proj(x, c):
    """Spectral idempotent P_c = (1/3)(1 + w^{-c} x + w^{-2c} x^2) of the letter x."""
    X = NC.var(x)
    return (NC.const(Eis(1)) + X * wpow(-c) + X * X * wpow(-2 * c)) * Eis(Fr(1, 3))


def operator_bayer_identity():
    """Theorem 7.2: identities in the free algebra C<x_u, x_v> (exact, over Q(omega))."""
    xu, xv = NC.var("u"), NC.var("v")
    q = xu * xu + xu * xv + xv * xv
    rhs = nsum(P_proj("u", c) * P_proj("v", c) * (wpow(2 * c) * 3) for c in range(3))
    free_identity = (q - rhs).is_zero()
    unity = (nsum(P_proj("u", c) for c in range(3)) - 1).is_zero()
    powers = all((nsum(P_proj("u", c) * wpow(a * c) for c in range(3)) - xu ** a).is_zero()
                 for a in range(3))
    orth = all(reduce_cubes(P_proj("u", c) * P_proj("u", c2)
                            - (P_proj("u", c) if c == c2 else NC()), {"u"}).is_zero()
               for c in range(3) for c2 in range(3))
    lift = all(reduce_cubes(P_proj("u", c) * q - P_proj("u", c) * P_proj("v", c) * (wpow(2 * c) * 3),
                            {"u", "v"}).is_zero() for c in range(3))
    reversed_order = (xv * xv + xv * xu + xu * xu
                      - nsum(P_proj("v", c) * P_proj("u", c) * (wpow(2 * c) * 3) for c in range(3))).is_zero()
    return {"q_uv = 3 sum w^{2c} P_uc P_vc (free algebra)": free_identity,
            "sum_c P_c = 1 (free algebra)": unity,
            "x^a = sum_c w^{ac} P_c, a=0,1,2 (free algebra)": powers,
            "P_c P_c' = delta P_c mod x^3=1": orth,
            "P_uc q_uv = 3 w^{2c} P_uc P_vc mod x^3=1": lift,
            "reversed order q_vu identity": reversed_order}


def random_order3_unitary(d, rng):
    Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(Z)
    ph = np.exp(2j * np.pi * rng.integers(0, 3, size=d) / 3)
    return Q @ np.diag(ph) @ Q.conj().T


def spectral_projections(X):
    I = np.eye(len(X))
    return [(I + W3 ** (-c) * X + W3 ** (-2 * c) * X @ X) / 3 for c in range(3)]


def operator_numeric_check(d=7, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    err_id, err_proj, noncomm = 0.0, 0.0, 0.0
    for _ in range(trials):
        Xu, Xv = random_order3_unitary(d, rng), random_order3_unitary(d, rng)
        Pu, Pv = spectral_projections(Xu), spectral_projections(Xv)
        lhs = Xu @ Xu + Xu @ Xv + Xv @ Xv
        rhs = 3 * sum(W3 ** (2 * c) * Pu[c] @ Pv[c] for c in range(3))
        err_id = max(err_id, np.abs(lhs - rhs).max())
        err_proj = max(err_proj, max(np.abs(P @ P - P).max() + np.abs(P - P.conj().T).max() for P in Pu))
        noncomm = max(noncomm, np.abs(Xu @ Xv - Xv @ Xu).max())
    # a solution of the edge relation: X_v = w X_u
    Xu = random_order3_unitary(d, rng)
    Xv = W3 * Xu
    Pu, Pv = spectral_projections(Xu), spectral_projections(Xv)
    sol_rel = np.abs(Xu @ Xu + Xu @ Xv + Xv @ Xv).max()
    sol_proj = max(np.abs(Pu[c] @ Pv[c]).max() for c in range(3))
    return {"identity error (non-commuting)": err_id, "projection error": err_proj,
            "max |[X_u,X_v]|": noncomm, "edge relation on X_v=wX_u": sol_rel,
            "max |P_uc P_vc| there": sol_proj}


# ---- degree-bounded noncommutative Nullstellensatz ---------------------------
def _words(letters, L):
    return list(itertools.product(letters, repeat=L))


def _cert_columns(gens, letters, D, commutative=False):
    """Columns w g w' (deg w + deg g + deg w' <= D) as dicts word -> coefficient."""
    cols, labels = [], []
    for gi, g in enumerate(gens):
        room = D - g.degree()
        if room < 0:
            continue
        seen = set()
        for L in range(room + 1):
            for a in range(L + 1):
                for wl in _words(letters, a):
                    for wr in _words(letters, L - a):
                        if commutative:
                            key = tuple(sorted(wl + wr))
                            if key in seen:
                                continue
                            seen.add(key)
                        col = NC({wl: 1}) * g * NC({wr: 1})
                        if commutative:
                            col = col.abelianize()
                        cols.append(col.d)
                        labels.append((wl, gi, wr))
    return cols, labels


def nc_certificate(gens, letters, D, target=None, commutative=False):
    """Search target (default 1) = sum c * w g w' with deg w + deg g + deg w' <= D, exactly over Q.
    Returns a list of (left word, generator index, right word, coefficient) or None."""
    target = target if target is not None else NC.const(1)
    cols, labels = _cert_columns(gens, letters, D, commutative)
    tgt = target.abelianize().d if commutative else target.d
    sol = solve_rational(cols, tgt)
    if sol is None:
        return None
    return [(wl, gi, wr, c) for (wl, gi, wr), c in zip(labels, sol) if c != 0]


def rank_mod_p_np(A, p):
    A = np.array(A, dtype=np.int64) % p
    m, n = A.shape
    r = 0
    for c in range(n):
        if r == m:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        col = A[:, c].copy()
        col[r] = 0
        rows = np.nonzero(col)[0]
        if len(rows):
            A[rows] = (A[rows] - np.outer(col[rows], A[r])) % p
        r += 1
    return r


def three_idempotent_gens(lam):
    a, b = NC.var("a"), NC.var("b")
    c = NC.const(Fr(lam)) - a - b
    return [a * a - a, b * b - b, c * c - c]


def three_idempotent_scan(p=1009, nc_degrees=(4, 5), com_degrees=(3, 4)):
    """For every lambda in F_p: is 1 in the degree-D span?  c^2 - c = G0 + lambda G1 + lambda^2 G2."""
    a, b = NC.var("a"), NC.var("b")
    G0 = a * a + b * b + a * b + b * a + a + b
    G1 = NC.const(-1) - a * 2 - b * 2
    G2 = NC.const(1)
    parts = [a * a - a, b * b - b, G0, G1, G2]
    out = {}
    for commutative, degrees in ((False, nc_degrees), (True, com_degrees)):
        for D in degrees:
            cols, labels = _cert_columns(parts, "ab", D, commutative)
            # columns of G1, G2 must use the same multipliers as G0: index by (wl, wr)
            by_gen = {}
            for col, (wl, gi, wr) in zip(cols, labels):
                by_gen.setdefault(gi, {})[(wl, wr)] = col
            keys = sorted(set(k for c in cols for k in c) | {()}, key=repr)
            idx = {k: i for i, k in enumerate(keys)}

            def vec(col):
                v = np.zeros(len(keys), dtype=np.int64)
                for k, x in col.items():
                    v[idx[k]] = int(x) % p
                return v
            fixed = [vec(c) for gi in (0, 1) for c in by_gen.get(gi, {}).values()]
            mults = list(by_gen.get(2, {}).keys())
            P0 = [vec(by_gen[2][m]) for m in mults]
            P1 = [vec(by_gen[3].get(m, {})) if m in by_gen.get(3, {}) else np.zeros(len(keys), np.int64) for m in mults]
            P2 = [vec(by_gen[4].get(m, {})) if m in by_gen.get(4, {}) else np.zeros(len(keys), np.int64) for m in mults]
            tgt = np.zeros(len(keys), dtype=np.int64)
            tgt[idx[()]] = 1
            bad = []
            for lam in range(p):
                var = [(x + lam * y + lam * lam * z) % p for x, y, z in zip(P0, P1, P2)]
                M = np.array(fixed + var).T
                if rank_mod_p_np(M, p) == rank_mod_p_np(np.column_stack([M, tgt]), p):
                    continue
                bad.append(lam)
            out[("NC" if not commutative else "comm", D)] = bad
    exc = sorted({0, 1, 2, 3, (3 * pow(2, p - 2, p)) % p})
    return {"p": p, "exceptional set mod p": exc,
            "NC deg 4 infeasible for all lambda": len(out[("NC", 4)]) == p,
            "NC deg 5 infeasible exactly at exceptional set": out[("NC", 5)] == exc,
            "comm deg 3 infeasible for all lambda": len(out[("comm", 3)]) == p,
            "comm deg 4 infeasible exactly at {0,1,2,3}": out[("comm", 4)] == [0, 1, 2, 3]}


def three_idempotents(lams=(Fr(-1), Fr(0), Fr(1, 2), Fr(1), Fr(3, 2), Fr(2), Fr(5, 2), Fr(3), Fr(4), Fr(5)),
                      Dmax=6):
    """Lemma 7.4: three idempotents with scalar sum lambda in a nonzero algebra force
    lambda in {0,1,3/2,2,3}.  Otherwise 1 lies in the two-sided ideal; the minimal
    certificate degree is computed (noncommutative: 5, commutative: 4)."""
    rows = []
    for lam in lams:
        gens = three_idempotent_gens(lam)
        deg_nc = next((D for D in range(2, Dmax + 1) if nc_certificate(gens, "ab", D) is not None), None)
        deg_c = next((D for D in range(2, Dmax + 1)
                      if nc_certificate(gens, "ab", D, commutative=True) is not None), None)
        rows.append((lam, deg_nc, deg_c))
    # explicit nonzero points for the exceptional values
    pts = {}
    pts[Fr(0)] = (np.zeros((1, 1)), np.zeros((1, 1)))
    pts[Fr(1)] = (np.ones((1, 1)), np.zeros((1, 1)))
    pts[Fr(2)] = (np.ones((1, 1)), np.ones((1, 1)))
    pts[Fr(3)] = (np.ones((1, 1)), np.ones((1, 1)))
    v0, v1 = np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3) / 2])
    pts[Fr(3, 2)] = (np.outer(v0, v0), np.outer(v1, v1))
    point_ok = {}
    for lam, (a, b) in pts.items():
        c = float(lam) * np.eye(len(a)) - a - b
        point_ok[str(lam)] = max(np.abs(x @ x - x).max() for x in (a, b, c)) < 1e-12
    qa, qb = pts[Fr(3, 2)]
    return {"rows": rows, "points": point_ok, "noncommutative at 3/2": float(np.abs(qa @ qb - qb @ qa).max())}


# ---- explicit certificates in the game-algebra generators e_{v,c} ----------------------
def _e(v, c):
    return NC.var(("e", v, c))


def rel_idem(v, c):
    return _e(v, c) * _e(v, c) - _e(v, c)


def rel_sum(v):
    return nsum(_e(v, c) for c in range(3)) - 1


def rel_edge(u, v, c):
    return _e(u, c) * _e(v, c)


class Cert:
    """sum_k left_k * rel_k * right_k, every rel_k a generator relation of the game algebra."""

    def __init__(self, terms=None):
        self.terms = list(terms or [])

    def add(self, left, rel, right):
        self.terms.append((_nc(left), rel, _nc(right)))
        return self

    def mul(self, left=None, right=None, scale=1):
        L = _nc(left) if left is not None else NC.const(1)
        R = _nc(right) if right is not None else NC.const(1)
        return Cert([(L * l * scale, r, rr * R) for l, r, rr in self.terms])

    def __add__(self, other):
        return Cert(self.terms + other.terms)

    def value(self):
        return nsum(l * r * rr for l, r, rr in self.terms)

    def degree(self):
        return max(l.degree() + r.degree() + rr.degree() for l, r, rr in self.terms)


def _phi_relation(gi, tri, lam):
    """Image of the {a,b}-generator gi under a -> Q_0, b -> Q_1 (c -> Q_2 - eps) as a Cert,
    for a clique `tri` with sum_c Q_c = lam + eps."""
    def idem_part(c):
        cert = Cert()
        for v in tri:
            cert.add(1, rel_idem(v, c), 1)
        for u in tri:
            for v in tri:
                if u != v:
                    cert.add(1, rel_edge(u, v, c), 1)
        return cert
    if gi in (0, 1):
        return idem_part(gi)
    Q2 = nsum(_e(v, 2) for v in tri)
    eps = nsum(rel_sum(v) for v in tri)
    cert = idem_part(2)
    for v in tri:
        cert.add(-Q2, rel_sum(v), 1)
        cert.add(-1, rel_sum(v), Q2)
        cert.add(1, rel_sum(v), eps)
        cert.add(1, rel_sum(v), 1)
    return cert


def lift_certificate(cert, tri, lam, swap=False):
    """Lift an {a,b}-certificate along a -> Q_0, b -> Q_1 (or swapped)."""
    images = {"a": nsum(_e(v, 1 if swap else 0) for v in tri), "b": nsum(_e(v, 0 if swap else 1) for v in tri)}
    out = Cert()
    for wl, gi, wr, cf in cert:
        g = {0: 1, 1: 0, 2: 2}[gi] if swap else gi
        out = out + _phi_relation(g, tri, lam).mul(NC({wl: cf}).substitute(images), NC({wr: 1}).substitute(images))
    return out


def k4_certificates():
    """Proposition 7.5: algebraic certificate of degree 5 in the e_{v,c} (no trace, no positivity);
    C*-certificates of degree 2 (game algebra) and 4 (Bayer variables with unitarity only)."""
    tri = list(range(4))
    Q = [nsum(_e(v, c) for v in tri) for c in range(3)]
    eps = Q[0] + Q[1] + Q[2] - 4
    gens = three_idempotent_gens(4)
    cert = nc_certificate(gens, "ab", 5)
    lifted = lift_certificate(cert, tri, 4)
    lift_is_one = (lifted.value() - 1).is_zero()
    # C*-certificate in the game algebra
    sos = nsum((NC.const(1) - Q[c]) * (NC.const(1) - Q[c]) for c in range(3))
    ideal_part = nsum(Q[c] * Q[c] - Q[c] for c in range(3)) - eps
    cstar_ok = (sos + 1 - ideal_part).is_zero()
    ab_ok = (nsum(NC({wl: cf}) * gens[gi] * NC({wr: 1}) for wl, gi, wr, cf in cert).abelianize() - 1).is_zero()
    return {"lambda=4 certificate terms": len(cert),
            "lifted certificate (relations of A(K_4)) equals 1": lift_is_one,
            "certificate degree in e": lifted.degree(),
            "C* (game algebra): sum(1-Q_c)^2 + 1 in I, degree 2": cstar_ok,
            "abelianized certificate valid": ab_ok,
            "C* (Bayer, unitarity only): identity holds": k4_cstar_bayer()[0],
            "C* (Bayer, unitarity only): degree": k4_cstar_bayer()[1],
            "lift of e_uc e_vc: degree of P_uc P_vc - w^{-2c}/3 P_uc q_uv": bayer_edge_lift_degree()}


def _star(p):
    """Involution of the free *-algebra on x_v, s_v = x_v^*: reverse, x <-> s, conjugate."""
    swap = {"x": "s", "s": "x"}
    out = {}
    for w, c in p.d.items():
        w2 = tuple((swap[l[0]], l[1]) for l in reversed(w))
        out[w2] = _eis(c).conj()
    return NC(out)


def k4_cstar_bayer():
    X = {v: NC.var(("x", v)) for v in range(4)}
    S = {v: NC.var(("s", v)) for v in range(4)}
    third = Eis(Fr(1, 3))
    E = {(v, c): (NC.const(Eis(1)) + X[v] * wpow(-c) + S[v] * wpow(c)) * third for v in range(4) for c in range(3)}
    h = {c: NC.const(Eis(1)) - nsum(E[v, c] for v in range(4)) for c in range(3)}
    selfadj = all((_star(h[c]) - h[c]).is_zero() for c in range(3))
    sos = nsum(_star(h[c]) * h[c] for c in range(3))
    U1 = {v: X[v] * S[v] - 1 for v in range(4)}           # x x^* - 1
    U2 = {v: S[v] * X[v] - 1 for v in range(4)}           # x^* x - 1

    def q(u, v):
        return X[u] * X[u] + X[u] * X[v] + X[v] * X[v]
    terms = []
    for v in range(4):
        terms += [(third, U1[v], 1), (third, U2[v], 1)]
    for u in range(4):
        for v in range(4):
            if u != v:
                terms += [(S[u] * third, q(u, v), S[v]), (-third, U2[u], X[u] * S[v]), (-third, U2[u], 1),
                          (-third, U1[v], 1), (-third, U2[u], U1[v]), (S[u] * X[v] * (-third), U1[v], 1)]
    ideal = nsum(_nc(l) * r * _nc(rr) for l, r, rr in terms)
    ok = selfadj and (sos + 1 - ideal).is_zero()
    deg = max([_nc(l).degree() + r.degree() + _nc(rr).degree() for l, r, rr in terms] + [sos.degree()])
    return ok, deg


def bayer_edge_lift_degree():
    xu, xv = NC.var("u"), NC.var("v")
    q = xu * xu + xu * xv + xv * xv
    degs = []
    for c in range(3):
        Dc = P_proj("u", c) * P_proj("v", c) - P_proj("u", c) * q * (wpow(-2 * c) * Eis(Fr(1, 3)))
        assert reduce_cubes(Dc, {"u", "v"}).is_zero()
        degs.append(Dc.degree())
    return max(degs)


def triangle_rigidity_and_transport():
    """Proposition 7.6: in A(K_3), Q^tau_c - 1 is in the ideal (lambda = 3 case of Lemma 7.4);
    linear Vieta transport then refutes every non-colourable triangle component in bounded
    degree.  Explicit certificate for the odd wheel W_5 (K_4-free)."""
    a, b = NC.var("a"), NC.var("b")
    gens = three_idempotent_gens(3)
    deg_nc = next(D for D in range(2, 8) if nc_certificate(gens, "ab", D, target=a - 1) is not None)
    deg_c = next(D for D in range(2, 8) if nc_certificate(gens, "ab", D, target=a - 1, commutative=True) is not None)
    cert_a = nc_certificate(gens, "ab", deg_nc, target=a - 1)

    def Qm1(tri):
        """Certs for Q^tau_c - 1, c = 0, 1, 2."""
        c0 = lift_certificate(cert_a, tri, 3)
        c1 = lift_certificate(cert_a, tri, 3, swap=True)
        c2 = c0.mul(scale=-1) + c1.mul(scale=-1)
        for v in tri:
            c2.add(1, rel_sum(v), 1)
        return [c0, c1, c2]
    tri0 = (0, 1, 2)
    certs0 = Qm1(tri0)
    tri_ok = all((certs0[c].value() - (nsum(_e(v, c) for v in tri0) - 1)).is_zero() for c in range(3))
    # W_5: hub 0, rim 1..5; triangles t_i = {0, i, i+1}
    tris = {i: (0, i, i % 5 + 1) for i in range(1, 6)}
    Qc = {i: Qm1(t) for i, t in tris.items()}
    total = Cert()
    for c in range(3):
        # e_{1,c} - e_{2,c} = -(Q^{t2}_c - 1) + (Q^{t3}_c - 1) - (Q^{t4}_c - 1) + (Q^{t5}_c - 1)
        D = Qc[2][c].mul(scale=-1) + Qc[3][c] + Qc[4][c].mul(scale=-1) + Qc[5][c]
        total = total + Cert().add(-1, rel_idem(1, c), 1) + D.mul(left=_e(1, c)) + Cert().add(1, rel_edge(1, 2, c), 1)
    total.add(-1, rel_sum(1), 1)
    w5_ok = (total.value() - 1).is_zero()
    return {"lambda=3: degree of a-1 in I (NC, comm)": (deg_nc, deg_c),
            "triangle: Q_c - 1 in I(K_3) (explicit)": tri_ok,
            "triangle certificate degree in e": max(c.degree() for c in certs0),
            "W_5 certificate equals 1": w5_ok, "W_5 certificate degree in e": total.degree(),
            "W_5 certificate terms": len(total.terms)}


# =============================================================================
# Part B. Anyonic braids, Kauffman bracket, Jones polynomial, Fox colourings
# =============================================================================
# ---- Temperley-Lieb diagrams: partner arrays on 2n points (top 0..n-1, bottom n..2n-1)
def tl_id(n):
    return tuple([n + i for i in range(n)] + [i for i in range(n)])


def tl_e(n, i):
    p = list(tl_id(n))
    p[i], p[i + 1] = i + 1, i
    p[n + i], p[n + i + 1] = n + i + 1, n + i
    return tuple(p)


def tl_compose(p, q, n):
    """p stacked on top of q; returns (diagram, number of closed loops)."""
    res = [None] * (2 * n)
    seen = [False] * n

    def walk(side, k):
        while True:
            if side == "p":
                j = p[k]
                if j < n:
                    return j
                m = j - n
                seen[m] = True
                side, k = "q", m
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


def tl_braid_element(word, n):
    """Image of a braid word (+i = sigma_i, -i = sigma_i^{-1}, 1-based) in TL_n(d),
    sigma = A + A^{-1} e,  sigma^{-1} = A^{-1} + A e."""
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
    return elem


def bracket_closure(word, n):
    """Kauffman bracket of the braid closure, normalised so that <O> = 1."""
    total = {}
    for D, c in tl_braid_element(word, n).items():
        total = ladd(total, lmul(c, lpow(D_LP, closure_loops(D, n) - 1)))
    return total


def jones_f(word, n):
    """f(A) = (-A^3)^{-w} <closure>; V(t) = f(t^{-1/4})."""
    w = sum(1 if g > 0 else -1 for g in word)
    return lscale(bracket_closure(word, n), (-1) ** (w % 2), -3 * w)


def jones_eval(f, theta):
    """V(e^{i theta}) with the choice A = e^{-i theta/4}."""
    return leval(f, cmath.exp(-1j * theta / 4))


def jones_in_t(f):
    return {Fr(-e, 4): c for e, c in f.items()}


# ---- Fox colourings of braid closures -----------------------------------------
def fox_matrix(word, n, p=3):
    """Action on the colours of the n positions: sigma_i (strand i over):
    (a,b) -> (2a-b, a);  sigma_i^{-1}: (a,b) -> (b, 2b-a)."""
    M = np.eye(n, dtype=np.int64)
    for g in word:
        i = abs(g) - 1
        G = np.eye(n, dtype=np.int64)
        if g > 0:
            G[i, i], G[i, i + 1], G[i + 1, i], G[i + 1, i + 1] = 2, -1, 1, 0
        else:
            G[i, i], G[i, i + 1], G[i + 1, i], G[i + 1, i + 1] = 0, 1, -1, 2
        M = (G @ M) % p
    return M


def fox_dim(word, n, p=3):
    M = (fox_matrix(word, n, p) - np.eye(n, dtype=np.int64)) % p
    return n - rank_mod_p(M.tolist(), p)


def fox_colourings(word, n):
    M = fox_matrix(word, n)
    return [x for x in itertools.product(range(3), repeat=n)
            if all(int(v) % 3 == x[k] for k, v in enumerate(M @ np.array(x)))]


def crossing_types(word, n, x):
    """Number of distinct colours (1 or 3) at each crossing for the colouring x."""
    x = list(x)
    out = []
    for g in word:
        i = abs(g) - 1
        a, b = x[i], x[i + 1]
        if g > 0:
            over, uin, uout = a, b, (2 * a - b) % 3
            x[i], x[i + 1] = uout, a
        else:
            over, uin, uout = b, a, (2 * b - a) % 3
            x[i], x[i + 1] = b, uout
        out.append(len({over, uin, uout}))
    return out


def components(word, n):
    perm = list(range(n))
    for g in word:
        i = abs(g) - 1
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    seen, c = set(), 0
    for s in range(n):
        if s not in seen:
            c += 1
            k = s
            while k not in seen:
                seen.add(k)
                k = perm[k]
    return c


def burau_reduced_minus1(word, n):
    """Product of reduced Burau matrices at t = -1 (integer, unimodular)."""
    m = n - 1
    B = np.eye(m, dtype=np.int64)
    for g in word:
        i = abs(g)              # 1-based generator index
        G = np.eye(m, dtype=np.int64)
        t = -1
        if m == 1:
            G[0, 0] = -t
        elif i == 1:
            G[0, 0], G[0, 1] = -t, 1
        elif i == n - 1:
            G[m - 1, m - 2], G[m - 1, m - 1] = t, -t
        else:
            r = i - 1
            G[r, r - 1], G[r, r], G[r, r + 1] = t, -t, 1
        if g < 0:
            G = np.rint(np.linalg.inv(G)).astype(np.int64)
        B = B @ G
    return B


def link_determinant(word, n):
    """|Delta(-1)| via det(I - Burau_reduced(beta)) at t = -1 (stabilised to odd n)."""
    word = list(word)
    if n % 2 == 0:
        word.append(n)
        n += 1
    if n == 1:
        return 1
    B = burau_reduced_minus1(word, n)
    return abs(det_int((np.eye(n - 1, dtype=np.int64) - B).tolist()))


KNOT_PANEL = [
    # name, braid word, strands, expected determinant (None = not asserted)
    ("unknot", [1], 2, 1),
    ("3_1 trefoil", [1, 1, 1], 2, 3),
    ("4_1 figure-eight", [1, -2, 1, -2], 3, 5),
    ("5_1", [1] * 5, 2, 5),
    ("5_2", [1, 1, 1, 2, -1, 2], 3, 7),
    ("6_1", [1, 1, 2, -1, -3, 2, -3], 4, 9),
    ("8_19 = T(3,4)", [1, 2] * 4, 3, 3),
    ("3_1 # 3_1 granny", [1, 1, 1, 2, 2, 2], 3, 9),
    ("3_1 # -3_1 square", [1, 1, 1, -2, -2, -2], 3, 9),
    ("Hopf link", [1, 1], 2, 2),
    ("T(2,4)", [1] * 4, 2, 4),
    ("T(2,6)", [1] * 6, 2, 6),
    ("Borromean rings", [1, -2] * 3, 3, 16),
    ("T(3,3)", [1, 2] * 3, 3, None),
]


def knot_panel():
    rows = []
    for name, word, n, det_exp in KNOT_PANEL:
        f = jones_f(word, n)
        c = components(word, n)
        dfox = fox_dim(word, n)
        det = link_determinant(word, n)
        v_fox = jones_eval(f, math.pi / 3)
        v_m1 = jones_eval(f, math.pi)
        v_1 = jones_eval(f, 0.0)
        row = {"name": name, "c": c, "V(t)": lstr(jones_in_t(f), "t"),
               "#Fox": 3 ** dfox, "det": det,
               "LM: |V(e^{i pi/3})|^2 = #Fox/3": abs(abs(v_fox) ** 2 - 3 ** (dfox - 1)) < 1e-9,
               "|V(-1)| = det": abs(abs(v_m1) - det) < 1e-9,
               "V(1) = (-2)^{c-1}": abs(v_1 - (-2) ** (c - 1)) < 1e-9,
               "det as expected": det_exp is None or det == det_exp,
               "3 | det <=> #Fox > 3": (det % 3 == 0) == (dfox > 1)}
        if c == 1:
            arf = 0 if det % 8 in (1, 7) else 1
            row["V(i) = (-1)^Arf"] = abs(jones_eval(f, math.pi / 2) - (-1) ** arf) < 1e-9
            row["V(e^{2 pi i/3}) = 1"] = abs(jones_eval(f, 2 * math.pi / 3) - 1) < 1e-9
        rows.append(row)
    return rows


# ---- Fox = Vieta; the affine wall ---------------------------------------------------
def fox_vieta_and_affine_wall():
    sols = [s for s in itertools.product(range(3), repeat=3) if sum(s) % 3 == 0]
    consts = [s for s in sols if len(set(s)) == 1]
    bij = [s for s in sols if len(set(s)) == 3]
    mult = all((sum(s) % 3 == 0) == (abs(W3 ** s[0] * W3 ** s[1] * W3 ** s[2] - 1) < 1e-12)
               for s in itertools.product(range(3), repeat=3))
    fox_form = all(((2 * a - b - c) % 3 == 0) == ((a + b + c) % 3 == 0)
                   for a, b, c in itertools.product(range(3), repeat=3))
    neq = {(a, b) for a in range(3) for b in range(3) if a != b}
    malcev_witness = ((0, 1), (1, 0), (1, 2))
    m = tuple((x - y + z) % 3 for x, y, z in zip(*malcev_witness))
    # boundary relations of random braids (tangles) are graphs of linear maps fixing constants
    rng = random.Random(3)
    lin_ok = True
    for _ in range(50):
        n = rng.randint(2, 5)
        word = [rng.choice([1, -1]) * rng.randint(1, n - 1) for _ in range(rng.randint(1, 12))]
        M = fox_matrix(word, n)
        lin_ok &= bool(np.all((M @ np.ones(n, dtype=np.int64)) % 3 == 1))
    return {"solutions of a+b+c=0": len(sols), "constants": len(consts), "bijections": len(bij),
            "a+b+c=0 <=> w^a w^b w^c = 1": mult, "Fox 2a=b+c <=> a+b+c=0": fox_form,
            "|!=| = 6 is not a power of 3": len(neq) == 6,
            "Mal'tsev x-y+z on (0,1),(1,0),(1,2)": m, "result in '!='": m in neq,
            "braid boundary relations linear and fix constants": lin_ok}


def reidemeister_one_check():
    out = {}
    for name, word, n in [("trefoil sigma_1^3", [1, 1, 1], 2), ("trefoil sigma_1^3 sigma_2 (kink)", [1, 1, 1, 2], 3)]:
        cols = fox_colourings(word, n)
        proper = [x for x in cols if all(t == 3 for t in crossing_types(word, n, x))]
        out[name] = (len(cols), len(proper))
    return out


# ---- vertex-strand / edge-clasp encoding --------------------------------------------
def pure_generator(i, j):
    """A_{ij} (1-based, i<j) = s_{j-1}...s_{i+1} s_i^2 s_{i+1}^{-1}...s_{j-1}^{-1}."""
    return list(range(j - 1, i, -1)) + [i, i] + [-k for k in range(i + 1, j)]


def clasp_encoding(nv, edges):
    word = []
    for u, v in edges:
        a, b = min(u, v) + 1, max(u, v) + 1
        word += pure_generator(a, b)
    return word


def linking_matrix(word, n):
    pos = list(range(n))          # pos[k] = strand at position k
    lk = np.zeros((n, n))
    for g in word:
        i = abs(g) - 1
        a, b = pos[i], pos[i + 1]
        s = 0.5 if g > 0 else -0.5
        lk[a, b] += s
        lk[b, a] += s
        pos[i], pos[i + 1] = b, a
    return lk


def proper_colourings(nv, edges):
    return [c for c in itertools.product(range(3), repeat=nv) if all(c[u] != c[v] for u, v in edges)]


def clasp_report():
    graphs = {"K3": (3, [(0, 1), (1, 2), (0, 2)]), "P3": (3, [(0, 1), (1, 2)]),
              "C5": (5, [(i, (i + 1) % 5) for i in range(5)]),
              "K4": (4, [(u, v) for u in range(4) for v in range(u + 1, 4)])}
    rows = {}
    for name, (nv, edges) in graphs.items():
        word = clasp_encoding(nv, edges)
        lk = linking_matrix(word, nv)
        adj = np.zeros((nv, nv))
        for u, v in edges:
            adj[u, v] = adj[v, u] = 1
        fox = set(fox_colourings(word, nv))
        col = set(proper_colourings(nv, edges))
        f = jones_f(word, nv)
        lm = abs(abs(jones_eval(f, math.pi / 3)) ** 2 - len(fox) / 3) < 1e-9
        rows[name] = {"crossings": len(word), "linking matrix = adjacency": bool(np.allclose(lk, adj)),
                      "#Fox": len(fox), "#proper": len(col), "#proper & Fox": len(fox & col),
                      "constants are Fox": all((c,) * nv in fox for c in range(3)),
                      "Lickorish-Millett": lm}
    return rows


# ---- B_3 dichotomy; standard modules of TL_n -------------------------------------------
def link_states(n, k):
    """Planar matchings of n points with k defects, no cup enclosing a defect."""
    out = []

    def rec(i, s, stack):
        if i == n:
            if not stack and s.count(-1) == k:
                out.append(tuple(s))
            return
        if not stack:
            rec(i + 1, s + [-1], stack)
        rec(i + 1, s + [None], stack + [i])
        if stack:
            j = stack[-1]
            s2 = list(s) + [j]
            s2[j] = i
            rec(i + 1, s2, stack[:-1])
    rec(0, [], [])
    return out


def standard_module(n, k, A):
    """Matrices of e_1..e_{n-1} on the standard module W_{n,k} of TL_n(d), d = -A^2 - A^{-2}."""
    d = -A ** 2 - A ** -2
    S = link_states(n, k)
    idx = {s: j for j, s in enumerate(S)}
    mats = []
    for i in range(n - 1):
        M = np.zeros((len(S), len(S)), complex)
        for col, s in enumerate(S):
            if s[i] == i + 1:
                M[col, col] += d
                continue
            a, b = s[i], s[i + 1]
            if a == -1 and b == -1:
                continue
            t = list(s)
            if a >= 0 and b >= 0:
                t[a], t[b] = b, a
            elif a == -1:
                t[b] = -1
            else:
                t[a] = -1
            t[i], t[i + 1] = i + 1, i
            M[idx[tuple(t)], col] += 1
        mats.append(M)
    return mats


def braid_matrices(n, k, A):
    es = standard_module(n, k, A)
    I = np.eye(es[0].shape[0])
    return [A * I + e / A for e in es]


def _pkey(M, digits=6):
    m = M.shape[0]
    M = M / (np.linalg.det(M) + 0j) ** (1.0 / m)
    flat = M.ravel()
    k = int(np.argmax(np.abs(flat) > 1e-6))
    M = M * (abs(flat[k]) / flat[k])
    v = np.round(M.ravel(), digits)
    return tuple(np.concatenate([v.real, v.imag]) + 0.0)


def projective_image_size(gens, cap=30000):
    I = np.eye(gens[0].shape[0], dtype=complex)
    seen = {_pkey(I)}
    frontier = [I]
    allg = gens + [np.linalg.inv(g) for g in gens]
    while frontier:
        nxt = []
        for M in frontier:
            for g in allg:
                N = M @ g
                key = _pkey(N)
                if key not in seen:
                    seen.add(key)
                    nxt.append(N)
                    if len(seen) > cap:
                        return None
        frontier = nxt
    return len(seen)


def galois_As(r):
    return [cmath.exp(-1j * math.pi * k / (2 * r)) for k in range(1, 4 * r) if math.gcd(k, 4 * r) == 1]


def word_trace_ratio(word, n, k, A):
    gs = braid_matrices(n, k, A)
    M = np.eye(gs[0].shape[0], dtype=complex)
    for w in word:
        g = gs[abs(w) - 1]
        M = M @ (g if w > 0 else np.linalg.inv(g))
    return abs(np.trace(M)) / abs(np.linalg.det(M)) ** (1.0 / M.shape[0])


def infinite_order_certificate(word, n, k, r):
    """Kronecker: a finite-order projective image has |tr|/|det|^{1/dim} <= dim under every
    Galois conjugate A -> A^k.  Returns the maximum over conjugates."""
    return max(word_trace_ratio(word, n, k, A) for A in galois_As(r))


def kronecker_exceptional_set(R):
    return [r for r in range(1, R + 1)
            if not any(math.gcd(j, r) == 1 and r < 3 * j < 2 * r for j in range(1, r))]


def b3_dichotomy(rs=tuple(range(1, 13)), cap=3000):
    """W_{3,1} at A_r.  Infinite order of g = s1 s2^{-1} is certified either by a Galois conjugate
    of tr g exceeding 2 (Kronecker) or, at r = 3, by g being a non-trivial unipotent."""
    rows = []
    for r in rs:
        A = cmath.exp(-1j * math.pi / (2 * r))
        s1, s2 = braid_matrices(3, 1, A)
        g = s1 @ np.linalg.inv(s2)
        size = projective_image_size([s1, s2], cap=cap)
        g_cert = infinite_order_certificate([1, -2], 3, 1, r)
        formula = max(abs(1 - 2 * math.cos(2 * math.pi * j / r)) for j in range(1, r + 1) if math.gcd(j, r) == 1)
        unipotent = abs(np.trace(g) - 2) < 1e-9 and abs(np.linalg.det(g) - 1) < 1e-9 and not np.allclose(g, np.eye(2))
        rows.append({"r": r, "projective image": size if size else f">{cap}",
                     "max conj |tr(s1 s2^-1)|": round(float(g_cert), 6),
                     "= max|1-2cos(2 pi j/r)|": bool(abs(g_cert - formula) < 1e-9),
                     "Kronecker certificate (>2)": bool(g_cert > 2 + 1e-9),
                     "non-trivial unipotent": bool(unipotent)})
    return rows


BN_CERT_WORDS = {(4, 2, 10): [2, 2, -3, 1, 1, 3, 2, -3], (5, 1, 10): [2, 1, 1, 2, 1, -3, 2],
                 (5, 3, 10): [-2, 3, 1]}


def bn_modules(full=True):
    rows = []
    cases = [(4, 2, 6), (4, 0, 6), (4, 0, 10)]
    if full:
        cases += [(5, 1, 6), (5, 3, 6)]
    for n, k, r in cases:
        A = cmath.exp(-1j * math.pi / (2 * r))
        size = projective_image_size(braid_matrices(n, k, A), cap=40000)
        rows.append({"n": n, "defects": k, "r": r, "dim": len(link_states(n, k)), "projective image": size})
    for (n, k, r), word in BN_CERT_WORDS.items():
        val = infinite_order_certificate(word, n, k, r)
        rows.append({"n": n, "defects": k, "r": r, "dim": len(link_states(n, k)),
                     "certificate word": word, "max conj ratio": round(val, 6),
                     "infinite": val > len(link_states(n, k)) + 1e-6})
    return rows


# ---- plane graphs, ribbon state sums, medial links -------------------------------------
def _poly_xy(n, edges):
    return n, tuple(edges)


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


def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def is_straight_line_embedding(coords, edges):
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) == 4 and _segments_cross(coords[a], coords[b], coords[c], coords[d]):
            return False
    return True


def rotation_system(coords, edges):
    nv = len(coords)
    out = [[] for _ in range(nv)]
    for j, (u, v) in enumerate(edges):
        out[u].append(2 * j)
        out[v].append(2 * j + 1)

    def head_tail(dart):
        u, v = edges[dart // 2]
        return (v, u) if dart % 2 == 0 else (u, v)
    rot = []
    for x in range(nv):
        def ang(dart):
            h, t = head_tail(dart)
            return math.atan2(coords[h][1] - coords[t][1], coords[h][0] - coords[t][0])
        rot.append(sorted(out[x], key=ang))
    return rot


def ribbon_loops(nv, edges, rot, S):
    """Boundary components of the ribbon subgraph (V, S) of a plane graph."""
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


def components_count(nv, edges_sub):
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges_sub:
        a, b = find(u), find(v)
        if a != b:
            parent[a] = b
    return len({find(x) for x in range(nv)})


def state_table(coords, edges):
    """N[(|S|, loops(S), k(S))] over all S subset E."""
    nv, m = len(coords), len(edges)
    rot = rotation_system(coords, edges)
    table = {}
    for mask in range(1 << m):
        S = [j for j in range(m) if mask >> j & 1]
        key = (len(S), ribbon_loops(nv, edges, rot, S), components_count(nv, [edges[j] for j in S]))
        table[key] = table.get(key, 0) + 1
    return table


def medial_bracket(coords, edges):
    """<D_G> = sum_S A^{|S|-|E-S|} d^{loops(S)-1} (normalised, A-smoothing <-> S)."""
    m = len(edges)
    br = {}
    for (s, l, _), cnt in state_table(coords, edges).items():
        br = ladd(br, lscale(lpow(D_LP, l - 1), cnt, 2 * s - m))
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


def _padd(p, q):
    r = dict(p)
    for k, c in q.items():
        r[k] = r.get(k, 0) + c
        if r[k] == 0:
            del r[k]
    return r


@lru_cache(maxsize=None)
def tutte(edges):
    """Tutte polynomial of a connected multigraph by deletion-contraction: {(i,j): coeff of x^i y^j}."""
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
    return _padd(tutte(_canon(rest)), tutte(_canon(con)))


def tutte_eval(T, x, y):
    return sum(c * x ** i * y ** j for (i, j), c in T.items())


def tutte_on_jones_curve(T, nv, m):
    """A^{2|V|-2-|E|} T(-A^{-4}, -A^4) as a Laurent polynomial in A."""
    out = {}
    for (i, j), c in T.items():
        out = ladd(out, {4 * j - 4 * i + 2 * nv - 2 - m: c * (-1) ** (i + j)})
    return out


def laplacian_rank_mod3(nv, edges):
    L = [[0] * nv for _ in range(nv)]
    for u, v in edges:
        L[u][u] += 1
        L[v][v] += 1
        L[u][v] -= 1
        L[v][u] -= 1
    return rank_mod_p(L, 3)


def medial_fox_dim(coords, edges):
    """Fox 3-colourings of the alternating medial diagram: arcs <-> edges of G, and at the
    crossing of e = uv:  x_e + x_{prev_u(e)} + x_{prev_v(e)} = 0 (mod 3)."""
    rot = rotation_system(coords, edges)
    pos = {}
    for x in range(len(coords)):
        for i, d in enumerate(rot[x]):
            pos[d] = (x, i)
    m = len(edges)
    M = [[0] * m for _ in range(m)]
    for j in range(m):
        M[j][j] += 1
        for dart in (2 * j, 2 * j + 1):
            x, i = pos[dart]
            prev = rot[x][(i - 1) % len(rot[x])]
            M[j][prev // 2] += 1
    return m - rank_mod_p(M, 3)


def medial_components(coords, edges):
    """Components of the medial link: orbits of <alpha, sigma^{-1} alpha sigma> on corners (darts)."""
    rot = rotation_system(coords, edges)
    nxt, prv = {}, {}
    for x in range(len(coords)):
        k = len(rot[x])
        for i, d in enumerate(rot[x]):
            nxt[d] = rot[x][(i + 1) % k]
            prv[d] = rot[x][(i - 1) % k]
    parent = {d: d for d in nxt}

    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d
    for d in nxt:
        for e in (d ^ 1, prv[nxt[d] ^ 1]):
            a, b = find(d), find(e)
            if a != b:
                parent[a] = b
    return len({find(d) for d in nxt})


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


def medial_report(names=None):
    rows = {}
    for name, (coords, edges) in PLANE_GRAPHS.items():
        if names and name not in names:
            continue
        nv, m = len(coords), len(edges)
        T = tutte(_canon(edges))
        table = state_table(coords, edges)
        br = {}
        for (s, l, _), cnt in table.items():
            br = ladd(br, lscale(lpow(D_LP, l - 1), cnt, 2 * s - m))
        thist = br == tutte_on_jones_curve(T, nv, m)
        faces = max(l for (s, l, _) in table if s == m)
        P3 = count_3_colourings(nv, edges)
        fk = sum(cnt * (-1) ** s * 3 ** k for (s, l, k), cnt in table.items())
        loopform = sum(cnt * (-1 / math.sqrt(3)) ** s * math.sqrt(3) ** l for (s, l, k), cnt in table.items()) * 3 ** (nv / 2)
        # topological point on the same loop weight sqrt(3): A^2 = e^{5 pi i/6}
        A6 = cmath.exp(5j * math.pi / 12)
        Z_top = sum(cnt * A6 ** s * A6 ** (-(m - s)) * math.sqrt(3) ** l for (s, l, k), cnt in table.items())
        top_ok = abs(Z_top / math.sqrt(3) - leval(br, A6)) < 1e-8 and abs(A6 ** -4 - cmath.exp(1j * math.pi / 3)) < 1e-12
        dfox = medial_fox_dim(coords, edges)
        rk = laplacian_rank_mod3(nv, edges)
        r = nv - 1
        Tfox = tutte_eval(T, W3 ** 2, W3)
        vfox = leval(br, cmath.exp(-1j * math.pi / 12))
        rows[name] = {
            "|V|,|E|": (nv, m), "planar coordinates": is_straight_line_embedding(coords, edges),
            "faces = |E|-|V|+2": faces == m - nv + 2,
            "Thistlethwaite <D>=A^(2V-2-E)T(-A^-4,-A^4)": thist,
            "P(G,3)": P3, "T(-2,0) = (-1)^{V-1} P/3": tutte_eval(T, -2, 0) * 3 * (-1) ** (nv - 1) == P3,
            "FK sum = P(G,3)": fk == P3, "loop form (d=sqrt3, a/b=-1/sqrt3) = P": abs(loopform - P3) < 1e-6,
            "topological point (d=sqrt3, a/b=e^{5pi i/6}) = <D>(t=e^{i pi/3})": top_ok,
            "#Fox(L(G)) (arcs)": 3 ** dfox, "3^(1+corank3 L~)": 3 ** (1 + r - rk),
            "|T(w^2,w)|^2 = 3^(r-rank3 L)": abs(abs(Tfox) ** 2 - 3 ** (r - rk)) < 1e-8,
            "Lickorish-Millett 3|<D>(e^{i pi/3})|^2 = #Fox": abs(3 * abs(vfox) ** 2 - 3 ** dfox) < 1e-8,
            "components of L(G)": medial_components(coords, edges),
            "|T(-1,-1)| = 2^(c-1)": abs(tutte_eval(T, -1, -1)) == 2 ** (medial_components(coords, edges) - 1),
            "bracket": br, "tutte": T}
    return rows


def _complete(n):
    return n, [(u, v) for u in range(n) for v in range(u + 1, n)]


GAUSS_GRAPHS = {
    "K5": _complete(5),
    "K33": (6, [(u, v) for u in range(3) for v in range(3, 6)]),
    "Petersen": (10, [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
                 + [(i, i + 5) for i in range(5)]),
    "K4 + C5 (disconnected)": (9, [(u, v) for u in range(4) for v in range(u + 1, 4)] + [(4 + i, 4 + (i + 1) % 5) for i in range(5)]),
}


def _random_graph(n, prob, seed):
    rng = random.Random(seed)
    return n, [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < prob]


def gauss_phase_check():
    """Theorem 8.5, full complex value: T_G(w^2,w) = w^{|E|} 3^{-k} (w-1)^{-r} sum_sigma w^{-sigma^T L sigma}."""
    graphs = dict(GAUSS_GRAPHS)
    for s_ in (1, 2):
        graphs[f"G(8,1/2) seed {s_}"] = _random_graph(8, 0.5, s_)
    rows = {}
    for name, (n, edges) in graphs.items():
        T = tutte(_canon(edges))
        k = components_count(n, edges)
        r = n - k
        L = np.zeros((n, n), dtype=np.int64)
        for u, v in edges:
            L[u, u] += 1
            L[v, v] += 1
            L[u, v] -= 1
            L[v, u] -= 1
        sig = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int64)
        quad = np.einsum("ij,jk,ik->i", sig, L, sig) % 3
        S = np.sum(W3 ** (-quad))
        rhs = W3 ** len(edges) / (3 ** k * (W3 - 1) ** r) * S
        lhs = tutte_eval(T, W3 ** 2, W3)
        rk = rank_mod_p(L.tolist(), 3)
        rows[name] = {"|E|": len(edges), "k": k, "|lhs - rhs|": float(abs(lhs - rhs)),
                      "|T|^2 = 3^(r-rank3 L)": bool(abs(abs(lhs) ** 2 - 3 ** (r - rk)) < 1e-8)}
    return rows


def jones_blindness():
    rep = medial_report({"W6", "H7", "prism", "bipyramid"})
    W6, H7 = rep["W6"], rep["H7"]
    pr, bp = rep["prism"], rep["bipyramid"]
    TW, TH = W6["tutte"], H7["tutte"]

    def restrict(T):
        out = {}
        for (i, j), c in T.items():
            out = ladd(out, {i - j: c})
        return out
    return {"W6: P(G,3)": W6["P(G,3)"], "H7: P(G,3)": H7["P(G,3)"],
            "T_W6(x,1/x) vs T_H7(x,1/x) (sign, shift)": lunit_ratio(restrict(TW), restrict(TH)),
            "<L(W6)> = unit * <L(H7)>": lunit_ratio(W6["bracket"], H7["bracket"]),
            "Tutte polynomials differ": TW != TH,
            "T_W6(x,1/x)": lstr(restrict(TW), "x"),
            "prism: P(G,3)": pr["P(G,3)"], "bipyramid: P(G,3)": bp["P(G,3)"],
            "<L(prism)>(A) = <L(bipyramid)>(A^-1)": pr["bracket"] == lmirror(bp["bracket"]),
            "<L(prism)> amphichiral?": lunit_ratio(pr["bracket"], lmirror(pr["bracket"])) is not None}


def search_jones_blind_pairs():
    """Exhaustive search over bridgeless connected planar graphs with <= 7 vertices
    (networkx graph atlas).  Optional."""
    try:
        import networkx as nx
    except ImportError:
        return None
    groups = {}
    count = 0
    for G in nx.graph_atlas_g()[1:]:
        if G.number_of_edges() == 0 or not nx.is_connected(G) or not nx.check_planarity(G)[0]:
            continue
        if any(True for _ in nx.bridges(G)):
            continue
        count += 1
        T = tutte(_canon(list(G.edges())))
        J = {}
        for (i, j), c in T.items():
            J[i - j] = J.get(i - j, 0) + c
        J = {k: v for k, v in J.items() if v}
        lo, hi = min(J), max(J)
        s = 1 if J[hi] > 0 else -1
        key = tuple(s * J.get(k, 0) for k in range(lo, hi + 1))
        c3 = count_3_colourings(G.number_of_nodes(), list(G.edges())) > 0
        groups.setdefault(key, []).append((G.number_of_nodes(), G.number_of_edges(), c3, sorted(G.edges())))
    mixed = {k: v for k, v in groups.items() if len({c for _, _, c, _ in v}) == 2}
    return {"graphs": count, "classes": len(groups), "mixed classes": mixed}


# =============================================================================
# Part C. Quantum Hopf algebras
# =============================================================================
def kron_list(ms):
    return reduce(np.kron, ms)


def uq_tensor(q, n):
    """U_q(sl_2) on V^{(x)n}: Delta(E)=E(x)K+1(x)E, Delta(F)=F(x)1+K^{-1}(x)F, Delta(K)=K(x)K."""
    E = np.array([[0, 1], [0, 0]], complex)
    F = np.array([[0, 0], [1, 0]], complex)
    K = np.diag([q, 1 / q]).astype(complex)
    Ki = np.diag([1 / q, q]).astype(complex)
    I = np.eye(2)
    En = sum(kron_list([I] * i + [E] + [K] * (n - i - 1)) for i in range(n))
    Fn = sum(kron_list([Ki] * i + [F] + [I] * (n - i - 1)) for i in range(n))
    Kn = kron_list([K] * n)
    return En, Fn, Kn


def tl_on_tensor(q, n, i):
    e = np.zeros((4, 4), complex)
    e[1, 1], e[1, 2], e[2, 1], e[2, 2] = q, -1, -1, 1 / q
    return kron_list([np.eye(2)] * i + [e] + [np.eye(2)] * (n - i - 2))


def hopf_checks(q=0.83 * cmath.exp(0.41j), n=4):
    En, Fn, Kn = uq_tensor(q, n)
    Kinv = np.linalg.inv(Kn)
    rel1 = np.abs(Kn @ En @ Kinv - q ** 2 * En).max()
    rel2 = np.abs(Kn @ Fn @ Kinv - q ** -2 * Fn).max()
    rel3 = np.abs(En @ Fn - Fn @ En - (Kn - Kinv) / (q - 1 / q)).max()
    # antipode: S(E) = -E K^{-1}, S(F) = -K F, S(K) = K^{-1}; the dual module x -> rho(S(x))^T
    E, F, K = uq_tensor(q, 1)
    Ki = np.linalg.inv(K)
    SE, SF, SK = -E @ Ki, -K @ F, Ki
    dE, dF, dK = SE.T, SF.T, SK.T
    dual_rel = max(np.abs(dK @ dE @ np.linalg.inv(dK) - q ** 2 * dE).max(),
                   np.abs(dE @ dF - dF @ dE - (dK - np.linalg.inv(dK)) / (q - 1 / q)).max())
    # m(S (x) id) Delta = eps: S(E)K + E = 0, S(F) + S(K^{-1})F = S(F) + K F = 0, S(K)K = 1
    antipode_axiom = max(np.abs(SE @ K + E).max(), np.abs(SF + K @ F).max(), np.abs(SK @ K - np.eye(2)).max())
    # S^2(E) = S(-E K^{-1}) = -S(K^{-1}) S(E) = K E K^{-1} = q^2 E  (S is not involutive)
    S2E = -K @ SE
    s2 = np.abs(S2E - K @ E @ Ki).max()
    s2_not_id = np.abs(S2E - E).max()
    # commutant
    comm = 0.0
    tl = 0.0
    d = q + 1 / q
    es = [tl_on_tensor(q, n, i) for i in range(n - 1)]
    for e in es:
        comm = max(comm, max(np.abs(e @ X - X @ e).max() for X in (En, Fn, Kn)))
        tl = max(tl, np.abs(e @ e - d * e).max())
    for i in range(n - 2):
        tl = max(tl, np.abs(es[i] @ es[i + 1] @ es[i] - es[i]).max(), np.abs(es[i + 1] @ es[i] @ es[i + 1] - es[i + 1]).max())
    # zig-zag identities: (<w'| (x) 1)(1 (x) |w>) = (1 (x) <w'|)(|w> (x) 1) = -1
    w = np.zeros(4, complex)
    w[1], w[2] = q, -1
    wp = np.zeros(4, complex)
    wp[1], wp[2] = 1, -1 / q
    I2 = np.eye(2)
    z1 = np.kron(wp.reshape(1, 4), I2) @ np.kron(I2, w.reshape(4, 1))
    z2 = np.kron(I2, wp.reshape(1, 4)) @ np.kron(w.reshape(4, 1), I2)
    zig = max(np.abs(z1 + I2).max(), np.abs(z2 + I2).max())
    E2, F2, K2 = uq_tensor(q, 2)
    inv = max(np.abs(E2 @ w).max(), np.abs(F2 @ w).max(), np.abs(K2 @ w - w).max(),
              np.abs(wp @ E2).max(), np.abs(wp @ F2).max(), np.abs(wp @ K2 - wp).max())
    return {"w, w' invariant": inv, "zig-zags = -1": zig, "K E K^-1 = q^2 E": rel1, "K F K^-1 = q^-2 F": rel2, "[E,F] = (K-K^-1)/(q-q^-1)": rel3,
            "dual module via S satisfies relations": dual_rel, "antipode axiom on generators": antipode_axiom,
            "S^2(E) - K E K^-1": s2, "S^2(E) - E (not involutive)": s2_not_id,
            "[e_i, U_q] on V^(x)n": comm, "TL relations": tl}


def quantum_trace_bracket(word, n, A):
    q = -A ** 2
    En, Fn, Kn = uq_tensor(q, n)
    M = np.eye(2 ** n, dtype=complex)
    I = np.eye(2 ** n)
    for g in word:
        e = tl_on_tensor(q, n, abs(g) - 1)
        s = A * I + e / A if g > 0 else I / A + A * e
        M = M @ s
    d = -A ** 2 - A ** -2
    return np.trace(Kn @ M) / d


def quantum_trace_checks():
    out = {}
    for A in (0.77 * cmath.exp(0.29j), cmath.exp(-1j * math.pi / 12), cmath.exp(-1j * math.pi / 20)):
        err = 0.0
        for name, word, n, _ in KNOT_PANEL:
            if n > 5:
                continue
            err = max(err, abs(quantum_trace_bracket(word, n, A) - leval(bracket_closure(word, n), A)))
        out[f"A={A:.3f}"] = err
    return out


def highest_weight_multiplicities(n, q):
    En, _, _ = uq_tensor(q, n)
    weight = [n - 2 * bin(b).count("1") for b in range(2 ** n)]
    out = {}
    for m in range(n, -1, -2):
        cols = [b for b in range(2 ** n) if weight[b] == m]
        rows = [b for b in range(2 ** n) if weight[b] == m + 2]
        if not rows:
            out[m] = len(cols)
            continue
        sub = En[np.ix_(rows, cols)]
        out[m] = len(cols) - np.linalg.matrix_rank(sub, tol=1e-9)
    return out


def lusztig_rosso(nmax=7):
    rows = []
    for n in range(2, nmax + 1):
        classical = {m: math.comb(n, (n - m) // 2) - (math.comb(n, (n - m) // 2 - 1) if (n - m) // 2 >= 1 else 0)
                     for m in range(n, -1, -2)}
        gen = [highest_weight_multiplicities(n, q) for q in (0.71 * cmath.exp(0.53j), 1.3 * cmath.exp(-1.1j), 2.0)]
        q1 = highest_weight_multiplicities(n, 1.0)
        root = highest_weight_multiplicities(n, cmath.exp(1j * math.pi / 3))
        rows.append({"n": n, "classical": classical, "generic q equal": all(g == classical for g in gen),
                     "q=1 equal": q1 == classical, "q=e^{i pi/3}": root, "jump at root of unity": root != classical})
    return rows


# ---- S_3^+ = S_3 --------------------------------------------------------------------
def magic_certificate():
    """Explicit degree-3 certificate: [u00, u11] = rho_B - rev(rho_B) in the ideal of the
    3x3 magic unitary relations; all pairs by row/column permutations."""
    def build(rp, cp):
        U = lambda i, j: NC.var(("u", rp[i], cp[j]))
        R = lambda i: nsum(U(i, j) for j in range(3)) - 1
        C = lambda j: nsum(U(i, j) for i in range(3)) - 1
        rhoA = -U(0, 0) * U(1, 0) + U(0, 0) * U(0, 2) - U(0, 0) * C(2) + U(0, 0) * R(1)
        identA = (U(0, 0) * U(1, 1) - U(0, 0) * U(2, 2) - rhoA).is_zero()
        rhoB = U(0, 0) * U(1, 1) * U(0, 1) + U(0, 0) * U(2, 2) * U(0, 2) + rhoA * U(0, 2) - U(0, 0) * U(1, 1) * R(0)
        identB = (U(0, 0) * U(1, 1) - U(0, 0) * U(1, 1) * U(0, 0) - rhoB).is_zero()
        comm = (U(0, 0) * U(1, 1) - U(1, 1) * U(0, 0) - (rhoB - rhoB.reverse())).is_zero()
        return identA and identB and comm, max(rhoB.degree(), rhoB.reverse().degree())
    results = [build(rp, cp) for rp in itertools.permutations(range(3)) for cp in itertools.permutations(range(3))]
    # 4x4 magic unitary with non-commuting entries
    p = np.array([[1, 0], [0, 0]], float)
    v = np.array([1, 1]) / math.sqrt(2)
    qq = np.outer(v, v)
    I2, Z2 = np.eye(2), np.zeros((2, 2))
    u = [[p, I2 - p, Z2, Z2], [I2 - p, p, Z2, Z2], [Z2, Z2, qq, I2 - qq], [Z2, Z2, I2 - qq, qq]]
    magic = all(np.allclose(sum(u[i][j] for j in range(4)), I2) and np.allclose(sum(u[j][i] for j in range(4)), I2)
                for i in range(4)) and all(np.allclose(x @ x, x) and np.allclose(x, x.T) for row in u for x in row)
    return {"all 36 relabelled identities hold": all(r for r, _ in results), "certificate degree": results[0][1],
            "4x4 magic unitary valid": magic, "||[u00,u22]|| (n=4)": float(np.abs(p @ qq - qq @ p).max())}


def set_partitions(k):
    def rec(i, blocks):
        if i == k:
            yield [list(b) for b in blocks]
            return
        for b in blocks:
            b.append(i)
            yield from rec(i + 1, blocks)
            b.pop()
        blocks.append([i])
        yield from rec(i + 1, blocks)
        blocks.pop()
    yield from rec(0, [])


def is_noncrossing(part):
    lab = {}
    for bi, b in enumerate(part):
        for x in b:
            lab[x] = bi
    k = len(lab)
    for a, b, c, d in itertools.combinations(range(k), 4):
        if lab[a] == lab[c] and lab[b] == lab[d] and lab[a] != lab[b]:
            return False
    return True


def partition_vectors(parts, n, k):
    """T_pi in (C^n)^{(x)k}: indicator of {i : i_a = i_b whenever a ~_pi b}."""
    grid = np.array(list(itertools.product(range(n), repeat=k)))
    out = []
    for part in parts:
        mask = np.ones(len(grid), bool)
        for b in part:
            for x in b[1:]:
                mask &= grid[:, b[0]] == grid[:, x]
        out.append(mask.astype(float))
    return np.array(out)


def partition_spans(ks=(4, 5, 6), ns=(3, 4)):
    rows = []
    for k in ks:
        parts = list(set_partitions(k))
        nc = [p for p in parts if is_noncrossing(p)]
        for n in ns:
            V_all = partition_vectors(parts, n, k)
            V_nc = partition_vectors(nc, n, k)
            r_all, r_nc = np.linalg.matrix_rank(V_all), np.linalg.matrix_rank(V_nc)
            le_n = sum(1 for p in parts if len(p) <= n)
            rows.append({"k": k, "n": n, "#P(k)": len(parts), "#NC(k)": len(nc),
                         "rank span P": r_all, "rank span NC": r_nc, "#partitions <= n blocks": le_n,
                         "S_n^+ = S_n at level k": r_all == r_nc})
    return rows


# ---- Kneser graphs -------------------------------------------------------------------
def kneser_spectrum(n, k):
    lam = [(-1) ** j * math.comb(n - k - j, k - j) for j in range(k + 1)]
    mult = [math.comb(n, j) - (math.comb(n, j - 1) if j else 0) for j in range(k + 1)]
    return lam, mult


def kneser_report():
    rows = []
    for n, k in [(5, 2), (6, 2), (9, 3), (12, 4), (15, 5)]:
        lam, mult = kneser_spectrum(n, k)
        theta = 1 - lam[0] / min(lam)
        num_ok = None
        if math.comb(n, k) <= 300:
            V = list(itertools.combinations(range(n), k))
            Adj = np.array([[1.0 if not set(a) & set(b) else 0.0 for b in V] for a in V])
            ev = np.round(np.linalg.eigvalsh(Adj)).astype(int)
            num_ok = all(int(np.sum(ev == l)) == m for l, m in zip(lam, mult))
        rows.append({"KG(n,k)": (n, k), "eigenvalues": lam, "multiplicities": mult,
                     "theta(complement) = 1 - l_max/l_min": theta, "n/k": n / k,
                     "chi = n-2k+2 (Lovasz)": n - 2 * k + 2, "numerical spectrum agrees": num_ok})
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


def report(quick=False):
    t0 = time.time()
    print("quantum_braid_assault.py  --  Nonlocal Operator Fields, Anyonic Braids, Quantum Minimax Collapse")
    print("=" * 96)
    print("PART A  operator Bayer fields")
    _print("A.1 free-algebra identities (exact over Q(omega))", operator_bayer_identity())
    _print("A.2 random non-commuting order-3 unitaries (d=7)", operator_numeric_check())
    ti = three_idempotents()
    _print("A.3 three idempotents with sum lambda: (lambda, NC certificate degree, commutative degree)",
           [(str(l), dn, dc) for l, dn, dc in ti["rows"]])
    _print("    nonzero matrix points at the exceptional values", ti["points"])
    _print("    ||[a,b]|| at lambda = 3/2", ti["noncommutative at 3/2"])
    _print("    exact pattern for every lambda in F_p", three_idempotent_scan(p=101 if quick else 1009))
    _print("A.4 K_4 certificates", k4_certificates())
    _print("A.5 triangle rigidity and Vieta transport (W_5, K_4-free)", triangle_rigidity_and_transport())
    print("\nPART B  anyonic braids and knots")
    rows = knot_panel()
    print("\n--- B.3 knot panel (V(t) from the Temperley-Lieb state sum; Arf and V(e^{2 pi i/3}) = 1 for the knots)")
    print(f"    {sum(1 for r in rows if r['c'] == 1)} knots, {sum(1 for r in rows if r['c'] > 1)} links")
    for r in rows:
        checks = {k: v for k, v in r.items() if isinstance(v, bool)}
        print(f"    {r['name']:<20} c={r['c']}  det={r['det']:<3} #Fox={r['#Fox']:<3} all checks: {all(checks.values())}")
        print(f"        V = {r['V(t)']}")
    _print("B.4 Fox = Vieta and the affine wall", fox_vieta_and_affine_wall())
    _print("B.5 Reidemeister I: (#Fox colourings, #trichromatic at every crossing)", reidemeister_one_check())
    _print("B.6 vertex-strand / edge-clasp encoding", clasp_report())
    _print("B.7 B_3 on the 2-dim Jones module", b3_dichotomy())
    _print("    Kronecker exceptional set r <= 1000", kronecker_exceptional_set(1000))
    _print("B.8 B_4, B_5 standard modules", bn_modules(full=not quick))
    med = medial_report()
    print("\n--- B.9 medial links of plane graphs")
    for name, r in med.items():
        rr = {k: v for k, v in r.items() if k not in ("bracket", "tutte")}
        print(f"    {name}: {rr}")
        print(f"        <L({name})> = {lstr(r['bracket'])}")
    _print("B.9' Gauss sum at (w^2, w), full complex value (planar or not)", gauss_phase_check())
    _print("B.10 Jones blindness", jones_blindness())
    if not quick:
        s = search_jones_blind_pairs()
        if s is None:
            _print("B.10 exhaustive search", "networkx not available; skipped")
        else:
            _print("B.10 exhaustive search (bridgeless connected planar graphs, <= 7 vertices)",
                   {"graphs": s["graphs"], "classes of T(x,1/x) up to units": s["classes"],
                    "classes mixing 3-colourable and not": list(s["mixed classes"].values())})
    print("\nPART C  quantum Hopf algebras")
    _print("C.1 U_q(sl_2), antipode, Temperley-Lieb commutant (max errors)", hopf_checks())
    _print("C.2 quantum trace with pivotal K vs Temperley-Lieb bracket (max error over panel)", quantum_trace_checks())
    _print("C.3 Lusztig-Rosso rigidity: highest-weight multiplicities in V^(x)n", lusztig_rosso())
    _print("C.4 S_3^+ = S_3", magic_certificate())
    _print("    spans of partition maps", partition_spans())
    _print("C.5 Kneser graphs", kneser_report())
    print(f"\nTotal time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    report(quick=args.quick)
