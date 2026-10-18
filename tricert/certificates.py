"""Nullstellensatz certificates in R = K[x]/(x_v^3 - 1) (Section 3 of the paper).

- `vieta_certificate`: the identity of Lemma 3.4, 3 s_tau = sum of multipliers times q_e, multipliers of degree 2.
- `propagation_certificate`: the explicit certificate of Theorem 3.5 for a graph with a non-colourable triangle
  component: an identity c = sum_e beta_e q_e in R with c in {9, 27} and deg beta_e <= 4, so of degree 6.
- `floor_functional_check`: the functional of Theorem 3.3 vanishes on every x_w q_e and is 1 on 1.
- `ns_degree_f2`: exact Nullstellensatz degree over F_2 by Gaussian elimination on bitsets (Lemma 3.2).
"""
import itertools
from collections import deque
from fractions import Fraction

from . import ring
from .components import triangle_components, component_colourable

__all__ = ["vieta_certificate", "check_certificate", "propagation_certificate", "floor_functional_check",
           "ns_certificate_exists_f2", "ns_degree_f2"]


def _lin(n, coeffs):
    """sum_v coeffs[v] x_v."""
    out = {}
    for v, c in coeffs.items():
        if c:
            out = ring.add(out, ring.scale(ring.var(n, v), c))
    return out


def vieta_certificate(n, u, v, w):
    """{edge: multiplier} with 3(x_u + x_v + x_w) = sum multiplier * q_edge in R (Lemma 3.4)."""
    xw2 = ring.var(n, w, 2)
    m1 = ring.add(ring.scale(xw2, 2), ring.mul(ring.var(n, v), ring.var(n, w)))   # 2x_w^2 + x_v x_w
    m2 = ring.add(ring.scale(xw2, 2), ring.mul(ring.var(n, u), ring.var(n, w)))   # 2x_w^2 + x_u x_w
    e = lambda a, b: (min(a, b), max(a, b))  # noqa: E731
    return {e(u, w): m1, e(u, v): ring.scale(m1, -1), e(v, w): m2}


def check_certificate(n, cert, target, p=None):
    """Whether sum_e cert[e] q_e equals `target` in R (over Z, or over F_p); returns (ok, max multiplier degree)."""
    total = {}
    for (a, b), mult in cert.items():
        total = ring.add(total, ring.mul(mult, ring.q_edge(n, a, b)), p)
    diff = ring.sub(total, target, p)
    return ring.is_zero(diff, p), max((ring.deg(m) for m in cert.values()), default=-1)


def _add_cert(acc, cert, scale_poly):
    for e, m in cert.items():
        acc[e] = ring.add(acc.get(e, {}), ring.mul(m, scale_poly))
    return acc


def propagation_certificate(n, edges):
    """Theorem 3.5.  Returns (constant c, certificate {edge: beta_e}) with sum beta_e q_e = c in R, or None if
    every triangle component is colourable."""
    comps = triangle_components(n, edges)
    bad = next((K for K in comps if not component_colourable(K)), None)
    if bad is None:
        return None
    by_edge = {}
    for t in bad:
        for pr in itertools.combinations(t, 2):
            by_edge.setdefault(pr, []).append(t)
    t0 = bad[0]
    v0, v1, v2 = t0
    # forms f_0 = x_v0, f_1 = x_v1, f_2 = -x_v0 - x_v1, as coefficient pairs on (x_v0, x_v1)
    F = {0: (1, 0), 1: (0, 1), 2: (-1, -1)}
    form = {v0: 0, v1: 1, v2: 2}
    S = {v0: {}, v1: {}, v2: {t0: 1}}               # x_w = f_i(w) + sum_tau S_w[tau] s_tau
    seen = {t0}
    queue = deque([t0])
    conflict = None
    while queue and conflict is None:
        t = queue.popleft()
        for pr in itertools.combinations(t, 2):
            for s in by_edge[pr]:
                if s in seen or conflict is not None:
                    continue
                seen.add(s)
                u, v = pr
                w = next(x for x in s if x not in pr)
                i_new = 3 - form[u] - form[v]          # the third index
                S_new = {s: 1}
                for tau, c in S[u].items():
                    S_new[tau] = S_new.get(tau, 0) - c
                for tau, c in S[v].items():
                    S_new[tau] = S_new.get(tau, 0) - c
                if w in form:
                    if form[w] != i_new:
                        # f_old - f_new = S_new - S_old
                        diffS = dict(S_new)
                        for tau, c in S[w].items():
                            diffS[tau] = diffS.get(tau, 0) - c
                        conflict = (form[w], i_new, diffS)
                        continue
                else:
                    form[w], S[w] = i_new, S_new
                queue.append(s)
    assert conflict is not None
    i_old, i_new, diffS = conflict
    a = F[i_old][0] - F[i_new][0]
    b = F[i_old][1] - F[i_new][1]              # ell = a x_v0 + b x_v1
    # certificate for 3*ell = sum_tau diffS[tau] * 3 s_tau
    cert3 = {}
    for tau, c in diffS.items():
        if c:
            for e, m in vieta_certificate(n, *tau).items():
                cert3[e] = ring.add(cert3.get(e, {}), ring.scale(m, c))
    x0, x1 = ring.var(n, v0), ring.var(n, v1)
    e01 = (min(v0, v1), max(v0, v1))
    if (a, b) in ((1, -1), (-1, 1)):
        # 3 = x0 q01 + x0 (x1 + 2x0)(x0 - x1)  =>  9 = 3 x0 q01 + sgn * x0 (x1 + 2x0) * (3 ell)
        sgn = 1 if a == 1 else -1
        cert = _add_cert({}, cert3, ring.scale(ring.mul(x0, ring.add(x1, ring.scale(x0, 2))), sgn))
        cert[e01] = ring.add(cert.get(e01, {}), ring.scale(x0, 3))
        return 9, cert
    if (a, b) in ((2, 1), (-2, -1)):
        # 9 = (x1 + 2x0)(x1^2 - 2 x0 x1 + 4 x0^2)  =>  27 = sgn * (x1^2 - 2x0x1 + 4x0^2) * (3 ell)
        sgn = 1 if a == 2 else -1
        quad = ring.add(ring.add(ring.var(n, v1, 2), ring.scale(ring.mul(x0, x1), -2)), ring.scale(ring.var(n, v0, 2), 4))
        return 27, _add_cert({}, cert3, ring.scale(quad, sgn))
    if (a, b) in ((1, 2), (-1, -2)):
        sgn = 1 if a == 1 else -1
        quad = ring.add(ring.add(ring.var(n, v0, 2), ring.scale(ring.mul(x0, x1), -2)), ring.scale(ring.var(n, v1, 2), 4))
        return 27, _add_cert({}, cert3, ring.scale(quad, sgn))
    raise AssertionError("unexpected linear form %r" % ((a, b),))


def floor_functional_check(n, edges):
    """Theorem 3.3: L(1) = 1, L(x_a^2 x_b) = -1/2, L(x_a x_b x_c) = 1 vanishes on every x_w q_e (w in V)."""
    def L(f):
        tot = Fraction(0)
        for e, c in f.items():
            s = sum(e)
            if s == 0:
                tot += c
            elif s == 3:
                tot += c * (Fraction(-1, 2) if 2 in e else 1)
            else:
                raise ValueError("not a grade-0 element of degree <= 3")
        return tot
    return all(L(ring.mul(ring.var(n, w), ring.q_edge(n, a, b))) == 0 for w in range(n) for (a, b) in edges)


def ns_certificate_exists_f2(n, edges, d):
    """Whether 1 = sum_e beta_e q_e in R over F_2 with deg beta_e <= d - 2 (Lemma 3.2(a)); by the grading of
    Lemma 3.2(b) the multipliers may be taken of grade 1.  Gaussian elimination over F_2 on bitsets."""
    mons = ring.monomials(n, d - 2, grades={1})
    cols = []
    for (a, b) in edges:
        q = ring.q_edge(n, a, b)
        for m in mons:
            cols.append(ring.mul({m: 1}, q, 2))
    rows_index = {}
    for f in cols:
        for e in f:
            rows_index.setdefault(e, len(rows_index))
    one = (0,) * n
    rows_index.setdefault(one, len(rows_index))
    # build row bitsets: bit j set if column j has that monomial; augmented bit = target
    nr = len(rows_index)
    rows = [0] * nr
    for j, f in enumerate(cols):
        for e, c in f.items():
            if c % 2:
                rows[rows_index[e]] |= 1 << j
    aug = len(cols)
    rows[rows_index[one]] |= 1 << aug
    # elimination
    pivots = []
    rank_rows = []
    for r in rows:
        for (pc, pr) in zip(pivots, rank_rows):
            if (r >> pc) & 1:
                r ^= pr
        low = r & ((1 << aug) - 1)
        if low:
            pc = low.bit_length() - 1
            pivots.append(pc)
            rank_rows.append(r)
        elif (r >> aug) & 1:
            return False        # inconsistent: 0 = 1
    return True


def ns_degree_f2(n, edges, dmax=12):
    for d in range(3, dmax + 1, 3):
        if ns_certificate_exists_f2(n, edges, d):
            return d
    return None
