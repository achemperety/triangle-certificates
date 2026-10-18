"""Theorem 7.4: the Cayley graph of the binary icosahedral group 2I with respect to its elements of order 3.

Everything is exact.  Elements of Q(sqrt5) are pairs (a, b) = a + b*sqrt5 with rational a, b.
Ranks over Q are computed with Fractions; the rank of the norm system, whose entries lie in Q(sqrt5),
is computed modulo a prime l = 10009 (5 is a square mod l), which bounds it from below, and an explicit
kernel vector bounds it from above.
"""
from fractions import Fraction as F
import itertools

# ---------------------------------------------------------------- Q(sqrt5)
class Q5:
    __slots__ = ("a", "b")

    def __init__(self, a, b=0):
        self.a, self.b = F(a), F(b)

    def __add__(self, o):
        o = o if isinstance(o, Q5) else Q5(o)
        return Q5(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return Q5(-self.a, -self.b)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Q5) else Q5(o)))

    def __mul__(self, o):
        o = o if isinstance(o, Q5) else Q5(o)
        return Q5(self.a * o.a + 5 * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def __eq__(self, o):
        o = o if isinstance(o, Q5) else Q5(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def conj(self):
        """Galois conjugation sqrt5 -> -sqrt5."""
        return Q5(self.a, -self.b)

    def __repr__(self):
        return "(%s%+s*r5)" % (self.a, self.b)


PHI = Q5(F(1, 2), F(1, 2))        # golden ratio
IPHI = Q5(F(-1, 2), F(1, 2))      # 1/phi = phi - 1
HALF = Q5(F(1, 2))


def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)


def qconj(p):
    return (p[0], -p[1], -p[2], -p[3])


def binary_icosahedral():
    """The 120 unit quaternions of 2I: 24 Hurwitz units and 96 even permutations of (0, +-1, +-phi, +-1/phi)/2."""
    els = set()
    for i in range(4):
        for s in (1, -1):
            v = [Q5(0)] * 4
            v[i] = Q5(s)
            els.add(tuple(v))
    for signs in itertools.product((1, -1), repeat=4):
        els.add(tuple(Q5(F(s, 2)) for s in signs))
    even = [p for p in itertools.permutations(range(4))
            if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
    for s1, s2, s3 in itertools.product((1, -1), repeat=3):
        base = (Q5(0), HALF * s1, PHI * HALF * s2, IPHI * HALF * s3)
        for p in even:
            els.add(tuple(base[p[i]] for i in range(4)))
    els = sorted(els, key=lambda q: tuple((c.a, c.b) for c in q))
    assert len(els) == 120
    S = set(els)
    for p in els:                                   # closure under multiplication
        for q in els:
            assert qmul(p, q) in S
    return els


def cayley_graph():
    P = binary_icosahedral()
    minus_half = Q5(F(-1, 2))
    E = [(a, b) for a in range(120) for b in range(a + 1, 120) if qmul(P[a], qconj(P[b]))[0] == minus_half]
    return P, E


def triangles(n, E):
    adj = [set() for _ in range(n)]
    for a, b in E:
        adj[a].add(b)
        adj[b].add(a)
    T = sorted({tuple(sorted((a, b, c))) for a, b in E for c in adj[a] & adj[b]})
    return adj, T


def three_colourable(n, adj):
    """Exact backtracking search with domains, minimum-remaining-values ordering and forward checking.
    Returns a proper 3-colouring or None.  Complete: every branch is explored unless it is refuted."""
    adj = [sorted(a) for a in adj]
    dom = [7] * n
    col = [-1] * n

    def pick():
        best, bd = -1, 9
        for v in range(n):
            if col[v] < 0:
                d = bin(dom[v]).count("1")
                if d < bd or (d == bd and len(adj[v]) > len(adj[best])):
                    best, bd = v, d
        return best

    def rec(k):
        if k == n:
            return True
        v = pick()
        for c in range(3):
            if not (dom[v] >> c) & 1:
                continue
            changed, ok = [], True
            for w in adj[v]:
                if col[w] < 0 and (dom[w] >> c) & 1:
                    dom[w] &= ~(1 << c)
                    changed.append(w)
                    if dom[w] == 0:
                        ok = False
                        break
            if ok:
                col[v] = c
                if rec(k + 1):
                    return True
                col[v] = -1
            for w in changed:
                dom[w] |= 1 << c
        return False

    return list(col) if rec(0) else None


def rank_fraction(rows):
    """Exact rank over Q of a list of rows (lists of Fractions/ints)."""
    M = [[F(x) for x in r] for r in rows]
    rank, ncol = 0, len(M[0]) if M else 0
    for c in range(ncol):
        piv = next((r for r in range(rank, len(M)) if M[r][c] != 0), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][c]
        for r in range(len(M)):
            if r != rank and M[r][c] != 0:
                f = M[r][c] / pv
                M[r] = [x - f * y for x, y in zip(M[r], M[rank])]
        rank += 1
    return rank


def rank_mod(rows, l):
    M = [[x % l for x in r] for r in rows]
    rank, ncol = 0, len(M[0])
    for c in range(ncol):
        piv = next((r for r in range(rank, len(M)) if M[r][c]), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        inv = pow(M[rank][c], l - 2, l)
        M[rank] = [(x * inv) % l for x in M[rank]]
        for r in range(len(M)):
            if r != rank and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % l for x, y in zip(M[r], M[rank])]
        rank += 1
    return rank


def sqrt_mod(a, l):
    for x in range(l):
        if x * x % l == a % l:
            return x
    raise ValueError("no square root")


def report():
    P, E = cayley_graph()
    adj, T = triangles(120, E)
    out = {"vertices": 120, "edges": len(E), "triangles": len(T)}
    # every edge in exactly one triangle
    cnt = {}
    for t in T:
        for e in itertools.combinations(t, 2):
            cnt[e] = cnt.get(e, 0) + 1
    out["every_edge_in_exactly_one_triangle"] = all(cnt.get(tuple(sorted(e)), 0) == 1 for e in E)
    out["three_colourable"] = three_colourable(120, adj) is not None
    # triangle-sum-free space W
    A = []
    for t in T:
        row = [0] * 120
        for v in t:
            row[v] = 1
        A.append(row)
    rA = rank_fraction(A)
    out["dim_W"] = 120 - rA
    # the 8 coordinates of p -> (p, sigma(p)) lie in W and are independent
    coords = [[P[v][i] for v in range(120)] for i in range(4)] + [[P[v][i].conj() for v in range(120)] for i in range(4)]
    out["coordinates_in_W"] = all(sum((f[v] for v in t), Q5(0)) == Q5(0) for f in coords for t in T)
    # independence over R: split into rational and sqrt5 parts (independence over Q(sqrt5) <=> over R here
    # because we test the 8 x 120 matrix over Q(sqrt5) by rank modulo l, lower bound)
    l = 10009
    r5 = sqrt_mod(5, l)

    def red(x):
        return (int(x.a.numerator) * pow(int(x.a.denominator), l - 2, l) +
                int(x.b.numerator) * pow(int(x.b.denominator), l - 2, l) * r5) % l
    out["coordinate_rank_mod_l"] = rank_mod([[red(x) for x in f] for f in coords], l)
    # norm system: x_p^T B x_p = 1 for symmetric 8x8 B (36 unknowns)
    iu = [(i, j) for i in range(8) for j in range(i, 8)]
    rows = []
    for v in range(120):
        x = [P[v][i] for i in range(4)] + [P[v][i].conj() for i in range(4)]
        rows.append([red(x[i] * x[j] * (1 if i == j else 2)) for (i, j) in iu])
    out["norm_system_rank_mod_l"] = rank_mod(rows, l)
    # kernel vector I4 (+) -I4 (exact): x_p^T (I4 + -I4) x_p = |p|^2 - |sigma p|^2 = 0
    out["kernel_I4_minus_I4"] = all(
        sum((P[v][i] * P[v][i] for i in range(4)), Q5(0)) - sum((P[v][i].conj() * P[v][i].conj() for i in range(4)), Q5(0)) == Q5(0)
        for v in range(120))
    out["inclusion_is_strict_vector_colouring"] = all(
        sum((P[v][i] * P[v][i] for i in range(4)), Q5(0)) == Q5(1) for v in range(120))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1))
