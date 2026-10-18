"""The ring R = K[x_v : v in V]/(x_v^3 - 1), with K = Q or F_p, and the edge polynomials q_uv.

An element is a dict {e: c} with e a tuple in {0,1,2}^n (the reduced monomial x^e) and c an int or Fraction
(or an int mod p when `p` is given to the functions that take it).  Degrees are those of reduced
representatives (Lemma 3.2(a) of the paper).
"""
from fractions import Fraction

__all__ = ["var", "const", "add", "sub", "mul", "scale", "deg", "q_edge", "is_zero", "reduce_mod", "monomials",
           "grade"]


def const(n, c):
    return {(0,) * n: c} if c else {}


def var(n, v, power=1):
    e = [0] * n
    e[v] = power % 3
    return {tuple(e): 1}


def add(f, g, p=None):
    out = dict(f)
    for e, c in g.items():
        out[e] = out.get(e, 0) + c
        if p is not None:
            out[e] %= p
    return {e: c for e, c in out.items() if c}


def scale(f, a, p=None):
    if p is not None:
        return {e: (c * a) % p for e, c in f.items() if (c * a) % p}
    return {e: c * a for e, c in f.items() if c * a}


def sub(f, g, p=None):
    return add(f, scale(g, -1, p), p)


def mul(f, g, p=None):
    out = {}
    for e1, c1 in f.items():
        for e2, c2 in g.items():
            e = tuple((a + b) % 3 for a, b in zip(e1, e2))
            out[e] = out.get(e, 0) + c1 * c2
    if p is not None:
        out = {e: c % p for e, c in out.items()}
    return {e: c for e, c in out.items() if c}


def deg(f):
    return max((sum(e) for e in f), default=-1)


def grade(e):
    return sum(e) % 3


def is_zero(f, p=None):
    if p is None:
        return not any(f.values())
    return all(c % p == 0 for c in f.values())


def reduce_mod(f, p):
    return {e: c % p for e, c in f.items() if c % p}


def q_edge(n, u, v):
    """q_uv = x_u^2 + x_u x_v + x_v^2."""
    return add(add(var(n, u, 2), mul(var(n, u), var(n, v))), var(n, v, 2))


def monomials(n, maxdeg, grades=None):
    """Reduced monomials of degree <= maxdeg, optionally only those whose degree is in `grades` modulo 3."""
    out = []

    def rec(i, cur, s):
        if i == n:
            if grades is None or s % 3 in grades:
                out.append(tuple(cur))
            return
        for a in range(3):
            if s + a <= maxdeg:
                cur.append(a)
                rec(i + 1, cur, s + a)
                cur.pop()
    rec(0, [], 0)
    return out


def to_fraction(f):
    return {e: Fraction(c) for e, c in f.items()}
