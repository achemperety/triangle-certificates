"""Strict vector 3-colourings of Kneser graphs KG(3k, k) (Theorem 6.3 of the paper).

For k <= 4 we check, with ranks modulo a large prime (lower bounds for the ranks over Q) matched by explicit
upper bounds, that
  (i)  the space W of functions f on k-sets with f(S) + f(T) + f(U) = 0 on every partition has dimension 3k-1;
  (ii) the linear map B -> (1_S^T B 1_S)_S on symmetric 3k x 3k matrices with zero row sums is injective;
and that v_S = sqrt(3/2k) (1_S - 1/3) is a strict vector 3-colouring (exactly, in rational arithmetic).
"""
import itertools
from fractions import Fraction

__all__ = ["partitions_into_three", "w_dimension", "gram_map_rank", "standard_colouring_ok", "report"]

PRIME = 2 ** 31 - 1


def _rank_mod(rows, p=PRIME):
    """Rank modulo p (a lower bound for the rank over Q), with numpy int64 arithmetic (p < 2^31)."""
    import numpy as np
    A = np.array(rows, dtype=np.int64) % p
    nr, nc = A.shape
    rank = 0
    for col in range(nc):
        if rank == nr:
            break
        nz = np.nonzero(A[rank:, col])[0]
        if len(nz) == 0:
            continue
        piv = rank + nz[0]
        if piv != rank:
            A[[rank, piv]] = A[[piv, rank]]
        inv = pow(int(A[rank, col]), p - 2, p)
        A[rank] = (A[rank] * inv) % p
        f = A[:, col].copy()
        f[rank] = 0
        nzr = np.nonzero(f)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - (f[nzr, None] * A[rank][None, :]) % p) % p
        rank += 1
    return rank


def partitions_into_three(N, k):
    """Partitions of range(N) (N = 3k) into three k-sets, as triples of k-sets (sorted tuples)."""
    out = []
    first = 0
    for A in itertools.combinations(range(N), k):
        if first not in A:
            continue
        rest = [x for x in range(N) if x not in A]
        for B in itertools.combinations(rest, k):
            if rest[0] not in B:
                continue
            C = tuple(x for x in rest if x not in B)
            out.append((A, B, C))
    return out


def w_dimension(k):
    N = 3 * k
    sets = list(itertools.combinations(range(N), k))
    idx = {s: i for i, s in enumerate(sets)}
    rows = []
    for A, B, C in partitions_into_three(N, k):
        r = [0] * len(sets)
        for s in (A, B, C):
            r[idx[s]] = 1
        rows.append(r)
    return len(sets) - _rank_mod(rows), len(sets), len(rows)


def gram_map_rank(k):
    """Rank of B -> (1_S^T B 1_S)_S on symmetric matrices with B 1 = 0, and the dimension of that space."""
    N = 3 * k
    # basis of symmetric matrices with zero row sums: E_ij = e_ie_j^T + e_je_i^T - e_ie_i^T - e_je_j^T, i < j
    basis = list(itertools.combinations(range(N), 2))
    sets = list(itertools.combinations(range(N), k))
    rows = []
    for S in sets:
        Sset = set(S)
        row = []
        for (i, j) in basis:
            # 1_S^T E_ij 1_S = 2[i in S][j in S] - [i in S] - [j in S]
            a, b = i in Sset, j in Sset
            row.append(2 * (a and b) - a - b)
        rows.append(row)
    return _rank_mod(rows), len(basis)


def standard_colouring_ok(k):
    """<v_S, v_T> = (3|S cap T| - k)/(2k): 1 on the diagonal, -1/2 on disjoint pairs; rank 3k - 1."""
    N = 3 * k
    sets = list(itertools.combinations(range(N), k))
    ok = True
    for S, T in itertools.combinations_with_replacement(sets, 2):
        inner = (Fraction(len(set(S) & set(T))) - Fraction(k, 3)) * Fraction(3, 2 * k)
        if S == T:
            ok &= inner == 1
        elif not set(S) & set(T):
            ok &= inner == Fraction(-1, 2)
    return ok


def report(kmax=4):
    out = {}
    for k in range(1, kmax + 1):
        dimW, nsets, nparts = w_dimension(k)
        r, dom = gram_map_rank(k)
        out[k] = {"k-sets": nsets, "partitions": nparts, "dim W": dimW, "expected 3k-1": 3 * k - 1,
                  "Gram map rank": r, "domain dimension": dom, "injective": r == dom,
                  "standard colouring": standard_colouring_ok(k)}
    return out
