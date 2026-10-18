"""Named graphs of the paper.  A graph is a pair (n, edges) with vertices 0..n-1."""
import itertools

__all__ = ["complete", "cycle", "wheel", "moser_spindle", "grotzsch", "icosahedron", "torus", "strip", "kneser",
           "petersen", "NAMED"]


def complete(m):
    return m, list(itertools.combinations(range(m), 2))


def cycle(m):
    return m, [(i, (i + 1) % m) if i < (i + 1) % m else ((i + 1) % m, i) for i in range(m)]


def wheel(m):
    """Hub 0 joined to the cycle 1..m."""
    return m + 1, [(0, i) for i in range(1, m + 1)] + [tuple(sorted((i, i % m + 1))) for i in range(1, m + 1)]


def moser_spindle():
    return 7, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (0, 4), (0, 5), (4, 5), (4, 6), (5, 6), (3, 6)]


def grotzsch():
    E = [tuple(sorted((i, (i + 1) % 5))) for i in range(5)]
    E += [tuple(sorted((5 + i, (i + 1) % 5))) for i in range(5)] + [tuple(sorted((5 + i, (i - 1) % 5))) for i in range(5)]
    E += [(5 + i, 10) for i in range(5)]
    return 11, sorted(E)


def icosahedron():
    import networkx as nx
    G = nx.icosahedral_graph()
    return 12, sorted(tuple(sorted(e)) for e in G.edges())


def torus(p, q):
    """The triangular lattice modulo p Z x q Z: (i,j) ~ (i+1,j), (i,j+1), (i+1,j+1); p, q >= 3."""
    idx = lambda i, j: (i % p) * q + (j % q)  # noqa: E731
    E = set()
    for i in range(p):
        for j in range(q):
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                a, b = idx(i, j), idx(i + di, j + dj)
                E.add((min(a, b), max(a, b)))
    return p * q, sorted(E)


def strip(n, edges):
    """The strip G^nat of Definition 2.5: for u -> v (u < v) new vertices p, q, s, t and the triangles
    {u,p,q}, {p,q,s}, {s,v,t}.  Returns (n', edges', gadgets) with gadgets[k] = (u, v, p, q, s, t)."""
    E, gadgets, m = [], [], n
    for (u, v) in edges:
        p, q, s, t = m, m + 1, m + 2, m + 3
        m += 4
        gadgets.append((u, v, p, q, s, t))
        for a, b in ((u, p), (u, q), (p, q), (p, s), (q, s), (s, v), (s, t), (v, t)):
            E.append((min(a, b), max(a, b)))
    return m, sorted(E), gadgets


def kneser(N, k):
    sets = list(itertools.combinations(range(N), k))
    E = [(i, j) for i, j in itertools.combinations(range(len(sets)), 2) if not set(sets[i]) & set(sets[j])]
    return len(sets), E, sets


def petersen():
    n, E, _ = kneser(5, 2)
    return n, E


NAMED = {
    "K3": complete(3), "K4": complete(4), "W5": wheel(5), "W7": wheel(7), "Moser spindle": moser_spindle(),
    "Grotzsch": grotzsch(), "T(4,4)": torus(4, 4), "T(5,7)": torus(5, 7), "T(3,6)": torus(3, 6),
}
