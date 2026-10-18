"""Fast tests for the `tricert` package (python3 -m pytest -q, about 10 s)."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tricert import (brute_force_count, check_certificate, component_colourable, compression_count,  # noqa: E402
                     floor_functional_check, ns_certificate_exists_f2, propagation_certificate, ring,
                     triangle_components, vieta_certificate)
from tricert import graphs, kneser  # noqa: E402


def test_vieta():
    s = {(1, 0, 0): 3, (0, 1, 0): 3, (0, 0, 1): 3}
    ok, d = check_certificate(3, vieta_certificate(3, 0, 1, 2), s)
    assert ok and d == 2


def test_component_counts():
    rng = random.Random(3)
    for _ in range(30):
        n = rng.randint(4, 8)
        tris = [tuple(rng.sample(range(n), 3)) for _ in range(rng.randint(1, 5))]
        E = sorted({tuple(sorted(p)) for t in tris for p in itertools.combinations(t, 2)})
        used = sorted({v for e in E for v in e})
        idx = {v: i for i, v in enumerate(used)}
        E = [(idx[a], idx[b]) for a, b in E]
        assert compression_count(len(used), E)[0] == brute_force_count(len(used), E)


def test_torus_colourability():
    for p, q in ((3, 3), (3, 6), (4, 4), (5, 7), (6, 9)):
        n, E = graphs.torus(p, q)
        (K,) = triangle_components(n, E)
        assert component_colourable(K) == (p % 3 == 0 and q % 3 == 0)


def test_propagation_certificates():
    for name in ("K4", "W5", "T(4,4)"):
        n, E = graphs.NAMED[name]
        c, cert = propagation_certificate(n, E)
        ok, d = check_certificate(n, cert, ring.const(n, c))
        assert ok and d <= 4


def test_floor_and_f2():
    n, E = graphs.complete(4)
    assert floor_functional_check(n, E)
    assert ns_certificate_exists_f2(n, E, 3)
    n, E = graphs.moser_spindle()
    assert not ns_certificate_exists_f2(n, E, 3) and ns_certificate_exists_f2(n, E, 6)


def test_strip():
    n, E = graphs.complete(4)
    n2, E2, _ = graphs.strip(n, E)
    assert n2 == 28 and len(E2) == 48
    assert len(triangle_components(n2, E2)) == 12


def test_kneser_small():
    for k in (1, 2, 3):
        r = kneser.report(k)[k]
        assert r["dim W"] == 3 * k - 1 and r["injective"] and r["standard colouring"]
