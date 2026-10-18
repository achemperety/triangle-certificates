#!/usr/bin/env python3
"""Verify the computer-assisted statements of
    R. Baiandin, "Triangles and certificates of non-3-colourability: Nullstellensatz degrees, operator
    refutations and vector colourings".

    python3 scripts/verify_all.py               # Sections 2, 3 and 6 with the package tricert (about 1 minute)
    python3 scripts/verify_all.py --reference   # also re-runs the earlier reference programs (exact Nullstellensatz
                                                # degrees over Q, operator and Cuntz certificates, quaternionic
                                                # encodings; about 3 minutes) and compares their output with the
                                                # logs in reference/earlier, ignoring timings

Every line reports PASS or FAIL; the exit status is the number of failures.
"""
import argparse
import itertools
import os
import random
import re
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from tricert import (brute_force_count, check_certificate, component_colourable, compression_count,  # noqa: E402
                     every_edge_in_triangle, floor_functional_check, ns_certificate_exists_f2, ns_degree_f2,
                     propagate, propagation_certificate, ring, triangle_components, triangles, vieta_certificate)
from tricert import graphs, icosahedral, kneser  # noqa: E402

FAILS = []


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)


def section(t):
    print("\n== " + t)


def random_triangle_union(rng, nmin=5, nmax=10, tmin=2, tmax=8):
    n = rng.randint(nmin, nmax)
    tris = [tuple(rng.sample(range(n), 3)) for _ in range(rng.randint(tmin, tmax))]
    E = sorted({tuple(sorted(p)) for t in tris for p in itertools.combinations(t, 2)})
    used = sorted({v for e in E for v in e})
    idx = {v: i for i, v in enumerate(used)}
    return len(used), [(idx[a], idx[b]) for a, b in E]


def colourable(n, E):
    adj = [set() for _ in range(n)]
    for a, b in E:
        adj[a].add(b)
        adj[b].add(a)
    return icosahedral.three_colourable(n, adj) is not None


def sec2(rng):
    section("Section 2: triangle components")
    check("Lemma 2.1: a+b+c = 0 in F_3 iff a=b=c or {a,b,c} = F_3",
          all(((a + b + c) % 3 == 0) == (a == b == c or {a, b, c} == {0, 1, 2})
              for a, b, c in itertools.product(range(3), repeat=3)))
    ok_a = ok_c = True
    n_comp = 0
    for _ in range(80):
        n, E = random_triangle_union(rng)
        for K in triangle_components(n, E):
            n_comp += 1
            verts = sorted({x for t in K for x in t})
            if len(verts) > 8:
                continue
            idx = {v: i for i, v in enumerate(verts)}
            # colourings of K: injective on every triangle of K
            cnt = 0
            for c in itertools.product(range(3), repeat=len(verts)):
                if all(len({c[idx[x]] for x in t}) == 3 for t in K):
                    cnt += 1
            ok_a &= cnt in (0, 6)
            ok_c &= (cnt == 6) == component_colourable(K)
    check("Theorem 2.2(a),(b): |Col(K)| in {0, 6} for every triangle component (random unions of triangles)", ok_a,
          f"{n_comp} components")
    check("Theorem 2.2(c): propagation decides Col(K) != 0", ok_c)
    tor = all(component_colourable(triangle_components(*graphs.torus(p, q))[0]) == (p % 3 == 0 and q % 3 == 0)
              for p in range(3, 10) for q in range(3, 10))
    check("Remark 2.3: T(p,q) is 3-colourable iff 3 | p and 3 | q (3 <= p, q <= 9); a single component", tor and all(
        len(triangle_components(*graphs.torus(p, q))) == 1 for p in range(3, 8) for q in range(3, 8)))
    ok = True
    m = 0
    for _ in range(120):
        n, E = random_triangle_union(rng)
        P, k, dims = compression_count(n, E)
        ok &= P == brute_force_count(n, E) and all(d in (dims[-1], dims[-1] - 1) for d in dims[:-1])
        m += 1
    check("Theorem 2.4(a),(b): H_i = L or a hyperplane; inclusion-exclusion = brute force", ok, f"{m} random graphs")
    ok = True
    for _ in range(200):
        n, E = random_triangle_union(rng, tmax=6)
        P, k, dims = compression_count(n, E)
        if k <= 3:
            ok &= (P > 0) == all(d < dims[-1] for d in dims[:-1])
    check("Theorem 2.4(c): for k <= 3 components, colourable iff every H_i != L", ok)
    ok = True
    for name, (n, E) in (("K4", graphs.complete(4)), ("C5", graphs.cycle(5)), ("W5", graphs.wheel(5)),
                         ("K3", graphs.complete(3)), ("Petersen", graphs.petersen()), ("Moser", graphs.moser_spindle())):
        n2, E2, gad = graphs.strip(n, E)
        T = triangles(n2, E2)
        comps = triangle_components(n2, E2)
        cnt = {}
        for t in T:
            for p in itertools.combinations(t, 2):
                cnt[p] = cnt.get(p, 0) + 1
        ok &= colourable(n2, E2) == colourable(n, E)
        ok &= len(T) == 3 * len(E) and len(comps) == 2 * len(E) and every_edge_in_triangle(n2, E2)
        ok &= all(c in (1, 2) for c in cnt.values()) and all(len(K) <= 2 for K in comps)
    check("Proposition 2.6: the strip preserves 3-colourability; 3|E| triangles, 2|E| components of <= 2 triangles,"
          " every edge in 1 or 2 triangles", ok)


def sec3(rng):
    section("Section 3: Nullstellensatz certificates")
    for name, (n, E) in graphs.NAMED.items():
        check(f"Theorem 3.3: the degree-3 functional vanishes on every x_w q_e: {name}", floor_functional_check(n, E))
    # Lemma 3.4
    s = {(1, 0, 0): 3, (0, 1, 0): 3, (0, 0, 1): 3}
    ok, d = check_certificate(3, vieta_certificate(3, 0, 1, 2), s)
    check("Lemma 3.4: 3(x_u+x_v+x_w) = (2x_w^2+x_vx_w)(q_uw-q_uv) + (2x_w^2+x_ux_w)q_vw in R", ok and d == 2)
    try:
        import sympy as sp
        xu, xv, xw = sp.symbols("x_u x_v x_w")
        q = lambda a, b: a ** 2 + a * b + b ** 2  # noqa: E731
        G1 = sp.groebner([xu ** 3 - 1, xv ** 3 - 1, xw ** 3 - 1, q(xu, xv), q(xu, xw), q(xv, xw)], xu, xv, xw,
                         order="lex")
        G2 = sp.groebner([xu + xv + xw, xu * xv + xu * xw + xv * xw, xu * xv * xw - 1], xu, xv, xw, order="lex")
        check("Lemma 3.4: I_K3 = <e1, e2, e3 - 1>: equal reduced lex Groebner bases",
              list(G1.exprs) == list(G2.exprs), str(list(G1.exprs)))
    except ImportError:
        print("[SKIP] sympy not installed")
    # Theorem 3.5: explicit certificates
    targets = dict(graphs.NAMED)
    targets["icosahedron"] = graphs.icosahedron()
    for name, (n, E) in targets.items():
        pc = propagation_certificate(n, E)
        has_bad = any(not component_colourable(K) for K in triangle_components(n, E))
        if pc is None:
            check(f"Theorem 3.5: {name}: every triangle component colourable, no propagation certificate", not has_bad)
            continue
        c, cert = pc
        okq, d = check_certificate(n, cert, ring.const(n, c))
        ok7, _ = check_certificate(n, cert, ring.const(n, c), p=7)
        ok2, _ = check_certificate(n, cert, ring.const(n, c), p=2)
        check(f"Theorem 3.5: {name}: explicit certificate {c} = sum beta_e q_e in R over Z (hence over every field of "
              f"characteristic != 3), multipliers of degree <= 4, so NS <= 6", okq and ok7 and ok2 and d <= 4)
    ok = True
    m = 0
    for _ in range(300):
        n, E = random_triangle_union(rng, 6, 12, 4, 12)
        pc = propagation_certificate(n, E)
        if pc:
            m += 1
            ok &= check_certificate(n, pc[1], ring.const(n, pc[0]))[0] and check_certificate(
                n, pc[1], ring.const(n, pc[0]), p=7)[0]
    check("Theorem 3.5: explicit certificates on random unions of triangles with a non-colourable component", ok,
          f"{m} graphs")
    # Theorem 3.6: the strip map
    try:
        import sympy as sp
        w = sp.Symbol("w")
        xu, xv = sp.symbols("x_u x_v")
        red = lambda f: sp.rem(sp.expand(f), w ** 2 + w + 1, w)  # noqa: E731
        phi = {"u": xu, "v": xv, "p": w * xu, "q": w ** 2 * xu, "s": xu, "t": -xu - xv}
        q = lambda a, b: phi[a] ** 2 + phi[a] * phi[b] + phi[b] ** 2  # noqa: E731
        quv = xu ** 2 + xu * xv + xv ** 2
        ok = all(red(q(a, b)) == 0 for a, b in (("u", "p"), ("u", "q"), ("p", "q"), ("p", "s"), ("q", "s")))
        ok &= all(red(q(a, b) - quv) == 0 for a, b in (("s", "v"), ("s", "t"), ("v", "t")))
        ok &= all(red(phi[a] ** 3 - xu ** 3) == 0 for a in "pqs")
        ok &= red(phi["t"] ** 3 - 1 - (2 * (xu ** 3 - 1) - (xv ** 3 - 1) - 3 * xu * quv)) == 0
        check("Theorem 3.7: the strip map phi sends the generators of I_{G^nat} into I_G without raising degree", ok)
    except ImportError:
        pass
    # Table 1 over F_2 (exact) and the rows with a non-colourable component over Q
    expected_f2 = {"K4": 3, "W5": 3, "Moser spindle": 6, "Grotzsch": 3, "W7": 3, "icosahedron": 3, "T(4,4)": 6,
                   "T(5,7)": 6}
    for name, val in expected_f2.items():
        n, E = graphs.icosahedron() if name == "icosahedron" else graphs.NAMED[name]
        if val == 3:
            got = 3 if ns_certificate_exists_f2(n, E, 3) else None
        else:
            # NS_F2 >= 6 by the absence of a degree-3 certificate; <= 6 by a degree-6 one
            pc = propagation_certificate(n, E)
            up = (pc is not None and check_certificate(n, pc[1], ring.const(n, pc[0]), p=2)[0]) or \
                ns_certificate_exists_f2(n, E, 6)
            got = 6 if (not ns_certificate_exists_f2(n, E, 3) and up) else None
        check(f"Table 1: NS_F2({name}) = {val}", got == val, f"{got}")
    n, E = graphs.complete(3)
    check("Table 1: NS(K3 with target s_tau) = 4 (Lemma 3.4), by the grading the multipliers have grade 2",
          check_certificate(3, vieta_certificate(3, 0, 1, 2), s)[1] == 2)


def sec6():
    section("Section 7: strict vector colourings")
    rep = kneser.report(4)
    for k, r in rep.items():
        check(f"Theorem 7.3 (k = {k}): dim W = 3k-1 = {3 * k - 1}; B -> (1_S^T B 1_S)_S injective; the standard "
              f"colouring is strict", r["dim W"] == 3 * k - 1 and r["injective"] and r["standard colouring"],
              f"{r['k-sets']} k-sets, {r['partitions']} partitions")
    t = time.time()
    r = icosahedral.report()
    check("Theorem 7.4: 120 vertices, 1200 edges, 400 triangles, every edge in exactly one triangle",
          r["vertices"] == 120 and r["edges"] == 1200 and r["triangles"] == 400 and r["every_edge_in_exactly_one_triangle"])
    check("Theorem 7.4: not 3-colourable (exhaustive search with forward checking)", not r["three_colourable"])
    check("Theorem 7.4: dim W = 8, spanned by the coordinates of p -> (p, sigma(p))",
          r["dim_W"] == 8 and r["coordinates_in_W"] and r["coordinate_rank_mod_l"] == 8)
    check("Theorem 7.4: the norm system has rank 35 with kernel I4 + (-I4); the inclusion is a strict colouring",
          r["norm_system_rank_mod_l"] == 35 and r["kernel_I4_minus_I4"] and r["inclusion_is_strict_vector_colouring"],
          f"{time.time() - t:.1f} s")


def run_reference():
    section("Reference programs (reference/earlier): re-run and compare with the stored logs")
    ref = os.path.join(ROOT, "reference", "earlier")
    jobs = [("root3_collapse.py", [], "root3_collapse.log"),
            ("hypercomplex_assault.py", [], "hypercomplex_assault.log"),
            ("quantum_braid_assault.py", [], "quantum_braid_assault.log"),
            ("quantum_nonlocality_assault.py", [], "quantum_nonlocality_assault.log")]
    timing = re.compile(r"(\d+(\.\d+)?\s*s\b|time|Time|Total|\d+\.\d+s)")
    for script, args, log in jobs:
        t = time.time()
        r = subprocess.run([sys.executable, script] + args, cwd=ref, capture_output=True, text=True)
        new = [l for l in r.stdout.splitlines() if not timing.search(l)]
        old = [l for l in open(os.path.join(ref, log)).read().splitlines() if not timing.search(l)]
        check(f"{script}: output identical to {log} (timings ignored)", r.returncode == 0 and new == old,
              f"{time.time() - t:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args()
    rng = random.Random(2026)
    t0 = time.time()
    sec2(rng)
    sec3(rng)
    sec6()
    if a.reference:
        run_reference()
    print(f"\n{len(FAILS)} failure(s); total time {time.time() - t0:.1f} s")
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
