# triangle-certificates

Companion code for

> R. Baiandin, *Triangles and certificates of non-3-colourability: Nullstellensatz degrees, operator refutations
> and vector colourings* (Graduate School of Economics and Management, Ural Federal University).

## Contents

- `tricert/`: a package written for this version of the paper. It checks the statements of Sections 2, 3 and 7 exactly:
  - triangle components and the compression theorem;
  - the strip;
  - the Vieta identity;
  - the explicit degree-6 certificates of Theorem 3.5, constructed by propagation and verified as identities in `Z[x]/(x_v^3 - 1)`;
  - the functional of Theorem 3.3;
  - exact Nullstellensatz degrees over F_2 (bitset Gaussian elimination);
  - Kneser graphs;
  - the binary icosahedral Cayley graph.
- `reference/earlier/`: the earlier programs. They compute the exact Nullstellensatz degrees over Q (modular screening, CRT and rational reconstruction, exact verification of certificates and dual solutions). They also cover the noncommutative certificates in the free algebra (game algebra, three idempotents, rigid triangles, the 4675-term certificate for W_5), the Cuntz word calculus, and the Clifford and Pauli computations of Section 6. They are kept unmodified, together with the logs of the runs used for the paper. These exploratory scripts cover more than this paper; their reports also contain sections that the paper does not use.

## Requirements

- Python 3.9 or later.
- `numpy`.
- `sympy`: Lemma 3.4 and Theorem 3.7.
- `networkx`: the icosahedron.
- `pytest`.

The reference programs need `numpy` and `sympy`.

```
pip install -r requirements.txt
```

## Reproducing the paper

| Command | Time | Statements |
|---|---|---|
| `python3 -m pytest -q tests` | 1 s | unit tests |
| `python3 scripts/verify_all.py` | 15 s | Lemma 2.1, Thm 2.2, Remark 2.3, Thm 2.4, Prop. 2.6, Thm 3.3, Lemma 3.4, Thm 3.5, Thm 3.7, Table 1 over F_2, Thm 7.3 (k <= 4), Thm 7.4 |
| `python3 scripts/verify_all.py --reference` | 4 min | additionally re-runs the four reference programs and compares their output with the stored logs (timings ignored): Table 1 over Q, Thm 4.1, Lemma 4.3, Props 4.4-4.6, Lemma 4.8, Thms 5.3, 5.4, 6.1, 6.2, the Peres rays |

Every line prints `PASS` or `FAIL` and names the statement it checks. The exit status is the number of failures. `logs/` holds the outputs of the runs used for the paper.

## Layout

```
tricert/
  ring.py          R = K[x]/(x^3 - 1) over Z, Q or F_p; edge polynomials q_uv
  graphs.py        K_m, wheels, Moser spindle, Groetzsch graph, icosahedron, tori T(p,q), strips, Kneser graphs
  components.py    triangles, triangle components, propagation, F_3 linear algebra, inclusion-exclusion
  certificates.py  Vieta certificates, propagation certificates, the degree-3 functional, NS degrees over F_2
  kneser.py        dim W and the Gram map for KG(3k, k)
  icosahedral.py   the binary icosahedral group over Q(sqrt5), its Cayley graph, colourability, ranks
scripts/verify_all.py
tests/
reference/earlier/ root3_collapse.py, hypercomplex_assault.py, quantum_braid_assault.py,
                   quantum_nonlocality_assault.py, and their .log files
logs/
```

## License

MIT (see `LICENSE`). `MANIFEST.md` lists SHA-256 checksums of all files.
