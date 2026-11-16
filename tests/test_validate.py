"""Validate the Rust core against plain numpy / scipy on synthetic data.
    cargo build --release && python tests/test_validate.py
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import slide_rs as sr  # noqa: E402


def main():
    rng = np.random.default_rng(0); n, G, L = 20000, 60, 500
    E1 = rng.normal(size=(n, 30)).astype(np.float32); E2 = rng.normal(size=(n, 20)).astype(np.float32); g = rng.integers(0, G, n)
    lm = np.sort(rng.choice(n, L, replace=False))
    # leverage
    Z = np.hstack([E1, E2]); Q, _ = np.linalg.qr(Z.astype(float) - Z.mean(0)); ref = (Q ** 2).sum(1)
    lev_err = np.max(np.abs(sr.leverage(Z) - ref) / ref)
    # density
    aff = 0
    for E, w in ((E1, .5), (E2, .5)):
        d = cdist(E, E[lm]); bw = np.partition(d, 20, 1)[:, 20][:, None]; aff = aff + w * np.exp(-d / bw)
    nb = np.argpartition(-aff, 5, 1)[:, :5]; C = np.zeros((L, G)); np.add.at(C, (nb.ravel(), np.repeat(g, 5)), 1)
    same = (sr.density([(E1, .5), (E2, .5)], lm, g, G) == C).mean()
    # niche
    xy = rng.uniform(0, 2000, (n, 2)).astype(np.float32); ty = rng.integers(0, 12, n); sec = rng.integers(0, 4, n)
    ref = np.zeros((n, 24))
    for s in range(4):
        idx = np.flatnonzero(sec == s); tr = cKDTree(xy[idx])
        for r, rad in enumerate((30, 100)):
            for a, nbs in zip(idx, tr.query_ball_point(xy[idx], rad)):
                nbs = [idx[b] for b in nbs if idx[b] != a]
                if nbs: ref[a, r * 12:(r + 1) * 12] = np.bincount(ty[nbs], minlength=12) / len(nbs)
    niche_err = np.abs(sr.niche(xy, ty, sec, (30, 100), 12) - ref).max()
    print(f"leverage max rel err {lev_err:.1e}; density counts identical {same:.4f}; niche max abs err {niche_err:.1e}")
    assert lev_err < 1e-3 and same >= .995 and niche_err < 1e-5
    print("PASS")


if __name__ == "__main__":
    main()
