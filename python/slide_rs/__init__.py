"""slide_rs: landmark-density sample embeddings (scSLIDE-style) with spatial and temporal extensions.

The heavy steps run in the Rust binary (`cargo build --release`); this module writes inputs as raw little-endian
binaries, calls the binary and reads the outputs. Set SLIDE_RS_BIN to point at the binary if it is not in
<repo>/target/release/slide-rs.

Core (after Satija lab scSLIDE: "Reconstructing developmental and disease progression with sample-level embeddings",
bioRxiv 2025.12.10.693462):
    landmarks(Z, n)                    leverage-score sketch of n landmark cells from a cell embedding
    density(mods, landmarks, groups)   weighted-NN cell -> landmark graph (k nearest, binarised), counts per group
    chisq(N)                           Pearson residuals of a Poisson contingency model, centred per landmark
    diffusion_map(M)                   sample-level diffusion components
Spatial:
    niche(xy, types, sections, radii)  per-cell neighbourhood composition, usable as an extra modality
    region_density(...)                densities counted per sample x region, chi-squared per region, concatenated
Temporal:
    stage_embedding(N, stages)         landmark counts pooled per stage / time point -> stage-level embedding
    paired_displacement(E, donor, t)   per-donor displacement between time points; leave-one-donor-out consistency
"""
import os
import subprocess
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("SLIDE_RS_BIN", os.path.join(_HERE, "..", "..", "target", "release", "slide-rs"))


def _run(*args):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stderr.strip()


def leverage(Z):
    """Leverage score of each row of the column-centred matrix Z (n x d)."""
    with tempfile.TemporaryDirectory() as t:
        np.ascontiguousarray(Z, np.float32).tofile(f"{t}/z.f32")
        _run("leverage", "--emb", f"{t}/z.f32", "--n", Z.shape[0], "--d", Z.shape[1], "--out", f"{t}/lev.f32")
        return np.fromfile(f"{t}/lev.f32", np.float32).astype(float)


def landmarks(Z, n=2000, seed=0):
    """Sample n landmark rows with probability proportional to their leverage score."""
    lev = np.clip(leverage(Z), 0, None)
    return np.sort(np.random.default_rng(seed).choice(len(lev), n, replace=False, p=lev / lev.sum()))


def niche(xy, types, sections, radii=(30, 100, 300), ntypes=None):
    """Per-cell share of each cell type among neighbours within each radius (same section, own cell excluded)."""
    types = np.asarray(types); ntypes = int(ntypes or types.max() + 1)
    with tempfile.TemporaryDirectory() as t:
        np.ascontiguousarray(xy, np.float32).tofile(f"{t}/xy.f32"); types.astype(np.uint32).tofile(f"{t}/t.u32")
        np.asarray(sections).astype(np.uint32).tofile(f"{t}/s.u32")
        _run("niche", "--xy", f"{t}/xy.f32", "--types", f"{t}/t.u32", "--ntypes", ntypes, "--sections", f"{t}/s.u32",
             "--radii", ",".join(map(str, radii)), "--out", f"{t}/n.f32")
        return np.fromfile(f"{t}/n.f32", np.float32).reshape(len(types), ntypes * len(radii))


def density(mods, landmark_idx, groups, ngroups=None, k=5, kbw=20):
    """mods: list of (embedding n x d, weight). Returns landmark x group counts."""
    groups = np.asarray(groups); ngroups = int(ngroups or groups.max() + 1)
    with tempfile.TemporaryDirectory() as t:
        groups.astype(np.uint32).tofile(f"{t}/g.u32"); np.asarray(landmark_idx).astype(np.uint32).tofile(f"{t}/l.u32")
        args = ["density", "--n", len(groups), "--groups", f"{t}/g.u32", "--ngroups", ngroups, "--landmarks", f"{t}/l.u32", "--k", k, "--kbw", kbw]
        for j, (E, w) in enumerate(mods):
            np.ascontiguousarray(E, np.float32).tofile(f"{t}/m{j}.f32"); args += ["--mod", f"{t}/m{j}.f32:{E.shape[1]}:{w}"]
        _run(*args, "--out", f"{t}/c.u32")
        return np.fromfile(f"{t}/c.u32", np.uint32).reshape(len(landmark_idx), ngroups)


def chisq(N):
    """Pearson residuals (landmark x sample) of a Poisson contingency model, centred per landmark."""
    N = np.asarray(N, float); E = np.outer(N.sum(1), N.sum(0)) / N.sum(); E[E == 0] = 1e-9
    R = (N - E) / np.sqrt(E); return R - R.mean(1, keepdims=True)


def diffusion_map(M, n=10, k=5):
    """M: samples x features. Adaptive Gaussian kernel, alpha = 1 normalisation; returns the first n components."""
    from scipy.spatial.distance import cdist
    D = cdist(M, M); s = np.sort(D, 1)[:, min(k, len(M) - 1)]; s[s == 0] = 1e-9
    K = np.exp(-D ** 2 / (s[:, None] * s[None, :])); q = K.sum(1); K = K / np.outer(q, q)
    d = K.sum(1); A = K / np.sqrt(np.outer(d, d)); w, v = np.linalg.eigh(A); o = np.argsort(-w); w, v = w[o], v[:, o]
    return (v / np.sqrt(d)[:, None])[:, 1:n + 1] * w[1:n + 1]


def sample_embedding(mods, groups, n_landmarks=2000, k=5, n_components=10, seed=0):
    """Full pipeline: landmarks on the standardised joint embedding -> densities -> chi-squared -> diffusion map.
    Returns (sample embedding, residual matrix samples x landmarks, landmark indices, counts)."""
    Z = np.hstack([E / np.where(E.std(0) == 0, 1, E.std(0)) for E, _ in mods])
    lm = landmarks(Z, n_landmarks, seed); N = density(mods, lm, groups, k=k); R = chisq(N).T
    return diffusion_map(R, n_components), R, lm, N


def region_density(mods, landmark_idx, groups, regions, k=5):
    """Densities counted per sample x region; chi-squared within each region; returns samples x (landmarks * regions)."""
    groups = np.asarray(groups); regions = np.asarray(regions); names = sorted(set(regions)); G = groups.max() + 1
    gi = groups * len(names) + np.searchsorted(names, regions)
    N = density(mods, landmark_idx, gi, G * len(names), k=k).reshape(len(landmark_idx), G, len(names))
    return np.hstack([chisq(N[:, :, r]).T for r in range(len(names))]), names


def stage_embedding(N, stages, n_components=3):
    """Pool landmark counts (landmark x sample) by stage label; return stage names and a PCA of their residuals."""
    stages = np.asarray(stages); names = list(dict.fromkeys(stages))
    S = np.column_stack([N[:, stages == s].sum(1) for s in names]); R = chisq(S).T
    U, s, _ = np.linalg.svd(R - R.mean(0), full_matrices=False)
    return names, U[:, :n_components] * s[:n_components]


def paired_displacement(E, donor, time):
    """For donors with two time points (time in {0, 1}), the displacement E[t1] - E[t0]; leave-one-donor-out projection
    of each donor's displacement on the mean of the others. Returns (projections, n positive, two-sided sign-test p)."""
    from scipy.stats import binomtest
    donor = np.asarray(donor); time = np.asarray(time); d = []
    for u in dict.fromkeys(donor):
        a, b = np.flatnonzero((donor == u) & (time == 0)), np.flatnonzero((donor == u) & (time == 1))
        if len(a) and len(b): d.append(E[b].mean(0) - E[a].mean(0))
    d = np.array(d); proj = np.array([d[i] @ np.delete(d, i, 0).mean(0) for i in range(len(d))])
    k = int((proj > 0).sum()); return proj, k, float(binomtest(k, len(proj)).pvalue)
