# slide-rs

Landmark-density **sample-level embeddings** for single-cell and spatial omics, in Rust with a thin Python wrapper.
It reimplements the core of **scSLIDE** and adds a spatial and a temporal layer:

- **Core (scSLIDE):** cells are embedded so both type and state are kept. A diverse set of landmark cells is sketched by
  leverage scores. Every cell is linked to its k nearest landmarks (weighted nearest neighbours over several embeddings).
  Each sample then becomes a density profile over landmarks, which is chi-squared normalised and reduced with a
  diffusion map. Distances and trajectories between samples come from that sample embedding.
- **Spatial:**
  - *Niche modality:* each cell's neighbourhood composition (cell-type shares within chosen radii) can be added as an
    extra embedding, so landmarks become "cell in context" types.
  - *Region-stratified densities:* counts per sample × region, normalised per region.
- **Temporal:**
  - *Stage-level embeddings:* landmark counts pooled per stage or time point.
  - *Paired displacements:* per-donor movement between two time points, with a leave-one-donor-out consistency test.

The method is from the Satija lab: *Reconstructing developmental and disease progression with sample-level
embeddings*, bioRxiv 2025.12.10.693462 ([vignette](https://satijalab.org/seurat/articles/scslide_vignette)). Please
cite the original work. This is an independent reimplementation, not affiliated with the authors, and it approximates
Seurat's weighted-NN graph with fixed modality weights and per-cell exponential kernels.

## Install

```bash
cargo build --release              # binary: target/release/slide-rs (only dependency: rayon)
python tests/test_validate.py      # checks the Rust output against numpy / scipy on synthetic data
```

From Python, put `python/` on the path, or set `SLIDE_RS_BIN` to the binary.

## Use

```python
import slide_rs as sr
# E_type: cells x d1 (e.g. PCA), E_state: cells x d2 (e.g. PLS on a phenotype), sample: int per cell
emb, R, lm, N = sr.sample_embedding([(E_type, .5), (E_state, .5)], sample, n_landmarks=2000, k=5)

# spatial: add the neighbourhood as a third modality
NI = sr.niche(xy, celltype, section, radii=(30, 100, 300))
emb_s, *_ = sr.sample_embedding([(E_type, 1/3), (E_state, 1/3), (NI / NI.std(0).clip(1e-9), 1/3)], sample)

# region-stratified densities, stage-level embedding, paired displacement
F, regions = sr.region_density([(E_type, .5), (E_state, .5)], lm, sample, region)
stages, S = sr.stage_embedding(N, stage_of_sample)
proj, n_pos, p = sr.paired_displacement(emb, donor_of_sample, timepoint_of_sample)
```

## Command line

```
slide-rs leverage --emb E.f32 --n N --d D --out lev.f32
slide-rs niche    --xy XY.f32 --types T.u32 --ntypes K --sections S.u32 --radii 30,100 --out niche.f32
slide-rs density  --n N --groups G.u32 --ngroups M --landmarks L.u32 --k 5 --kbw 20 \
                  --mod E1.f32:D1:W1 [--mod E2.f32:D2:W2 ...] --out counts.u32
```

Inputs and outputs are raw little-endian arrays (numpy `.tofile()`): f32 row-major matrices and u32 vectors.

## Validation

The synthetic check (20,000 cells, 2 modalities, 500 landmarks, 60 samples) gives:

| Output | Agreement with numpy / scipy |
|---|---|
| Leverage | Max relative error ~1e-7 |
| Landmark × sample counts | Identical in 100% of entries |
| Niche shares | Max error ~1e-8 |

## License

MIT
