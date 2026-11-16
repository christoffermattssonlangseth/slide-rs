//! slide-rs: scSLIDE-style landmark-density sample embeddings with a spatial niche modality.
//!
//! All arrays are raw little-endian binaries (numpy `.tofile()`): f32 row-major matrices, u32 vectors.
//! Subcommands
//!   leverage  --emb E.f32 --n N --d D --out lev.f32
//!       Leverage score of each row of the column-centred matrix: h_i = z_i^T (Z^T Z)^-1 z_i.
//!   niche     --xy XY.f32 --types T.u32 --ntypes K --sections S.u32 --radii 30,100 --out niche.f32
//!       Per cell, the cell-type composition (shares, own cell excluded) within each radius, same section only.
//!       Output n x (K * len(radii)).
//!   density   --n N --groups G.u32 --ngroups M --landmarks L.u32 --k 5 --kbw 20
//!             --mod E1.f32:D1:W1 [--mod E2.f32:D2:W2 ...] --out counts.u32
//!       Weighted-NN cell -> landmark graph: per modality an exponential kernel exp(-d / bw_i) with bw_i the distance
//!       to the kbw-th nearest landmark; affinities mixed with fixed weights; the k landmarks with highest affinity
//!       are neighbours (binarised). Output landmark x group counts (L x M, row-major).
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;

fn read_f32(p: &str) -> Vec<f32> {
    let b = fs::read(p).expect(p);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u32(p: &str) -> Vec<u32> {
    let b = fs::read(p).expect(p);
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn write_f32(p: &str, v: &[f32]) {
    fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect(p)
}
fn write_u32(p: &str, v: &[u32]) {
    fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect(p)
}

struct Args(HashMap<String, Vec<String>>);
impl Args {
    fn parse() -> (String, Args) {
        let a: Vec<String> = std::env::args().collect();
        let mut m: HashMap<String, Vec<String>> = HashMap::new();
        let mut i = 2;
        while i + 1 < a.len() {
            m.entry(a[i].trim_start_matches("--").to_string()).or_default().push(a[i + 1].clone());
            i += 2;
        }
        (a.get(1).cloned().unwrap_or_default(), Args(m))
    }
    fn s(&self, k: &str) -> String { self.0.get(k).and_then(|v| v.first()).cloned().unwrap_or_else(|| panic!("--{k} missing")) }
    fn u(&self, k: &str) -> usize { self.s(k).parse().unwrap() }
    fn all(&self, k: &str) -> Vec<String> { self.0.get(k).cloned().unwrap_or_default() }
}

/// Cholesky solve helper: returns inverse of a small SPD matrix (d x d).
fn spd_inverse(a: &[f64], d: usize) -> Vec<f64> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j { s -= l[i * d + k] * l[j * d + k]; }
            if i == j { l[i * d + i] = s.max(1e-12).sqrt(); } else { l[i * d + j] = s / l[j * d + j]; }
        }
    }
    // inverse of L, then A^-1 = L^-T L^-1
    let mut li = vec![0.0; d * d];
    for i in 0..d {
        li[i * d + i] = 1.0 / l[i * d + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i { s -= l[i * d + k] * li[k * d + j]; }
            li[i * d + j] = s / l[i * d + i];
        }
    }
    let mut inv = vec![0.0; d * d];
    for i in 0..d { for j in 0..d { let mut s = 0.0; for k in i.max(j)..d { s += li[k * d + i] * li[k * d + j]; } inv[i * d + j] = s; } }
    inv
}

fn leverage(a: &Args) {
    let (n, d) = (a.u("n"), a.u("d"));
    let z = read_f32(&a.s("emb"));
    let mut mu = vec![0.0f64; d];
    for i in 0..n { for j in 0..d { mu[j] += z[i * d + j] as f64; } }
    mu.iter_mut().for_each(|m| *m /= n as f64);
    let g = (0..n).into_par_iter().fold(|| vec![0.0f64; d * d], |mut g, i| {
        for p in 0..d { let x = z[i * d + p] as f64 - mu[p]; for q in 0..d { g[p * d + q] += x * (z[i * d + q] as f64 - mu[q]); } }
        g
    }).reduce(|| vec![0.0f64; d * d], |mut a, b| { for (x, y) in a.iter_mut().zip(b) { *x += y; } a });
    let gi = spd_inverse(&g, d);
    let lev: Vec<f32> = (0..n).into_par_iter().map(|i| {
        let x: Vec<f64> = (0..d).map(|p| z[i * d + p] as f64 - mu[p]).collect();
        let mut s = 0.0;
        for p in 0..d { let mut t = 0.0; for q in 0..d { t += gi[p * d + q] * x[q]; } s += x[p] * t; }
        s as f32
    }).collect();
    write_f32(&a.s("out"), &lev);
    eprintln!("leverage: n {n}, d {d}, sum {:.3} (expected {d})", lev.iter().map(|&x| x as f64).sum::<f64>());
}

fn niche(a: &Args) {
    let xy = read_f32(&a.s("xy")); let ty = read_u32(&a.s("types")); let sec = read_u32(&a.s("sections"));
    let k = a.u("ntypes"); let n = ty.len();
    let radii: Vec<f32> = a.s("radii").split(',').map(|r| r.parse().unwrap()).collect();
    let rmax = radii.iter().cloned().fold(0.0, f32::max);
    // grid hash per section, cell size = rmax
    let mut grid: HashMap<(u32, i64, i64), Vec<u32>> = HashMap::new();
    for i in 0..n {
        let key = (sec[i], (xy[2 * i] / rmax).floor() as i64, (xy[2 * i + 1] / rmax).floor() as i64);
        grid.entry(key).or_default().push(i as u32);
    }
    let w = k * radii.len();
    let out: Vec<f32> = (0..n).into_par_iter().flat_map_iter(|i| {
        let (x, y, s) = (xy[2 * i], xy[2 * i + 1], sec[i]);
        let (gx, gy) = ((x / rmax).floor() as i64, (y / rmax).floor() as i64);
        let mut cnt = vec![0f32; w]; let mut tot = vec![0f32; radii.len()];
        for dx in -1..=1 { for dy in -1..=1 {
            if let Some(v) = grid.get(&(s, gx + dx, gy + dy)) { for &j in v {
                let j = j as usize; if j == i { continue; }
                let d = ((xy[2 * j] - x).powi(2) + (xy[2 * j + 1] - y).powi(2)).sqrt();
                for (r, &rad) in radii.iter().enumerate() { if d <= rad { cnt[r * k + ty[j] as usize] += 1.0; tot[r] += 1.0; } }
            }}
        }}
        for r in 0..radii.len() { if tot[r] > 0.0 { for t in 0..k { cnt[r * k + t] /= tot[r]; } } }
        cnt.into_iter()
    }).collect();
    write_f32(&a.s("out"), &out);
    eprintln!("niche: n {n}, types {k}, radii {:?}", radii);
}

fn density(a: &Args) {
    let n = a.u("n"); let groups = read_u32(&a.s("groups")); let m = a.u("ngroups");
    let lm = read_u32(&a.s("landmarks")); let nl = lm.len(); let k = a.u("k"); let kbw = a.u("kbw");
    let mods: Vec<(Vec<f32>, usize, f32)> = a.all("mod").iter().map(|s| {
        let p: Vec<&str> = s.split(':').collect(); (read_f32(p[0]), p[1].parse().unwrap(), p[2].parse().unwrap())
    }).collect();
    let nbrs: Vec<Vec<u32>> = (0..n).into_par_iter().map(|i| {
        let mut aff = vec![0f32; nl];
        for (e, d, w) in &mods {
            let ci = &e[i * d..(i + 1) * d];
            let dist: Vec<f32> = lm.iter().map(|&l| {
                let cl = &e[l as usize * d..(l as usize + 1) * d];
                ci.iter().zip(cl).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt()
            }).collect();
            let mut s = dist.clone(); let kk = kbw.min(nl - 1);
            s.select_nth_unstable_by(kk, |a, b| a.partial_cmp(b).unwrap());
            let bw = s[kk].max(1e-6);
            for (x, dd) in aff.iter_mut().zip(&dist) { *x += w * (-dd / bw).exp(); }
        }
        let mut idx: Vec<u32> = (0..nl as u32).collect();
        idx.select_nth_unstable_by(k, |&a, &b| aff[b as usize].partial_cmp(&aff[a as usize]).unwrap());
        idx[..k].to_vec()
    }).collect();
    let mut counts = vec![0u32; nl * m];
    for (i, nb) in nbrs.iter().enumerate() { for &l in nb { counts[l as usize * m + groups[i] as usize] += 1; } }
    write_u32(&a.s("out"), &counts);
    eprintln!("density: n {n}, landmarks {nl}, groups {m}, k {k}, modalities {}", mods.len());
}

fn main() {
    let (cmd, a) = Args::parse();
    match cmd.as_str() {
        "leverage" => leverage(&a),
        "niche" => niche(&a),
        "density" => density(&a),
        _ => eprintln!("usage: slide-rs leverage|niche|density ... (see source header)"),
    }
}
