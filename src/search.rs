use rayon::prelude::*;
use wide::f32x8;

use crate::index::Index;

pub struct Scored {
    pub score: f32,
    pub path: String,
}

pub fn top_k(index: &Index, query: &[f32], k: usize) -> Vec<Scored> {
    let dim = index.dim() as usize;
    let n = index.len() as usize;
    if n == 0 || k == 0 || query.len() != dim {
        return Vec::new();
    }
    let k = k.min(n);

    let q_norm = l2_norm(query);
    if q_norm == 0.0 {
        return Vec::new();
    }

    let mut results: Vec<(f32, u32)> = Vec::with_capacity(n);
    results.par_extend((0..n).into_par_iter().map(|i| {
        let v = index.vector(i as u32);
        let (dot, norm) = dot_and_norm(query, v);
        let score = if norm == 0.0 { 0.0 } else { dot / (q_norm * norm) };
        (score, i as u32)
    }));

    results.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
    results[..k].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    results[..k]
        .iter()
        .map(|&(score, idx)| Scored {
            score,
            path: index.path(idx).to_string(),
        })
        .collect()
}

fn dot_and_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let mut i = 0;

    let mut dot_acc = f32x8::splat(0.0);
    let mut norm_acc = f32x8::splat(0.0);

    while i + 8 <= n {
        let va = f32x8::from([
            a[i], a[i + 1], a[i + 2], a[i + 3],
            a[i + 4], a[i + 5], a[i + 6], a[i + 7],
        ]);
        let vb = f32x8::from([
            b[i], b[i + 1], b[i + 2], b[i + 3],
            b[i + 4], b[i + 5], b[i + 6], b[i + 7],
        ]);
        dot_acc = va.mul_add(vb, dot_acc);
        norm_acc = vb.mul_add(vb, norm_acc);
        i += 8;
    }

    let mut dot = dot_acc.reduce_add();
    let mut norm = norm_acc.reduce_add();

    for j in i..n {
        dot += a[j] * b[j];
        norm += b[j] * b[j];
    }

    (dot, norm.sqrt())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
