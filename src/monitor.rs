use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::Result;
use rayon::prelude::*;
use thiserror::Error;

use crate::embed::Embedder;
use crate::index::Index;
use crate::search::{top_k, Scored};

#[derive(Debug, Error)]
pub enum MonitorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("embedding error: {0}")]
    Embed(String),
}

struct Record {
    path: String,
    _mtime: SystemTime,
    hash: blake3::Hash,
    embedding: Vec<f32>,
}

#[allow(dead_code)]
pub struct LiveIndex {
    inner: Arc<RwLock<Index>>,
    index_path: PathBuf,
    data_dir: PathBuf,
}

impl LiveIndex {
    pub fn new(data_dir: PathBuf, index_path: PathBuf, embedder: Arc<Embedder>) -> Result<Self> {
        let records = scan_dir(&data_dir, &embedder)?;

        let mut paths: Vec<String> = Vec::new();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for r in &records {
            paths.push(r.path.clone());
            vectors.push(r.embedding.clone());
        }
        let dim = vectors.first().map(|v| v.len() as u32).unwrap_or(384);
        Index::save(&index_path, dim, &vectors, &paths)?;
        let index = Index::open(&index_path)?;
        let inner = Arc::new(RwLock::new(index));

        let records = Arc::new(RwLock::new(records));
        let thr_inner = inner.clone();
        let thr_index = index_path.clone();
        let thr_dir = data_dir.clone();
        let thr_emb = embedder.clone();
        let thr_recs = records.clone();
        std::thread::spawn(move || {
            poll_loop(thr_dir, thr_index, thr_emb, thr_inner, thr_recs);
        });

        Ok(Self {
            inner,
            index_path,
            data_dir,
        })
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<Scored> {
        let index = self.inner.read().unwrap();
        top_k(&index, query, k)
    }

    pub fn get_index(&self) -> Arc<RwLock<Index>> {
        self.inner.clone()
    }
}

fn scan_dir(dir: &Path, embedder: &Embedder) -> Result<Vec<Record>> {
    let mut entries: Vec<PathBuf> = Vec::new();
    collect_files(dir, &mut entries)?;

    let records: Vec<Result<Record>> = entries
        .par_iter()
        .map(|p| {
            let meta = std::fs::metadata(p)?;
            let _mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let content = std::fs::read_to_string(p)?;
            let hash = blake3::hash(content.as_bytes());
            let text = content.trim().to_string();
            let embedding = if text.is_empty() {
                vec![0.0f32; 384]
            } else if text.len() > 10000 {
                embedder.embed_chunked(&text).map_err(|e| MonitorError::Embed(e.to_string()))?
            } else {
                embedder.embed(&text).map_err(|e| MonitorError::Embed(e.to_string()))?
            };
            Ok(Record {
                path: p.to_string_lossy().to_string(),
                _mtime,
                hash,
                embedding,
            })
        })
        .collect();

    records.into_iter().collect()
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with('.') || name.ends_with(".tmp") || name.ends_with(".swp") {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() && path.extension().map(|e| e == "txt").unwrap_or(false) {
            out.push(path);
        }
    }
    Ok(())
}

fn disk_snapshot(dir: &Path) -> Result<HashMap<String, blake3::Hash>> {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(dir, &mut files)?;
    let mut map = HashMap::new();
    for p in &files {
        let content = match std::fs::read_to_string(p) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let hash = blake3::hash(content.as_bytes());
        map.insert(p.to_string_lossy().to_string(), hash);
    }
    Ok(map)
}

fn poll_loop(
    data_dir: PathBuf,
    index_path: PathBuf,
    embedder: Arc<Embedder>,
    inner: Arc<RwLock<Index>>,
    records: Arc<RwLock<Vec<Record>>>,
) {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));

        let disk = match disk_snapshot(&data_dir) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("poll error: {e}");
                continue;
            }
        };

        let changed = {
            let recs = records.read().unwrap();
            if recs.len() != disk.len() {
                true
            } else {
                let mut diff = false;
                for (path, hash) in &disk {
                    if let Some(existing) = recs.iter().find(|r| r.path == *path) {
                        if existing.hash != *hash {
                            diff = true;
                            break;
                        }
                    } else {
                        diff = true;
                        break;
                    }
                }
                diff
            }
        };

        if changed {
            apply_changes(&disk, &embedder, &records, &index_path, &inner);
        }
    }
}

fn apply_changes(
    disk: &HashMap<String, blake3::Hash>,
    embedder: &Embedder,
    records: &Arc<RwLock<Vec<Record>>>,
    index_path: &Path,
    inner: &Arc<RwLock<Index>>,
) {
    let mut recs = records.write().unwrap();

    recs.retain(|r| disk.contains_key(&r.path));

    for (path, hash) in disk {
        if let Some(existing) = recs.iter_mut().find(|r| r.path == *path) {
            if existing.hash != *hash {
                if let Ok(content) = std::fs::read_to_string(path) {
                    let text = content.trim().to_string();
                    if let Ok(embedding) = if text.is_empty() {
                        Ok(vec![0.0f32; 384])
                    } else if text.len() > 10000 {
                        embedder.embed_chunked(&text)
                    } else {
                        embedder.embed(&text)
                    } {
                        existing.hash = *hash;
                        existing.embedding = embedding;
                    }
                }
            }
        } else {
            if let Ok(content) = std::fs::read_to_string(path) {
                let text = content.trim().to_string();
                let embedding = if text.is_empty() {
                    vec![0.0f32; 384]
                } else if text.len() > 10000 {
                    embedder.embed_chunked(&text).unwrap_or(vec![0.0f32; 384])
                } else {
                    embedder.embed(&text).unwrap_or(vec![0.0f32; 384])
                };
                let _mtime = SystemTime::UNIX_EPOCH;
                let hash = blake3::hash(content.as_bytes());
                recs.push(Record {
                    path: path.clone(),
                    _mtime,
                    hash,
                    embedding,
                });
            }
        }
    }
    drop(recs);

    let (paths, vectors) = {
        let recs = records.read().unwrap();
        let paths: Vec<String> = recs.iter().map(|r| r.path.clone()).collect();
        let vectors: Vec<Vec<f32>> = recs.iter().map(|r| r.embedding.clone()).collect();
        (paths, vectors)
    };

    let dim = vectors.first().map(|v| v.len() as u32).unwrap_or(384);
    let tmp = index_path.with_extension("tmp");
    if let Ok(()) = Index::save(&tmp, dim, &vectors, &paths) {
        let _ = std::fs::rename(&tmp, index_path);
        if let Ok(new_index) = Index::open(index_path) {
            *inner.write().unwrap() = new_index;
        }
    }
}
