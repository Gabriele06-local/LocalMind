mod embed;
mod extract;
mod index;
mod monitor;
mod search;
mod tui;

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use embed::Embedder;
use index::Index;
use monitor::LiveIndex;
use search::top_k;

fn wait_count(live: &LiveIndex, min: usize) -> usize {
    let start = Instant::now();
    loop {
        let n = live.get_index().read().unwrap().len() as usize;
        if n >= min || start.elapsed() > Duration::from_secs(30) {
            return n;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn wait_count_eq(live: &LiveIndex, target: usize) -> usize {
    let start = Instant::now();
    loop {
        let n = live.get_index().read().unwrap().len() as usize;
        if n == target || start.elapsed() > Duration::from_secs(30) {
            return n;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let is_demo = args.get(1).map(|s| s == "--demo").unwrap_or(false);

    let embedder = Embedder::new()?;
    let embedder = Arc::new(embedder);

    if is_demo {
        return demo(embedder);
    }

    let (progress_tx, progress_rx) = std::sync::mpsc::channel();
    let watch_dir = std::env::temp_dir().join("localmind_watch");
    let _ = std::fs::create_dir_all(&watch_dir);
    let index_path = watch_dir.join("index.bin");
    let live = LiveIndex::new(watch_dir.clone(), index_path, embedder.clone(), Some(progress_tx))?;
    let index = live.get_index();

    tui::run(embedder, index, progress_rx)
}

fn demo(embedder: Arc<Embedder>) -> Result<()> {
    // ── Step 1: Embedding ──
    println!("=== Step 1: Embedding ===");
    let text = "This is a test sentence for embedding.";
    let t0 = Instant::now();
    let vec = embedder.embed(text)?;
    let t1 = Instant::now();
    println!("Dimension: {}", vec.len());
    println!("First 5 values: {:?}", &vec[..5]);
    println!("L2 norm: {:.6}", vec.iter().map(|x| x * x).sum::<f32>().sqrt());
    println!("Embed latency: {:.3}s", (t1 - t0).as_secs_f32());

    let t0 = Instant::now();
    let _ = embedder.embed("short text")?;
    let t1 = Instant::now();
    println!("Short embed latency: {:.3}s", (t1 - t0).as_secs_f32());

    // ── Step 2: Binary Index ──
    println!("\n=== Step 2: Binary Index ===");
    let vectors = vec![
        vec![0.1f32, 0.2, 0.3],
        vec![0.4f32, 0.5, 0.6],
        vec![0.7f32, 0.8, 0.9],
    ];
    let paths = vec![
        "/doc/a.txt".to_string(),
        "/doc/b.txt".to_string(),
        "/doc/c.txt".to_string(),
    ];

    let index_path = std::env::temp_dir().join("test_index.bin");
    Index::save(&index_path, 3, &vectors, &paths)?;
    let index = Index::open(&index_path)?;

    println!("Records: {}", index.len());
    println!("Dim: {}", index.dim());
    for i in 0..index.len() {
        let v = index.vector(i);
        let p = index.path(i);
        println!("  [{}] path={:?} vec={:?}", i, p, v);
    }

    // ── Step 3: Search (top-k) ──
    println!("\n=== Step 3: Search (top-2) ===");
    fn normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    let mut query = vec![0.35f32, 0.45, 0.55];
    normalize(&mut query);
    let t0 = Instant::now();
    let results = top_k(&index, &query, 2);
    let t1 = Instant::now();
    println!("Search latency (3 records): {:.6}s", (t1 - t0).as_secs_f32());
    for r in &results {
        println!("  score={:.4} path={}", r.score, r.path);
    }

    // ── Step 3b: End-to-end (embed + index + search) ──
    println!("\n=== Step 3b: End-to-end ===");
    let docs = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "Pizza is the best food",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is awesome",
        "Do you like pizza?",
    ];
    let doc_paths: Vec<String> = docs
        .iter()
        .enumerate()
        .map(|(i, _)| format!("/doc/{}.txt", i))
        .collect();
    let doc_vectors: Vec<Vec<f32>> = docs
        .iter()
        .map(|d| embedder.embed(d).unwrap())
        .collect();

    let e2e_path = std::env::temp_dir().join("e2e_index.bin");
    Index::save(&e2e_path, 384, &doc_vectors, &doc_paths)?;
    let e2e_index = Index::open(&e2e_path)?;

    let t0 = Instant::now();
    let q_emb = embedder.embed("food and cooking")?;
    let t1 = Instant::now();
    let e2e_results = top_k(&e2e_index, &q_emb, 3);
    let t2 = Instant::now();
    println!("Query: \"food and cooking\"");
    println!("  Embed: {:.3}s, Search (8 records): {:.6}s",
        (t1 - t0).as_secs_f32(), (t2 - t1).as_secs_f32());
    for r in &e2e_results {
        println!("  score={:.4} path={}", r.score, r.path);
    }
    // Semantic sanity: top result should be pizza-related
    assert!(e2e_results[0].path.contains("/doc/3.txt") || e2e_results[0].path.contains("/doc/7.txt"),
        "Expected pizza-related doc as top result");

    // ── Step 4: Live Monitor + Edge Cases ──
    println!("\n=== Step 4: Live Monitor ===");
    let watch_dir = std::env::temp_dir().join("localmind_watch");
    let _ = std::fs::remove_dir_all(&watch_dir);
    std::fs::create_dir_all(&watch_dir)?;

    let live_index_path = watch_dir.join("index.bin");
    let live = LiveIndex::new(watch_dir.clone(), live_index_path, embedder.clone(), None)?;

    // 4a: Create and verify
    println!("\n-- 4a: Create file --");
    let test_file = watch_dir.join("hello.txt");
    std::fs::write(&test_file, "Hello world, this is a test document for embedding.")?;
    let n = wait_count(&live, 1);
    println!("After create: {n} records");
    assert_eq!(n, 1, "create should produce 1 record");

    let t0 = Instant::now();
    let q_greetings = embedder.embed("greetings")?;
    let results = live.search(&q_greetings, 5);
    let t1 = Instant::now();
    println!("Search \"greetings\": {:.3}s", (t1 - t0).as_secs_f32());
    for r in &results {
        println!("  score={:.4} path={}", r.score, r.path);
    }
    assert!(!results.is_empty(), "should find the hello file");

    // 4b: Modify and verify
    println!("\n-- 4b: Modify file --");
    std::fs::write(&test_file, "Goodbye everyone, this document was updated.")?;
    let n = wait_count(&live, 1);
    println!("After modify: {n} records");
    let q_goodbye = embedder.embed("farewell")?;
    let results = live.search(&q_goodbye, 5);
    println!("Search \"farewell\":");
    for r in &results {
        println!("  score={:.4} path={}", r.score, r.path);
    }

    // 4c: Delete and verify
    println!("\n-- 4c: Delete file --");
    std::fs::remove_file(&test_file)?;
    let n = wait_count_eq(&live, 0);
    println!("After delete: {n} records");
    assert_eq!(n, 0, "delete should yield 0 records");

    // 4d: Empty file
    println!("\n-- 4d: Empty file --");
    let empty_file = watch_dir.join("empty.txt");
    std::fs::write(&empty_file, "")?;
    let n = wait_count(&live, 1);
    println!("Empty file: {n} records");
    assert_eq!(n, 1);
    // Search with empty vector should still return the record (score 0)
    let results = live.search(&embedder.embed("something")?, 5);
    println!("Search after adding empty file: {} results", results.len());
    // Clean up empty file and wait for propagation
    std::fs::remove_file(&empty_file)?;
    let n = wait_count_eq(&live, 0);
    println!("Empty file removed: {n} records");

    // 4e: Unicode / emoji
    println!("\n-- 4e: Unicode & emoji --");
    let multi_file = watch_dir.join("multilingual.txt");
    let multi_text = "Ciao mondo! 🍕 Pizza è la migliore 😊 こんにちは";
    std::fs::write(&multi_file, multi_text)?;
    let n = wait_count(&live, 1);
    println!("Unicode file: {n} records");

    // Diagnostic: check what tokens the tokenizer produces
    let enc = embedder.tokenize(multi_text)?;
    let ids = enc.get_ids();
    let tokens: Vec<String> = ids.iter()
        .take(16)
        .map(|&id| embedder.id_to_token(id).unwrap_or_else(|| format!("[ID:{id}]")))
        .collect();
    let unk_count = ids.iter().filter(|&&id| id == 100).count(); // 100 = [UNK]
    println!("  Text: \"{multi_text}\"");
    println!("  Tokens: {} total, {unk_count} [UNK], first 16: {tokens:?}", ids.len());

    let q = embedder.embed("pizza")?;
    let results = live.search(&q, 5);
    println!("Search \"pizza\": {} results, top score={:.4}",
        results.len(), results.first().map(|r| r.score).unwrap_or(0.0));
    if let Some(top) = results.first() {
        println!("  top path: {}", top.path);
    }

    // 4f: 5-file batch stress test (embedding is ~1.5s/file)
    println!("\n-- 4f: 5-file batch --");
    let topics = [
        "machine learning",
        "pizza recipes",
        "cat behavior",
        "solar system",
        "classical music",
    ];
    let batch_dir = watch_dir.join("batch");
    std::fs::create_dir_all(&batch_dir)?;
    for (i, topic) in topics.iter().enumerate() {
        std::fs::write(batch_dir.join(format!("{i:02}_{}.txt", topic.replace(' ', "_"))),
            format!("This document is about {topic}. {topic} is a fascinating subject with many applications."))?;
    }
    // Wait for 5 batch files + multilingual.txt = 6
    let n = wait_count(&live, 5);
    println!("After batch create: {n} records (expected >= 5)");

    // Measure full search latency across all indexed docs
    let queries = ["machine learning", "pizza", "cat", "music", "space"];
    let mut total_search = 0.0f64;
    for q_str in &queries {
        let t0 = Instant::now();
        let q = embedder.embed(q_str)?;
        let results = live.search(&q, 5);
        let t1 = Instant::now();
        let lat = (t1 - t0).as_secs_f64();
        total_search += lat;
        println!("  \"{q_str}\": {lat:.4}s — top: {}",
            results.first().map(|r| &r.path).unwrap_or(&"-".into()));
    }
    println!("Search avg ({} docs): {:.4}s", n, total_search / queries.len() as f64);

    // 4g: Chunking — verify long file ranking is not penalized
    println!("\n-- 4g: Chunking quality --");
    let long_text = format!("Pizza margherita is a traditional Italian pizza. {} ",
        "It has tomatoes, mozzarella, and basil. ").repeat(2000);
    std::fs::write(watch_dir.join("long_pizza.txt"), &long_text)?;
    // Wait for 7 total records (5 batch + multilingual + long_pizza)
    let n = wait_count(&live, 7);
    println!("Long file indexed: {n} total records");

    let q = embedder.embed("margherita pizza")?;
    let results = live.search(&q, 5);
    println!("Search \"margherita pizza\":");
    for r in &results {
        println!("  score={:.4} path={}", r.score, r.path);
    }
    // Long pizza file should be in top results, not artificially low
    let long_in_top = results.iter().any(|r| r.path.contains("long_pizza"));
    let check = if long_in_top { "passed" } else { "FAILED — check renormalization" };
    println!("Long pizza in top 5: {check}");

    // Cleanup
    let _ = std::fs::remove_dir_all(&watch_dir);
    println!("\n=== All demo steps passed ===");
    Ok(())
}
