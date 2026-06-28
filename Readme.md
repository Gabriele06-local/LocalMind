# LocalMind

A **local semantic search engine** written in Rust. Indexes `.txt`, `.pdf`, and `.docx` files, embeds them into vectors via a BERT model (all-MiniLM-L6-v2), and provides real-time semantic search through a terminal UI.

No external database, no cloud service, no API key. Everything runs locally on CPU.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  embed   │ -> │  index   │ -> │  search  │ <- │   monitor    │
│ (BERT)   │    │ (binary) │    │ (SIMD)   │    │ (polling)    │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
                                                    │
                                                    v
                                              ┌──────────┐
                                              │   tui    │
                                              │ (ratatui)│
                                              └──────────┘
```

## Screenshots

![Empty TUI with search bar](img/tui_empty.png)
*Empty TUI — write a query and press Enter (or wait for automatic search)*

![Searching in progress](img/tui_searching.png)
*Query submitted, embedding in progress (~55–65ms on CPU, SIMD)*

![Results](img/tui_results.png)
*Results with similarity score, keyboard navigation via arrows*

## Design Principles

### No external databases

Every "semantic search" demo on the internet assumes you run a separate database — Pinecone, Qdrant, Weaviate, or pgvector. This adds operational complexity and often recurring costs. LocalMind stores its vectors in a **custom binary format** designed for memory-mapped O(1) access (detailed below). The entire index is a single file you can `cp`, `rsync`, or commit to a repo.

This choice comes with trade-offs: you don't get built-in replication, sharding, or incremental compaction. For a local tool indexing a few thousand files, those features are irrelevant. For millions of vectors, an approximate index (HNSW) is the right answer — but even that should be a local file, not a network service.

### The index file format

Most vector indexes store metadata alongside vectors in a database table, which means every search query goes through a serialization boundary (SQL or protobuf). LocalMind's binary format is designed around a simple observation: **a search reads every vector but only needs a few paths**.

```
┌─────────────────────────────────────────────────┐
│ Header: [u8;4] magic + u32 ver + u32 n + u32 d │  16 bytes
├─────────────────────────────────────────────────┤
│ vectors: [f32; d] × n, 8-byte aligned           │  384 × 4 × n bytes
├─────────────────────────────────────────────────┤
│ entries: [{offset: u64, len: u32}] × n          │  12 × n bytes
├─────────────────────────────────────────────────┤
│ strings: concatenated path bytes                │  variable
└─────────────────────────────────────────────────┘
```

- **Header**: magic `LMND` (0x4C4D4E44), format version, record count `n`, dimension `d` (always 384).
- **Vectors**: flat array of `n × d` floats, padded to 8-byte alignment. Each vector sits at a known offset — `header_size + i * d * 4` — so reading vector `i` is a single pointer dereference through the memory map.
- **Entry table**: fixed-size records of `offset: u64 + length: u32` pointing into the string block. Since every entry is the same size (12 bytes), locating entry `i` is O(1): `entry_table_offset + i * 12`.
- **Strings**: concatenated path strings with no separator — each entry's `(offset, len)` tells you exactly where its path lives.

The whole file is opened with `memmap2`, so the operating system manages paging. A search touches every vector (sequential read, prefetcher-friendly) but only reads `k` paths for the final results. You never deserialize the entire index into heap memory.

### Polling instead of filesystem events

The `notify` crate on Windows produces unreliable `is_file` events — rapid writes (common with editors) trigger spurious creation/deletion cycles. A polling loop with a 1-second interval and `blake3` hash comparison is simpler to reason about: it compares the *actual content* of every file at a fixed cadence, with zero false positives.

Polling burns CPU when nothing changes (~1% of a core on an SSD directory scan). On Linux/macOS, an `inotify`/`FSEvents` backend would be more efficient — but the polling loop is correct on every platform, and correctness beats efficiency when you're indexing someone's documents.

### SIMD by hand instead of BLAS

Most cosine similarity implementations call `dot` from a BLAS library — fine for large matrices, but for a hot loop that compares 384-dimensional vectors, you spend more time on function call overhead and bounds checks than on actual math. LocalMind uses the [`wide`](https://crates.io/crates/wide) crate to do SIMD directly:

```rust
// Distilled from search.rs — dot product on 8 f32s at once
let a8 = f32x8::from(a);
let b8 = f32x8::from(b);
acc = f32x8::mul_add(a8, b8, acc);  // FMA: a8 * b8 + acc
```

This avoids a BLAS dependency (saving binary size), makes the code obviously correct, and compiles down to a single `VFMADD231PS` instruction on any CPU with FMA support. The remainder (< 8 dimensions) falls back to a scalar loop. No SIMD dispatch, no runtime detection — `wide` generates the best instruction set available at compile time.

### Thread safety as architecture

Index updates happen in a polling thread (or any background thread in the future). Searches happen in the TUI thread. The shared state is `Arc<RwLock<Index>>`, where `Index` is the memory-mapped file + a few bookkeeping fields.

The update protocol:
1. Build the new index to a temporary file.
2. Memory-map the temp file.
3. Acquire a **write** lock on the `RwLock`.
4. Swap the `Arc` pointer (clone of the old `Arc` is cheap).
5. Drop the write lock.

Searches hold only a **read** lock, which never blocks another reader. The write lock is held for less than a microsecond — just the pointer swap and drop. This means embedding 20 files (taking seconds) does not pause the TUI or block ongoing searches. The old index continues to serve queries until the new one is ready, then transitions atomically.

### Binary size is a feature

A release binary of ~8 MB means you can `scp` it to a server, include it in a Docker image, or distribute it as a single file. This is achieved with three Cargo profile settings:

```toml
[profile.release]
lto = true
codegen-units = 1
strip = true
```

`lto = true` enables cross-crate inlining — the SIMD hot path gets inlined and specialized. `codegen-units = 1` lets LLVM see the entire crate at once (better optimization at the cost of compile time). `strip = true` removes debug symbols. The result is a self-contained binary that includes the BERT model weights loader, the tokenizer, a TUI framework, and the entire search pipeline in ~8 MB.

### Why MiniLM-L6-v2

The all-MiniLM-L6-v2 model produces 384-dimensional embeddings, compared to 768 for BERT-base or 1024 for BERT-large. This is a deliberate space/performance trade-off:
- **Memory**: 384 floats per vector × 4 bytes × 10,000 documents = ~15 MB for the vector block.
- **Search speed**: the SIMD dot product processes 48 iterations (384/8) per vector; smaller dimensions mean fewer iterations.
- **Quality retention**: MiniLM retains ~95% of BERT-base's semantic quality on STS benchmarks, despite being 50% smaller and 2× faster at inference.

## Technical Deep-Dive

### Hybrid search (BM25 + RRF)

Hybrid search combines the exact-term precision of BM25 with the semantic power of vector embeddings. The two subsystems are independent and are fused at query time using **Reciprocal Rank Fusion (RRF)**:

```
RRF(doc) = 1 / (60 + rank_vec(doc))  +  1 / (60 + rank_bm25(doc))
```

RRF works purely on ordinal rank, not raw scores — so there is no normalization problem, no parameter to retune when the collection changes, and either subsystem can be absent (if BM25 finds no matches, its contribution is zero). A document that ranks #1 in both systems scores highest; a document that BM25 doesn't match at all still appears if its vector score is strong enough.

Two separate tokenizers live alongside each other:
- **Vector search**: WordPiece (BERT's subword tokenizer). "Running", "runs", and "ran" all map to nearby embeddings.
- **BM25**: whitespace + punctuation split, lowercased. `RSSMRA85M01H501U` stays intact as a single token — no subword fragmentation.

The BM25 index is an inverted file: `HashMap<String, Vec<(u32, u32)>>` mapping each whitespace token to `(doc_id, term_frequency)` pairs, with per-document total term counts for BM25 length normalization (k1=1.2, b=0.75). The index is rebuilt from scratch whenever a file changes — IDF shifts globally, and incremental updates are too error-prone to justify the complexity.

### The binary index format — O(1) access without deserialization

Every "embedding file" in LocalMind is a single file with four contiguous regions:

```
 Offset  │ Content
─────────┼──────────────────────────────────────────────
       0  │ Magic "LMND" (4 bytes) + version (u32)
          │ + num_records (u32) + dimension (u32)      = 16 bytes
─────────┼──────────────────────────────────────────────
      16  │ Vectors: [f32; dim] × num_records
          │ 8-byte aligned, packed contiguously
          │ dim = 384, so each vector = 1536 bytes
─────────┼──────────────────────────────────────────────
 16 + V  │ Entry table: {offset: u64, len: u32} × n
          │ 12 bytes per entry, fixed stride
─────────┼──────────────────────────────────────────────
 16+ V+E │ String block: concatenated path bytes
          │ No separator — each entry's (offset, len)
          │ tells you exactly where its path lives
```

The file is opened with `memmap2`, so the OS pages it in on demand:

```rust
// Reading vector i — one pointer dereference
let vec_offset = 16 + i * dim * 4;
let vector: &[f32] = bytemuck::cast_slice(
    &mmap[vec_offset..vec_offset + dim * 4]
);

// Reading path i — through the entry table
let entry_offset = 16 + vec_block_size + i * 12;
let path_offset = u64::from_ne_bytes(...);
let path_len = u32::from_ne_bytes(...);
let path = &mmap[path_offset..path_offset + path_len];
```

No deserialization, no heap allocation, no `serde`. Every access is a pointer into the memory map. A search reads every vector (sequential scan, prefetcher-friendly) but only resolves paths for the top-k results.

The `Index` struct held by `Arc<RwLock<Index>>` is just `(memmap, len, dim)` — three fields, trivially cloneable via `Arc`. Swapping the entire index (during live re-indexing) means building a new file, memory-mapping it, and swapping one `Arc` pointer under a write lock. The old index continues serving queries from its memory map until the last `Arc` reference drops, at which point the OS unmaps it.

### Candle and CPU embedding — no GPU, no Python

LocalMind runs BERT inference on CPU using [`candle`](https://github.com/huggingface/candle), HuggingFace's Rust framework. The decision to avoid Python (and GPU) is deliberate:

- **Zero runtime dependencies**: no CUDA toolkit, no Python interpreter, no `libomp`. The binary is self-contained.
- **Deterministic performance**: GPU inference has variable latency due to driver scheduling and VRAM contention. CPU inference at ~55ms is predictable and good enough for interactive search.
- **Small model, big effect**: all-MiniLM-L6-v2 is 6 transformer layers × 384 hidden dims — ~22M parameters, ~90MB in safetensors format. Candle loads the weights directly with no conversion step.

The actual matrix multiplication is handled by the [`gemm`](https://crates.io/crates/gemm) crate, which candle delegates to at build time. `gemm` detects the CPU's SIMD capabilities (AVX-512, AVX2, SSE) and selects the optimal kernel at compile time. On an Intel i7-12700H, the attention layers run at AVX2+FMA, producing an embedding in ~55–65ms.

The tokenizer runs separately via the `tokenizers` crate (Rust bindings over the HuggingFace tokenizer Rust implementation in `tokenizers` 0.23). WordPiece encoding adds ~2-5ms per query for short text, dominated by the transformer forward pass.

### Syntax-highlighted preview — instant file exploration

When the user navigates results with arrow keys, the selected file is loaded, run through [`syntect`](https://crates.io/crates/syntect) with language detection by extension, and rendered in a right-side panel with syntax coloring. Query terms are highlighted in bold/yellow. The highlighted lines are cached in a `PreviewCache` struct — re-highlighting only happens when the selection changes, not on every frame.

## Benchmarks (CPU, no hardware acceleration)

| Operation | Latency |
|---|---|
| Query embedding (BERT forward pass) | ~55–65ms |
| Index search (8 documents) | ~0.05–0.13ms |
| Initial model download | ~90MB (one-time) |
| Release binary | ~8 MB |
| Monitor polling interval | 1 scan/s |

## Known limitations

- **Brute-force search**: O(n) in the number of documents — parallelized but not approximated. Fine for tens of thousands of local files; for millions you'd want an approximate index (e.g., HNSW).
- **English-centric tokenizer**: MiniLM handles Italian correctly via subword tokenization (e.g., "migliore" → `mig + ##lio + ##re`). Tests with mixed Italian/emoji/Japanese text produced valid embeddings (score 0.44 for "pizza"). The only `[UNK]` tokens were the emoji themselves. Semantic quality is still best for English.
- **Polling monitor**: a pragmatic choice on Windows (`notify` produces unreliable `is_file` events). On Linux/macOS, `inotify`/`FSEvents` would be more responsive and consume zero CPU when nothing changes.
- **No INT8 quantization**: candle 0.11.0 does not support quantized BERT forward passes (only quantized LLaMA/Mistral/Qwen2). The model runs at full FP32 precision (~55–65ms per embed).
- **DOCX extraction is minimal**: the inline parser strips XML tags from `word/document.xml`. Tables, headers, footers, and embedded images are ignored.

## Getting started

```bash
git clone https://github.com/Gabriele06-local/LocalMind.git
cd LocalMind
cargo build --release
```

Requires: stable Rust (2021 edition+), internet for initial model download (cached afterwards).

```bash
cargo run --release              # Launch TUI, watching %TEMP%/localmind_watch/
cargo run --release -- --demo    # Headless demo: timing, edge cases, assertions
```

The default watch directory is `%TEMP%\localmind_watch\` (Windows) or `/tmp/localmind_watch/` (Unix). Place `.txt`, `.pdf`, or `.docx` files there, then search.

### TUI controls

| Key | Action |
|---|---|
| Typing | Updates query (search triggers after 200ms of inactivity) |
| `↑` / `↓` | Navigate results |
| `←` / `→` | Move cursor in search bar |
| `Enter` | Open selected file with system default application |
| `Esc` | Clear query, or quit if already empty |
| `Ctrl+C` | Quit |

Similarity scores are color-coded: green (>0.5), yellow (>0.3), gray (<0.3).

## Project structure

```
src/
├── bm25.rs     # BM25 inverted index, whitespace tokenizer, scoring
├── embed.rs    # model loading, tokenization, mean pooling, L2 norm
├── extract.rs  # text extraction for .txt, .pdf, .docx
├── index.rs    # custom binary format, save/load via memmap2
├── search.rs   # top-k vector SIMD + BM25 hybrid fused via RRF
├── monitor.rs  # polling loop, blake3 hashing, thread-safe reindex
├── tui.rs      # ratatui/crossterm terminal interface
└── main.rs     # entry point, demo harness
```

## Roadmap

- [x] **Hybrid search (BM25 + RRF)** — inverted index, whitespace tokenizer, reciprocal rank fusion with k=60
- [ ] Approximate nearest neighbor index (HNSW) for large-scale datasets
- [ ] Native `inotify`/`FSEvents` monitoring on Linux/macOS
- [ ] Support for additional formats (Markdown, ODT, HTML)

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome. The project started as an exercise in building a "from scratch" semantic search engine without heavy dependencies — contributions that preserve this philosophy (minimalism, no external databases, measured performance) are especially appreciated.
