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

## Why

Most "semantic search" tools require a vector database (Pinecone, Qdrant, Weaviate...) or run via Python with heavy dependencies. LocalMind does the same with a single self-contained Rust binary: a custom binary index format, memory-mapped access, hand-written SIMD vector math, and a responsive terminal UI.

## Features

- **Local embedding**: all-MiniLM-L6-v2 (384 dimensions) via [`candle`](https://github.com/huggingface/candle) — no network calls after the initial weight download.
- **Custom binary index format**: no external database. Header + aligned vectors + offset table + concatenated path strings, memory-mapped with `memmap2` for O(1) access without loading everything into RAM.
- **Parallel + SIMD search**: cosine similarity computed with `rayon` across all cores, dot product accelerated with SIMD (`wide::f32x8` + FMA).
- **Live monitoring**: a polling loop (1s interval) watches a directory, re-hashes files with `blake3`, and re-indexes only on actual changes. Searches are never blocked during reindexing (atomic `Arc<RwLock<Index>>` swap).
- **Responsive TUI**: search bar with debounce, live results, keyboard navigation, file opening via `Enter`.
- **Multi-format support**: indexes `.txt`, `.pdf`, and `.docx` files. PDF text extraction via [`pdf-extract`](https://crates.io/crates/pdf-extract), DOCX via inline ZIP/XML parsing.
- **Small binary**: ~8 MB release build (LTO, strip, codegen-units=1), despite bundled BERT inference.

## How it works

0. **`extract.rs`** — extracts text from `.txt`, `.pdf`, and `.docx` files. TXT uses plain `read_to_string`, PDF uses `pdf-extract` with `lopdf` backend, DOCX reads `word/document.xml` from the ZIP archive and strips XML tags.

1. **`embed.rs`** — loads the model and tokenizer, runs the forward pass, applies mean pooling (masked by the attention mask) and L2 normalization to produce a 384-dimensional vector. Long texts are split into overlapping chunks (256-token windows, overlap 50), embedded separately, averaged, and re-normalized.

2. **`index.rs`** — saves vectors and paths to disk in a custom binary format: header with magic number and version, aligned vector block, fixed-size offset table for random access, concatenated string block. The file is opened via `memmap2`, so reading vector N is always O(1) with no linear scan.

3. **`search.rs`** — computes cosine similarity between the query and every vector in the index, in parallel with `rayon`. The dot product and norm are computed in blocks of 8 floats using SIMD (FMA), with a scalar fallback for the remainder. Top-k results are extracted via `select_nth_unstable_by` to avoid a full sort.

4. **`monitor.rs`** — a polling loop (every 1s) scans the watched directory for `.txt`, `.pdf`, and `.docx` files, computes blake3 hashes from raw bytes, and compares against the previous state. Text is extracted via `extract.rs` before embedding. Only changed files are re-embedded. The new index is written to a temp file, atomically renamed, then the `Arc` pointer is swapped under a brief write-lock — in-flight searches are never blocked for the duration of embedding.

5. **`tui.rs`** — `ratatui`/`crossterm` interface: the query is sent with a 200ms debounce to a `tokio::spawn_blocking` task that performs embedding + search without blocking the UI render loop.

## Benchmarks (CPU, no hardware acceleration)

| Operation | Latency |
|---|---|
| Query embedding (BERT forward pass) | ~55–65ms |
| Index search (8 documents) | ~0.05–0.13ms |
| Initial model download | ~90MB (one-time) |
| Release binary | ~8 MB |
| Monitor polling interval | 1 scan/s |

Embedding is accelerated by `candle`'s matrix multiplication backend ([`gemm` crate](https://crates.io/crates/gemm)), which detects AVX-512/AVX2/SSE at runtime. Search is sub-millisecond even on thousands of documents.

## Known limitations

- **Brute-force search**: O(n) in the number of documents — parallelized but not approximated. Fine for tens of thousands of local files; for millions you'd want an approximate index (e.g., HNSW).
- **English-centric tokenizer**: MiniLM handles Italian correctly via subword tokenization (e.g., "migliore" → `mig + ##lio + ##re`). Tests with mixed Italian/emoji/Japanese text produced valid embeddings (score 0.44 for "pizza"). The only `[UNK]` tokens were the emoji themselves. Semantic quality is still best for English.
- **Polling monitor**: a pragmatic choice on Windows (`notify` produces unreliable `is_file` events). On Linux/macOS, `inotify`/`FSEvents` would be more responsive and consume zero CPU when nothing changes.
- **No INT8 quantization**: candle 0.11.0 does not support quantized BERT forward passes (only quantized LLaMA/Mistral/Qwen2). The model runs at full FP32 precision (~55–65ms per embed).
- **DOCX extraction is minimal**: the inline parser strips XML tags from `word/document.xml`. Tables, headers, footers, and embedded images are ignored.

## Installation

Requirements: stable Rust (2021 edition+), internet connection for the initial model download (cached locally afterwards).

```bash
git clone https://github.com/Gabriele06-local/LocalMind.git
cd LocalMind
cargo build --release
```

## Usage

```bash
# Launch the TUI, monitoring the default directory
cargo run --release

# Demo mode: step-by-step tests without the interface
cargo run --release -- --demo
```

The default watch directory is `%TEMP%\localmind_watch\` (Windows) or `/tmp/localmind_watch/`. Place `.txt`, `.pdf`, or `.docx` files there, then search.

In the TUI:

| Key | Action |
|---|---|
| Typing | Updates the query (search triggers after 200ms of inactivity) |
| `↑` / `↓` | Navigate results |
| `←` / `→` | Move cursor in the search bar |
| `Enter` | Open the selected file with the system default application |
| `Esc` | Clear the query, or quit if already empty |
| `Ctrl+C` | Quit |

Similarity scores are color-coded: green (>0.5), yellow (>0.3), gray (<0.3).

## Project structure

```
src/
├── embed.rs    # model loading, tokenization, mean pooling, L2 norm
├── extract.rs  # text extraction for .txt, .pdf, .docx
├── index.rs    # custom binary format, save/load via memmap2
├── search.rs   # parallel cosine similarity + SIMD, top-k
├── monitor.rs  # polling loop, blake3 hashing, thread-safe reindex
├── tui.rs      # ratatui/crossterm terminal interface
└── main.rs     # entry point, demo harness
```

## Roadmap

- [ ] Approximate nearest neighbor index (HNSW) for large-scale datasets
- [ ] Native `inotify`/`FSEvents` monitoring on Linux/macOS (current polling is a pragmatic workaround for Windows)
- [ ] Support for additional formats (Markdown, ODT, HTML)

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome. The project started as an exercise in building a "from scratch" semantic search engine without heavy dependencies — contributions that preserve this philosophy (minimalism, no external databases, measured performance) are especially appreciated.
