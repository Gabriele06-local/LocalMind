use std::collections::HashMap;

/// A simple whitespace-based tokenizer for BM25.
/// Splits on whitespace and punctuation, lowercases, filters short tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::with_capacity(32);
    for c in text.chars() {
        if c.is_alphanumeric() || c == '-' || c == '_' {
            current.push(c);
        } else {
            if current.len() >= 2 {
                tokens.push(current.to_lowercase());
            }
            current.clear();
        }
    }
    if current.len() >= 2 {
        tokens.push(current.to_lowercase());
    }
    tokens
}

/// BM25 inverted index.
pub struct Bm25Index {
    postings: HashMap<String, Vec<(u32, u32)>>, // term -> [(doc_id, tf)]
    doc_lengths: Vec<u32>,
    num_docs: u32,
    avg_doc_len: f32,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: Vec::new(),
            num_docs: 0,
            avg_doc_len: 0.0,
        }
    }

    pub fn add_document(&mut self, id: u32, text: &str) {
        let tokens = tokenize(text);
        let len = tokens.len() as u32;
        if id as usize >= self.doc_lengths.len() {
            self.doc_lengths.resize(id as usize + 1, 0);
        }
        self.doc_lengths[id as usize] = len;
        self.num_docs += 1;
        self.avg_doc_len = ((self.avg_doc_len * (self.num_docs - 1) as f32) + len as f32)
            / self.num_docs as f32;

        let mut tf = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0u32) += 1;
        }
        for (term, freq) in tf {
            self.postings.entry(term).or_default().push((id, freq));
        }
    }

    /// Score a single document against the query.
    fn score(&self, query_tokens: &[String], doc_id: u32) -> f32 {
        let doc_len = self.doc_lengths.get(doc_id as usize).copied().unwrap_or(0) as f32;
        if doc_len == 0.0 || self.avg_doc_len == 0.0 {
            return 0.0;
        }
        let n = self.num_docs as f32;
        let k1 = 1.2f32;
        let b = 0.75f32;
        let mut total = 0.0f32;

        for term in query_tokens {
            if let Some(postings) = self.postings.get(term) {
                let docs_with_term = postings.len() as f32;
                let idf = ((n - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0).ln();
                let tf = postings
                    .iter()
                    .find(|&&(id, _)| id == doc_id)
                    .map(|&(_, f)| f as f32)
                    .unwrap_or(0.0);
                if tf > 0.0 {
                    total += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / self.avg_doc_len));
                }
            }
        }
        total
    }

    /// Search the BM25 index, return up to `k` results as `(doc_id, score)`.
    /// Score is the raw BM25 score (unbounded).
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || self.num_docs == 0 {
            return Vec::new();
        }

        // Collect candidate doc_ids from matching postings
        let mut candidates: Vec<u32> = Vec::new();
        for term in &query_tokens {
            if let Some(postings) = self.postings.get(term) {
                for &(doc_id, _) in postings {
                    if !candidates.contains(&doc_id) {
                        candidates.push(doc_id);
                    }
                }
            }
        }

        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|doc_id| (doc_id, self.score(&query_tokens, doc_id)))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.postings.clear();
        self.doc_lengths.clear();
        self.num_docs = 0;
        self.avg_doc_len = 0.0;
    }

    pub fn len(&self) -> usize {
        self.num_docs as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "the cat sits on the mat");
        idx.add_document(1, "the dog runs in the park");
        idx.add_document(2, "cats and dogs are pets");

        let results = idx.search("cat", 3);
        assert!(!results.is_empty(), "cat should match document 0");
        assert_eq!(results[0].0, 0, "document 0 should be top for 'cat'");

        let results = idx.search("dog", 3);
        assert!(!results.is_empty(), "dog should match");
        assert_eq!(results[0].0, 1, "document 1 should be top for 'dog'");
    }

    #[test]
    fn test_bm25_empty_query() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "some text");
        let results = idx.search("", 5);
        assert!(results.is_empty(), "empty query yields no results");
    }

    #[test]
    fn test_bm25_no_match() {
        let mut idx = Bm25Index::new();
        idx.add_document(0, "hello world");
        let results = idx.search("zzzznotfound", 5);
        assert!(results.is_empty(), "no match yields empty results");
    }
}
