use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::HFClientSync;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("model error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("HF hub error: {0}")]
    Hub(#[from] hf_hub::HFError),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("empty input")]
    EmptyInput,
}

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;

        let client = HFClientSync::new()?;
        let repo = client.model("sentence-transformers", "all-MiniLM-L6-v2");

        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .send()?;
        let config_path = repo
            .download_file()
            .filename("config.json")
            .send()?;
        let weights_path = repo
            .download_file()
            .filename("model.safetensors")
            .send()?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn max_tokens(&self) -> usize {
        256
    }

    // Diagnostic helpers — not used in production paths
    pub fn tokenize(&self, text: &str) -> Result<tokenizers::Encoding> {
        self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(EmbedError::Tokenizer(e.to_string())))
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            anyhow::bail!(EmbedError::EmptyInput);
        }
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        self.embed_encoding(&encoding)
    }

    pub fn embed_chunked(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        let ids = encoding.get_ids();
        let max_len = self.max_tokens();

        if ids.len() <= max_len {
            return self.embed_encoding(&encoding);
        }

        let content = &ids[1..ids.len() - 1];
        let overlap = 50usize;
        let stride = max_len - 2 - overlap;
        let mut chunk_embs: Vec<Vec<f32>> = Vec::new();
        let mut pos = 0;

        while pos < content.len() {
            let end = (pos + max_len - 2).min(content.len());
            let mut chunk_ids = Vec::with_capacity(end - pos + 2);
            chunk_ids.push(101);
            chunk_ids.extend_from_slice(&content[pos..end]);
            chunk_ids.push(102);
            let mask = vec![1u32; chunk_ids.len()];
            let type_ids = vec![0u32; chunk_ids.len()];
            chunk_embs.push(self.embed_tokens(&chunk_ids, &mask, &type_ids)?);
            if end >= content.len() {
                break;
            }
            pos = (end - stride).min(content.len() - 1);
        }

        let dim = chunk_embs[0].len();
        let mut result = vec![0.0f32; dim];
        for emb in &chunk_embs {
            for i in 0..dim {
                result[i] += emb[i];
            }
        }
        let n = chunk_embs.len() as f32;
        for x in &mut result {
            *x /= n;
        }
        let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut result {
            *x /= norm;
        }
        Ok(result)
    }

    fn embed_encoding(&self, encoding: &tokenizers::Encoding) -> Result<Vec<f32>> {
        self.embed_tokens(encoding.get_ids(), encoding.get_attention_mask(), encoding.get_type_ids())
    }

    fn embed_tokens(&self, ids: &[u32], mask: &[u32], type_ids: &[u32]) -> Result<Vec<f32>> {
        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(mask, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;

        let embeddings = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let mask_f = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let summed = (embeddings.broadcast_mul(&mask_f)?).sum(1)?;
        let count = mask_f.sum(1)?;
        let pooled = summed.broadcast_div(&count)?;
        let normalized = pooled.broadcast_div(&pooled.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        Ok(normalized.flatten_all()?.to_vec1()?)
    }
}
