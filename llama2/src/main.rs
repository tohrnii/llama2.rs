#[derive(Debug, Clone)]
pub struct Config {
    dim: usize, // transformer dimension
    hidden_dim: usize, // ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads
    vocab_size: usize, // vocabulary size,
    seq_len: usize // max seq length
}

pub struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    
    // weights for rmsnorms
    rms_attn_weights: Vec<f32>, // (n_layers, dim)
    rms_ffn_weights: Vec<f32>, // (n_layers, dim)
    
    // weights for matmuls. dim == n_heads * head_size
    wq: Vec<f32>, // (layer, dim, n_heads * head_size)
    wk: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wv: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wo: Vec<f32>, // (layer, n_heads * head_size, dim)

    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)

    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim)

    // (optional) classifier weights for the logits, on the last layer
    wcls: Option<Vec<f32>>,
}

pub struct RunState {
    // current wave of activations
    x: Vec<f32>, // activation at current time stamp
    xb: Vec<f32>, // activation inside a residual branch
    xb2: Vec<f32>, // additional buffer for convencience
    hb: Vec<f32>, // buffer for hidden dimension in the ffn
    hb2: Vec<f32>, // buffer for hidden dimension in the ffn
    q: Vec<f32>, // query
    k: Vec<f32>, // key
    v: Vec<f32>, // value
    att: Vec<f32>, // buffer for attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

pub struct Transformer {
    config: Config, // hyperparameters
    weights: TransformerWeights,
    state: RunState, // buffers for the activations in the forward pass
}

impl Transformer {
    /// TODO
    pub fn build_transformer(transformer: Transformer, checkpoint_path: &str) {
    }
}


/// TODO
fn rmsnorm() {

}

/// TODO
fn softmax() {

}


/// TODO
fn matmul() {

}

/// TODO
fn forward() -> Vec<f32> {
    vec![]
}


pub fn main() {
    // load model and config

    // load weights

    // run
}