//! Versioned binary checkpoints for the HKL-2 Spiking Transformer.
//!
//! Layout (all little-endian):
//!
//! ```text
//! magic             u8[8]     b"HKLKCPT"
//! version           u16       = 1
//! num_layers        u16
//! flags             u32       = 0
//! step_count        u64
//! bpe_len           u32
//! bpe               u8[bpe_len]         (BpeTokenizer::to_bytes)
//! embedding         i32[VOCAB*EMBED]    (row-major)
//! head_weights      i32[VOCAB*EMBED]
//! head_bias         i32[VOCAB]
//! per layer:
//!   wq/wk/wv/wo     i32[EMBED*EMBED]    (each)
//!   w1/w2           i32[EMBED*FFN] each
//!   norm gamma/beta i32[EMBED]          (x4: norm1 g/b, norm2 g/b)
//! checksum          u32                 (wrapping sum over all prior bytes)
//! ```

use crate::core::math::FixedPoint;
use crate::embedding::bpe_tokenizer::BpeTokenizer;
use crate::embedding::spike_embedding::{EMBED_DIM, VOCAB_SIZE};
use crate::training::corpus::fnv_hash;
use crate::transformer::backbone::SpikingTransformer;
use crate::transformer::block::SpikingTransformerBlock;
use crate::transformer::feed_forward::SpikingFeedForward;
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

pub const CHECKPOINT_MAGIC: &[u8; 7] = b"HKLKCPT";
pub const CHECKPOINT_VERSION: u16 = 1;
pub const SNAPSHOT_SLOTS: usize = 3;

/// A loaded model checkpoint plus its tokenizer.
pub struct Checkpoint {
    pub model: SpikingTransformer,
    pub tokenizer: BpeTokenizer,
    pub step_count: u64,
    pub vocab_hash: u64,
}

/// Serialize the full model + tokenizer into a checkpoint blob.
pub fn serialize_model(
    model: &SpikingTransformer,
    tokenizer: &BpeTokenizer,
    step_count: u64,
) -> Vec<u8> {
    let tokenizer_blob = tokenizer.to_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(CHECKPOINT_MAGIC);
    out.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
    out.extend_from_slice(&(model.blocks.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&step_count.to_le_bytes());
    out.extend_from_slice(&(tokenizer_blob.len() as u32).to_le_bytes());
    out.extend_from_slice(&tokenizer_blob);

    for row in &model.embedding.weights {
        for fp in row {
            extend_fp(&mut out, *fp);
        }
    }
    for fp in &model.head.weights {
        extend_fp(&mut out, *fp);
    }
    for fp in &model.head.bias {
        extend_fp(&mut out, *fp);
    }
    for block in &model.blocks {
        serialize_block(&mut out, block);
    }

    let checksum = compute_checksum(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

/// Deserialize a checkpoint blob produced by [`serialize_model`].
pub fn deserialize_model(bytes: &[u8]) -> Option<Checkpoint> {
    let mut r = Reader::new(bytes)?;
    if r.take(7)? != CHECKPOINT_MAGIC {
        return None;
    }
    let version = r.u16()?;
    if version > CHECKPOINT_VERSION {
        return None;
    }
    let num_layers = r.u16()? as usize;
    let _flags = r.u32()?;
    let step_count = r.u64()?;
    let tokenizer_len = r.u32()? as usize;
    let tokenizer_blob = r.take(tokenizer_len)?;
    let tokenizer = BpeTokenizer::from_bytes(tokenizer_blob)?;

    let mut model = SpikingTransformer::new(num_layers);

    for row in model.embedding.weights.iter_mut() {
        for fp in row.iter_mut() {
            *fp = r.fp()?;
        }
    }
    for fp in model.head.weights.iter_mut() {
        *fp = r.fp()?;
    }
    for fp in model.head.bias.iter_mut() {
        *fp = r.fp()?;
    }
    for block in model.blocks.iter_mut() {
        deserialize_block(&mut r, block)?;
    }

    let checksum = r.u32()?;
    if r.remaining() != 0 {
        return None;
    }
    let expected = compute_checksum(&bytes[..bytes.len() - 4]);
    if checksum != expected {
        return None;
    }

    let vocab_hash = fnv_hash(tokenizer_blob);
    Some(Checkpoint {
        model,
        tokenizer,
        step_count,
        vocab_hash,
    })
}

fn extend_fp(out: &mut Vec<u8>, fp: FixedPoint) {
    out.extend_from_slice(&fp.0.to_le_bytes());
}

fn serialize_block(out: &mut Vec<u8>, block: &SpikingTransformerBlock) {
    for w in [
        &block.attention.wq,
        &block.attention.wk,
        &block.attention.wv,
        &block.attention.wo,
    ] {
        for fp in w {
            extend_fp(out, *fp);
        }
    }
    serialize_ffn(out, &block.feed_forward);
    for norm in [&block.norm1, &block.norm2] {
        for fp in norm.gamma.iter().take(EMBED_DIM) {
            extend_fp(out, *fp);
        }
        for fp in norm.beta.iter().take(EMBED_DIM) {
            extend_fp(out, *fp);
        }
    }
}

fn serialize_ffn(out: &mut Vec<u8>, ffn: &SpikingFeedForward) {
    for fp in &ffn.w1 {
        extend_fp(out, *fp);
    }
    for fp in &ffn.w2 {
        extend_fp(out, *fp);
    }
}

fn deserialize_block(r: &mut Reader<'_>, block: &mut SpikingTransformerBlock) -> Option<()> {
    let mut matrices = [
        block.attention.wq.as_mut_slice(),
        block.attention.wk.as_mut_slice(),
        block.attention.wv.as_mut_slice(),
        block.attention.wo.as_mut_slice(),
    ];
    for m in matrices.iter_mut() {
        for fp in m.iter_mut() {
            *fp = r.fp()?;
        }
    }
    for fp in block.feed_forward.w1.iter_mut() {
        *fp = r.fp()?;
    }
    for fp in block.feed_forward.w2.iter_mut() {
        *fp = r.fp()?;
    }
    for norm in [&mut block.norm1, &mut block.norm2] {
        for fp in norm.gamma.iter_mut().take(EMBED_DIM) {
            *fp = r.fp()?;
        }
        for fp in norm.beta.iter_mut().take(EMBED_DIM) {
            *fp = r.fp()?;
        }
    }
    Some(())
}

fn compute_checksum(bytes: &[u8]) -> u32 {
    let mut sum = 0u32;
    for &b in bytes {
        sum = sum.wrapping_add(b as u32);
    }
    sum
}

/// Save a checkpoint to a file.
pub fn save_checkpoint(
    path: &str,
    model: &SpikingTransformer,
    tokenizer: &BpeTokenizer,
    step_count: u64,
) -> Result<(), String> {
    let blob = serialize_model(model, tokenizer, step_count);
    std::fs::write(path, &blob).map_err(|e| format!("write {}: {}", path, e))
}

/// Load a checkpoint from a file. The tokenizer vocabulary must match the
/// blob's unless `verify_hash` is false.
pub fn load_checkpoint(path: &str) -> Result<Checkpoint, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    deserialize_model(&bytes).ok_or_else(|| format!("corrupt checkpoint: {}", path))
}

/// Rotate snapshot slot paths (slot 2 <- slot 1 <- slot 0 <- current).
/// Mirrors the flash `PERSISTENCE_SLOTS` rotation used on bare metal.
pub fn rotate_slot_paths(base: &str, current: &str) -> Vec<String> {
    let mut paths = Vec::with_capacity(SNAPSHOT_SLOTS);
    for i in 1..SNAPSHOT_SLOTS {
        paths.push(format!("{}.{}.hklk", base, SNAPSHOT_SLOTS - i));
    }
    paths.push(current.to_string());
    paths
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        Some(Self { bytes, pos: 0 })
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.bytes.len() {
            return None;
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }

    fn u16(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn fp(&mut self) -> Option<FixedPoint> {
        let b = self.take(4)?;
        Some(FixedPoint(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec as AllocVec;

    fn make_model() -> (SpikingTransformer, BpeTokenizer) {
        let mut model = SpikingTransformer::new(1);
        model.init_random(7);

        let mut tokenizer = BpeTokenizer::new();
        tokenizer.train(b"the quick brown fox and the quick hound dog", 8);

        (model, tokenizer)
    }

    fn forward_tokens(
        model: &mut SpikingTransformer,
        tokens: &[u16],
    ) -> AllocVec<[FixedPoint; VOCAB_SIZE]> {
        model.reset_state();
        let logits = model.forward(tokens);
        model.reset_state();
        logits
    }

    #[test]
    fn test_checkpoint_roundtrip_preserves_forward() {
        let (mut a, tokenizer) = make_model();
        let tokens: Vec<u16> = tokenizer.encode_bytes(b"the quick fox");
        let before = forward_tokens(&mut a, &tokens);

        let blob = serialize_model(&a, &tokenizer, 123);
        let mut checkpoint = deserialize_model(&blob).expect("checkpoint failed to parse");

        assert_eq!(checkpoint.step_count, 123);
        assert_eq!(checkpoint.tokenizer.merges, tokenizer.merges);
        assert_eq!(checkpoint.vocab_hash, fnv_hash(&tokenizer.to_bytes()));

        let after = forward_tokens(&mut checkpoint.model, &tokens);
        assert_eq!(before, after);
    }

    #[test]
    fn test_checkpoint_rejects_truncated_blob() {
        let (model, tokenizer) = make_model();
        let blob = serialize_model(&model, &tokenizer, 0);
        assert!(deserialize_model(&blob).is_some());
        assert!(deserialize_model(&blob[..blob.len() / 2]).is_none());
    }

    #[test]
    fn test_rotate_slot_paths_rotation_order() {
        let slots = rotate_slot_paths("data/snap", "current.hklk");
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0], "data/snap.2.hklk");
        assert_eq!(slots[1], "data/snap.1.hklk");
        assert_eq!(slots[2], "current.hklk");
    }
}
