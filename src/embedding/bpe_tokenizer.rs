use alloc::vec::Vec;
use std::collections::HashMap;

pub const BPE_MAGIC: &[u8; 6] = b"HKLBPE";
pub const MAX_MERGES: usize = crate::embedding::spike_embedding::VOCAB_SIZE - 256;

#[derive(Clone)]
pub struct BpeTokenizer {
    pub merges: Vec<(u16, u16, u16)>,
    pub merge_count: usize,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            merges: Vec::new(),
            merge_count: 0,
        }
    }

    pub fn add_merge(&mut self, a: u16, b: u16, merged_id: u16) -> bool {
        if merged_id < 256 || a == merged_id || b == merged_id {
            return false;
        }
        if !self.is_known_token(a) || !self.is_known_token(b) {
            return false;
        }
        if self.merges.iter().any(|&(_, _, id)| id == merged_id) {
            return false;
        }

        self.merges.push((a, b, merged_id));
        self.merge_count += 1;
        true
    }

    pub fn is_known_token(&self, token: u16) -> bool {
        token < 256 || self.merges.iter().any(|&(_, _, merged)| merged == token)
    }

    /// Train a byte-level BPE merge table on a raw byte corpus.
    /// Merge ids are assigned sequentially from 256 upward.
    /// Returns the number of merges actually learned.
    pub fn train(&mut self, corpus: &[u8], num_merges: usize) -> usize {
        let mut tokens: Vec<u16> = corpus.iter().map(|&b| b as u16).collect();
        let mut learned = 0usize;
        let budget = num_merges.min(MAX_MERGES);

        while learned < budget {
            let mut counts: HashMap<(u16, u16), usize> = HashMap::new();
            for w in tokens.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
            }

            let best = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));

            let Some((&(a, b), &count)) = best else {
                break;
            };
            if count < 2 {
                break;
            }

            let merged_id = (256 + learned) as u16;
            if !self.add_merge(a, b, merged_id) {
                break;
            }
            learned += 1;

            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == a && tokens[i + 1] == b {
                    tokens[i] = merged_id;
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }

        learned
    }

    /// Serialize the merge table into a self-describing binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.merges.len() * 6);
        out.extend_from_slice(BPE_MAGIC);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for &(a, b, merged) in &self.merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
            out.extend_from_slice(&merged.to_le_bytes());
        }
        out
    }

    /// Deserialize a merge table previously written by [`BpeTokenizer::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 || &bytes[0..6] != BPE_MAGIC {
            return None;
        }
        if u16::from_le_bytes([bytes[6], bytes[7]]) != 1 {
            return None;
        }
        let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let expected = 12 + count.checked_mul(6)?;
        if bytes.len() != expected || count > MAX_MERGES {
            return None;
        }

        let mut tokenizer = Self::new();
        for i in 0..count {
            let off = 12 + i * 6;
            let a = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            let b = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]);
            let merged = u16::from_le_bytes([bytes[off + 4], bytes[off + 5]]);
            if !tokenizer.add_merge(a, b, merged) {
                return None;
            }
        }
        Some(tokenizer)
    }

    pub fn encode_bytes(&self, text: &[u8]) -> Vec<u16> {
        let mut tokens: Vec<u16> = text.iter().map(|&b| b as u16).collect();

        if tokens.is_empty() {
            return tokens;
        }

        loop {
            let mut best_pair = None;
            let mut best_idx = 0;
            let mut best_merge_idx = usize::MAX;

            // Find the earliest matching merge rule for adjacent tokens
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(pos) = self
                    .merges
                    .iter()
                    .position(|&(a, b, _)| a == pair.0 && b == pair.1)
                {
                    if pos < best_merge_idx {
                        best_merge_idx = pos;
                        best_pair = Some(self.merges[pos].2);
                        best_idx = i;
                    }
                }
            }

            if let Some(merged_id) = best_pair {
                tokens[best_idx] = merged_id;
                tokens.remove(best_idx + 1);
            } else {
                break;
            }
        }

        tokens
    }

    pub fn decode_tokens(&self, tokens: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &t in tokens {
            self.decode_token(t, &mut bytes, 0);
        }
        bytes
    }

    fn decode_token(&self, token: u16, bytes: &mut Vec<u8>, depth: usize) {
        if token < 256 {
            bytes.push(token as u8);
            return;
        }

        if depth >= self.merges.len() {
            return;
        }

        if let Some(&(left, right, _)) = self.merges.iter().find(|&&(_, _, merged)| merged == token)
        {
            self.decode_token(left, bytes, depth + 1);
            self.decode_token(right, bytes, depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encode_decode_bytes() {
        let tokenizer = BpeTokenizer::new();
        let text = b"hello";
        let tokens = tokenizer.encode_bytes(text);
        assert_eq!(tokens, alloc::vec![104, 101, 108, 108, 111]);

        let decoded = tokenizer.decode_tokens(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_add_merge() {
        let mut tokenizer = BpeTokenizer::new();
        let text = b"hello";
        assert!(tokenizer.add_merge(108, 108, 256));
        let tokens = tokenizer.encode_bytes(text);
        assert_eq!(tokens, alloc::vec![104, 101, 256, 111]);

        let decoded = tokenizer.decode_tokens(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_nested_merge_decodes_original_bytes() {
        let mut tokenizer = BpeTokenizer::new();
        assert!(tokenizer.add_merge(b'h' as u16, b'e' as u16, 256));
        assert!(tokenizer.add_merge(256, b'l' as u16, 257));

        let tokens = tokenizer.encode_bytes(b"hel");
        assert_eq!(tokens, alloc::vec![257]);
        assert_eq!(tokenizer.decode_tokens(&tokens), b"hel");
    }

    #[test]
    fn test_bpe_rejects_ambiguous_or_recursive_merges() {
        let mut tokenizer = BpeTokenizer::new();

        assert!(!tokenizer.add_merge(b'a' as u16, b'b' as u16, 42));
        assert!(tokenizer.add_merge(b'a' as u16, b'b' as u16, 256));
        assert!(!tokenizer.add_merge(b'c' as u16, b'd' as u16, 256));
        assert!(!tokenizer.add_merge(257, b'e' as u16, 257));
        assert!(!tokenizer.add_merge(999, b'e' as u16, 258));
        assert_eq!(tokenizer.merge_count, 1);
    }

    #[test]
    fn test_bpe_train_learns_repeated_pairs_and_roundtrips() {
        let corpus = b"ab ab ab ab cd cd cd cd ab ab ab ab";
        let mut tokenizer = BpeTokenizer::new();
        let learned = tokenizer.train(corpus, 8);

        assert!(learned > 0);
        assert!(tokenizer.merge_count <= 8);
        assert!(tokenizer.merges.iter().all(|&(_, _, id)| id >= 256));

        let tokens = tokenizer.encode_bytes(corpus);
        assert!(tokens.len() < corpus.len());
        assert_eq!(tokenizer.decode_tokens(&tokens), corpus);
    }

    #[test]
    fn test_bpe_train_respects_merge_budget_and_vocab_limit() {
        let corpus = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mut tokenizer = BpeTokenizer::new();
        let learned = tokenizer.train(corpus, MAX_MERGES + 100);

        assert!(learned <= MAX_MERGES);
        assert!(tokenizer.merges.iter().all(|&(_, _, id)| id <= 4095));
    }

    #[test]
    fn test_bpe_train_empty_or_tiny_corpus_learns_nothing() {
        let mut tokenizer = BpeTokenizer::new();
        assert_eq!(tokenizer.train(b"", 64), 0);
        assert_eq!(tokenizer.train(b"abc", 64), 0);
    }

    #[test]
    fn test_bpe_serialization_roundtrip() {
        let corpus = b"the quick brown fox jumps over the lazy dog the quick";
        let mut tokenizer = BpeTokenizer::new();
        tokenizer.train(corpus, 16);

        let bytes = tokenizer.to_bytes();
        let restored = BpeTokenizer::from_bytes(&bytes).expect("BPE blob failed to parse");

        assert_eq!(restored.merges, tokenizer.merges);
        let encoded = tokenizer.encode_bytes(corpus);
        assert_eq!(restored.encode_bytes(corpus), encoded);
        assert_eq!(restored.decode_tokens(&encoded), corpus);
    }

    #[test]
    fn test_bpe_serialization_rejects_corrupt_blobs() {
        let mut tokenizer = BpeTokenizer::new();
        tokenizer.train(b"hello world hello world hello", 4);
        let mut bytes = tokenizer.to_bytes();

        assert!(BpeTokenizer::from_bytes(&bytes).is_some());
        bytes[0] = b'X';
        assert!(BpeTokenizer::from_bytes(&bytes).is_none());

        let tokenizer = BpeTokenizer::new();
        assert!(BpeTokenizer::from_bytes(&tokenizer.to_bytes()).is_some());
    }
}
