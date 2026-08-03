use alloc::vec::Vec;

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
}
