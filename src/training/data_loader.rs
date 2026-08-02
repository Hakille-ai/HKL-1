//! Data loader for token sequences.

use alloc::vec::Vec;

pub struct TextDataLoader {
    pub tokens: Vec<u16>,
    pub seq_len: usize,
    pub cursor: usize,
}

impl TextDataLoader {
    pub fn new(tokens: Vec<u16>, seq_len: usize) -> Self {
        Self {
            tokens,
            seq_len,
            cursor: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() || self.seq_len == 0 || self.tokens.len() <= self.seq_len
    }

    pub fn remaining_samples(&self) -> usize {
        if self.seq_len == 0 || self.cursor + self.seq_len + 1 > self.tokens.len() {
            return 0;
        }

        let remaining_tokens = self.tokens.len() - self.cursor;
        (remaining_tokens - 1) / self.seq_len
    }

    /// Fetch next batch sample (inputs, targets) for autoregressive language modeling
    pub fn next_sample(&mut self) -> Option<(Vec<u16>, Vec<u16>)> {
        if self.seq_len == 0 {
            return None;
        }
        if self.cursor + self.seq_len + 1 > self.tokens.len() {
            return None;
        }

        let input = self.tokens[self.cursor..self.cursor + self.seq_len].to_vec();
        let target = self.tokens[self.cursor + 1..self.cursor + self.seq_len + 1].to_vec();
        self.cursor += self.seq_len;

        Some((input, target))
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader_next_sample() {
        let tokens = alloc::vec![10, 20, 30, 40, 50, 60, 70];
        let mut loader = TextDataLoader::new(tokens, 3);
        assert!(!loader.is_empty());
        assert_eq!(loader.remaining_samples(), 2);

        let sample1 = loader.next_sample().unwrap();
        assert_eq!(sample1.0, alloc::vec![10, 20, 30]);
        assert_eq!(sample1.1, alloc::vec![20, 30, 40]);
        assert_eq!(loader.remaining_samples(), 1);

        let sample2 = loader.next_sample().unwrap();
        assert_eq!(sample2.0, alloc::vec![40, 50, 60]);
        assert_eq!(sample2.1, alloc::vec![50, 60, 70]);
        assert_eq!(loader.remaining_samples(), 0);

        assert!(loader.next_sample().is_none());
    }

    #[test]
    fn test_data_loader_zero_seq_len_is_empty_and_non_advancing() {
        let tokens = alloc::vec![10, 20, 30];
        let mut loader = TextDataLoader::new(tokens, 0);

        assert!(loader.is_empty());
        assert_eq!(loader.remaining_samples(), 0);
        assert!(loader.next_sample().is_none());
        assert_eq!(loader.cursor, 0);
    }

    #[test]
    fn test_data_loader_requires_target_token() {
        let tokens = alloc::vec![10, 20, 30];
        let mut loader = TextDataLoader::new(tokens, 3);

        assert!(loader.is_empty());
        assert_eq!(loader.remaining_samples(), 0);
        assert!(loader.next_sample().is_none());
    }
}
