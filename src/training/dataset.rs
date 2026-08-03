//! Dataset abstractions for the HKL-2 training pipeline.
//! Defines the single data interface consumed by the trainer: every
//! source of samples (in-memory corpus, corpus files, live streams)
//! implements [`DataSource`], so training code never cares where data
//! comes from.

use crate::core::math::XorShift64Star;
use alloc::vec::Vec;

/// A source of autoregressive training samples `(inputs, targets)`.
pub trait DataSource {
    /// Fetch the next sample, or `None` when exhausted.
    fn next_sample(&mut self) -> Option<(Vec<u16>, Vec<u16>)>;
    /// Number of samples still available (0 when exhausted).
    fn remaining_samples(&self) -> usize;
    /// Rewind the source to the beginning.
    fn reset(&mut self);
}

/// Deterministically split a contiguous token stream into train/validation
/// streams using fixed-size windows. Splits are reproducible for a given seed.
pub fn split_tokens(
    tokens: &[u16],
    seq_len: usize,
    val_ratio: f32,
    seed: u64,
) -> (Vec<u16>, Vec<u16>) {
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut rng = XorShift64Star::new(seed);
    let step = seq_len.max(1);

    let mut cursor = 0usize;
    while cursor + step + 1 <= tokens.len() {
        let window = &tokens[cursor..cursor + step + 1];
        if rng.next_f32() < val_ratio {
            val.extend_from_slice(window);
        } else {
            train.extend_from_slice(window);
        }
        cursor += step;
    }
    (train, val)
}

/// Batches samples from any [`DataSource`], with optional seeded shuffling.
pub struct BatchSampler<S: DataSource> {
    pub source: S,
    pub batch_size: usize,
    pub pool_size: usize,
    pub shuffle: bool,
    pub rng: Option<XorShift64Star>,
    pool: Vec<(Vec<u16>, Vec<u16>)>,
    served: u64,
}

impl<S: DataSource> BatchSampler<S> {
    pub fn new(source: S, batch_size: usize) -> Self {
        Self {
            source,
            batch_size,
            pool_size: 32,
            shuffle: false,
            rng: None,
            pool: Vec::new(),
            served: 0,
        }
    }

    /// Enable deterministic shuffling with a seed.
    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.rng = Some(XorShift64Star::new(seed));
        self
    }

    fn refill_pool(&mut self) {
        while self.pool.len() < self.pool_size {
            match self.source.next_sample() {
                Some(sample) => self.pool.push(sample),
                None => break,
            }
        }
    }

    fn shuffle_pool(&mut self) {
        let Some(rng) = self.rng.as_mut() else {
            return;
        };
        for i in (1..self.pool.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            self.pool.swap(i, j);
        }
    }

    /// Fetch the next batch. Returns an empty Vec when exhausted.
    pub fn next_batch(&mut self) -> Vec<(Vec<u16>, Vec<u16>)> {
        if self.shuffle {
            if self.pool.is_empty() {
                self.refill_pool();
                self.shuffle_pool();
            }
        } else if self.pool.is_empty() {
            while let Some(sample) = self.source.next_sample() {
                self.pool.push(sample);
                if self.pool.len() == self.batch_size {
                    break;
                }
            }
        }

        let take = self.pool.len().min(self.batch_size);
        if take == 0 {
            return Vec::new();
        }

        let batch: Vec<(Vec<u16>, Vec<u16>)> = self.pool.drain(0..take).collect();
        self.served += take as u64;
        batch
    }

    /// Total samples served so far.
    pub fn served_samples(&self) -> u64 {
        self.served
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::data_loader::TextDataLoader;

    #[test]
    fn test_text_data_loader_implements_data_source() {
        let tokens = alloc::vec![1u16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut loader = TextDataLoader::new(tokens, 3);
        assert_eq!(loader.remaining_samples(), 3);

        let (inputs, targets) = loader.next_sample().unwrap();
        assert_eq!(inputs, alloc::vec![1, 2, 3]);
        assert_eq!(targets, alloc::vec![2, 3, 4]);
        loader.reset();
        assert_eq!(loader.remaining_samples(), 3);
    }

    #[test]
    fn test_batch_sampler_sequential_batching() {
        let tokens: Vec<u16> = (0..40).collect();
        let loader = TextDataLoader::new(tokens, 4);
        let mut sampler = BatchSampler::new(loader, 3);

        let b1 = sampler.next_batch();
        assert_eq!(b1.len(), 3);
        let b2 = sampler.next_batch();
        assert_eq!(b2.len(), 3);
        let b3 = sampler.next_batch();
        assert_eq!(b3.len(), 3);
        let b4 = sampler.next_batch();
        assert_eq!(b4.len(), 0);
        assert_eq!(sampler.served_samples(), 9);
    }

    #[test]
    fn test_batch_sampler_shuffling_is_deterministic_for_seed() {
        let tokens: Vec<u16> = (0..60).collect();
        let make = || {
            let loader = TextDataLoader::new(tokens.clone(), 4);
            BatchSampler::new(loader, 2).with_shuffle(42)
        };

        let mut s1 = make();
        let mut s2 = make();
        let first_order: Vec<Vec<u16>> = s1.next_batch().into_iter().map(|(i, _)| i).collect();
        let second_order: Vec<Vec<u16>> = s2.next_batch().into_iter().map(|(i, _)| i).collect();
        assert_eq!(first_order, second_order);

        let mut s3 = BatchSampler::new(TextDataLoader::new(tokens, 4), 2).with_shuffle(7);
        let other_order: Vec<Vec<u16>> = s3.next_batch().into_iter().map(|(i, _)| i).collect();
        assert_ne!(first_order, other_order);
    }

    #[test]
    fn test_split_tokens_is_reproducible_and_roughly_balanced() {
        let tokens: Vec<u16> = (0..2000).collect();
        let (a1, b1) = split_tokens(&tokens, 8, 0.2, 1234);
        let (a2, b2) = split_tokens(&tokens, 8, 0.2, 1234);

        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert!(!a1.is_empty() && !b1.is_empty());
        // Note: split_tokens produces overlapping windows; total tokens in splits
        // may exceed original length. We only check reproducibility here.

        let total = a1.len() + b1.len();
        let val_frac = b1.len() as f32 / total as f32;
        assert!((val_frac - 0.2).abs() < 0.1);
    }

    #[test]
    fn test_split_tokens_different_seed_different_split() {
        let tokens: Vec<u16> = (0..2000).collect();
        let (a1, _) = split_tokens(&tokens, 8, 0.2, 1);
        let (a2, _) = split_tokens(&tokens, 8, 0.2, 2);
        assert_ne!(a1, a2);
    }
}
