use crate::core::math::{FixedPoint, XorShift64Star};
use crate::embedding::spike_embedding::EMBED_DIM;
use alloc::vec;
use alloc::vec::Vec;

pub const FFN_DIM: usize = 512; // 2x EMBED_DIM

pub struct SpikingFeedForward {
    pub w1: Vec<FixedPoint>, // EMBED_DIM x FFN_DIM (row-major: FFN_DIM x EMBED_DIM)
    pub w2: Vec<FixedPoint>, // FFN_DIM x EMBED_DIM (row-major: EMBED_DIM x FFN_DIM)
    pub sn1_membranes: Vec<FixedPoint>,
    pub sn2_membranes: Vec<FixedPoint>,
    pub threshold: FixedPoint,
    pub decay: FixedPoint,
}

impl SpikingFeedForward {
    pub fn new() -> Self {
        Self {
            w1: vec![FixedPoint::ZERO; EMBED_DIM * FFN_DIM],
            w2: vec![FixedPoint::ZERO; FFN_DIM * EMBED_DIM],
            sn1_membranes: vec![FixedPoint::ZERO; FFN_DIM],
            sn2_membranes: vec![FixedPoint::ZERO; EMBED_DIM],
            threshold: FixedPoint::from_f32(1.0),
            decay: FixedPoint::from_f32(0.5),
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        let mut rng = XorShift64Star::new(seed);

        let k1_f32 = 1.0 / (EMBED_DIM as f32).sqrt();
        for v in self.w1.iter_mut() {
            let rand_val = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 * k1_f32 - k1_f32;
            *v = FixedPoint::from_f32(rand_val);
        }

        let k2_f32 = 1.0 / (FFN_DIM as f32).sqrt();
        for v in self.w2.iter_mut() {
            let rand_val = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 * k2_f32 - k2_f32;
            *v = FixedPoint::from_f32(rand_val);
        }
        self.reset_state();
    }

    pub fn reset_state(&mut self) {
        for membrane in self
            .sn1_membranes
            .iter_mut()
            .chain(self.sn2_membranes.iter_mut())
        {
            *membrane = FixedPoint::ZERO;
        }
    }

    fn lif_step(
        membrane: &mut FixedPoint,
        input: FixedPoint,
        threshold: FixedPoint,
        decay: FixedPoint,
    ) -> FixedPoint {
        *membrane = *membrane * decay + input;
        if *membrane >= threshold {
            *membrane = *membrane - threshold; // Soft reset
            FixedPoint::ONE
        } else {
            FixedPoint::ZERO
        }
    }

    pub fn forward(&mut self, x: &[FixedPoint; EMBED_DIM]) -> [FixedPoint; EMBED_DIM] {
        let mut hidden_raw = [FixedPoint::ZERO; FFN_DIM];
        // linear_transform equivalent
        for r in 0..FFN_DIM {
            let mut sum = FixedPoint::ZERO;
            for c in 0..EMBED_DIM {
                sum = sum + self.w1[r * EMBED_DIM + c] * x[c];
            }
            hidden_raw[r] = sum;
        }

        let mut hidden_spike = [FixedPoint::ZERO; FFN_DIM];
        for i in 0..FFN_DIM {
            hidden_spike[i] = Self::lif_step(
                &mut self.sn1_membranes[i],
                hidden_raw[i],
                self.threshold,
                self.decay,
            );
        }

        let mut output_raw = [FixedPoint::ZERO; EMBED_DIM];
        for r in 0..EMBED_DIM {
            let mut sum = FixedPoint::ZERO;
            for c in 0..FFN_DIM {
                sum = sum + self.w2[r * FFN_DIM + c] * hidden_spike[c];
            }
            output_raw[r] = sum;
        }

        let mut output_spike = [FixedPoint::ZERO; EMBED_DIM];
        for i in 0..EMBED_DIM {
            output_spike[i] = Self::lif_step(
                &mut self.sn2_membranes[i],
                output_raw[i],
                self.threshold,
                self.decay,
            );
        }

        output_spike
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward() {
        let mut ff = SpikingFeedForward::new();
        ff.init_random(42);
        let x = [FixedPoint::ZERO; EMBED_DIM];
        let out = ff.forward(&x);
        assert_eq!(out.len(), EMBED_DIM);
    }

    #[test]
    fn test_feed_forward_reset_state_clears_all_membranes() {
        let mut ff = SpikingFeedForward::new();
        ff.sn1_membranes[0] = FixedPoint::ONE;
        ff.sn2_membranes[0] = FixedPoint::ONE;

        ff.reset_state();

        assert!(ff.sn1_membranes.iter().all(|v| *v == FixedPoint::ZERO));
        assert!(ff.sn2_membranes.iter().all(|v| *v == FixedPoint::ZERO));
    }
}
