use crate::core::math::{FixedPoint, XorShift64Star};
use crate::embedding::spike_embedding::EMBED_DIM;
use alloc::vec;
use alloc::vec::Vec;

pub const NUM_HEADS: usize = 4;
pub const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
pub const MAX_SEQ_LEN: usize = 128;

pub struct SpikingSelfAttention {
    pub wq: Vec<FixedPoint>,
    pub wk: Vec<FixedPoint>,
    pub wv: Vec<FixedPoint>,
    pub wo: Vec<FixedPoint>,
    pub q_membranes: Vec<FixedPoint>,
    pub k_membranes: Vec<FixedPoint>,
    pub v_membranes: Vec<FixedPoint>,
    pub o_membranes: Vec<FixedPoint>,
    pub threshold: FixedPoint,
    pub decay: FixedPoint,
    pub scale: FixedPoint,
}

impl SpikingSelfAttention {
    pub fn new() -> Self {
        let size = EMBED_DIM * EMBED_DIM;
        Self {
            wq: vec![FixedPoint::ZERO; size],
            wk: vec![FixedPoint::ZERO; size],
            wv: vec![FixedPoint::ZERO; size],
            wo: vec![FixedPoint::ZERO; size],
            q_membranes: vec![FixedPoint::ZERO; EMBED_DIM],
            k_membranes: vec![FixedPoint::ZERO; EMBED_DIM],
            v_membranes: vec![FixedPoint::ZERO; EMBED_DIM],
            o_membranes: vec![FixedPoint::ZERO; EMBED_DIM],
            threshold: FixedPoint::from_f32(1.0),
            decay: FixedPoint::from_f32(0.5),
            scale: FixedPoint::ONE / FixedPoint::from_int(HEAD_DIM as i32).sqrt(),
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        let mut rng = XorShift64Star::new(seed);
        let k_f32 = 1.0 / (EMBED_DIM as f32).sqrt();

        for w in [&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo] {
            for v in w.iter_mut() {
                let rand_val = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 * k_f32 - k_f32;
                *v = FixedPoint::from_f32(rand_val);
            }
        }
        self.reset_state();
    }

    pub fn reset_state(&mut self) {
        for membrane in self
            .q_membranes
            .iter_mut()
            .chain(self.k_membranes.iter_mut())
            .chain(self.v_membranes.iter_mut())
            .chain(self.o_membranes.iter_mut())
        {
            *membrane = FixedPoint::ZERO;
        }
    }

    pub fn linear_transform(
        weights: &[FixedPoint],
        input: &[FixedPoint],
        output: &mut [FixedPoint],
        rows: usize,
        cols: usize,
    ) -> bool {
        if weights.len() < rows.saturating_mul(cols) || input.len() < cols || output.len() < rows {
            return false;
        }

        for r in 0..rows {
            let mut sum = FixedPoint::ZERO;
            for c in 0..cols {
                sum = sum + weights[r * cols + c] * input[c];
            }
            output[r] = sum;
        }
        true
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

    pub fn forward(
        &mut self,
        input: &[[FixedPoint; EMBED_DIM]],
        seq_len: usize,
    ) -> Vec<[FixedPoint; EMBED_DIM]> {
        let active_len = seq_len.min(input.len()).min(MAX_SEQ_LEN);
        let mut outputs = vec![[FixedPoint::ZERO; EMBED_DIM]; active_len];

        let mut q_seq = vec![[FixedPoint::ZERO; EMBED_DIM]; active_len];
        let mut k_seq = vec![[FixedPoint::ZERO; EMBED_DIM]; active_len];
        let mut v_seq = vec![[FixedPoint::ZERO; EMBED_DIM]; active_len];

        // 1. For each position: compute Q, K, V
        for i in 0..active_len {
            let x = &input[i];

            let mut q_raw = [FixedPoint::ZERO; EMBED_DIM];
            let mut k_raw = [FixedPoint::ZERO; EMBED_DIM];
            let mut v_raw = [FixedPoint::ZERO; EMBED_DIM];

            Self::linear_transform(&self.wq, x, &mut q_raw, EMBED_DIM, EMBED_DIM);
            Self::linear_transform(&self.wk, x, &mut k_raw, EMBED_DIM, EMBED_DIM);
            Self::linear_transform(&self.wv, x, &mut v_raw, EMBED_DIM, EMBED_DIM);

            for j in 0..EMBED_DIM {
                q_seq[i][j] = Self::lif_step(
                    &mut self.q_membranes[j],
                    q_raw[j],
                    self.threshold,
                    self.decay,
                );
                k_seq[i][j] = Self::lif_step(
                    &mut self.k_membranes[j],
                    k_raw[j],
                    self.threshold,
                    self.decay,
                );
                v_seq[i][j] = Self::lif_step(
                    &mut self.v_membranes[j],
                    v_raw[j],
                    self.threshold,
                    self.decay,
                );
            }
        }

        let mut context = vec![[FixedPoint::ZERO; EMBED_DIM]; active_len];

        // 2. For each head
        for h in 0..NUM_HEADS {
            let head_offset = h * HEAD_DIM;

            let mut attn = vec![vec![FixedPoint::ZERO; active_len]; active_len];

            for i in 0..active_len {
                for j in 0..active_len {
                    let mut dot = FixedPoint::ZERO;
                    for d in 0..HEAD_DIM {
                        dot = dot + q_seq[i][head_offset + d] * k_seq[j][head_offset + d];
                    }
                    let score = dot * self.scale;

                    // 3. Apply LIF spiking to attention scores
                    attn[i][j] = if score >= self.threshold {
                        FixedPoint::ONE
                    } else {
                        FixedPoint::ZERO
                    };
                }
            }

            // 4. Compute context
            for i in 0..active_len {
                for d in 0..HEAD_DIM {
                    let mut sum = FixedPoint::ZERO;
                    for j in 0..active_len {
                        sum = sum + attn[i][j] * v_seq[j][head_offset + d];
                    }
                    context[i][head_offset + d] = sum;
                }
            }
        }

        // 5. Concatenate heads, apply W_o projection + LIF
        for i in 0..active_len {
            let mut o_raw = [FixedPoint::ZERO; EMBED_DIM];
            Self::linear_transform(&self.wo, &context[i], &mut o_raw, EMBED_DIM, EMBED_DIM);

            for j in 0..EMBED_DIM {
                outputs[i][j] = Self::lif_step(
                    &mut self.o_membranes[j],
                    o_raw[j],
                    self.threshold,
                    self.decay,
                );
            }
        }

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_transform() {
        let mut weights = vec![FixedPoint::ZERO; 4];
        weights[0] = FixedPoint::from_f32(1.0);
        weights[1] = FixedPoint::from_f32(2.0);
        weights[2] = FixedPoint::from_f32(3.0);
        weights[3] = FixedPoint::from_f32(4.0);

        let input = [FixedPoint::from_f32(1.0), FixedPoint::from_f32(2.0)];
        let mut output = [FixedPoint::ZERO; 2];

        assert!(SpikingSelfAttention::linear_transform(
            &weights,
            &input,
            &mut output,
            2,
            2
        ));

        assert_eq!(output[0].to_f32(), 5.0); // 1*1 + 2*2 = 5
        assert_eq!(output[1].to_f32(), 11.0); // 3*1 + 4*2 = 11
    }

    #[test]
    fn linear_transform_rejects_short_buffers_without_mutating_output() {
        let weights = [FixedPoint::ONE; 3];
        let input = [FixedPoint::ONE; 2];
        let sentinel = FixedPoint::from_f32(7.0);
        let mut output = [sentinel; 2];

        assert!(!SpikingSelfAttention::linear_transform(
            &weights,
            &input,
            &mut output,
            2,
            2
        ));

        assert_eq!(output, [sentinel; 2]);
    }

    #[test]
    fn linear_transform_rejects_short_input_or_output() {
        let weights = [FixedPoint::ONE; 4];
        let input = [FixedPoint::ONE; 1];
        let mut output = [FixedPoint::ZERO; 2];

        assert!(!SpikingSelfAttention::linear_transform(
            &weights,
            &input,
            &mut output,
            2,
            2
        ));

        let input = [FixedPoint::ONE; 2];
        let mut output = [FixedPoint::ZERO; 1];
        assert!(!SpikingSelfAttention::linear_transform(
            &weights,
            &input,
            &mut output,
            2,
            2
        ));
    }

    #[test]
    fn test_attention_forward() {
        let mut attn = SpikingSelfAttention::new();
        attn.init_random(42);
        let seq = vec![[FixedPoint::ZERO; EMBED_DIM]; 4];
        let output = attn.forward(&seq, 4);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_attention_forward_clamps_to_input_len() {
        let mut attn = SpikingSelfAttention::new();
        let seq = vec![[FixedPoint::ZERO; EMBED_DIM]; 1];
        let output = attn.forward(&seq, MAX_SEQ_LEN + 1);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_attention_reset_state_clears_all_membranes() {
        let mut attn = SpikingSelfAttention::new();
        attn.q_membranes[0] = FixedPoint::ONE;
        attn.k_membranes[1] = FixedPoint::ONE;
        attn.v_membranes[2] = FixedPoint::ONE;
        attn.o_membranes[3] = FixedPoint::ONE;

        attn.reset_state();

        assert!(attn.q_membranes.iter().all(|v| *v == FixedPoint::ZERO));
        assert!(attn.k_membranes.iter().all(|v| *v == FixedPoint::ZERO));
        assert!(attn.v_membranes.iter().all(|v| *v == FixedPoint::ZERO));
        assert!(attn.o_membranes.iter().all(|v| *v == FixedPoint::ZERO));
    }
}
