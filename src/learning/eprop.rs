//! E-prop eligibility propagation engine.

use crate::core::math::FixedPoint;
use crate::learning::surrogate::fast_sigmoid;

/// E-prop learning engine for updating eligibility traces and computing weight deltas.
pub struct EpropEngine {
    pub lr: FixedPoint,
    pub surrogate_alpha: FixedPoint,
    pub decay: FixedPoint,
}

impl EpropEngine {
    pub fn new() -> Self {
        Self {
            lr: FixedPoint::from_f32(0.001),
            surrogate_alpha: FixedPoint::from_f32(25.0),
            decay: FixedPoint::from_f32(0.95),
        }
    }

    /// Update the eligibility trace for a synapse.
    #[inline(always)]
    pub fn update_eligibility_trace(
        &self,
        trace: &mut FixedPoint,
        pre_spiked: bool,
        post_membrane: FixedPoint,
        post_threshold: FixedPoint,
        decay: FixedPoint,
    ) {
        *trace = *trace * decay;
        if pre_spiked {
            *trace = *trace + fast_sigmoid(post_membrane, post_threshold, self.surrogate_alpha);
        }
    }

    /// Compute the weight delta based on learning signal and eligibility trace.
    /// Formula: -lr * learning_signal * eligibility_trace
    #[inline(always)]
    pub fn compute_weight_delta(
        &self,
        learning_signal: FixedPoint,
        eligibility_trace: FixedPoint,
        lr: FixedPoint,
    ) -> FixedPoint {
        let neg_lr = FixedPoint::ZERO - lr;
        neg_lr * learning_signal * eligibility_trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eprop_engine_new() {
        let engine = EpropEngine::new();
        assert!((engine.lr.to_f32() - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_trace_decay() {
        let engine = EpropEngine::new();
        let mut trace = FixedPoint::from_f32(1.0);
        let decay = FixedPoint::from_f32(0.5);
        engine.update_eligibility_trace(
            &mut trace,
            false,
            FixedPoint::ZERO,
            FixedPoint::ONE,
            decay,
        );
        assert_eq!(trace.to_f32(), 0.5);
    }

    #[test]
    fn test_trace_accumulation() {
        let engine = EpropEngine::new();
        let mut trace = FixedPoint::ZERO;
        let decay = FixedPoint::from_f32(0.95);
        // pre_spiked = true, post_mem = 1.0, post_th = 1.0 -> max surrogate
        engine.update_eligibility_trace(
            &mut trace,
            true,
            FixedPoint::from_f32(1.0),
            FixedPoint::from_f32(1.0),
            decay,
        );
        assert_eq!(trace, FixedPoint::ONE);
    }

    #[test]
    fn test_weight_delta_sign() {
        let engine = EpropEngine::new();
        let learning_signal = FixedPoint::from_f32(1.0);
        let trace = FixedPoint::from_f32(1.0);
        let lr = FixedPoint::from_f32(0.1);
        let dw = engine.compute_weight_delta(learning_signal, trace, lr);
        // dw = -0.1 * 1.0 * 1.0 = -0.1
        assert!((dw.to_f32() - (-0.1)).abs() < 0.001);
    }
}
