//! Loss functions for spiking neural networks

use crate::core::math::FixedPoint;

/// Spiking Cross-Entropy Loss
pub struct SpikingCrossEntropyLoss;

impl SpikingCrossEntropyLoss {
    /// Smallest representable positive firing-rate floor used for numerical
    /// smoothing. Silent/random initial models should receive a finite loss so
    /// guarded training can observe and throttle them instead of treating every
    /// cold-start batch as corrupted input.
    const EPSILON_RATE: FixedPoint = FixedPoint::from_bits(1);

    /// Compute simplified cross-entropy loss over non-negative firing rates.
    pub fn compute_loss(
        output_rates: &[FixedPoint],
        target_idx: usize,
        vocab_size: usize,
    ) -> FixedPoint {
        let active_len = core::cmp::min(output_rates.len(), vocab_size);
        if active_len == 0 || target_idx >= active_len {
            return FixedPoint::MAX;
        }

        let target_rate = output_rates[target_idx]
            .max(FixedPoint::ZERO)
            .max(Self::EPSILON_RATE);

        let mut sum = FixedPoint::ZERO;
        for rate in output_rates.iter().take(active_len) {
            sum = sum + (*rate).max(FixedPoint::ZERO).max(Self::EPSILON_RATE);
        }

        if sum == FixedPoint::ZERO {
            return FixedPoint::MAX;
        }

        let p = (target_rate / sum).max(Self::EPSILON_RATE);

        let loss = -p.ln();
        if loss < FixedPoint::ZERO {
            FixedPoint::ZERO
        } else {
            loss
        }
    }

    /// Compute learning signals: output_rates[i] - (1 if i == target else 0)
    pub fn compute_learning_signals(
        output_rates: &[FixedPoint],
        target_idx: usize,
        signals: &mut [FixedPoint],
    ) {
        let len = core::cmp::min(output_rates.len(), signals.len());
        for (i, rate) in output_rates.iter().take(len).enumerate() {
            let target = if i == target_idx {
                FixedPoint::ONE
            } else {
                FixedPoint::ZERO
            };
            signals[i] = (*rate).max(FixedPoint::ZERO) - target;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_loss() {
        let rates = [
            FixedPoint::from_f32(0.1),
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.1),
        ];

        let loss = SpikingCrossEntropyLoss::compute_loss(&rates, 1, 3);
        // p = 0.8
        // -ln(0.8) approx 0.223
        assert!(loss > FixedPoint::ZERO);
        assert!(loss < FixedPoint::from_f32(0.3));
    }

    #[test]
    fn test_compute_learning_signals() {
        let rates = [
            FixedPoint::from_f32(0.2),
            FixedPoint::from_f32(0.7),
            FixedPoint::from_f32(0.1),
        ];
        let mut signals = [FixedPoint::ZERO; 3];

        SpikingCrossEntropyLoss::compute_learning_signals(&rates, 1, &mut signals);

        assert!((signals[0].to_f32() - 0.2).abs() < 0.001);
        assert!((signals[1].to_f32() - (-0.3)).abs() < 0.001); // 0.7 - 1.0
        assert!((signals[2].to_f32() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_compute_loss_rejects_out_of_range_target() {
        let rates = [FixedPoint::ONE, FixedPoint::ZERO];

        assert_eq!(
            SpikingCrossEntropyLoss::compute_loss(&rates, 2, rates.len()),
            FixedPoint::MAX
        );
        assert_eq!(
            SpikingCrossEntropyLoss::compute_loss(&rates, 1, 1),
            FixedPoint::MAX
        );
    }

    #[test]
    fn test_compute_loss_ignores_negative_rates() {
        let rates = [
            FixedPoint::from_f32(-4.0),
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.2),
        ];

        let loss = SpikingCrossEntropyLoss::compute_loss(&rates, 1, rates.len());

        assert!(loss > FixedPoint::ZERO);
        assert!(loss < FixedPoint::from_f32(0.3));
        let smoothed_loss = SpikingCrossEntropyLoss::compute_loss(&rates, 0, rates.len());
        assert!(smoothed_loss > FixedPoint::from_f32(10.0));
        assert!(smoothed_loss < FixedPoint::MAX);
    }

    #[test]
    fn test_compute_loss_smooths_silent_outputs() {
        let rates = [FixedPoint::ZERO; 4];

        let loss = SpikingCrossEntropyLoss::compute_loss(&rates, 2, rates.len());

        assert!(loss > FixedPoint::ZERO);
        assert!(loss < FixedPoint::MAX);
        assert!(loss < FixedPoint::from_f32(2.0));
    }

    #[test]
    fn test_compute_learning_signals_truncates_to_output_slice() {
        let rates = [FixedPoint::from_f32(0.2), FixedPoint::from_f32(0.7)];
        let mut signals = [FixedPoint::ZERO; 1];

        SpikingCrossEntropyLoss::compute_learning_signals(&rates, 1, &mut signals);

        assert!((signals[0].to_f32() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_compute_learning_signals_clamps_negative_rates() {
        let rates = [FixedPoint::from_f32(-0.5), FixedPoint::from_f32(0.7)];
        let mut signals = [FixedPoint::ZERO; 2];

        SpikingCrossEntropyLoss::compute_learning_signals(&rates, 0, &mut signals);

        assert_eq!(signals[0], -FixedPoint::ONE);
        assert_eq!(signals[1], FixedPoint::from_f32(0.7));
    }
}
