//! Surrogate gradients for spiking neural networks
//! Non-differentiable spike threshold approximations for e-prop.

use crate::core::math::FixedPoint;

/// Fast sigmoid surrogate gradient
/// Formula: 1 / (1 + alpha * |u - threshold|)^2
#[inline(always)]
pub fn fast_sigmoid(u: FixedPoint, threshold: FixedPoint, alpha: FixedPoint) -> FixedPoint {
    let diff = (u - threshold).abs();
    let denom_part = FixedPoint::ONE + (alpha * diff);
    let denom = denom_part * denom_part;
    FixedPoint::ONE / denom
}

/// Arctangent surrogate gradient (ATan)
/// Formula: alpha / (1 + (beta * (u - threshold))^2)
#[inline(always)]
pub fn atan_surrogate(
    u: FixedPoint,
    threshold: FixedPoint,
    alpha: FixedPoint,
    beta: FixedPoint,
) -> FixedPoint {
    let diff = u - threshold;
    let beta_diff = beta * diff;
    let squared = beta_diff * beta_diff;
    let denom = FixedPoint::ONE + squared;
    alpha / denom
}

/// Straight-through estimator
/// Returns 1.0 if |u - threshold| < 0.5, else 0.0
#[inline(always)]
pub fn straight_through(u: FixedPoint, threshold: FixedPoint) -> FixedPoint {
    let diff = (u - threshold).abs();
    if diff < FixedPoint::HALF {
        FixedPoint::ONE
    } else {
        FixedPoint::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_sigmoid() {
        let alpha = FixedPoint::from_f32(10.0);
        let threshold = FixedPoint::from_f32(1.0);

        // Exact match
        let u1 = FixedPoint::from_f32(1.0);
        assert_eq!(fast_sigmoid(u1, threshold, alpha), FixedPoint::ONE);

        // Slightly off
        let u2 = FixedPoint::from_f32(1.1);
        // diff = 0.1, alpha = 10, denom_part = 1 + 10*0.1 = 2, denom = 4, res = 0.25
        assert!((fast_sigmoid(u2, threshold, alpha).to_f32() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_atan_surrogate() {
        let alpha = FixedPoint::from_f32(0.5);
        let beta = FixedPoint::from_f32(1.0);
        let threshold = FixedPoint::from_f32(1.0);

        let u1 = FixedPoint::from_f32(1.0);
        assert_eq!(atan_surrogate(u1, threshold, alpha, beta).to_f32(), 0.5);
    }

    #[test]
    fn test_straight_through() {
        let threshold = FixedPoint::from_f32(1.0);
        assert_eq!(
            straight_through(FixedPoint::from_f32(1.0), threshold),
            FixedPoint::ONE
        );
        assert_eq!(
            straight_through(FixedPoint::from_f32(1.49), threshold),
            FixedPoint::ONE
        );
        assert_eq!(
            straight_through(FixedPoint::from_f32(1.5), threshold),
            FixedPoint::ZERO
        );
        assert_eq!(
            straight_through(FixedPoint::from_f32(0.5), threshold),
            FixedPoint::ZERO
        );
    }
}
