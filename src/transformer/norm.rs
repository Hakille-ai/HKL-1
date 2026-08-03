use crate::core::math::FixedPoint;

pub const MAX_DIM: usize = 256;

/// Simple Layer Normalization
pub struct LayerNorm {
    pub gamma: [FixedPoint; MAX_DIM], // scale
    pub beta: [FixedPoint; MAX_DIM],  // shift
    pub dim: usize,
    pub epsilon: FixedPoint,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        let gamma = [FixedPoint::ONE; MAX_DIM];
        let beta = [FixedPoint::ZERO; MAX_DIM];
        Self {
            gamma,
            beta,
            dim: dim.min(MAX_DIM),
            epsilon: FixedPoint::from_f32(1e-5),
        }
    }

    pub fn forward(&self, x: &mut [FixedPoint]) {
        let active_dim = self.dim.min(x.len()).min(MAX_DIM);
        if active_dim == 0 {
            return;
        }

        let mut sum = FixedPoint::ZERO;
        for i in 0..active_dim {
            sum = sum + x[i];
        }
        let mean = sum / FixedPoint::from_int(active_dim as i32);

        let mut var_sum = FixedPoint::ZERO;
        for i in 0..active_dim {
            let diff = x[i] - mean;
            var_sum = var_sum + diff * diff;
        }
        let var = var_sum / FixedPoint::from_int(active_dim as i32);
        let std_dev = (var + self.epsilon).sqrt();

        for i in 0..active_dim {
            let normalized = (x[i] - mean) / std_dev;
            x[i] = normalized * self.gamma[i] + self.beta[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4);
        let mut x = [
            FixedPoint::from_f32(1.0),
            FixedPoint::from_f32(2.0),
            FixedPoint::from_f32(3.0),
            FixedPoint::from_f32(4.0),
        ];
        ln.forward(&mut x);

        let mut sum = FixedPoint::ZERO;
        for v in x.iter() {
            sum = sum + *v;
        }
        let mean = sum / FixedPoint::from_int(4);
        assert!(mean.to_f32().abs() < 0.1);

        let mut var_sum = FixedPoint::ZERO;
        for v in x.iter() {
            let diff = *v - mean;
            var_sum = var_sum + diff * diff;
        }
        let var = var_sum / FixedPoint::from_int(4);
        assert!((var.to_f32() - 1.0).abs() < 0.1);
    }

    #[test]
    fn layer_norm_new_clamps_oversized_dimension() {
        let ln = LayerNorm::new(MAX_DIM + 8);

        assert_eq!(ln.dim, MAX_DIM);
    }

    #[test]
    fn layer_norm_forward_handles_shorter_slice() {
        let ln = LayerNorm::new(4);
        let mut x = [FixedPoint::from_f32(1.0), FixedPoint::from_f32(3.0)];

        ln.forward(&mut x);

        let mean = (x[0] + x[1]) / FixedPoint::from_int(2);
        assert!(mean.to_f32().abs() < 0.1);
    }

    #[test]
    fn layer_norm_forward_ignores_extra_tail() {
        let ln = LayerNorm::new(2);
        let tail = FixedPoint::from_f32(9.0);
        let mut x = [FixedPoint::from_f32(1.0), FixedPoint::from_f32(3.0), tail];

        ln.forward(&mut x);

        assert_eq!(x[2], tail);
    }

    #[test]
    fn layer_norm_forward_handles_empty_slice() {
        let ln = LayerNorm::new(4);
        let mut x: [FixedPoint; 0] = [];

        ln.forward(&mut x);

        assert!(x.is_empty());
    }
}
