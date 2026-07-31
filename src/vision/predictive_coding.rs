//! Visual Predictive Coding module (Friston Free Energy principle).
//! Generates top-down predicted visual frame I_pred based on physical dynamics,
//! computes visual prediction error E_vis = |I_actual - I_pred|, and couples error to attention/reflexes.

use crate::core::math::FixedPoint;
use crate::vision::retina::{VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};

/// Visual Predictive Coding Engine
pub struct VisualPredictiveCoding {
    pub predicted_frame: [FixedPoint; VISION_PIXELS],
    pub error_map: [FixedPoint; VISION_PIXELS],
    pub mean_prediction_error: FixedPoint,
    pub prev_mean_error: FixedPoint,
    pub visual_novelty: FixedPoint,
}

impl VisualPredictiveCoding {
    pub fn new() -> Self {
        Self {
            predicted_frame: [FixedPoint::ZERO; VISION_PIXELS],
            error_map: [FixedPoint::ZERO; VISION_PIXELS],
            mean_prediction_error: FixedPoint::ZERO,
            prev_mean_error: FixedPoint::ZERO,
            visual_novelty: FixedPoint::ZERO,
        }
    }

    /// Predict next visual frame I_pred based on motion & physics displacement
    pub fn predict_next_frame(&mut self, current_frame: &[u8; VISION_PIXELS], dx: i32, dy: i32) {
        for y in 0..VISION_HEIGHT {
            for x in 0..VISION_WIDTH {
                let p_x = (x as i32 - dx).clamp(0, VISION_WIDTH as i32 - 1) as usize;
                let p_y = (y as i32 - dy).clamp(0, VISION_HEIGHT as i32 - 1) as usize;
                let src_idx = p_y * VISION_WIDTH + p_x;
                let dst_idx = y * VISION_WIDTH + x;

                self.predicted_frame[dst_idx] = FixedPoint::from_f32(current_frame[src_idx] as f32 / 255.0);
            }
        }
    }

    /// Compute Visual Prediction Error E_vis = |I_actual - I_pred|
    pub fn compute_prediction_error(&mut self, actual_frame: &[u8; VISION_PIXELS]) -> FixedPoint {
        let mut total_error = FixedPoint::ZERO;

        for i in 0..VISION_PIXELS {
            let actual = FixedPoint::from_f32(actual_frame[i] as f32 / 255.0);
            let err = (actual - self.predicted_frame[i]).abs();
            self.error_map[i] = err;
            total_error += err;
        }

        let mean_err = total_error * FixedPoint::from_f32(1.0 / (VISION_PIXELS as f32));

        // Compute Visual Novelty as absolute derivative of visual prediction error
        self.visual_novelty = (mean_err - self.prev_mean_error).abs();
        self.prev_mean_error = self.mean_prediction_error;
        self.mean_prediction_error = mean_err;

        mean_err
    }
}
