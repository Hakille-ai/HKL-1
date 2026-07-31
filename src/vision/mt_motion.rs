//! Visual Motion Cortex MT / MSTd module.
//! Implements Reichardt Elementary Motion Detectors (EMD) and 3D optical flow vectors (Vx, Vy, Vz looming).

use crate::core::math::FixedPoint;
use crate::vision::retina::{GanglionResponse, VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DominantDirection {
    Stationary,
    Left,
    Right,
    Up,
    Down,
    LoomingApproach,
    LoomingRecede,
}

/// Global 3D Motion & Optical Flow Vector
#[derive(Clone, Copy)]
pub struct MotionVector {
    pub vx: FixedPoint, // Horizontal velocity (+Right, -Left)
    pub vy: FixedPoint, // Vertical velocity (+Down, -Up)
    pub vz: FixedPoint, // Radial expansion / Looming velocity (+Approaching, -Receding)
    pub magnitude: FixedPoint,
    pub direction: DominantDirection,
}

/// Reichardt Elementary Motion Detector Engine
pub struct MotionEngine {
    pub prev_retina_frame: [FixedPoint; VISION_PIXELS],
    pub motion_threshold: FixedPoint,
}

impl MotionEngine {
    pub fn new() -> Self {
        Self {
            prev_retina_frame: [FixedPoint::ZERO; VISION_PIXELS],
            motion_threshold: FixedPoint::from_f32(0.02),
        }
    }

    /// Process current retinal contrast map to extract optical flow velocity vectors
    pub fn process_motion(
        &mut self,
        retina_output: &[GanglionResponse; VISION_PIXELS],
    ) -> MotionVector {
        let mut sum_vx = FixedPoint::ZERO;
        let mut sum_vy = FixedPoint::ZERO;
        let mut sum_looming = FixedPoint::ZERO;

        let mut curr_frame = [FixedPoint::ZERO; VISION_PIXELS];
        for i in 0..VISION_PIXELS {
            curr_frame[i] = retina_output[i].on_response + retina_output[i].off_response;
        }

        let center_x = VISION_WIDTH as i32 / 2;
        let center_y = VISION_HEIGHT as i32 / 2;

        for y in 1..(VISION_HEIGHT - 1) {
            for x in 1..(VISION_WIDTH - 1) {
                let idx = y * VISION_WIDTH + x;

                let c_now = curr_frame[idx];

                // Reichardt EMD correlation cross-multiplication:
                // M_right = I(x, y, t) * I(x-1, y, t-1) - I(x-1, y, t) * I(x, y, t-1)
                let left_prev = self.prev_retina_frame[y * VISION_WIDTH + (x - 1)];
                let right_prev = self.prev_retina_frame[y * VISION_WIDTH + (x + 1)];
                let up_prev = self.prev_retina_frame[(y - 1) * VISION_WIDTH + x];
                let down_prev = self.prev_retina_frame[(y + 1) * VISION_WIDTH + x];

                let emd_right = c_now * left_prev - curr_frame[y * VISION_WIDTH + (x - 1)] * self.prev_retina_frame[idx];
                let emd_left = c_now * right_prev - curr_frame[y * VISION_WIDTH + (x + 1)] * self.prev_retina_frame[idx];
                let emd_down = c_now * up_prev - curr_frame[(y - 1) * VISION_WIDTH + x] * self.prev_retina_frame[idx];
                let emd_up = c_now * down_prev - curr_frame[(y + 1) * VISION_WIDTH + x] * self.prev_retina_frame[idx];

                let vx_local = emd_right - emd_left;
                let vy_local = emd_down - emd_up;

                sum_vx += vx_local;
                sum_vy += vy_local;

                // Radial Expansion / Looming (MSTd component):
                // Objects moving towards the viewer expand outwards from center (r_vec dot v_vec > 0)
                let dx = FixedPoint::from_int(x as i32 - center_x);
                let dy = FixedPoint::from_int(y as i32 - center_y);
                let radial_flow = dx * vx_local + dy * vy_local;
                sum_looming += radial_flow;
            }
        }

        self.prev_retina_frame = curr_frame;

        let scale = FixedPoint::from_f32(1.0 / (VISION_PIXELS as f32));
        let vx = sum_vx * scale;
        let vy = sum_vy * scale;
        let vz = sum_looming * scale * FixedPoint::from_f32(0.1);

        let magnitude = (vx * vx + vy * vy + vz * vz).sqrt();

        let direction = if magnitude < self.motion_threshold {
            DominantDirection::Stationary
        } else if vz.abs() > vx.abs() && vz.abs() > vy.abs() {
            if vz > FixedPoint::ZERO {
                DominantDirection::LoomingApproach
            } else {
                DominantDirection::LoomingRecede
            }
        } else if vx.abs() > vy.abs() {
            if vx > FixedPoint::ZERO {
                DominantDirection::Right
            } else {
                DominantDirection::Left
            }
        } else if vy > FixedPoint::ZERO {
            DominantDirection::Down
        } else {
            DominantDirection::Up
        };

        MotionVector {
            vx,
            vy,
            vz,
            magnitude,
            direction,
        }
    }
}
