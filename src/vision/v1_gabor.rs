//! Primary Visual Cortex V1 orientation selectivity module.
//! Implements multi-angle 2D Gabor kernels (0°, 45°, 90°, 135°) to extract directional edges and contours.

use crate::core::math::FixedPoint;
use crate::vision::retina::{GanglionResponse, VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};

pub const GABOR_ORIENTATIONS: usize = 4; // 0° (Horiz), 45° (Diag Asc), 90° (Vert), 135° (Diag Desc)

/// Pre-computed 5x5 Gabor Filter Kernels in Q16.16
/// Oriented at 0°, 45°, 90°, and 135°
pub const GABOR_KERNELS_5X5: [[[FixedPoint; 5]; 5]; GABOR_ORIENTATIONS] = [
    // 0° — Horizontal Edges
    [
        [FixedPoint(-6554), FixedPoint(-13107), FixedPoint(-19661), FixedPoint(-13107), FixedPoint(-6554)],
        [FixedPoint(     0), FixedPoint(      0), FixedPoint(      0), FixedPoint(      0), FixedPoint(     0)],
        [FixedPoint( 13107), FixedPoint( 26214), FixedPoint( 39322), FixedPoint( 26214), FixedPoint( 13107)],
        [FixedPoint(     0), FixedPoint(      0), FixedPoint(      0), FixedPoint(      0), FixedPoint(     0)],
        [FixedPoint(-6554), FixedPoint(-13107), FixedPoint(-19661), FixedPoint(-13107), FixedPoint(-6554)],
    ],
    // 45° — Diagonal Ascending Edges
    [
        [FixedPoint(-19661), FixedPoint(-13107), FixedPoint(     0), FixedPoint( 13107), FixedPoint( 39322)],
        [FixedPoint(-13107), FixedPoint(     0), FixedPoint( 26214), FixedPoint( 39322), FixedPoint( 13107)],
        [FixedPoint(     0), FixedPoint( 26214), FixedPoint( 39322), FixedPoint( 26214), FixedPoint(     0)],
        [FixedPoint( 13107), FixedPoint( 39322), FixedPoint( 26214), FixedPoint(     0), FixedPoint(-13107)],
        [FixedPoint( 39322), FixedPoint( 13107), FixedPoint(     0), FixedPoint(-13107), FixedPoint(-19661)],
    ],
    // 90° — Vertical Edges
    [
        [FixedPoint(-6554), FixedPoint(    0), FixedPoint( 13107), FixedPoint(    0), FixedPoint(-6554)],
        [FixedPoint(-13107), FixedPoint(   0), FixedPoint( 26214), FixedPoint(   0), FixedPoint(-13107)],
        [FixedPoint(-19661), FixedPoint(   0), FixedPoint( 39322), FixedPoint(   0), FixedPoint(-19661)],
        [FixedPoint(-13107), FixedPoint(   0), FixedPoint( 26214), FixedPoint(   0), FixedPoint(-13107)],
        [FixedPoint(-6554), FixedPoint(    0), FixedPoint( 13107), FixedPoint(    0), FixedPoint(-6554)],
    ],
    // 135° — Diagonal Descending Edges
    [
        [FixedPoint( 39322), FixedPoint( 13107), FixedPoint(     0), FixedPoint(-13107), FixedPoint(-19661)],
        [FixedPoint( 13107), FixedPoint( 39322), FixedPoint( 26214), FixedPoint(     0), FixedPoint(-13107)],
        [FixedPoint(     0), FixedPoint( 26214), FixedPoint( 39322), FixedPoint( 26214), FixedPoint(     0)],
        [FixedPoint(-13107), FixedPoint(     0), FixedPoint( 26214), FixedPoint( 39322), FixedPoint( 13107)],
        [FixedPoint(-19661), FixedPoint(-13107), FixedPoint(     0), FixedPoint( 13107), FixedPoint( 39322)],
    ],
];

/// V1 Orientation Selective Map per pixel
#[derive(Clone, Copy)]
pub struct V1OrientationResponse {
    pub responses: [FixedPoint; GABOR_ORIENTATIONS],
    pub dominant_orientation: u8, // 0..3
    pub dominant_energy: FixedPoint,
}

/// Gabor Bank for Cortex V1 Orientation Hypercolumns
pub struct GaborBank;

impl GaborBank {
    /// Convolve retinal ON/OFF response with 4 Gabor orientation kernels
    pub fn process_retina(
        retina_output: &[GanglionResponse; VISION_PIXELS],
    ) -> [V1OrientationResponse; VISION_PIXELS] {
        let mut v1_map = [V1OrientationResponse {
            responses: [FixedPoint::ZERO; GABOR_ORIENTATIONS],
            dominant_orientation: 0,
            dominant_energy: FixedPoint::ZERO,
        }; VISION_PIXELS];

        for y in 0..VISION_HEIGHT {
            for x in 0..VISION_WIDTH {
                let idx = y * VISION_WIDTH + x;
                let mut max_energy = FixedPoint::ZERO;
                let mut best_orient = 0u8;

                for angle in 0..GABOR_ORIENTATIONS {
                    let mut sum = FixedPoint::ZERO;
                    let kernel = &GABOR_KERNELS_5X5[angle];

                    for ky in 0..5 {
                        for kx in 0..5 {
                            let px = (x + kx).saturating_sub(2).min(VISION_WIDTH - 1);
                            let py = (y + ky).saturating_sub(2).min(VISION_HEIGHT - 1);
                            let pidx = py * VISION_WIDTH + px;
                            // Combined ON + OFF retinal contrast
                            let contrast = retina_output[pidx].on_response + retina_output[pidx].off_response;
                            sum += contrast * kernel[ky][kx];
                        }
                    }

                    let energy = sum.abs();
                    v1_map[idx].responses[angle] = energy;

                    if energy > max_energy {
                        max_energy = energy;
                        best_orient = angle as u8;
                    }
                }

                v1_map[idx].dominant_orientation = best_orient;
                v1_map[idx].dominant_energy = max_energy;
            }
        }

        v1_map
    }
}
