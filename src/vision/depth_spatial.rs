//! Spatial Depth & Stereo Parallax mapping module.
//! Computes binocular disparity maps Z = (f * B) / d and 3D spatial coordinates (X, Y, Z).

use crate::core::math::FixedPoint;
use crate::vision::retina::{VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};

/// Spatial 3D Coordinate in centimeters relative to agent
#[derive(Clone, Copy)]
pub struct SpatialPoint3D {
    pub x: FixedPoint, // Lateral offset
    pub y: FixedPoint, // Vertical offset
    pub z: FixedPoint, // Depth distance
}

/// Depth Engine for binocular stereo parallax and spatial 3D mapping
pub struct DepthEngine {
    pub focal_length: FixedPoint, // Virtual focal length (f)
    pub baseline: FixedPoint,     // Interpupillary baseline (B) in cm
    pub max_disparity: usize,
}

impl DepthEngine {
    pub fn new() -> Self {
        Self {
            focal_length: FixedPoint::from_f32(50.0), // f = 50
            baseline: FixedPoint::from_f32(6.5),      // B = 6.5 cm
            max_disparity: 8,
        }
    }

    /// Compute 3D Depth Map Z(x, y) from Left and Right stereo frame arrays
    pub fn compute_depth_map(
        &self,
        left_frame: &[u8; VISION_PIXELS],
        right_frame: &[u8; VISION_PIXELS],
    ) -> [SpatialPoint3D; VISION_PIXELS] {
        let mut depth_map = [SpatialPoint3D {
            x: FixedPoint::ZERO,
            y: FixedPoint::ZERO,
            z: FixedPoint::from_f32(500.0), // Default far depth
        }; VISION_PIXELS];

        for y in 0..VISION_HEIGHT {
            for x in 0..VISION_WIDTH {
                let idx = y * VISION_WIDTH + x;

                let left_val = left_frame[idx] as i32;
                let mut best_disp = 0;
                let mut min_diff = i32::MAX;

                // Match left pixel with shifted right pixel across max_disparity window
                for d in 0..self.max_disparity {
                    if x >= d {
                        let right_idx = y * VISION_WIDTH + (x - d);
                        let right_val = right_frame[right_idx] as i32;
                        let diff = (left_val - right_val).abs();
                        if diff < min_diff {
                            min_diff = diff;
                            best_disp = d;
                        }
                    }
                }

                // Stereo Parallax Z = (f * B) / d
                let disp_fp = FixedPoint::from_int(best_disp as i32 + 1);
                let depth_z = (self.focal_length * self.baseline).div(disp_fp);

                // Convert pixel (x, y) to spatial 3D coordinates (X, Y, Z) in cm
                let spatial_x = FixedPoint::from_int(x as i32 - 16) * depth_z / self.focal_length;
                let spatial_y = FixedPoint::from_int(y as i32 - 16) * depth_z / self.focal_length;

                depth_map[idx] = SpatialPoint3D {
                    x: spatial_x,
                    y: spatial_y,
                    z: depth_z,
                };
            }
        }

        depth_map
    }
}
