//! Spiking Convolutional Neural Network (S-CNN) module.
//! Implements SpikingConv2D, SpikingConv3D (spatio-temporal), and SpikingMaxPool (2x2 spatial rate reduction).

use crate::core::math::{FixedPoint, Weight};
use crate::vision::retina::{VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};

pub const CONV_KERNEL_SIZE: usize = 3;
pub const TEMPORAL_HISTORY_FRAMES: usize = 4;

/// Spiking Convolution 2D Layer with shared weights Q8.8
pub struct SpikingConv2D {
    pub kernel: [[Weight; CONV_KERNEL_SIZE]; CONV_KERNEL_SIZE],
    pub threshold: FixedPoint,
}

impl SpikingConv2D {
    pub fn new() -> Self {
        Self {
            kernel: [
                [Weight::from_f32(-0.5), Weight::from_f32(1.0), Weight::from_f32(-0.5)],
                [Weight::from_f32( 1.0), Weight::from_f32(2.0), Weight::from_f32( 1.0)],
                [Weight::from_f32(-0.5), Weight::from_f32(1.0), Weight::from_f32(-0.5)],
            ],
            threshold: FixedPoint::from_f32(0.3),
        }
    }

    /// Convolve 2D spike activation grid
    pub fn convolve_2d(
        &self,
        input_grid: &[FixedPoint; VISION_PIXELS],
    ) -> [FixedPoint; VISION_PIXELS] {
        let mut output_grid = [FixedPoint::ZERO; VISION_PIXELS];

        for y in 0..VISION_HEIGHT {
            for x in 0..VISION_WIDTH {
                let mut sum = FixedPoint::ZERO;
                for ky in 0..CONV_KERNEL_SIZE {
                    for kx in 0..CONV_KERNEL_SIZE {
                        let px = (x + kx).saturating_sub(1).min(VISION_WIDTH - 1);
                        let py = (y + ky).saturating_sub(1).min(VISION_HEIGHT - 1);
                        let idx = py * VISION_WIDTH + px;
                        let w = self.kernel[ky][kx].to_fixed();
                        sum += input_grid[idx] * w;
                    }
                }
                output_grid[y * VISION_WIDTH + x] = sum.max(FixedPoint::ZERO);
            }
        }

        output_grid
    }
}

/// Spiking Convolution 3D Layer (Spatio-Temporal)
pub struct SpikingConv3D {
    pub history: [[FixedPoint; VISION_PIXELS]; TEMPORAL_HISTORY_FRAMES],
    pub history_idx: usize,
    pub spatial_conv: SpikingConv2D,
}

impl SpikingConv3D {
    pub fn new() -> Self {
        Self {
            history: [[FixedPoint::ZERO; VISION_PIXELS]; TEMPORAL_HISTORY_FRAMES],
            history_idx: 0,
            spatial_conv: SpikingConv2D::new(),
        }
    }

    /// Ingest current temporal spike frame and compute 3D spatio-temporal convolution
    pub fn convolve_3d(
        &mut self,
        input_grid: &[FixedPoint; VISION_PIXELS],
    ) -> [FixedPoint; VISION_PIXELS] {
        self.history[self.history_idx] = *input_grid;
        self.history_idx = (self.history_idx + 1) % TEMPORAL_HISTORY_FRAMES;

        let mut output_grid = [FixedPoint::ZERO; VISION_PIXELS];

        for t in 0..TEMPORAL_HISTORY_FRAMES {
            let temporal_decay = FixedPoint::from_f32(1.0 - (t as f32 * 0.2));
            let frame_conv = self.spatial_conv.convolve_2d(&self.history[t]);

            for i in 0..VISION_PIXELS {
                output_grid[i] += frame_conv[i] * temporal_decay;
            }
        }

        output_grid
    }
}

/// Spiking Max-Pooling (2x2 spatial rate reduction: 32x32 -> 16x16)
pub struct SpikingMaxPool;

impl SpikingMaxPool {
    pub fn pool_2x2(input_grid: &[FixedPoint; VISION_PIXELS]) -> [FixedPoint; 16 * 16] {
        let mut pooled = [FixedPoint::ZERO; 16 * 16];

        for py in 0..16 {
            for px in 0..16 {
                let x0 = px * 2;
                let y0 = py * 2;

                let v00 = input_grid[y0 * VISION_WIDTH + x0];
                let v01 = input_grid[y0 * VISION_WIDTH + (x0 + 1)];
                let v10 = input_grid[(y0 + 1) * VISION_WIDTH + x0];
                let v11 = input_grid[(y0 + 1) * VISION_WIDTH + (x0 + 1)];

                pooled[py * 16 + px] = v00.max(v01).max(v10).max(v11);
            }
        }

        pooled
    }
}
