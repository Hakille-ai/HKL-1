//! Retinal processing module for HKL-1.
//! Implements Difference of Gaussians (DoG) spatial contrast filtering,
//! dual ON/OFF ganglion cell pathways, and asynchronous DVS (Dynamic Vision Sensor) event encoding.

use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::io::buffers::{EncodedSpike, Modality, ingest_spike};

pub const VISION_WIDTH: usize = 32;
pub const VISION_HEIGHT: usize = 32;
pub const VISION_PIXELS: usize = VISION_WIDTH * VISION_HEIGHT;

/// Fixed 5x5 Difference of Gaussians (DoG) kernel in Q16.16
/// Outer surround negative, inner center positive
pub const DOG_KERNEL_5X5: [[FixedPoint; 5]; 5] = [
    [
        FixedPoint(-1310),
        FixedPoint(-2621),
        FixedPoint(-3932),
        FixedPoint(-2621),
        FixedPoint(-1310),
    ],
    [
        FixedPoint(-2621),
        FixedPoint(6553),
        FixedPoint(19660),
        FixedPoint(6553),
        FixedPoint(-2621),
    ],
    [
        FixedPoint(-3932),
        FixedPoint(19660),
        FixedPoint(52428),
        FixedPoint(19660),
        FixedPoint(-3932),
    ],
    [
        FixedPoint(-2621),
        FixedPoint(6553),
        FixedPoint(19660),
        FixedPoint(6553),
        FixedPoint(-2621),
    ],
    [
        FixedPoint(-1310),
        FixedPoint(-2621),
        FixedPoint(-3932),
        FixedPoint(-2621),
        FixedPoint(-1310),
    ],
];

/// Dual ON/OFF Ganglion Cell Response
#[derive(Clone, Copy)]
pub struct GanglionResponse {
    pub on_response: FixedPoint,  // Light enhancement contrast
    pub off_response: FixedPoint, // Dark enhancement contrast
}

/// Retinal Engine - processes raw frame arrays (32x32) into spatial contrast & DVS events
pub struct RetinalEngine {
    pub prev_log_frame: [FixedPoint; VISION_PIXELS],
    pub dvs_threshold: FixedPoint,
    pub base_neuron_id: NeuronId,
    pub event_count: u32,
}

impl RetinalEngine {
    pub const fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            prev_log_frame: [FixedPoint::ZERO; VISION_PIXELS],
            dvs_threshold: FixedPoint(3276), // ~0.05 contrast threshold
            base_neuron_id,
            event_count: 0,
        }
    }

    /// Compute 5x5 DoG spatial contrast at pixel location (x, y)
    pub fn compute_dog(&self, frame: &[u8; VISION_PIXELS], x: usize, y: usize) -> FixedPoint {
        let mut sum = FixedPoint::ZERO;
        for ky in 0..5 {
            for kx in 0..5 {
                let px = (x + kx).saturating_sub(2).min(VISION_WIDTH - 1);
                let py = (y + ky).saturating_sub(2).min(VISION_HEIGHT - 1);
                let idx = py * VISION_WIDTH + px;
                let val = FixedPoint::from_f32(frame[idx] as f32 / 255.0);
                sum += val * DOG_KERNEL_5X5[ky][kx];
            }
        }
        sum
    }

    /// Process a full 32x32 frame: computes ON/OFF spatial contrast & DVS temporal events
    pub fn process_frame(
        &mut self,
        frame: &[u8; VISION_PIXELS],
        timestamp: u32,
    ) -> [GanglionResponse; VISION_PIXELS] {
        let mut responses = [GanglionResponse {
            on_response: FixedPoint::ZERO,
            off_response: FixedPoint::ZERO,
        }; VISION_PIXELS];
        self.event_count = 0;

        for y in 0..VISION_HEIGHT {
            for x in 0..VISION_WIDTH {
                let idx = y * VISION_WIDTH + x;

                // 1. Spatial DoG Contrast Filtering
                let dog = self.compute_dog(frame, x, y);
                let on_val = dog.max(FixedPoint::ZERO);
                let off_val = (-dog).max(FixedPoint::ZERO);

                responses[idx] = GanglionResponse {
                    on_response: on_val,
                    off_response: off_val,
                };

                // 2. DVS Polarity Event Generator (Log-intensity difference)
                let pixel_fp = FixedPoint::from_f32(frame[idx] as f32 / 255.0);
                // Approximate log(I + 1) via FixedPoint linear/quadratic scaling
                let log_i = pixel_fp * (FixedPoint::ONE - pixel_fp * FixedPoint::from_f32(0.3));
                let delta_log = log_i - self.prev_log_frame[idx];

                if delta_log.abs() > self.dvs_threshold {
                    let neuron_idx = (self.base_neuron_id.index() + idx) % crate::MAX_NEURONS;
                    let spike = EncodedSpike {
                        neuron_id: NeuronId::new(neuron_idx as u16),
                        intensity: delta_log.abs(),
                        timestamp,
                        modality: Modality::Vision,
                    };
                    ingest_spike(spike);
                    self.event_count += 1;
                }

                self.prev_log_frame[idx] = log_i;
            }
        }

        responses
    }
}
