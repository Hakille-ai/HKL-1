//! Visual Cortex V4 & Inferotemporal (IT) Cortex module.
//! Extracts curvature, shape boundaries, corners, and online Hebbian visual object prototypes.

use crate::core::math::FixedPoint;
use crate::vision::retina::{VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};
use crate::vision::v1_gabor::V1OrientationResponse;

pub const MAX_VISUAL_PROTOTYPES: usize = 16;

/// Object shape features extracted from V4
#[derive(Clone, Copy)]
pub struct ShapeFeature {
    pub corner_density: FixedPoint,
    pub horizontal_bias: FixedPoint,
    pub vertical_bias: FixedPoint,
    pub diagonal_bias: FixedPoint,
    pub total_edge_energy: FixedPoint,
}

/// Learned Visual Object Prototype in IT Cortex
#[derive(Clone, Copy)]
pub struct VisualObjectPrototype {
    pub feature: ShapeFeature,
    pub confidence: FixedPoint,
    pub count: u32,
    pub valid: bool,
}

impl VisualObjectPrototype {
    pub const fn empty() -> Self {
        Self {
            feature: ShapeFeature {
                corner_density: FixedPoint::ZERO,
                horizontal_bias: FixedPoint::ZERO,
                vertical_bias: FixedPoint::ZERO,
                diagonal_bias: FixedPoint::ZERO,
                total_edge_energy: FixedPoint::ZERO,
            },
            confidence: FixedPoint::ZERO,
            count: 0,
            valid: false,
        }
    }

    /// Distance between prototype shape and observed shape feature
    pub fn distance(&self, feat: &ShapeFeature) -> FixedPoint {
        let d1 = (self.feature.corner_density - feat.corner_density).abs();
        let d2 = (self.feature.horizontal_bias - feat.horizontal_bias).abs();
        let d3 = (self.feature.vertical_bias - feat.vertical_bias).abs();
        let d4 = (self.feature.diagonal_bias - feat.diagonal_bias).abs();
        d1 + d2 + d3 + d4
    }

    /// Online Hebbian merge update
    pub fn merge(&mut self, feat: &ShapeFeature, lr: FixedPoint) {
        if !self.valid {
            self.feature = *feat;
            self.confidence = FixedPoint::from_f32(0.5);
            self.count = 1;
            self.valid = true;
            return;
        }

        let one_minus_lr = FixedPoint::ONE - lr;
        self.feature.corner_density =
            self.feature.corner_density * one_minus_lr + feat.corner_density * lr;
        self.feature.horizontal_bias =
            self.feature.horizontal_bias * one_minus_lr + feat.horizontal_bias * lr;
        self.feature.vertical_bias =
            self.feature.vertical_bias * one_minus_lr + feat.vertical_bias * lr;
        self.feature.diagonal_bias =
            self.feature.diagonal_bias * one_minus_lr + feat.diagonal_bias * lr;
        self.feature.total_edge_energy =
            self.feature.total_edge_energy * one_minus_lr + feat.total_edge_energy * lr;

        self.count += 1;
        self.confidence = (self.confidence + FixedPoint::from_f32(0.05)).min(FixedPoint::ONE);
    }
}

/// Cortex V4 / IT Object Recognizer
pub struct ShapeEngine {
    pub prototypes: [VisualObjectPrototype; MAX_VISUAL_PROTOTYPES],
    pub learning_rate: FixedPoint,
}

impl ShapeEngine {
    pub fn new() -> Self {
        Self {
            prototypes: [VisualObjectPrototype::empty(); MAX_VISUAL_PROTOTYPES],
            learning_rate: FixedPoint::from_f32(0.1),
        }
    }

    /// Extract shape features from V1 orientation map
    pub fn extract_shape(&self, v1_map: &[V1OrientationResponse; VISION_PIXELS]) -> ShapeFeature {
        let mut horiz_energy = FixedPoint::ZERO;
        let mut vert_energy = FixedPoint::ZERO;
        let mut diag_energy = FixedPoint::ZERO;
        let mut total_energy = FixedPoint::ZERO;
        let mut corners = 0u32;

        for y in 1..(VISION_HEIGHT - 1) {
            for x in 1..(VISION_WIDTH - 1) {
                let idx = y * VISION_WIDTH + x;
                let v1 = &v1_map[idx];

                horiz_energy += v1.responses[0];
                diag_energy += v1.responses[1] + v1.responses[3];
                vert_energy += v1.responses[2];
                total_energy += v1.dominant_energy;

                // Corner detection: orthogonal edge overlap (horizontal + vertical active)
                if v1.responses[0] > FixedPoint::from_f32(0.2)
                    && v1.responses[2] > FixedPoint::from_f32(0.2)
                {
                    corners += 1;
                }
            }
        }

        let inv_total = if total_energy > FixedPoint::ZERO {
            FixedPoint::ONE.div(total_energy)
        } else {
            FixedPoint::ZERO
        };

        ShapeFeature {
            corner_density: FixedPoint::from_f32(corners as f32 / VISION_PIXELS as f32),
            horizontal_bias: horiz_energy * inv_total,
            vertical_bias: vert_energy * inv_total,
            diagonal_bias: diag_energy * inv_total,
            total_edge_energy: total_energy,
        }
    }

    /// Recognize or learn object prototype in IT cortex
    pub fn process_visual_object(&mut self, feat: &ShapeFeature) -> (usize, FixedPoint) {
        let mut best_idx = 0;
        let mut min_dist = FixedPoint::MAX;

        // Find matching prototype
        for i in 0..MAX_VISUAL_PROTOTYPES {
            if self.prototypes[i].valid {
                let dist = self.prototypes[i].distance(feat);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = i;
                }
            }
        }

        let match_threshold = FixedPoint::from_f32(0.3);
        if min_dist <= match_threshold {
            // Update existing object prototype
            self.prototypes[best_idx].merge(feat, self.learning_rate);
            (best_idx, self.prototypes[best_idx].confidence)
        } else {
            // Find empty slot for new object prototype
            for i in 0..MAX_VISUAL_PROTOTYPES {
                if !self.prototypes[i].valid {
                    self.prototypes[i].merge(feat, self.learning_rate);
                    return (i, self.prototypes[i].confidence);
                }
            }
            // Overwrite worst if full
            self.prototypes[best_idx].merge(feat, self.learning_rate);
            (best_idx, self.prototypes[best_idx].confidence)
        }
    }
}
