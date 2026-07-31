//! Visual & Spatial Intelligence Module for HKL-1.
//! Integrates retinal processing (DoG, DVS), V1 Gabor orientation filters, V4 shape prototypes,
//! MT optical flow, 3D stereo depth, Intuitive Physics Engine (trajectory, gravity, collisions, occlusion),
//! visual predictive coding, and spiking convolutions (S-CNN).

pub mod conv;
pub mod depth_spatial;
pub mod mt_motion;
pub mod physics_engine;
pub mod predictive_coding;
pub mod retina;
pub mod v1_gabor;
pub mod v4_shape;

use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use conv::{SpikingConv2D, SpikingConv3D};
use depth_spatial::{DepthEngine, SpatialPoint3D};
use mt_motion::{MotionEngine, MotionVector};
use physics_engine::IntuitivePhysicsEngine;
use predictive_coding::VisualPredictiveCoding;
use retina::{GanglionResponse, RetinalEngine, VISION_PIXELS};
use v1_gabor::{GaborBank, V1OrientationResponse};
use v4_shape::{ShapeEngine, ShapeFeature};

/// Unified Visual & Physical Intelligence Engine
pub struct VisualEngine {
    pub retina: RetinalEngine,
    pub shape_engine: ShapeEngine,
    pub motion_engine: MotionEngine,
    pub depth_engine: DepthEngine,
    pub physics_engine: IntuitivePhysicsEngine,
    pub predictive_coding: VisualPredictiveCoding,
    pub conv2d: SpikingConv2D,
    pub conv3d: SpikingConv3D,
    pub last_motion: MotionVector,
    pub last_shape: ShapeFeature,
    pub last_recognized_object: (usize, FixedPoint),
}

impl VisualEngine {
    pub fn new() -> Self {
        Self {
            retina: RetinalEngine::new(NeuronId::new(0)),
            shape_engine: ShapeEngine::new(),
            motion_engine: MotionEngine::new(),
            depth_engine: DepthEngine::new(),
            physics_engine: IntuitivePhysicsEngine::new(),
            predictive_coding: VisualPredictiveCoding::new(),
            conv2d: SpikingConv2D::new(),
            conv3d: SpikingConv3D::new(),
            last_motion: MotionVector {
                vx: FixedPoint::ZERO,
                vy: FixedPoint::ZERO,
                vz: FixedPoint::ZERO,
                magnitude: FixedPoint::ZERO,
                direction: mt_motion::DominantDirection::Stationary,
            },
            last_shape: ShapeFeature {
                corner_density: FixedPoint::ZERO,
                horizontal_bias: FixedPoint::ZERO,
                vertical_bias: FixedPoint::ZERO,
                diagonal_bias: FixedPoint::ZERO,
                total_edge_energy: FixedPoint::ZERO,
            },
            last_recognized_object: (0, FixedPoint::ZERO),
        }
    }

    /// Process input video frame, compute 3D dynamics, intuitive physics, and predictive coding
    pub fn process_visual_scene(
        &mut self,
        frame: &[u8; VISION_PIXELS],
        timestamp: u32,
        dt_ms: u32,
    ) -> (
        [GanglionResponse; VISION_PIXELS],
        [V1OrientationResponse; VISION_PIXELS],
        MotionVector,
        FixedPoint, // Visual Prediction Error
    ) {
        // 1. Retinal DoG Filtering & DVS Polarity Event Generation
        let ganglion_responses = self.retina.process_frame(frame, timestamp);

        // 2. Cortex V1 Gabor Orientation Hypercolumns
        let v1_responses = GaborBank::process_retina(&ganglion_responses);

        // 3. Cortex V4/IT Shape & Object Recognition
        let shape_feat = self.shape_engine.extract_shape(&v1_responses);
        let recognized_obj = self.shape_engine.process_visual_object(&shape_feat);
        self.last_shape = shape_feat;
        self.last_recognized_object = recognized_obj;

        // 4. Cortex MT/MST Optical Flow & Looming Motion
        let motion_vec = self.motion_engine.process_motion(&ganglion_responses);
        self.last_motion = motion_vec;

        // 5. Intuitive Physics Engine (Trajectory Extrapolation & Collision Forecast)
        let observed_3d_pos = SpatialPoint3D {
            x: motion_vec.vx * FixedPoint::from_int(100),
            y: motion_vec.vy * FixedPoint::from_int(100),
            z: FixedPoint::from_f32(100.0) - motion_vec.vz * FixedPoint::from_int(100),
        };
        self.physics_engine.track_object(
            recognized_obj.0,
            observed_3d_pos,
            &motion_vec,
            true, // Visible
            dt_ms,
        );
        self.physics_engine.step(dt_ms);

        // 6. Visual Predictive Coding (Frame prediction & E_vis calculation)
        let dx = (motion_vec.vx * FixedPoint::from_int(10)).to_int();
        let dy = (motion_vec.vy * FixedPoint::from_int(10)).to_int();
        self.predictive_coding.predict_next_frame(frame, dx, dy);
        let prediction_error = self.predictive_coding.compute_prediction_error(frame);

        // 7. Spiking Convolution S-Conv2D & S-Conv3D
        let mut conv_input = [FixedPoint::ZERO; VISION_PIXELS];
        for i in 0..VISION_PIXELS {
            conv_input[i] = v1_responses[i].dominant_energy;
        }
        let _conv2d_out = self.conv2d.convolve_2d(&conv_input);
        let _conv3d_out = self.conv3d.convolve_3d(&conv_input);

        (ganglion_responses, v1_responses, motion_vec, prediction_error)
    }
}

// ---------------------------------------------------------------------------
// Global Instance
// ---------------------------------------------------------------------------
use core::mem::MaybeUninit;

pub static mut VISUAL_ENGINE: MaybeUninit<VisualEngine> = MaybeUninit::uninit();

static INITIALIZED_VISUAL_ENGINE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_visual_engine() {
    unsafe {
        if !INITIALIZED_VISUAL_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            VISUAL_ENGINE.write(VisualEngine::new());
            INITIALIZED_VISUAL_ENGINE.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn visual_engine() -> &'static mut VisualEngine {
    unsafe {
        if !INITIALIZED_VISUAL_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            init_visual_engine();
        }
        &mut *VISUAL_ENGINE.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;
    use crate::core::memory::NeuronId;
    use crate::vision::physics_engine::PhysicalObject;

    #[test]
    fn visual_engine_new_state() {
        let ve = VisualEngine::new();
        assert_eq!(ve.retina.base_neuron_id, NeuronId::new(0));
    }

    #[test]
    fn visual_retina_process_frame() {
        let mut retina = RetinalEngine::new(NeuronId::new(0));
        let frame = [0u8; VISION_PIXELS];
        let responses = retina.process_frame(&frame, 0);
        assert_eq!(responses.len(), VISION_PIXELS);
    }

    #[test]
    fn visual_gabor_responses() {
        let retina_output = [GanglionResponse { on_response: FixedPoint::ONE, off_response: FixedPoint::ZERO }; VISION_PIXELS];
        let v1_map = GaborBank::process_retina(&retina_output);
        assert_eq!(v1_map.len(), VISION_PIXELS);
    }

    #[test]
    fn visual_physics_extrapolate() {
        let obj = PhysicalObject::empty(1);
        let pos = obj.extrapolate_trajectory(100);
        assert!(pos.x >= FixedPoint::ZERO);
    }

    #[test]
    fn visual_collision_detection() {
        let obj = PhysicalObject::empty(2);
        let agent = SpatialPoint3D { x: FixedPoint::ZERO, y: FixedPoint::ZERO, z: FixedPoint::ZERO };
        let hit = obj.check_collision(&agent, FixedPoint::from_f32(10.0));
        assert!(hit, "object at origin should collide with agent at origin within radius 10");
    }
}
