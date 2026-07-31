#![cfg(feature = "std")]

use hkl1::core::math::FixedPoint;
use hkl1::core::memory::NeuronId;
use hkl1::vision::conv::{SpikingConv2D, SpikingConv3D, SpikingMaxPool};
use hkl1::vision::depth_spatial::{DepthEngine, SpatialPoint3D};
use hkl1::vision::mt_motion::{DominantDirection, MotionEngine};
use hkl1::vision::physics_engine::{IntuitivePhysicsEngine, PhysicalObject};
use hkl1::vision::predictive_coding::VisualPredictiveCoding;
use hkl1::vision::retina::{RetinalEngine, VISION_HEIGHT, VISION_PIXELS, VISION_WIDTH};
use hkl1::vision::v1_gabor::GaborBank;
use hkl1::vision::v4_shape::ShapeEngine;
use hkl1::vision::visual_engine;


#[test]
fn test_retinal_dog_and_dvs_encoding() {
    let mut retina = RetinalEngine::new(NeuronId::new(0));

    // Create frame with a high-contrast central square
    let mut frame1 = [0u8; VISION_PIXELS];
    for y in 10..22 {
        for x in 10..22 {
            frame1[y * VISION_WIDTH + x] = 255;
        }
    }

    let ganglion = retina.process_frame(&frame1, 100);

    // Verify ON-center contrast response at square edge
    let center_idx = 16 * VISION_WIDTH + 16;
    assert!(ganglion[center_idx].on_response > FixedPoint::ZERO);

    // Shift square right to trigger DVS polarity events
    let mut frame2 = [0u8; VISION_PIXELS];
    for y in 10..22 {
        for x in 15..27 {
            frame2[y * VISION_WIDTH + x] = 255;
        }
    }

    retina.process_frame(&frame2, 110);
    assert!(retina.event_count > 0, "DVS events must be generated on movement!");
}

#[test]
fn test_v1_gabor_orientation_kernels() {
    let mut retina = RetinalEngine::new(NeuronId::new(0));

    // Create vertical line frame (90° orientation)
    let mut vert_frame = [0u8; VISION_PIXELS];
    for y in 0..VISION_HEIGHT {
        vert_frame[y * VISION_WIDTH + 16] = 255;
    }

    let ganglion = retina.process_frame(&vert_frame, 100);
    let v1_map = GaborBank::process_retina(&ganglion);

    let line_idx = 16 * VISION_WIDTH + 16;
    // 90° (Index 2) response must be higher than 0° (Index 0) response for vertical line
    assert!(v1_map[line_idx].responses[2] >= v1_map[line_idx].responses[0]);
}

#[test]
fn test_v4_shape_and_object_prototypes() {
    let mut retina = RetinalEngine::new(NeuronId::new(0));
    let mut shape_engine = ShapeEngine::new();

    let mut frame = [0u8; VISION_PIXELS];
    for y in 8..24 {
        for x in 8..24 {
            frame[y * VISION_WIDTH + x] = 200;
        }
    }

    let ganglion = retina.process_frame(&frame, 100);
    let v1_map = GaborBank::process_retina(&ganglion);

    let feat = shape_engine.extract_shape(&v1_map);
    assert!(feat.total_edge_energy > FixedPoint::ZERO);

    let (proto_id, confidence) = shape_engine.process_visual_object(&feat);
    assert_eq!(proto_id, 0);
    assert!(confidence > FixedPoint::ZERO);
}

#[test]
fn test_mt_motion_and_looming() {
    let mut motion_engine = MotionEngine::new();
    let mut retina = RetinalEngine::new(NeuronId::new(0));

    // Frame 1: Circle at center
    let mut frame1 = [0u8; VISION_PIXELS];
    for y in 12..20 {
        for x in 12..20 {
            frame1[y * VISION_WIDTH + x] = 255;
        }
    }
    let g1 = retina.process_frame(&frame1, 100);
    motion_engine.process_motion(&g1);

    // Frame 2: Circle moved right
    let mut frame2 = [0u8; VISION_PIXELS];
    for y in 12..20 {
        for x in 16..24 {
            frame2[y * VISION_WIDTH + x] = 255;
        }
    }

    let g2 = retina.process_frame(&frame2, 110);
    let motion_vec = motion_engine.process_motion(&g2);

    assert!(motion_vec.vx > FixedPoint::ZERO, "Vx must be positive for rightward motion");
    assert_eq!(motion_vec.direction, DominantDirection::Right);
}

#[test]
fn test_stereo_depth_parallax() {
    let depth_engine = DepthEngine::new();

    let mut left_frame = [0u8; VISION_PIXELS];
    let mut right_frame = [0u8; VISION_PIXELS];

    // Left eye sees object at x=16, Right eye sees object shifted at x=12 (Disparity d=4)
    for y in 10..20 {
        for x in 14..18 {
            left_frame[y * VISION_WIDTH + x] = 255;
        }
        for x in 10..14 {
            right_frame[y * VISION_WIDTH + x] = 255;
        }
    }

    let depth_map = depth_engine.compute_depth_map(&left_frame, &right_frame);
    let idx = 15 * VISION_WIDTH + 16;

    // Depth Z = (f * B) / d = (50 * 6.5) / 4 = 81.25 cm
    assert!(depth_map[idx].z > FixedPoint::ZERO);
    assert!(depth_map[idx].z < FixedPoint::from_f32(200.0));
}

#[test]
fn test_intuitive_physics_engine_gravity_and_occlusion() {
    let mut physics = IntuitivePhysicsEngine::new();

    let initial_pos = SpatialPoint3D {
        x: FixedPoint::ZERO,
        y: FixedPoint::ZERO,
        z: FixedPoint::from_f32(100.0),
    };

    let mut obj = PhysicalObject::empty(0);
    obj.position = initial_pos;
    obj.velocity = SpatialPoint3D {
        x: FixedPoint::from_f32(10.0), // Moving right
        y: FixedPoint::ZERO,
        z: FixedPoint::ZERO,
    };
    obj.valid = true;

    // Extrapolate trajectory under gravity for 100ms
    let pred_pos = obj.extrapolate_trajectory(100);

    // Gravity accelerates downward (+Y)
    assert!(pred_pos.y > initial_pos.y, "Gravity must accelerate object downward (+y)");
    assert!((pred_pos.x - FixedPoint::from_int(1)).abs() < FixedPoint::from_f32(0.01)); // x = 0 + 10 * 0.1s = 1.0 cm



    // Test Object Permanence under Occlusion
    physics.track_object(0, initial_pos, &hkl1::vision::mt_motion::MotionVector {
        vx: FixedPoint::from_f32(10.0),
        vy: FixedPoint::ZERO,
        vz: FixedPoint::ZERO,
        magnitude: FixedPoint::from_f32(10.0),
        direction: DominantDirection::Right,
    }, true, 10);

    // Object disappears behind wall (is_visible = false)
    physics.track_object(0, initial_pos, &hkl1::vision::mt_motion::MotionVector {
        vx: FixedPoint::from_f32(10.0),
        vy: FixedPoint::ZERO,
        vz: FixedPoint::ZERO,
        magnitude: FixedPoint::from_f32(10.0),
        direction: DominantDirection::Right,
    }, false, 50);

    assert!(physics.objects[0].is_occluded, "Object must be marked occluded");
    assert!(physics.objects[0].valid, "Object permanence must keep object valid under occlusion");
}

#[test]
fn test_visual_predictive_coding_and_novelty() {
    let mut pred_coding = VisualPredictiveCoding::new();

    let frame1 = [100u8; VISION_PIXELS];
    let mut frame2 = [100u8; VISION_PIXELS];
    frame2[500] = 250; // Sudden anomaly/novelty

    pred_coding.predict_next_frame(&frame1, 0, 0);
    let error1 = pred_coding.compute_prediction_error(&frame1);
    assert_eq!(error1, FixedPoint::ZERO);

    let error2 = pred_coding.compute_prediction_error(&frame2);
    assert!(error2 > FixedPoint::ZERO, "Visual prediction error must increase for unexpected frame");
    assert!(pred_coding.visual_novelty > FixedPoint::ZERO, "Visual novelty derivative must trigger");
}

#[test]
fn test_spiking_convolutions_2d_3d_maxpool() {
    let conv2d = SpikingConv2D::new();
    let mut conv3d = SpikingConv3D::new();

    let mut input_grid = [FixedPoint::ZERO; VISION_PIXELS];
    input_grid[16 * VISION_WIDTH + 16] = FixedPoint::ONE;

    let out2d = conv2d.convolve_2d(&input_grid);
    assert!(out2d[16 * VISION_WIDTH + 16] > FixedPoint::ZERO);

    let out3d = conv3d.convolve_3d(&input_grid);
    assert!(out3d[16 * VISION_WIDTH + 16] > FixedPoint::ZERO);

    let pooled = SpikingMaxPool::pool_2x2(&input_grid);
    // 32x32 pooled to 16x16: (16, 16) is at pooled index (8, 8)
    assert!(pooled[8 * 16 + 8] > FixedPoint::ZERO);
}

#[test]
fn test_full_visual_engine_pipeline() {
    let engine = visual_engine();
    let frame = [128u8; VISION_PIXELS];

    let (_ganglion, _v1, motion, pred_err) = engine.process_visual_scene(&frame, 100, 10);
    assert_eq!(motion.direction, DominantDirection::Stationary);
    assert_eq!(pred_err, FixedPoint::ZERO);
}
