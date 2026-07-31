//! Intuitive Physics Engine module.
//! Simulates physical laws: ballistic trajectory extrapolation, gravity vector (g),
//! momentum conservation (p = m*v), collision boundary prediction, and object permanence under occlusion.

use crate::core::math::FixedPoint;
use crate::vision::depth_spatial::SpatialPoint3D;
use crate::vision::mt_motion::MotionVector;

pub const GRAVITY_G: FixedPoint = FixedPoint(642252); // ~9.81 m/s^2 scaled in cm/ms^2 (FixedPoint Q16.16)
pub const MAX_TRACKED_PHYSICAL_OBJECTS: usize = 8;

/// Physical Object State in 3D Space
#[derive(Clone, Copy)]
pub struct PhysicalObject {
    pub id: u8,
    pub position: SpatialPoint3D,
    pub velocity: SpatialPoint3D,
    pub predicted_position: SpatialPoint3D,
    pub mass: FixedPoint,
    pub is_occluded: bool,
    pub occlusion_duration_ms: u32,
    pub collision_imminent: bool,
    pub valid: bool,
}

impl PhysicalObject {
    pub const fn empty(id: u8) -> Self {
        Self {
            id,
            position: SpatialPoint3D {
                x: FixedPoint::ZERO,
                y: FixedPoint::ZERO,
                z: FixedPoint::ZERO,
            },
            velocity: SpatialPoint3D {
                x: FixedPoint::ZERO,
                y: FixedPoint::ZERO,
                z: FixedPoint::ZERO,
            },
            predicted_position: SpatialPoint3D {
                x: FixedPoint::ZERO,
                y: FixedPoint::ZERO,
                z: FixedPoint::ZERO,
            },
            mass: FixedPoint::ONE,
            is_occluded: false,
            occlusion_duration_ms: 0,
            collision_imminent: false,
            valid: false,
        }
    }

    /// Extrapolate 3D position at t + dt under velocity, momentum, and gravity g:
    /// x(t + dt) = x(t) + v*dt
    /// y(t + dt) = y(t) + vy*dt + 0.5*g*dt^2
    /// z(t + dt) = z(t) + vz*dt
    pub fn extrapolate_trajectory(&self, dt_ms: u32) -> SpatialPoint3D {
        let dt_sec = FixedPoint::from_f32(dt_ms as f32 / 1000.0);
        let dt_sq_half = dt_sec * dt_sec * FixedPoint::from_f32(0.5);

        let pred_x = self.position.x + self.velocity.x * dt_sec;
        // Gravity accelerates objects downwards (+y direction) in cm/s^2
        let pred_y =
            self.position.y + self.velocity.y * dt_sec + FixedPoint::from_f32(981.0) * dt_sq_half;
        let pred_z = self.position.z + self.velocity.z * dt_sec;

        SpatialPoint3D {
            x: pred_x,
            y: pred_y,
            z: pred_z,
        }
    }

    /// Check if object trajectory intersects with agent spatial boundary (collision check)
    pub fn check_collision(
        &self,
        agent_pos: &SpatialPoint3D,
        collision_radius: FixedPoint,
    ) -> bool {
        let dx = self.predicted_position.x - agent_pos.x;
        let dy = self.predicted_position.y - agent_pos.y;
        let dz = self.predicted_position.z - agent_pos.z;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        dist_sq < (collision_radius * collision_radius)
    }
}

/// Spiking Intuitive Physics Engine
pub struct IntuitivePhysicsEngine {
    pub objects: [PhysicalObject; MAX_TRACKED_PHYSICAL_OBJECTS],
    pub agent_position: SpatialPoint3D,
    pub collision_warning_active: bool,
}

impl IntuitivePhysicsEngine {
    pub fn new() -> Self {
        let mut objects = [PhysicalObject::empty(0); MAX_TRACKED_PHYSICAL_OBJECTS];
        for i in 0..MAX_TRACKED_PHYSICAL_OBJECTS {
            objects[i] = PhysicalObject::empty(i as u8);
        }
        Self {
            objects,
            agent_position: SpatialPoint3D {
                x: FixedPoint::ZERO,
                y: FixedPoint::ZERO,
                z: FixedPoint::ZERO,
            },
            collision_warning_active: false,
        }
    }

    /// Track visual object and update 3D physical dynamics
    pub fn track_object(
        &mut self,
        obj_id: usize,
        observed_pos: SpatialPoint3D,
        motion: &MotionVector,
        is_visible: bool,
        dt_ms: u32,
    ) {
        if obj_id >= MAX_TRACKED_PHYSICAL_OBJECTS {
            return;
        }

        let obj = &mut self.objects[obj_id];

        if is_visible {
            // Object is visible: update position & estimate velocity vector
            let dt = FixedPoint::from_int(dt_ms.max(1) as i32);
            if obj.valid && !obj.is_occluded {
                obj.velocity.x = (observed_pos.x - obj.position.x) / dt;
                obj.velocity.y = (observed_pos.y - obj.position.y) / dt;
                obj.velocity.z = (observed_pos.z - obj.position.z) / dt;
            } else {
                obj.velocity.x = motion.vx;
                obj.velocity.y = motion.vy;
                obj.velocity.z = motion.vz;
            }

            obj.position = observed_pos;
            obj.is_occluded = false;
            obj.occlusion_duration_ms = 0;
            obj.valid = true;
        } else if obj.valid {
            // OBJECT PERMANENCE UNDER OCCLUSION:
            // Object is hidden behind obstacle, maintain spatial trajectory simulation!
            obj.is_occluded = true;
            obj.occlusion_duration_ms += dt_ms;

            // Extrapolate position using physical trajectory laws
            obj.position = obj.extrapolate_trajectory(dt_ms);

            // Deactivate tracking if occluded for more than 5000 ms
            if obj.occlusion_duration_ms > 5000 {
                obj.valid = false;
            }
        }

        // Extrapolate 100ms into the future for collision forecasting
        obj.predicted_position = obj.extrapolate_trajectory(100);

        // Check collision forecast
        let collision_radius = FixedPoint::from_f32(30.0); // 30cm radius
        obj.collision_imminent = obj.check_collision(&self.agent_position, collision_radius);
    }

    /// Step physical simulation engine
    pub fn step(&mut self, dt_ms: u32) {
        self.collision_warning_active = false;
        for obj in &mut self.objects {
            if obj.valid {
                if obj.is_occluded {
                    obj.position = obj.extrapolate_trajectory(dt_ms);
                }
                obj.predicted_position = obj.extrapolate_trajectory(100);
                if obj.collision_imminent {
                    self.collision_warning_active = true;
                }
            }
        }
    }
}
