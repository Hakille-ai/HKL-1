//! Watchdog timer with graduated recovery actions. Monitors system health.
use crate::core::math::FixedPoint;
use crate::core::memory::{NEURON_COUNT, NeuronId, neuron_state_ref};

/// Neurological watchdog (Section 7.2)
/// Detects epileptic states and triggers rollback

pub struct NeurologicalWatchdog {
    pub firing_rate_threshold: FixedPoint, // Max spikes/sec before "epileptic"
    pub firing_rate_high_count: u32,
    pub max_high_count_before_reset: u32,
    pub last_reset_time: u32,
    pub reset_count: u32,
    pub watch_active: bool,
}

impl NeurologicalWatchdog {
    pub const fn new() -> Self {
        Self {
            firing_rate_threshold: FixedPoint::from_f32(100.0), // 100Hz is epileptic
            firing_rate_high_count: 0,
            max_high_count_before_reset: 10,
            last_reset_time: 0,
            reset_count: 0,
            watch_active: true,
        }
    }

    /// Check network health
    pub fn check_health(&mut self, avg_firing_rate: FixedPoint) -> WatchdogAction {
        if !self.watch_active {
            return WatchdogAction::None;
        }

        if avg_firing_rate > self.firing_rate_threshold {
            self.firing_rate_high_count += 1;
        } else {
            self.firing_rate_high_count = 0;
        }

        if self.firing_rate_high_count >= self.max_high_count_before_reset {
            self.reset_count += 1;
            self.firing_rate_high_count = 0;
            WatchdogAction::EmergencyReset
        } else if self.firing_rate_high_count > 3 {
            WatchdogAction::RollbackJ1
        } else {
            WatchdogAction::None
        }
    }

    /// Detect "epilepsy" - synchronous high-frequency spiking
    pub fn detect_epilepsy(&self) -> bool {
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        let mut synchronous_pairs = 0u32;
        let mut total_pairs = 0u32;

        for i in 0..count.min(100) as u16 {
            let id_a = NeuronId::new(i);
            let state_a = neuron_state_ref(id_a);
            for j in (i + 1)..count.min(100) as u16 {
                let id_b = NeuronId::new(j);
                let state_b = neuron_state_ref(id_b);
                total_pairs += 1;
                if (state_a.membrane_potential - state_b.membrane_potential).abs()
                    < FixedPoint::from_f32(0.1)
                {
                    synchronous_pairs += 1;
                }
            }
        }

        let sync_ratio = if total_pairs > 0 {
            (synchronous_pairs as f32) / (total_pairs as f32)
        } else {
            0.0
        };

        sync_ratio > 0.7 // >70% synchronous = epileptic
    }
}

pub enum WatchdogAction {
    None,
    RollbackJ1,     // Moderate: rollback to J-1
    EmergencyReset, // Severe: full reset
}

pub static mut WATCHDOG: NeurologicalWatchdog = NeurologicalWatchdog::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_new_defaults() {
        let wd = NeurologicalWatchdog::new();
        assert!(wd.watch_active);
        assert_eq!(wd.firing_rate_high_count, 0);
        assert_eq!(wd.reset_count, 0);
    }

    #[test]
    fn test_watchdog_returns_none_for_normal_rate() {
        let mut wd = NeurologicalWatchdog::new();
        let action = wd.check_health(FixedPoint::from_f32(50.0));
        assert!(matches!(action, WatchdogAction::None));
    }

    #[test]
    fn test_watchdog_rollback_after_consecutive_high() {
        let mut wd = NeurologicalWatchdog::new();
        // 4 consecutive high readings → RollbackJ1
        for _ in 0..4 {
            let action = wd.check_health(FixedPoint::from_f32(150.0));
            if action as u8 > 0 {
                break;
            }
        }
        let action = wd.check_health(FixedPoint::from_f32(150.0));
        assert!(matches!(
            action,
            WatchdogAction::RollbackJ1 | WatchdogAction::EmergencyReset
        ));
    }

    #[test]
    fn test_watchdog_emergency_after_many_high() {
        let mut wd = NeurologicalWatchdog::new();
        // EmergencyReset fires when count reaches max_high_count_before_reset
        for _ in 0..wd.max_high_count_before_reset {
            let action = wd.check_health(FixedPoint::from_f32(200.0));
            if matches!(action, WatchdogAction::EmergencyReset) {
                return;
            }
        }
        let action = wd.check_health(FixedPoint::from_f32(200.0));
        assert!(matches!(action, WatchdogAction::EmergencyReset));
    }

    #[test]
    fn test_watchdog_inactive_returns_none() {
        let mut wd = NeurologicalWatchdog::new();
        wd.watch_active = false;
        let action = wd.check_health(FixedPoint::from_f32(200.0));
        assert!(matches!(action, WatchdogAction::None));
    }
}
