use crate::core::entropy::CognitiveMode;
use crate::core::math::FixedPoint;

/// Evaluate whether cognitive system should suppress spinal reflexes.
/// Returns true when the situation is understood and reflexes can be safely suppressed.
pub fn evaluate_override() -> bool {
    // 1. Check noradrenaline — high NA means crisis, no override
    let na = crate::snn::neuron::neuromodulators().noradrenaline;
    if na > FixedPoint::from_f32(0.6) {
        return false;
    }

    // 2. Check cognitive mode — only override in Stable mode
    let mode = unsafe { crate::core::entropy::ENTROPY_MONITOR.derive_mode() };
    if mode != CognitiveMode::Stable {
        return false;
    }

    // 3. Check attention focus — must be dwelling on a target
    let attn = unsafe { &*crate::cognitive::attention::ATTENTION_ROUTER.as_mut_ptr() };
    if attn.focus.dwell_counter < 5 {
        return false;
    }

    true
}

/// Override strength — how much to suppress reflex output (0.0 = full suppression)
pub fn override_attenuation() -> FixedPoint {
    if !evaluate_override() {
        return FixedPoint::ONE;
    }
    let na = crate::snn::neuron::neuromodulators().noradrenaline;
    let base = FixedPoint::ONE - na;
    base.clamp(FixedPoint::ZERO, FixedPoint::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_override_false_when_high_na() {
        let nm = crate::snn::neuron::neuromodulators();
        nm.noradrenaline = FixedPoint::from_f32(0.8);
        assert!(!evaluate_override());
    }

    #[test]
    fn test_attenuation_zero_when_no_override() {
        let nm = crate::snn::neuron::neuromodulators();
        nm.noradrenaline = FixedPoint::from_f32(0.8);
        assert_eq!(override_attenuation(), FixedPoint::ONE);
    }

    #[test]
    fn test_override_false_when_not_stable_mode() {
        let nm = crate::snn::neuron::neuromodulators();
        nm.noradrenaline = FixedPoint::ZERO;
        let em = unsafe { &mut crate::core::entropy::ENTROPY_MONITOR };
        let mut net = crate::snn::network::Network::new();
        em.force_crystallization(&mut net);
        em.cognitive_update(FixedPoint::from_f32(5.0));
        let result = evaluate_override();
        assert!(!result, "Should be false when mode is not Stable");
    }

    #[test]
    fn test_attenuation_scales_with_na() {
        let nm = crate::snn::neuron::neuromodulators();
        nm.noradrenaline = FixedPoint::from_f32(0.3);
        let att = override_attenuation();
        assert!(att > FixedPoint::ZERO);
        assert!(att <= FixedPoint::ONE);
    }
}
