#![cfg(feature = "std")]

use hkl1::prelude::*;

#[test]
fn prelude_exposes_common_embedded_primitives() {
    let neuron = NeuronId::new(7);
    let synapse = SynapseId::new(3);
    let weight = Weight::from_f32(0.5);
    let fixed = FixedPoint::from_int(2);

    assert_eq!(neuron.index(), 7);
    assert_eq!(synapse.index(), 3);
    assert!(weight > Weight::ZERO);
    assert_eq!(fixed.to_int(), 2);
    assert_eq!(SIMULATION_DT_US, 1000);
    const {
        assert!(MAX_NEURONS >= 1);
        assert!(MAX_SYNAPSES >= 1);
    }
    assert!(!VERSION.is_empty());
}
