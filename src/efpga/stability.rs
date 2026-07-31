//! Sub-network Stability Analyzer for HKL-1 eFPGA Bio-Compilation.
//! Computes synaptic weight variance over time to identify immutable SNN sub-networks
//! suitable for hardware FPGA logic freezing.

use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;

pub const MAX_STABLE_SYNAPSE_CAPACITY: usize = 64;

/// Struct representing a frozen hardware-synthesizable synapse
#[derive(Clone, Copy, Debug)]
pub struct FrozenSynapse {
    pub source_id: NeuronId,
    pub target_id: NeuronId,
    pub weight: FixedPoint,
    pub delay_us: u32,
    pub variance: FixedPoint,
    pub age_cycles: u32,
}

/// Frozen Sub-network descriptor
#[derive(Clone)]
pub struct FrozenSubnetwork {
    pub id: u32,
    pub synapses: [Option<FrozenSynapse>; MAX_STABLE_SYNAPSE_CAPACITY],
    pub count: usize,
    pub avg_variance: FixedPoint,
}

/// Sub-network Stability Analyzer
pub struct SubnetworkStabilityAnalyzer {
    pub variance_threshold: FixedPoint,
    pub min_age_cycles: u32,
}

impl SubnetworkStabilityAnalyzer {
    pub fn new() -> Self {
        Self {
            variance_threshold: FixedPoint::from_f32(0.005),
            min_age_cycles: 100,
        }
    }

    /// Analyze a list of synapses and extract stable, immutable sub-networks for hardware freezing
    pub fn analyze_and_freeze_subnetwork(
        &self,
        synapse_data: &[(NeuronId, NeuronId, FixedPoint, u32, FixedPoint, u32)],
        subnetwork_id: u32,
    ) -> FrozenSubnetwork {
        let mut subnetwork = FrozenSubnetwork {
            id: subnetwork_id,
            synapses: [None; MAX_STABLE_SYNAPSE_CAPACITY],
            count: 0,
            avg_variance: FixedPoint::ZERO,
        };

        let mut total_var = FixedPoint::ZERO;

        for &(src, tgt, w, delay, var, age) in synapse_data {
            if var <= self.variance_threshold && age >= self.min_age_cycles {
                if subnetwork.count < MAX_STABLE_SYNAPSE_CAPACITY {
                    subnetwork.synapses[subnetwork.count] = Some(FrozenSynapse {
                        source_id: src,
                        target_id: tgt,
                        weight: w,
                        delay_us: delay,
                        variance: var,
                        age_cycles: age,
                    });
                    subnetwork.count += 1;
                    total_var += var;
                }
            }
        }

        if subnetwork.count > 0 {
            subnetwork.avg_variance =
                total_var * FixedPoint::from_f32(1.0 / subnetwork.count as f32);
        }

        subnetwork
    }
}
