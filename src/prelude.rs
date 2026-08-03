//! Compact public prelude for embedded HKL-1 applications.
//!
//! Import this module when examples or firmware code need the common fixed
//! point, neuron, synapse, network, safety, I/O, and telemetry types.

pub use crate::core::math::{FixedPoint, Matrix, Vector, Weight, XorShift64Star};
pub use crate::core::memory::{NeuronFlags, NeuronId, NeuronState, NeuronType, SynapseId};
pub use crate::io::decoder::{MotorDecoder, TextOutput, VoiceOutput};
pub use crate::io::encoder::{
    AudioEncoder, InternalEncoder, ModalityEncoder, ProprioceptionEncoder, RateEncoder,
    SensorEncoder, TextEncoder, VisionEncoder,
};
pub use crate::safety::reflexes::{ReflexRule, SpinalReflexes};
pub use crate::snn::network::{Network, SIMULATION_DT_US, SimulationResult};
pub use crate::snn::neuron::{LIFNeuron, Neuromodulators, SpikeEvent};
pub use crate::snn::synapse::Synapse;
pub use crate::telemetry::spike_trace::{
    BurstEvent, SpikeTraceLogger, SpikeTraceTextExport, TraceEvent,
};
pub use crate::telemetry::xai::{
    CausalEdge, CausalGraph, CausalTextExport, FeatureAttribution, SpikeTraceAnalyzer,
};
pub use crate::{
    MAX_NEURONS, MAX_SYNAPSES, PERSISTENCE_SLOTS, RING_BUFFER_SIZE, SPIKE_TRACE_BUFFER, VERSION,
};
