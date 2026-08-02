use hkl1::prelude::*;

fn main() {
    let neuron = NeuronId::new(0);
    let synapse = SynapseId::new(0);
    let threshold = FixedPoint::from_f32(0.75);
    let weight = Weight::from_f32(0.5);

    println!("HKL-1 {VERSION}");
    println!(
        "dt={}us neurons={} synapses={}",
        SIMULATION_DT_US, MAX_NEURONS, MAX_SYNAPSES
    );
    println!(
        "neuron={} synapse={} threshold={:.2} weight={:.2}",
        neuron.index(),
        synapse.index(),
        threshold.to_f32(),
        weight.to_f32()
    );
}
