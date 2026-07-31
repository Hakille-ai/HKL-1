# SNN Module (Spiking Neural Network)

The SNN module implements a complete spiking neural network with biological-plausibility features.

## Neuron (`snn/neuron.rs`)

### LIFNeuron (Leaky Integrate-and-Fire)

```
τ_m dV/dt = -V + I_syn + I_noise

V ≥ θ → spike, V → V_reset
```

| Parameter | Description |
|---|---|
| `tau_m` | Membrane time constant (decay) |
| `v_threshold` | Spiking threshold |
| `v_reset` | Reset potential after spike |
| `refractory_period` | Absolute refractory period (ms) |
| `adaptation` | Spike-frequency adaptation current |
| `noise_scale` | Stochastic noise injection amplitude |

### Neuron Types

| Type | Behavior |
|---|---|
| Excitatory | Standard excitatory (AMPA-like) |
| Inhibitory | Standard inhibitory (GABA-like) |
| Modulatory | Neuromodulator release on spike |
| Pacemaker | Intrinsic bursting (metabolic heartbeat) |
| Sensory | Encoder spike patterns from sensors |
| Motor | Decoder for actuator commands |

### Neuromodulators

Global neuromodulator concentrations that affect all neurons:

| Modulator | Effect |
|---|---|
| Dopamine | Reward prediction error, learning rate modulation |
| Serotonin | Mood, risk assessment, crystallization |
| Noradrenaline | Arousal, novelty, corrective action |
| Acetylcholine | Attention, memory encoding |

## Synapse (`snn/synapse.rs`)

- Weighted connections between neurons
- Configurable delay (1–255 ms)
- Depression/facilitation for short-term plasticity
- Tag-based marking for neurogenesis recycling

### Synaptic Senescence

Synapses undergo biological aging to prevent runaway growth:

| Feature | Description |
|---|---|
| `apply_senescence(max_age)` | Increments age; at `max_age` decays weight →50% and plasticity →20% |
| Pruning | If age exceeds `max_age` + grace period, synapse is pruned |
| Recycling | Pruned synapses return to the free pool |
| Integration | Called from `metabolic_maintenance()` via `NEUROGENESIS.maintenance_cycle()` |

### Reflex Arcs

Hard-wired synaptic connections bypassing plasticity:

| Connection | Source → Target | Weight | Plasticity |
|---|---|---|---|
| Sensor → Reflex | Layer 0 → Layer 6 | 1.0 Fixed | Disabled |
| Reflex → Motor | Layer 6 → Layer 4 | 0.8 Fixed | Disabled |

Initialized by `init_reflex_arcs()` — creates fixed-weight, non-plastic pathways for rapid unconditioned responses.

## Network (`snn/network.rs`)

Network topology and simulation orchestration:

- `Network::new()` — Static initialization of SNN network
- `Network::init_network()` — Static singleton initialization via `NETWORK_INSTANCE.write(Network::new())` (replaces previous buggy `core::ptr::write_bytes` that only zeroed 1 byte)
- `Network::connect(pre, post, weight)` — Create synapse
- `Network::step()` — Single-threaded deterministic simulation step (bare-metal `no_std`, >530,000 steps/sec, ~139,000 M neuron-evals/sec)
- `Network::step_parallel(num_threads)` — Multi-threaded parallel simulation engine (`std` feature)
- `Network::apply_plasticity_parallel(now, num_threads)` — Chunk-wise multi-threaded STDP trace decay (`decay_traces_chunk`) via `std::thread::scope`
- `Network::capture_simulation_snapshot()` — Zero-stack static snapshot storage (`SIMULATION_SAVE_SLOT`) eliminating stack allocation overhead during simulation steps
- `Network::propagate_spike(id, time)` — Propagate somatic spike

### eFPGA Bio-Compilation (`src/efpga/`)

Frozen SNN sub-networks with low weight variance ($\sigma_w^2 < 0.005$) and high age are automatically compiled to hardware logic:
- `SubnetworkStabilityAnalyzer`: Detects immutable sub-networks
- `HdlGenerator`: Generates synthesizable Verilog HDL RTL (`module efpga_snn_subnetwork(...)`)
- `BitstreamEncoder`: Compiles binary bitstream LUT4/LUT6 config arrays
- `EfpgaHardwareSimulator`: Cycle-accurate hardware evaluation achieving sub-nanosecond latency (< 1 ns per spike, >1000x speedup)



### Predictive Cycle

The full cognitive cycle at network level:

1. **Inhibit actor output** → `actor.output_inhibited = true`
2. **Generate hypotheses** → `generate_hypotheses()` → `predict_next()`
3. **TimeWarp simulation** → `activate_warp(100)` → `run_simulation()`
4. **Evaluate result** → dopamine signal + reconnect actor
5. **Record transition** → `record_transition()`
6. **Cooldown** → 5000 steps (success) / 2000 steps (failure)

### Novelty Computation

Novelty is computed as the derivative of prediction error:

```
novelty = |mean_prediction_error - prev_mean_prediction_error|
```

- `PredictorNetwork.prev_mean_prediction_error` stores the previous EMA error
- Enables the network to detect *changes* in uncertainty (not absolute error)
- Connected to curiosity and exploration drive

### Energy-Adaptive Threshold

`energy_adaption()` uses `PowerManager::threshold_multiplier()` + `battery_level` to dynamically adjust neuronal firing thresholds, reducing功耗 during low-energy states.

## Plasticity (`snn/plasticity.rs`)

### STDP (Spike-Timing-Dependent Plasticity)

```
Δw = A₊ exp(-Δt / τ₊)   if t_pre < t_post
Δw = -A₋ exp(Δt / τ₋)   if t_pre > t_post
```

- Pre-synaptic trace maintained per neuron
- Asymmetric learning window
- Neuromodulation-gated learning (dopamine modulates STDP)

### Calcium Model (Calcium Control Hypothesis)

Calcium concentration in postsynaptic spines gates LTP vs LTD:

| Parameter | Description |
|---|---|
| `concentration` | Current calcium level, accumulates on spikes |
| `ltp_threshold` | Above this → LTP (potentiation) |
| `ltd_threshold` | Between LTD and LTP threshold → LTD (depression) |
| `influx_per_spike` | Calcium influx per presynaptic spike |
| `decay_per_ms` | Exponential decay rate per ms |

### Plateau Potential

Prolonged depolarizations that amplify plasticity:

- Activates via `trigger_plateau(neuron, time)`
- Duration: configurable (default 100ms)
- During active plateau: LTD strength is boosted by amplitude
- Triggers extra calcium influx (3× multiplier)

Integration: `on_post_spike()` checks plateau state; both calcium gating and plateau boost multiply the STDP trace update.

### Hebbian Plasticity

- Rate-based weight adjustment
- Homeostatic normalization
- Oja's rule for stability

## Homeostasis (`snn/homeostasis.rs`)

- Target firing rate maintenance
- Gain control (threshold adaptation)
- Synaptic scaling (global weight normalization)
- Layer-specific compensation

## Neurogenesis (`snn/neurogenesis.rs`)

Structural plasticity through synapse creation and pruning:

- **Pruning**: Remove silent synapses (no activity for threshold period)
- **Senescence**: `apply_senescence()` with configurable `max_age`, tracks `total_senesced` count
- **Creation**: Add new synapses between active pre/post pairs
- **Recycling**: Reuse pruned synapse slots from free pool
- **Maintenance cycle**: `maintenance_cycle()` returns `(pruned, created, senesced)` tuple
- **Max connections**: Configurable per neuron
- **Adjacency**: New synapses created between geometrically nearby neurons
