# Changelog

All notable changes to HKL-1 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — Development Milestone

### Status
**378 unit & integration tests ✅ · 0 warnings · 0 errors · 0 dependencies**

### Added

#### 🧠 Apprentissage Continu & Anti-Oubli Catastrophique (`src/cognitive/continual.rs`)
- **Offline Experience Replay (`OfflineReplayEngine`)**: Sharp Wave-Ripples (SWR 150-250 Hz) hippocampal replay during rest/sleep.
- **Few-Shot Fast-Weights (`FewShotAdapter`)**: Fast-Weights 4x plasticity boosting for rapid 1-shot learning.
- **Meta-Learning Auto-Tuning (`MetaLearningEngine`)**: Dynamic tuning of $\eta_{\text{STDP}}$ and dopamine threshold $\theta_{\text{DA}}$.
- **Elastic Weight Consolidation (`ElasticWeightConsolidation`)**: Fisher Information Matrix $F_{ij}$ protection preventing catastrophic forgetting.

#### ⚡ Bio-Compilation eFPGA & Accélération Silicium (`src/efpga/`)

- **Subnetwork Stability Analyzer (`stability.rs`)**: Variance $\sigma_w^2$ thresholding & automated freezing of immutable SNN sub-networks.
- **Synthesizable Verilog RTL Generator (`hdl_gen.rs`)**: Verilog HDL code synthesis (`module efpga_snn_subnetwork(...)`) for hardware RTL integration.
- **eFPGA Bitstream Configurator (`bitstream.rs`)**: Binary bitstream compiler for embedded LUT4/LUT6 tables and routing switch matrices.
- **Nanosecond Hardware Simulator (`simulator.rs`)**: Cycle-accurate hardware logic simulator achieving sub-nanosecond propagation latency (< 1 ns per spike, >1000x speedup vs software).

#### 🎙️ Intelligence Auditive & Vocalisation Spiking (`src/audio/`)

- **Cochlear Filter Bank (`cochlea.rs`)**: 32-band logarithmic ERB Gammatone filter bank (80 Hz to 8000 Hz) with PFM hair cell spike encoding into `Modality::Audio`.
- **Cortex A1 & Formants (`a1_formants.rs`)**: Tonotopic frequency map, formant peak extractor ($F_1, F_2, F_3$), and vowel classifier (/a/, /i/, /u/, /e/, /o/).
- **Pitch & Rhythm Engine (`pitch_rhythm.rs`)**: Autocorrelation $F_0$ pitch estimator for voice categorization (Male/Female/Child) and speech onset energy detector.
- **Spike Voice Synthesizer (`voice_synth.rs`)**: Klatt-inspired formant resonator converting Layer 4 motor spikes into 16-bit 16kHz PCM audio sample streams.

#### 💬 Pont SNN ↔ LLM & Cognition Symbolique / NLP (`src/nlp/`)

- **Spike Tokenizer & Encoder (`spike_token.rs`)**: `SpikeVocabulary` indexing ASCII and 256 subword/token slots with temporal phase-timing position encoding ($\Delta t_{\text{pos}} = \text{pos} \cdot \tau_{\text{phase}}$) ingested into `Modality::Text`.
- **Spike Token Decoder (`spike_decoder.rs`)**: Winner-Take-All (WTA) firing rate integrator for Layer 4 motor/text neurons and sentence reconstruction buffer.
- **Neuromodulated Verbalizer (`verbalizer.rs`)**: `NeuromodulatedVerbalizer` generating real-time natural language explanations of dopamine (DA), serotonin (5-HT), noradrenaline (NA), acetylcholine (ACh), prediction error, curiosity, and boredom.
- **Symbolic Knowledge Graph (`symbolic_graph.rs`)**: `SymbolicKnowledgeGraph` managing concept nodes and triple relations $(S, R, O)$ with Spiking Hebbian concept binding and spreading activation.
- **Dialogue Engine (`dialogue_engine.rs`)**: `DialogueEngine` orchestrating text tokenization, neuro-symbolic reasoning, state verbalization, and text generation.

#### 🌊 Spiking Visual, Spatial & Intuitive Physics Engine (`src/vision/`)

- **Retinal Processing (`retina.rs`)**: Difference of Gaussians (DoG $5\times5$) spatial contrast filtering, ON/OFF dual ganglion cell channels, and DVS log-intensity event polarity encoder.
- **Cortex V1 & V4 (`v1_gabor.rs`, `v4_shape.rs`)**: Multi-angle 2D Gabor orientation kernels (0°, 45°, 90°, 135°) for edge extraction & online Hebbian IT visual object prototype clustering.
- **MT Motion & Stereo Depth (`mt_motion.rs`, `depth_spatial.rs`)**: Reichardt EMD pairs for 2D velocity vectors $(V_x, V_y)$, 3D looming expansion ($V_z$), and stereo parallax depth $Z = (f \cdot B) / d$.
- **Intuitive Physics Engine (`physics_engine.rs`)**: Ballistic trajectory extrapolation $\vec{x}(t+\Delta t) = \vec{x}(t) + \vec{v}\Delta t + \frac{1}{2}\vec{g}\Delta t^2$, gravity vector $g$, collision forecasting, and object permanence under occlusion.
- **Predictive Coding & S-CNN (`predictive_coding.rs`, `conv.rs`)**: Top-down frame prediction $I_{\text{pred}}$, visual prediction error $\mathcal{E}_{\text{vis}}$ calculation, `SpikingConv2D`, `SpikingConv3D`, and `SpikingMaxPool`.

#### ⚡ Parallel Execution & Hardware Adaptation Engine

- **Multi-Threaded Parallel Execution Engine (`step_parallel`)**: Parallelized SNN step evaluation and chunked STDP trace decay (`apply_plasticity_parallel`) via `std::thread::scope` for multi-core processors.
- **Dynamic Hardware Detection (`HardwareDetector`)**: Automatic runtime detection of available CPU cores and system RAM capacity using native OS hardware interfaces.
- **Adaptive Memory Scaling (`ADAPTIVE_MEMORY`)**: Dynamic allocation capacity adjustment (`set_capacity`) scaling SNN memory structures seamlessly up to hardware limits.
- **Host OS MMIO Protection**: Safe `#[cfg(feature = "std")]` guards for hardware register accesses (PWM, DAC, GPIO, I2C, SPI, ADC, PWR) allowing clean execution under host OS simulators.

#### 🧬 SNN (Spiking Neural Network)

- LIF neurons with 6 types: Excitatory, Inhibitory, Modulatory, Pacemaker, Sensory, Motor
- STDP plasticity with asymmetric learning window, eligibility traces, dopamine modulation
- Hebbian plasticity with Oja's rule, homeostatic normalization
- Homeostasis: target firing rate, gain control, synaptic scaling, layer compensation
- Neurogenesis: structural plasticity with pruning, creation, recycling, adjacency
- **Synaptic senescence**: `apply_senescence(max_age)` with weight decay (→50%), plasticity decay (→20%), and pruning
- **Reflex arcs**: hard-wired L0→L6→L4 connections with fixed weights, no plasticity
- Novelty computation as derivative of prediction error: `|mean_error - prev_error|`
- Energy-adaptive threshold: V_th = f(battery_level) via `PowerManager::threshold_multiplier()`
- Global neuromodulator system (`GLOBAL_NEUROMODULATORS`) with calibration

#### 🧠 Cognitive
- Actor-Critic RL: policy network, value table (64 buckets), TD-error, ε-greedy exploration
- Attention: bottom-up saliency map, top-down goal routing, WTA per layer
- Curiosity: habituation (32 slots, sigmoid familiarity), boredoom, monotony counter, thermal noise injection, dreaming mode
- **Predictor**: prototype-based forward model, `predict_next(state, last_action)`, online Hebbian learning, confidence EMA, transition buffer (256), prototype merging
- **Neuromodulation**: DA/5-HT/NA/ACh with auto-calibration (EMA error, volatility, adaptive decay, LTP/LTD sensitivity), `sync_to_snn()` bidirectional
- Proprioception: efference copy, prediction error, corrective current injection, body model (64 slots, online Hebbian)
- **Temporal cognition**: 64 time cells (1ms–50s, Gaussian activation), sequence buffer (256), interval timing, multi-scale integration (5 buffers), `last_triggered_ms` real-time tracking
- Reflex cognitive override: context-based suppression via NA + attention

#### 🌐 Swarm
- Federated learning: Gaussian DP noise, adaptive topology, node reliability
- Mesh networking: peer discovery (beacon/heartbeat), gossip protocol (64 messages, fanout=3, TTL=1000ms), clock sync (RTT, drift), remote spike propagation, weight deltas (2048 entries)

#### 🛡️ Safety
- 5 reflex types: nociceptive, startle, freeze, withdrawal, seizure detection
- Entropy monitor: Shannon entropy, adaptive thresholds (mean±2σ, EMA), 4 cognitive modes (Exploratory, Exploit, Crisis, Stability), neuromodulation correlation
- Hardware resilience: ECC (32 blocks, parity+syndrome, single-bit auto-fix), bad sector tracking, synaptic migration, `SenescenceStage` (Healthy→Aging→Degraded→EndOfLife)

#### ⚡ Power Management
- 5 power states: Active, Idle, Sleep, Deep Sleep, Shutdown
- DVFS: 5 OPP levels (16–216 MHz), auto PWR_CR VOS switching
- 6 power domains: CPU, Memory, Sensors, Actuators, Radio, Cognitive
- Energy harvesting: MPPT perturb-and-observe (`HarvestingType`)
- Auto mode switch: Survive/Explore based on battery + harvest
- Low-power idle: `idle_if_possible()` → deep sleep if idle > 10ms
- Clock gating: AHB1/APB1/APB2 ENR per domain
- V_th dynamic coupling: `threshold_multiplier()` → SNN

#### 📡 I/O & Real-Time
- Lock-free `RingBuffer<T, N>` with atomic head/tail, MPMU, mask-based modulo
- `GLOBAL_SPIKE_QUEUE`: 4096-entry ring buffer for ISR-to-main communication
- Modality→layer mapping: Text=0, Audio=1, Vision=2, Sensor=3, Proprioception=4, Internal=5
- Encoders: rate, temporal, population, place coding
- Decoders: rate, temporal, WTA, vector decoding
- Sensors: I2C/SPI/ADC MMIO drivers, `SensorManager` with `i2c_error_count` tracking
- Actuators: PWM/GPIO/DAC MMIO, `DacOutput::init()` (DAC_CR EN+BOFF)
- ISR handlers: TIM2, ADC, EXTI0–15, SPI1/2, I2C1/2 → `isr_push_spike()` with layer from intensity

#### 💾 Persistence & Boot
- **Boot sequence** (22ms): clock init → hardware peripherals (CPACR, SCB_VTOR, MPU, SCS_CCR) → UID OTP (96-bit @ 0x1FFF_7A10) → persistence load → entropy seed → spike logger init → sensor IRQ enable (NVIC_ISER0/1/2) → reflex arcs → emergency checks → main loop → OTA cycle
- **Emergency checks**: reflex thresholds, NEURON_COUNT validation, entropy range (high→dreaming, low→crystallize)
- Persistence: `BinaryDump` with header (64B), 3 slots (J-0/J-1/J-2), CRC32, optional ChaCha20 encryption, secure erase (XOR + verify), `commit_to_flash()` MMIO
- OTA: dual-bank, CRC32 validation, slot state machine (Empty→Filled→Validated→Applied→Stable), rollback, persistence before switch, no_std stack buffer `[u8; 1028]`

#### 🔍 Watchdog
- `NeurologicalWatchdog` with graduated escalation (10 levels)
- Level 0: pet (normal) → Level 1: warn → Level 2: fault counter → Level 3–4: soft reset → Level 5–9: rollback → Level 10+: full restore

#### 📊 Telemetry & XAI
- `SpikeTraceLogger`: 8192-event circular buffer, burst detection, per-neuron stats, UART export
- `CausalGraph`: 4096 edges, confidence EMA, `top_causal_paths()`, `export_uart_text()` (2048 bytes)
- `FeatureAttribution`: 128 slots with contribution + sign

#### 🎯 Platform Support
- **STM32F7** (ARM Cortex-M7, `thumbv7em-none-eabihf`): `stm32f746.ld` linker with TCM sections
- **HiFive1** (RISC-V RV32, `riscv32imac-unknown-none-elf`): `hifive1.ld` linker
- **ESP32-C6** (RISC-V RV32, `riscv32imc-unknown-none-elf`): `esp32c6.ld` linker
- BSP modules gated by Cargo features (`stm32f7`, `hifive1`, `esp32c6`)

### Infrastructure
- CI/CD: GitHub Actions (`check`, `test`, `clippy`, `fmt`, `cross`, `deny`)
- Cross-compilation targets for all 3 platforms
- `cargo deny` configuration (zero-dependency audit)
- Cargo aliases: `ct` (check test), `cr` (check release), `xt` (cross test), `xr` (cross release)
- Rust 2024 edition: `unsafe extern "C"`, `#[unsafe(no_mangle)]`, `#[unsafe(link_section)]`, explicit `unsafe {}`

### Documentation
- Architecture guide with ASCII diagram, data flow, boot timeline, memory model
- Module docs: Core, SNN, Cognitive, I/O, Swarm, Safety, System, Telemetry
- Getting started guide: build, flash (probe-rs, OpenOCD, espflash), feature reference
- ROADMAP.md: full TDD compliance analysis (454 lines, per-phase breakdown)
- CONTRIBUTING.md, GOVERNANCE.md, SECURITY.md, CODE_OF_CONDUCT.md, SUPPORT.md

---

## [0.1.0] — YYYY-MM-DD

### Added
- Initial project structure and crate skeleton
- Core fixed-point math (Q16.16) and weight types (Q8.8)
- Static memory pool allocators for neurons and synapses
- LIF neuron model with 6 types
- Spiking neural network simulation loop

---

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).
