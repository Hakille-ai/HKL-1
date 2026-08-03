# HKL-1 — Neuromorphic AI Engine

<h3 align="center">
  <em>A bare-metal, zero-dependency spiking neural network for embedded intelligence</em>
</h3>

<p align="center">
  <a href="https://github.com/Hakille-ai/HKL-1/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Hakille-ai/HKL-1/ci.yml?branch=main&style=flat-square&label=CI&logo=github" alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue?style=flat-square" alt="License">
  </a>
  <a href="https://www.rust-lang.org">
    <img src="https://img.shields.io/badge/Rust-1.95.0+-orange?style=flat-square&logo=rust" alt="Rust">
  </a>
  <img src="https://img.shields.io/badge/no__std-bare--metal-critical?style=flat-square" alt="no_std">
  <img src="https://img.shields.io/badge/dependencies-0-success?style=flat-square" alt="Zero deps">
  <img src="https://img.shields.io/badge/tests-764-green?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-Core%20%7C%20SNN%20%7C%20Cognitive%20%7C%20System%20%7C%20Safety%20%7C%20Swarm%20%7C%20HKL--2%20%7C%20API-brightgreen?style=flat-square" alt="Coverage">
  <img src="https://img.shields.io/badge/platform-ARM%20Cortex--M7%20%7C%20RISC--RV32%20%7C%20Multi--Core%20PC-informational?style=flat-square" alt="Platforms">
</p>

<p align="center">
  <b>English</b> · <a href="#-why-hkl-1">Why?</a> · <a href="#-features">Features</a> · <a href="#-quick-start">Quick Start</a> · <a href="#-architecture">Architecture</a> · <a href="#-roadmap">Roadmap</a> · <a href="#-contributing">Contributing</a>
</p>

---

## 🧠 What is HKL-1 & HKL-2?

**HKL-1 is a from-scratch neuromorphic AI engine** that runs on bare-metal microcontrollers and multi-core systems — no OS, no allocator, no external crates, no floating-point hardware required. It simulates a full spiking neural network with cognitive functions, swarm intelligence, and persistent memory in **~100 KB of Rust**.

**HKL-2** extends the system into a **Spiking Foundation Model** (`--features hkl2`), integrating eligibility propagation (e-prop) online learning, high-dimensional population coding, BPE tokenization, multi-modal audio/vision encoders, Softmax-free Spiking Transformer, and the **Native HKL Distributed Swarm API Protocol (`HKL-NP v1`)**.

```text
764 tests ✅  ·  0 warnings  ·  0 errors  ·  0 dependencies  ·  Swarm API & Spiking Transformer (HKL-2)
```

---

## ✨ Why HKL-1?

| The Problem | HKL-1 Solution |
|---|---|
| AI needs GPUs/cloud | Runs on **ARM Cortex-M7, RISC-V**, and **Multi-Core Host PCs** |
| Single-thread bottlenecks | **Parallel Engine (`step_parallel`)** auto-scales across all CPU cores |
| Fixed memory limits | **`ADAPTIVE_MEMORY`** dynamically scales SNN size to system RAM |
| ML frameworks are huge | **~100 KB** binary, **zero dependencies** |
| Embedded AI is static | **Online learning** — adapts continuously |
| Black-box decisions | Full **XAI causal graph** export |
| Single-point failure | **Swarm intelligence** with mesh networking |
| No persistence | **Flash state dumps** with rollback and encryption |

---

## 🚀 Features

### 🤖 HKL-2 — Spiking Foundation Model (`--features hkl2`)
| Component | Location | Capability |
|---|---|---|
| **e-prop Learning Engine** | `src/learning/` | Biologically plausible online global learning via eligibility propagation ($e_{ij}(t) = \alpha \cdot e_{ij}(t-1) + \text{surrogate}(U_j) \cdot \text{spike}_i$) and online weight updates ($\Delta w = -\eta \cdot L_j \cdot e_{ij}$) |
| **Surrogate Gradients** | `src/learning/surrogate.rs` | `Fast Sigmoid`, `ArcTan`, and `Straight Through` derivatives in Q16.16 FixedPoint arithmetic |
| **Spiking Cross-Entropy Loss** | `src/learning/loss.rs` | Cross-entropy loss & learning signal calculation for spiking rate outputs |
| **Spike Population Embedding** | `src/embedding/spike_embedding.rs` | 256-dimensional spatio-temporal spike pattern encoding over $T=4$ timesteps |
| **BPE Tokenizer** | `src/embedding/bpe_tokenizer.rs` | Byte-Pair Encoding engine with pair merges and lossless byte decoding |
| **Spiking Self-Attention (SSA)** | `src/transformer/attention.rs` | 4-head Softmax-free spiking self-attention operating directly on binary Q/K/V spike streams |
| **Spiking Feed-Forward (FFN)** | `src/transformer/feed_forward.rs` | 256D $\to$ 512D $\to$ 256D LIF spiking MLP with soft membrane resets |
| **Spiking Transformer Block** | `src/transformer/block.rs` | Residual Spiking Transformer block with dual LayerNorm & SSA |
| **Spiking Transformer Model** | `src/transformer/backbone.rs` | Full $N$-layer Spiking Transformer backbone with 4096-vocab `OutputProjection` head |
| **End-to-End Trainer** | `src/training/` | Complete `TextDataLoader` & `Trainer` pipeline connecting autoregressive data loading, loss evaluation, and e-prop updates |

### 🧠 Apprentissage Continu & Anti-Oubli Catastrophique (`src/cognitive/continual.rs`)
| Component | Capability |
|---|---|
| **Offline Replay (SWR)** | Sharp Wave-Ripples (150-250 Hz) hippocampal experience replay during rest/sleep phases |
| **Few-Shot Fast Adaptation** | Fast-Weights 4x plasticity booster for rapid 1-to-3 shot learning |
| **Meta-Learning Auto-Tuner** | Dynamic performance-driven tuning of $\eta_{\text{STDP}}$, $\theta_{\text{DA}}$ and decay constants |
| **Elastic Weight Consolidation** | Fisher Information Matrix $F_{ij}$ penalty preventing catastrophic forgetting of critical task synapses |

### ⚡ Bio-Compilation eFPGA & Accélération Silicium (`src/efpga/`)

| Component | Capability |
|---|---|
| **Subnetwork Stability Analyzer** | Synaptic weight variance $\sigma_w^2$ thresholding & automated freezing of immutable SNN sub-networks |
| **Synthesizable Verilog RTL Generator** | Automatic generation of Verilog HDL code (`module efpga_snn_subnetwork(...)`) for RTL hardware logic synthesis |
| **eFPGA Bitstream Configurator** | Binary bitstream compiler for embedded LUT4/LUT6 Look-Up Tables & Routing Switch Matrices |
| **Nanosecond Hardware Simulator** | Cycle-accurate hardware logic simulator achieving sub-nanosecond propagation latency (< 1 ns per spike, >1000x speedup) |

### 🎙️ Intelligence Auditive & Vocalisation Spiking (`src/audio/`)

| Component | Capability |
|---|---|
| **Gammatone Cochlear Engine** | 32-band logarithmic ERB Gammatone filter bank (80 Hz..8000 Hz) with PFM hair cell spiking |
| **Cortex A1 & Formants** | Tonotopic spatial map, formant peak extraction ($F_1, F_2, F_3$) & vowel classification (/a/, /i/, /u/, /e/, /o/) |
| **Pitch F0 & Rhythm** | Fundamental voice pitch $F_0$ autocorrelation (Male/Female/Child distinction) & speech onset rhythm detector |
| **Spike Voice Synthesizer** | Klatt-inspired formant resonator converting Layer 4 motor spikes to 16-bit 16kHz PCM audio waveforms |

### 💬 Pont SNN ↔ LLM & Cognition Symbolique / NLP (`src/nlp/`)

| Component | Capability |
|---|---|
| **Spike Tokenizer & Encoder** | `SpikeVocabulary` (ASCII + BPE 256) with temporal phase-timing position encoding ($\Delta t_{\text{pos}} = \text{pos} \cdot \tau_{\text{phase}}$) |
| **Spike Token Decoder** | Layer 4 firing rate integration with Winner-Take-All (WTA) token selection & sentence reconstruction |
| **Neuromodulated Verbalizer** | Real-time natural language synthesis of internal neuromodulator levels (DA, 5-HT, NA, ACh), prediction errors, curiosity & boredom |
| **Symbolic Knowledge Graph** | Neuro-symbolic concept nodes & triple relations $(S, R, O)$ with Spiking Hebbian concept binding & spreading activation |
| **Dialogue Engine** | Unified orchestrator integrating text tokenization, neuro-symbolic reasoning, state verbalization, and text generation |

### 🌊 Spiking Visual, Spatial & Intuitive Physics Engine (`src/vision/`)

| Component | Capability |
|---|---|
| **Retinal Engine & DVS** | Difference of Gaussians (DoG 5x5), ON/OFF Ganglion channels, and asynchronous DVS log-intensity polarity event encoding |
| **Cortex V1 & V4** | Multi-angle 2D Gabor kernels (0°, 45°, 90°, 135°) for edge extraction & online Hebbian IT visual object prototype clustering |
| **MT Motion & Stereo Depth** | Reichardt EMD pairs for optical flow $(V_x, V_y, V_z)$, 3D looming expansion, and binocular parallax depth $Z = f \cdot B / d$ |
| **Intuitive Physics Engine** | Ballistic trajectory extrapolation $\vec{x}(t+\Delta t) = \vec{x}(t) + \vec{v}\Delta t + \frac{1}{2}\vec{g}\Delta t^2$, gravity $g$, collision forecasting, & object permanence under occlusion |
| **Predictive Coding & S-CNN** | Top-down frame prediction $I_{\text{pred}}$, visual prediction error $\mathcal{E}_{\text{vis}}$, `SpikingConv2D`, `SpikingConv3D`, & `SpikingMaxPool` |

### ⚡ Parallel Engine & Dynamic Hardware Scaling
| Component | Capability |
|---|---|
| **Multi-Threaded Engine** | `step_parallel(num_threads)` splits STDP decay & SNN evaluation across threads via `std::thread::scope` |
| **Hardware Detection** | `HardwareDetector` auto-scans system CPU core count and available RAM capacity |
| **Adaptive Memory Engine** | `ADAPTIVE_MEMORY.set_capacity()` resizes neuron & synapse limits dynamically |

### 🧬 Spiking Neural Network
| Component | Capability |
|---|---|
| **LIF Neurons** | 6 types (Excitatory, Inhibitory, Modulatory, Pacemaker, Sensory, Motor) |
| **STDP Plasticity** | Spike-timing-dependent with asymmetric window + eligibility traces |
| **Homeostasis** | Target rate maintenance, gain control, synaptic scaling |
| **Neurogenesis** | Structural plasticity: pruning, creation, recycling |
| **Synaptic Senescence** | Biological aging: weight decay → plasticity decay → pruning |
| **Reflex Arcs** | Hard-wired L0→L6→L4 with fixed weights, no plasticity |


### 🧠 Cognitive Functions
| Component | Capability |
|---|---|
| **Actor-Critic RL** | Policy network, value table (64 buckets), TD-error → dopamine |
| **Attention** | Bottom-up saliency + top-down goal routing + WTA |
| **Curiosity** | Novelty derivative, habituation, thermal noise, boredoom, ε-greedy |
| **Predictor** | Prototype-based forward model with online Hebbian learning |
| **Neuromodulation** | DA/5-HT/NA/ACh with auto-calibration and SNN sync |
| **Proprioception** | Efference copy, prediction error, corrective current injection |
| **Temporal Cognition** | 64 time cells, 256-event buffer, interval timing, multi-scale integration |

### 🔄 Predictive Cycle
The brain's "mental simulation" loop:
1. Inhibit output → 2. Generate hypotheses → 3. TimeWarp (1000×) → 4. Evaluate → 5. Act

### 🌐 Swarm Intelligence
| Component | Capability |
|---|---|
| **Federated Learning** | DP noise, adaptive topology, node reliability |
| **Mesh Network** | Peer discovery, gossip protocol (fanout=3), clock sync, remote spikes |
| **Weight Deltas** | `(Weight, SynapseId)` entries shared across up to 2048 changes |

### 🔒 Safety & Monitoring
| Component | Capability |
|---|---|
| **Reflexes** | 5 hard-coded bypass responses (nociceptive, startle, freeze, withdrawal, seizure) |
| **Cognitive Override** | Context-based reflex suppression via NA + attention |
| **Entropy Monitor** | Shannon entropy, adaptive thresholds (mean±2σ), 4 cognitive modes |
| **Hardware Resilience** | ECC (32 blocks, syndrome, single-bit auto-fix), bad sector tracking, synaptic migration |
| **Watchdog** | Graduated: warn → soft reset → rollback → full restore (10 levels) |

### ⚡ Power Management
| Component | Capability |
|---|---|
| **5 Power States** | Active → Idle → Sleep → Deep Sleep → Shutdown |
| **DVFS** | 5 OPP levels (16–216 MHz), auto VOS switching |
| **6 Domains** | CPU/Memory/Sensors/Actuators/Radio/Cognitive |
| **Harvesting** | MPPT perturb-and-observe algorithm |
| **V_th Coupling** | Dynamic threshold = f(battery_level) |
| **Idle** | Deep sleep if idle > 10ms |

### 📡 I/O & Real-Time
| Component | Capability |
|---|---|
| **Sensors** | I2C, SPI, ADC with error counting and MMIO |
| **Actuators** | PWM, GPIO, DAC (DAC_CR EN+BOFF) |
| **ISR Handlers** | TIM2/ADC/EXTI/SPI/I2C → lock-free global spike queue |
| **Modalities** | Text=0, Audio=1, Vision=2, Sensor=3, Proprioception=4, Internal=5 |
| **Encoders** | Rate, temporal, population, place coding |
| **Decoders** | Rate, temporal, WTA, vector decoding |

### 💾 Persistence & Boot
| Component | Capability |
|---|---|
| **22ms Boot Sequence** | Clock → HW peripherals → UID OTP → persistence → entropy → IRQ → reflexes → emergencies → main loop |
| **OTA** | Dual-bank, CRC32, slot state machine, rollback, no_std stack buffer |
| **Flash Dump** | 3 rotating slots (J-0/J-1/J-2), CRC32, optional ChaCha20 |
| **Secure Erase** | XOR overwrite + verification |

### 📊 Telemetry & XAI
| Component | Capability |
|---|---|
| **Spike Trace** | 8192-event circular buffer, burst detection, per-neuron stats |
| **Causal Graph** | 4096 edges, confidence EMA, `top_causal_paths()` |
| **UART Export** | Structured text (2048 bytes) for external debugging |
| **Feature Attribution** | 128 slots with contribution + sign |

### 🎯 Platform Support
| BSP | Arch | Target | Linker Script |
|---|---|---|---|
| **STM32F7** | ARM Cortex-M7 | `thumbv7em-none-eabihf` | `stm32f746.ld` |
| **HiFive1** | RISC-V RV32 | `riscv32imac-unknown-none-elf` | `hifive1.ld` |
| **ESP32-C6** | RISC-V RV32 | `riscv32imac-unknown-none-elf` | `esp32c6.ld` |

---

## 📐 Architecture

```
src/
├── core/          Math (FixedPoint Q16.16, XorShift64*), Memory (static pools),
│                  Time (MetabolicClock 1Hz–1MHz, TemporalHierarchy, TimeWarper),
│                  Entropy (Shannon, adaptive thresholds), Crypto (ChaCha20, PUF)
│
├── snn/           LIF neurons (6 types), Synapses (delay, STDP, depression,
│                  facilitation), Network (step, propagate, predictive cycle),
│                  Plasticity (STDP Δw=A₊e^{-Δt/τ₊}, Hebbian, neuromodulated),
│                  Homeostasis (target rate, scaling), Neurogenesis (prune,
│                  create, senescence, recycle)
│
├── cognitive/     Actor-Critic (policy, value, TD-error, ε-greedy),
│                  Attention (saliency, top-down, WTA),
│                  Curiosity (habituation, boredom, thermal noise, dreaming),
│                  Predictor (prototypes, Hebbian, confidence, transition buffer),
│                  Neuromodulation (DA/5-HT/NA/ACh, calibration, SNN sync),
│                  Proprioception (efference, error, correction, body model),
│                  Temporal (time cells, sequences, intervals, multi-scale)
│
├── io/            Lock-free RingBuffer, Encoder/Decoder (rate, temporal,
│                  population, place), Sensors (I2C/SPI/ADC MMIO, error counting),
│                  Actuators (PWM/GPIO/DAC MMIO), ISR (TIM2/ADC/EXTI/SPI/I2C)
│
├── swarm/         FederatedLearning (DP noise, adaptive topology),
│                  MeshNetwork (discovery, gossip, clock sync, remote spikes)
│
├── safety/        Reflexes (5 types + cognitive override), EntropyMonitor
│                  (adaptive thresholds), HardwareResilience (ECC, migration,
│                  senescence, bad sectors)
│
├── system/        Boot (22ms sequence, HW init, UID, emergencies, OTA cycle),
│                  Persistence (3 slots, CRC32, ChaCha20, secure erase),
│                  Watchdog (escalating recovery, 10 levels),
│                  Power (5 states, DVFS, harvesting, V_th coupling, idle),
│                  OTA (dual-bank, CRC32, rollback, no_std)
│
└── telemetry/     SpikeTrace (8192 buffer, bursts, export),
                   XAI (CausalGraph 4096 edges, feature attribution, UART)
```

> **Full architecture diagram and data flow:** [docs/architecture.md](docs/architecture.md)

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/Hakille-ai/HKL-1
cd hkl1

# Build (host — for testing algorithms)
cargo build

# Build for bare-metal targets
cargo build --features stm32f7        # STM32F746 (ARM Cortex-M7)
cargo build --features hifive1        # SiFive HiFive1 (RISC-V)
cargo build --features esp32c6        # ESP32-C6 (RISC-V)

# Run complete host + HKL-2 test suite
cargo test --features std,alloc,simd,hkl2

# Run performance benchmark suite
cargo bench --bench snn_benchmark --features std,simd

# Run full cognitive integration demo
cargo run --example snn_cognitive_demo --features std,simd

# Run experimental HKL-2 training loop
cargo run --example hkl2_training_loop --features hkl2

# Build documentation
cargo doc --no-deps --open
```

### Feature Flags

| Flag | Enables |
|---|---|
| `stm32f7` | BSP STM32F746: CPACR, SCB_VTOR, MPU, NVIC, OTP UID |
| `hifive1` | BSP SiFive HiFive1 (RISC-V) |
| `esp32c6` | BSP ESP32-C6 (RISC-V + WiFi) |
| `encryption` | ChaCha20 dump encryption |
| `flash` | Flash persistence driver |
| `simd` | SIMD 4-way loop unrolling for FixedPoint vector & matrix math |
| `std` | Host mode for testing & multi-threading |
| `alloc` | Heap allocation |

---

## 🗺️ Roadmap

| Phase | Status |
|---|---|
| **Phase 1** — SNN Core (neurons, synapses, STDP, homeostasis, neurogenesis) | ✅ |
| **Phase 2** — Persistence, Security, Boot | ✅ |
| **Phase 3** — Cognitive Complete (actor-critic, attention, curiosity, predictor, NM, proprioception, temporal) | ✅ |
| **Phase 4** — I/O Hardware (sensors, actuators, ISR, encoders, decoders) | ✅ |
| **Phase 5** — Swarm Intelligence (federated, mesh, gossip) | ✅ |
| **Phase 6** — Safety & Resilience (reflexes, ECC, migration, senescence, watchdog) | ✅ |
| **Phase 7** — Power Management (DVFS, harvesting, V_th coupling) | ✅ |
| **Phase 8** — System Boot, OTA, Telemetry, XAI | ✅ |
| **Phase 9** — Bio-inspired (astrocytes, striosome, thalamus, hippocampus, cerebellum) | ✅ |
| **Phase 10** — CI/CD, QEMU testing, HIL benchmark | 📅 |

> **Full roadmap:** [ROADMAP.md](ROADMAP.md)

---

## 📚 Documentation

| Guide | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, data flow, boot timeline, platform support |
| [Core](docs/core.md) | FixedPoint math, static memory pools, time system, entropy, crypto/PUF |
| [SNN](docs/snn.md) | LIF neurons, synapses, STDP, homeostasis, neurogenesis, senescence |
| [Cognitive](docs/cognitive.md) | Actor-Critic RL, attention, curiosity, predictor, NM, temporal |
| [I/O](docs/io.md) | Ring buffers, modalities, encoders, sensors, actuators, ISR |
| [Swarm](docs/swarm.md) | Federated learning, mesh networking, gossip |
| [Safety](docs/safety.md) | Reflexes, entropy monitor, ECC, migration, senescence |
| [Industrial Safety Compliance](docs/industrial_safety_compliance.md) | Traceability matrix for ISO 26262 (ASIL-D), IEC 61508 (SIL 3), DO-178C (DAL A) |
| [System](docs/system.md) | Boot, persistence, watchdog, power management, OTA |
| [Telemetry](docs/telemetry.md) | Spike tracing, XAI causal graph, UART export |
| [Getting Started](docs/getting-started.md) | Build, flash, debug, contribute |
| [Roadmap](ROADMAP.md) | Full TDD compliance analysis |

---

## 🛡️ Security

We take security seriously. Please report vulnerabilities to **security@hkl1.dev** (or open a [draft advisory](https://github.com/Hakille-ai/HKL-1/security/advisories/new)).

See [SECURITY.md](SECURITY.md) for our full security policy and PGP key.

---

## 🤝 Contributing

We welcome contributors of all skill levels! HKL-1 is a complex project, but we've made it easy to get started:

- 🐛 **Found a bug?** [Open an issue](https://github.com/Hakille-ai/HKL-1/issues/new)
- 💡 **Have an idea?** Start a [Discussion](https://github.com/Hakille-ai/HKL-1/discussions)
- 🔧 **Want to contribute?** Read [CONTRIBUTING.md](CONTRIBUTING.md)
- 📖 **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

**375 tests pass — we maintain zero warnings at all times.**

---

## 📄 License

Licensed under either of:

- [MIT License](LICENSE)
- Apache License, Version 2.0

at your option.

---

<p align="center">
  <b>Built with ❤️ and Rust</b><br>
  <sub>Zero dependencies · Zero floating-point · Zero compromises</sub>
</p>
