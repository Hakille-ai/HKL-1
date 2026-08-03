# Architecture

## Overview

HKL-1 is a modular, layered neuromorphic AI system designed for bare-metal embedded environments. The architecture follows a publish-subscribe model with lock-free ring buffers for inter-module communication.

```
┌──────────────────────────────────────────────────────────────────┐
│                         TELEMETRY                                  │
│              (Spike Trace, XAI)                                    │
├──────────────────────────────────────────────────────────────────┤
│           HKL-2 SPIKING FOUNDATION MODEL (features = "hkl2")    │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │
│  │Eprop │ │Surrog│ │Spike │ │BPE   │ │Spike │ │Train │           │
│  │Engine│ │Grad  │ │Embed │ │Token │ │Trans │ │er/DL │           │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘           │
├──────────────────────────────────────────────────────────────────┤
│                         SAFETY                                     │
│    (Reflexes, Entropy Monitor, Hardware Resilience, Senescence)    │
├──────────────────────────────────────────────────────────────────┤
│                         BIO-INSPIRED MODULES (src/bio)            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Astro │ │Strio │ │Thal  │ │Hippo │ │Cereb │                    │
│  │cytes │ │some  │ │amus  │ │campus│ │ellum │                    │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                    │
├──────────────────────────────────────────────────────────────────┤
│                         COGNITIVE                                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │Actor │ │Attn  │ │Curios│ │Pred  │ │Neuro │ │Propr │ │Templ │  │
│  │Critic│ │      │ │ity   │ │ictor │ │Calib │ │iocpt │ │Cogn  │  │
│  ├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤  │
│  │Episod│ │Cont  │ │Refl  │ │      │ │      │ │      │ │      │  │
│  │ic Mem│ │inual │ │Override│     │ │      │ │      │ │      │  │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
├──────────────────────────────────────────────────────────────────┤
│             BIO-COMPILATION eFPGA (src/efpga)                    │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Stab  │ │Verilog││Bit   │ │HW    │ │eFPGA │                    │
│  │Analyz│ │Gen RTL││stream│ │Sim   │ │Engine│                    │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                    │
├──────────────────────────────────────────────────────────────────┤
│             NLP & SYMBOLIC COGNITION (src/nlp)                   │

│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Token │ │Decoder││Verbal│ │Symbol│ │Dialog│                    │
│  │Enc/Phase│WTA  │ │izer  │ │Graph │ │Engine│                    │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                    │
├──────────────────────────────────────────────────────────────────┤
│             AUDITORY & SPEECH INTELLIGENCE (src/audio)           │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                             │
│  │Cochlea││A1/Form││Pitch │ │Voice │                             │
│  │Gamma  ││ants  │ │F0/Rhy│ │Synth │                             │
│  └──────┘ └──────┘ └──────┘ └──────┘                             │
├──────────────────────────────────────────────────────────────────┤
│           VISION & INTUITIVE PHYSICS ENGINE (src/vision)         │


│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │Retina│ │V1    │ │V4    │ │MT    │ │Depth │ │Physic│ │Pred  │  │
│  │DoG/DVS││Gabor │ │Shape │ │Motion│ │Stereo│ │ Engine││Cod/Conv│
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
├──────────────────────────────────────────────────────────────────┤
│                      SNN (Spiking Neural Network)                  │

│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │
│  │Neuron│ │Synaps│ │Netwk │ │Plast │ │Homeo │ │Neuro │           │
│  │(LIF) │ │e     │ │ork   │ │(STDP)│ │stasis│ │genesis│           │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘           │
├──────────────────────────────────────────────────────────────────┤
│                         CORE                                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │
│  │Math  │ │Memory│ │Time  │ │Entrop│ │Crypto│ │Pool  │           │
│  │(FP)  │ │      │ │(Clk) │ │y     │ │      │ │Alloc │           │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘           │
├──────────────────────────────────────────────────────────────────┤
│                     SYSTEM & I/O                                    │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │Boot  │ │Persi │ │Watch │ │Power │ │OTA   │ │Enc/  │ │Sens/  │  │
│  │      │ │stence│ │dog   │ │      │ │      │ │Dec   │ │Act+ISR│  │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
├──────────────────────────────────────────────────────────────────┤
│                      SWARM                                          │
│           ┌──────────────┐  ┌──────────────┐                        │
│           │  Federated   │  │  Mesh        │                        │
│           │  Learning    │  │  Network     │                        │
│           └──────────────┘  └──────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Sensors + ISRs** capture environmental data and enqueue spikes via NVIC interrupt handlers → `GLOBAL_SPIKE_QUEUE`
2. **Encoder** converts sensor readings (I2C/SPI/ADC) into spike trains mapped to layers 0–5 by modality
3. **SNN** processes spikes through LIF neurons (6 types) with STDP plasticity, homeostatic scaling, and structural neurogenesis
4. **Senescence** ages synapses each metabolic cycle; aged synapses decay in weight and plasticity, then are pruned
5. **Bio-Inspired Modules** process SNN activity at bio-realistic timescales: thalamus sensory gating (every step), striosome dopamine-gated action selection (10ms), hippocampus memory consolidation (50ms), astrocyte glial modulation (100ms), cerebellum motor refinement (20ms). Hippocampus SWR events bridge to cognitive episodic memory.
6. **Cognitive** modules receive SNN + bio module output: actor-critic RL, attention, curiosity, prediction, neuromodulation, proprioception, temporal cognition, episodic memory, continual learning
6. **Safety** monitors entropy and triggers reflexes; `check_emergencies()` validates system integrity at boot
7. **Swarm** communicates with peers via mesh networking with gossip protocol and federated learning
8. **Telemetry** records spike traces and generates XAI explanations
9. **HKL-2 Foundation Model (`feature = "hkl2"`)** provides eligibility propagation (`src/learning/`), 256D spike population embeddings (`src/embedding/`), Spiking Self-Attention transformers (`src/transformer/`), end-to-end dataset training (`src/training/`), and bounded cognition control/planning/audit/evaluation/readiness/runtime gating/supervision/episode running (`src/cognition/`)
10. **Persistence** saves/restores full system state to flash (3 rotating slots, CRC32, optional ChaCha20)
11. **OTA** validates and applies firmware updates via dual-bank flash switching

## Boot Timeline (t=0 → 22ms)

| Time | Stage |
|---|---|
| 0.0 ms | Clock init (SysTick, PLL, metabolic clock) |
| 0.5 ms | Hardware peripherals (CPACR, SCB_VTOR, MPU, SCS_CCR) |
| 1.0 ms | Read boot config (UID 96-bit OTP, hardware version, PUF) |
| 2.0 ms | Init persistence (load J-0 checkpoint) |
| 2.5 ms | Init entropy (seed PRNG + TRNG) |
| 3.0 ms | Init spike logger (telemetry buffer) |
| 4.0 ms | Enable sensor interrupts (NVIC_ISER0/1/2) |
| 5.0 ms | Init reflex arcs (L0→L6→L4, fixed weights) |
| 5.5 ms | Check emergencies (NEURON_COUNT, entropy, reflexes) |
| 6.0 ms | Restore state from flash (or fresh init) |
| 17 ms | Init cognitive modules (actor, episodic memory, curiosity, predictor, neuromodulators, attention, temporal, continual learning) |
| 18 ms | Init bio-inspired modules (astrocytes, striosome, thalamus, hippocampus, cerebellum) |
| 19 ms | Init eFPGA bio-compilation engine |
| 19.5 ms | Init entropy monitor + XAI / telemetry |
| 20 ms | Main loop start (SNN step + bio module + eFPGA pipeline) |
| 21 ms | OTA check → apply → confirm (if candidate available) |

## Memory Model

Memory allocation supports both static compile-time bounds and dynamic hardware scaling:

- `MAX_NEURONS = 256,000` — Fixed-size maximum neuron array (`[MaybeUninit<NeuronState>; MAX_NEURONS]`)
- `MAX_SYNAPSES = 4,194,304` — Fixed-size maximum synapse array (`[MaybeUninit<SynapseSlot>; MAX_SYNAPSES]`)
- `ADAPTIVE_MEMORY` — Dynamic memory allocation engine (`set_capacity`) that auto-adjusts SNN capacity based on available system RAM scanned by `HardwareDetector`.
- `RING_BUFFER_SIZE = 4096` — Inter-module communication buffers
- `SPIKE_TRACE_BUFFER = 8192` — Telemetry buffer
- `PERSISTENCE_SLOTS = 3` — Flash save slots (J-0, J-1, J-2)
- `WeightDelta.entries = [(Weight, SynapseId); 2048]` — Swarm mesh delta buffer

## Concurrency & Parallel Execution Model

- **Bare-metal execution (`no_std`)**: Single-threaded deterministic loop driven by SysTick/interrupts.
- **Parallel Execution Engine (`std`)**: `step_parallel(num_threads)` leverages multi-core CPUs via `std::thread::scope`, splitting STDP eligibility trace updates and SNN evaluation across hardware worker threads.
- Interrupt-driven for time-critical paths (NVIC TIM2/ADC/EXTI/SPI/I2C)
- Lock-free ring buffers for ISR-to-main communication (`reserve_write`/`commit_write`)
- `AtomicU32`/`AtomicU64` for shared counters
- RISC-V RV32-IMC atomics via `FetchAtomic` trait (`core/atomic.rs`) — falls back to `UnsafeCell` when hardware atomic RMW instructions are absent (A extension)
- `MaybeUninit::uninit()` for zero-cost in-place static memory initialization (`init_network`)


## Platform Support

BSP modules are gated by Cargo features:

| Feature | Target | Linker Script |
|---|---|---|
| `stm32f7` | `thumbv7em-none-eabihf` (ARM Cortex-M7) | `stm32f746.ld` |
| `hifive1` | `riscv32imac-unknown-none-elf` (RISC-V) | `hifive1.ld` |
| `esp32c6` | `riscv32imc-unknown-none-elf` (RISC-V) | `esp32c6.ld` |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `#![no_std]` | Target microcontrollers, no OS dependency |
| FixedPoint (Q16.16) | No FPU required, deterministic across platforms |
| Weight (Q8.8) | Optimized for synaptic weights (±128 range, ~0.004 resolution) |
| Static pools | No runtime allocator, predictable latency |
| Lock-free rings | Safe ISR-to-main data transfer without mutexes |
| ChaCha20 | Lightweight, constant-time, no hardware crypto needed |
| MaybeUninit arrays | Zero-cost initialization of large static arrays |
| Rust 2024 edition | `unsafe extern "C"`, `#[unsafe(no_mangle)]`, `#[unsafe(link_section)]` |
| 3 BSP backends | Cross-platform support without trait bloat, gated by features |

## Test Status

**509 unit tests** — all passing — 0 warnings — 0 errors.

Integration tests (14 tests) pass with `--features std` — requires the `std` feature for test harness compatibility.
