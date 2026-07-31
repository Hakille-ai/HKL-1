# Core Module

The core module provides foundational primitives used by every other module in HKL-1.

## Atomic (`core/atomic.rs`)

Cross-platform atomics for RISC-V targets without the A extension (RV32-IMC).

- `FetchAtomic` trait adds `fetch_add`, `fetch_sub`, `fetch_or`, `swap`, `compare_exchange` to `AtomicU32` and `AtomicUsize`
- On targets with hardware atomics, inherent methods shadow the trait (zero overhead)
- On RISC-V without A (ESP32-C6), trait methods use `UnsafeCell` for plain load/store-modify-write
- `pub use core::sync::atomic::Ordering` re-exported for convenience

## Math (`core/math.rs`)

Fixed-point arithmetic (`Q16.16`) replacing all floating-point operations.

### FixedPoint (Q16.16)

- 32-bit signed integer (`i32`) with 16 fractional bits
- Range: ±32768, resolution: ~1.5 × 10⁻⁵
- Implements: `+`, `-`, `*`, `/`, `exp`, `ln`, `sqrt`, `pow`, `powf`, `sin`, `cos`, `fract`, `floor`, `ceil`
- Trigonometry: Bhaskara I approximation with robust 4-quadrant reduction (`rem_euclid(TAU)`) (~75.97 Million ops/sec)
- No floating-point hardware required

### Weight (Q8.8)

- 16-bit signed integer (`i16`) with 8 fractional bits
- Optimized for synaptic weights: range ±128, resolution ~0.004

### XorShift64Star

- 64-bit xorshift PRNG (passes BigCrush)
- `seed()` → `next_u32()` → `next_u64()` → `next_f32()` → `next_gaussian()`

### Matrix & Vector

- `Matrix<const N: usize>` — Fixed-size square matrix for weight operations
- `Vector<const N: usize>` — Fixed-size vector with 4-way SIMD chunk unrolling (`feature = "simd"` enabled) for `dot`, `add_assign`, `elementwise_mul`, and `sum` (~19.5 Million ops/sec on 64-dim vectors)
- Both use `FixedPoint` elements

## Memory (`core/memory.rs`)

Static pool allocators for neurons and synapses.

### Key Types

| Type | Description |
|---|---|
| `NeuronId(pub u16)` | Dense neuron identifier (0..MAX_NEURONS) |
| `SynapseId(pub u16)` | Dense synapse identifier (0..MAX_SYNAPSES) |
| `NeuronType` | Enum: Excitatory, Inhibitory, Modulatory, Pacemaker, Sensory, Motor |
| `NeuronFlags` | Bitfield: active, refractory, adapting, learning, bursting |
| `NeuronState` | Full neuron state: membrane potential, threshold, trace, etc. |
| `SynapseSlot` | Synapse with pre/post IDs, weight, delay, tag |

### Pool Allocators

- `GlobalPool` — Static pool for 4096 neurons (bitfield allocation)
- `SynapsePool` — Static pool for 65536 synapses (linked-list free list)
- Both use `MaybeUninit::uninit()` for zero-cost static initialization

### Memory Layout

```
NEURON_ARRAY: [MaybeUninit<NeuronState>; MAX_NEURONS]
SYNAPSE_ARRAY: [MaybeUninit<SynapseSlot>; MAX_SYNAPSES]
```

## Time (`core/time.rs`)

Multi-scale biological timing system.

### MetabolicClock

| Frequency | Purpose |
|---|---|
| 1 MHz | High-resolution timing |
| 1 kHz | Neural simulation tick |
| 100 Hz | Motor control |
| 10 Hz | Sensor fusion |
| 1 Hz | Metabolic heartbeat (senescence cycle) |

### TemporalHierarchy

Five circular buffers at different timescales (ultrafast, fast, medium, slow, ultraslow), each 1024 entries. Used for multi-scale temporal processing.

### TimeWarper

Accelerates simulation time for predictive coding. Up to 1000× speed. Used in `predictive_cycle()` for mental simulation.

## Entropy (`core/entropy.rs`)

Cognitive entropy monitoring and stochastic noise generation.

- Shannon entropy from weight histograms
- Thermal noise sampling (hardware sensor or PRNG fallback)
- TRNG sampling (hardware or PRNG fallback)
- Entropy health states: Healthy, LowEntropy, HighEntropy
- Adaptive thresholds: mean ± 2σ, smooth EMA
- Cognitive modes: `CognitiveMode` (Exploratory, Exploit, Crisis, Stability)
- Correlation with neuromodulation: `apply_cognitive_mode()` sets DA/5-HT/NA/ACh concentrations

### Boot Integration

`init_entropy(seed)` is called during boot (t=2.5ms) to seed the PRNG and initialize the TRNG interface.

## Crypto (`core/crypto.rs`)

### ChaCha20

- 20-round stream cipher
- 256-bit key, 96-bit nonce
- Constant-time operations
- No hardware crypto dependency

### PUF (Physically Unclonable Function)

- Interface for SRAM/ring-oscillator PUF
- Device-unique key generation
- Ephemeral key storage with secure erase
- UID extraction from OTP memory (96-bit STM32F7 UID at 0x1FFF_7A10) in `read_boot_config()`

### HMAC-SHA256

- For firmware validation (OTA integrity checks)
- Implemented from scratch, no external dependency

### Secure Erase

- XOR overwrite + verification
- Used for flash slot erasure in persistence module
