# Getting Started

## Prerequisites

- **Rust 1.95.0+** (edition 2024 — requires `unsafe extern "C"`, `#[unsafe(no_mangle)]`, `#[unsafe(link_section)]`)
- Target: `thumbv7em-none-eabihf` (ARM Cortex-M7), `riscv32imac-unknown-none-elf` (SiFive HiFive1), `riscv32imc-unknown-none-elf` (ESP32-C6)
- RISC-V RV32-IMC targets use the `FetchAtomic` trait (`core/atomic.rs`) for atomic operations when the A extension is absent — automatically handled at compile time
- Optional: JTAG/SWD debugger for flashing hardware

## Build

```bash
# Clone the repository
git clone <repo-url> hkl1
cd hkl1

# Build for host (native, for testing algorithms)
cargo build

# Run the minimal public API example
cargo run --example minimal_prelude --features std

# Run the experimental HKL-2 training loop
cargo run --example hkl2_training_loop --features hkl2

# Build for bare-metal target
cargo build --target thumbv7em-none-eabihf

# Build with BSP features
cargo build --features stm32f7        # STM32F746 (ARM Cortex-M7)
cargo build --features hifive1        # SiFive HiFive1 (RISC-V)
cargo build --features esp32c6        # ESP32-C6 (RISC-V)

# Release build (optimized)
cargo build --release
```

## Check

```bash
# Syntax and type checking
cargo check

# Lint using the host feature profile covered by CI
cargo clippy --features std,alloc,simd,hkl2 --all-targets -- -D warnings

# Format check
cargo fmt --check
```

## Test, Benchmark & Examples

```bash
# Run complete host + HKL-2 test suite
cargo test --features std,alloc,simd,hkl2

# Run specific test
cargo test --lib test_neuron_step

# Run performance benchmark suite
cargo bench --bench snn_benchmark --features std,simd

# Run cognitive engine integration demo
cargo run --example snn_cognitive_demo --features std,simd

# Run experimental HKL-2 training loop
cargo run --example hkl2_training_loop --features hkl2
```

Test mode uses `std` feature to enable `#[cfg(test)]` and test harness.

## Minimal API Prelude

Applications can start from the compact embedded prelude:

```rust
use hkl1::prelude::*;

let neuron = NeuronId::new(0);
let threshold = FixedPoint::from_f32(0.75);
let weight = Weight::from_f32(0.5);

assert_eq!(neuron.index(), 0);
assert!(threshold > FixedPoint::ZERO);
assert!(weight > Weight::ZERO);
```

The prelude is allocation-free and re-exports stable fixed-point, neuron,
synapse, network, safety, I/O, and telemetry types used by firmware and host
simulation examples. Runtime singleton accessors remain in their module paths
so firmware code can make initialization and aliasing boundaries explicit.

## Documentation

```bash
# Build and open docs
cargo doc --no-deps --open
```

The documentation is generated from source code doc comments and reflects the current implementation.

## Bare-Metal QEMU Emulation

```bash
# Run ARM Cortex-M7 bare-metal QEMU simulation harness
python scripts/run_qemu_arm.py

# Run RISC-V RV32 bare-metal QEMU simulation harness
python scripts/run_qemu_riscv.py
```

## Flashing (Hardware)

For ARM Cortex-M targets (STM32F7):

```bash
# Using probe-rs
cargo run --target thumbv7em-none-eabihf --features stm32f7

# Using openocd + gdb
openocd -f interface/stlink.cfg -f target/stm32f7x.cfg
arm-none-eabi-gdb target/thumbv7em-none-eabihf/debug/hkl1
(gdb) load
(gdb) continue
```

For RISC-V targets (HiFive1):

```bash
# Using OpenOCD + GDB
openocd -f board/hifive1.cfg
riscv64-unknown-elf-gdb target/riscv32imac-unknown-none-elf/debug/hkl1
(gdb) load
(gdb) continue
```

For ESP32-C6:

```bash
# Using espflash
espflash flash target/riscv32imac-unknown-none-elf/release/hkl1 --features esp32c6
```

## Feature Reference

| Feature | Cargo.toml flag | Description |
|---|---|---|
| Default | `--features default` | `alloc` enabled (required for RISC-V) |
| stm32f7 | `--features stm32f7` | BSP STM32F746 (ARM Cortex-M7) |
| hifive1 | `--features hifive1` | BSP SiFive HiFive1 (RISC-V) |
| esp32c6 | `--features esp32c6` | BSP ESP32-C6 (RISC-V) |
| std | `--features std` | Host/stdlib mode for testing |
| alloc | `--features alloc` | Enable heap allocation |
| simd | `--features simd` | SIMD matrix ops |
| flash | `--features flash` | Flash persistence driver |
| encryption | `--features encryption` | ChaCha20 dump encryption |

## Project Structure

```
hkl1/
├── src/
│   ├── lib.rs              # Crate root, global constants
│   ├── core/               # Atomic, Math (FixedPoint), Memory (pools), Time (clock), Entropy, Crypto
│   ├── snn/                # Neuron (LIF), Synapse, Network, Plasticity (STDP), Homeostasis, Neurogenesis
│   ├── cognitive/          # Actor-Critic, Attention, Curiosity, Predictor, NeuromodCalib, Proprioception, Temporal, ReflexOverride, Episodic, Continual
│   ├── bio/                # Astrocytes, Striosome, Thalamus, Hippocampus, Cerebellum
│   ├── efpga/              # Stability Analyzer, HDL Generator, Bitstream Encoder, HW Sim
│   ├── nlp/                # Token Encoder/Phase WTA, Decoder, Verbalizer, Symbol Graph, Dialogue Engine
│   ├── audio/              # Cochlea Gamma, A1/Formants, Pitch F0/Rhythm, Voice Synth
│   ├── vision/             # Retina DoG/DVS, V1 Gabor, V4 Shape, MT Motion, Depth Stereo, Physics Engine, Predictive Coding
│   ├── io/                 # Buffers (ring), Encoder, Decoder, Sensors (I2C/SPI/ADC), Actuators (PWM/DAC), ISR
│   ├── swarm/              # Federated Learning, Mesh Networking
│   ├── safety/             # Reflexes, Entropy Monitor, Hardware Resilience (ECC, migration, senescence)
│   ├── system/             # Boot, Persistence, Watchdog, Power (DVFS), OTA
│   └── telemetry/          # Spike Trace, XAI
├── bsp/                    # Board support packages (stm32f7, hifive1, esp32c6)
├── docs/                   # Documentation
├── tests/                  # Integration tests
├── stm32f746.ld            # Linker script (ARM Cortex-M7)
├── hifive1.ld              # Linker script (RISC-V)
├── esp32c6.ld              # Linker script (ESP32-C6)
├── Cargo.toml
├── deny.toml               # cargo deny configuration (zero deps)
└── .github/workflows/ci.yml
```

## Development Workflow

1. **Algorithm development**: Build for host (`cargo build`), test logic
2. **Integration testing**: Build for target, flash to hardware
3. **Performance tuning**: Use telemetry to identify bottlenecks
4. **Safety validation**: Verify reflexes with fault injection
5. **Field testing**: Deploy in swarm, monitor telemetry

## Code Style

- Follow existing patterns (look at neighbor files)
- Use `#![no_std]` compatible code
- No external dependencies
- Prefer `const fn` where possible
- Use `FixedPoint` over `f32` for new code
- Document public items with doc comments
- Rust 2024 edition: use `unsafe extern "C"`, `#[unsafe(no_mangle)]`, `#[unsafe(link_section)]`, explicit `unsafe {}` blocks
