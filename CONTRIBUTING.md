# Contributing to HKL-1

First off, thank you for considering contributing to HKL-1! 🧠

We're building a neuromorphic AI engine that runs on bare-metal microcontrollers — and we need your help. Whether you're a Rust expert, an embedded systems engineer, a neuroscientist, or just curious, there's a place for you here.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Feature Requests & Bug Reports](#feature-requests--bug-reports)
- [Documentation](#documentation)
- [Community](#community)

---

## Code of Conduct

This project adheres to the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to **conduct@hkl1.dev**.

---

## Getting Started

### 1. Understand the project

Read through the [README](../README.md) and [docs/architecture.md](../docs/architecture.md) to understand the system design.

### 2. Pick an area

Check the [open issues](https://github.com/Hakille-ai/HKL-1/issues) labeled:

| Label | Description |
|---|---|
| `good first issue` | Beginner-friendly, well-scoped tasks |
| `help wanted` | Tasks where we'd love assistance |
| `P5` | Synaptic migration / senescence enhancements |
| `QEMU` | QEMU-based integration testing |
| `documentation` | Docs improvements |
| `testing` | Test coverage expansion |

### 3. Fork & clone

```bash
git clone https://github.com/Hakille-ai/HKL-1
cd hkl1
```

### 4. Set up your environment

See [Development Environment](#development-environment) below.

---

## Development Environment

### Prerequisites

- **Rust 1.95.0+** (edition 2024)
- `rustup target add thumbv7em-none-eabihf` (for ARM builds)
- `rustup target add riscv32imac-unknown-none-elf` (for RISC-V builds)
- `rustup target add riscv32imc-unknown-none-elf` (for ESP32-C6 builds)
- `cargo install cargo-deny` (optional, for dependency auditing)

### Quick validation

```bash
cargo check             # No errors
cargo clippy            # No warnings
cargo test --lib        # 375 tests pass
```

### Rust 2024 Edition Notes

HKL-1 uses Rust 2024 edition syntax. If you're porting code from older editions:

| Old Syntax | New Syntax |
|---|---|
| `extern "C"` | `unsafe extern "C"` |
| `#[no_mangle]` | `#[unsafe(no_mangle)]` |
| `#[link_section = "..."]` | `#[unsafe(link_section = "...")]` |
| Implicit `unsafe` blocks | Explicit `unsafe {}` |

---

## Project Structure

```
hkl1/
├── src/
│   ├── core/             # Math, Memory, Time, Entropy, Crypto
│   │   ├── math.rs       # FixedPoint Q16.16, Weight Q8.8, PRNG, Matrix/Vector
│   │   ├── memory.rs     # NeuronState, SynapseSlot, GlobalPool, pools
│   │   ├── time.rs       # MetabolicClock, TemporalHierarchy, TimeWarper
│   │   ├── entropy.rs    # Shannon entropy, adaptive thresholds, CognitiveMode
│   │   └── crypto.rs     # ChaCha20, PUF, HMAC-SHA256, SecureErase
│   ├── snn/              # Spiking Neural Network
│   │   ├── neuron.rs     # LIFNeuron, neuron types, GLOBAL_NEUROMODULATORS
│   │   ├── synapse.rs    # Synapse, apply_senescence(), init_reflex_arcs()
│   │   ├── network.rs    # Network, predictive_cycle(), novelty, energy_adaption()
│   │   ├── plasticity.rs # STDP, Hebbian, eligibility traces, neuromodulated
│   │   ├── homeostasis.rs# Target rate, gain, scaling, layer compensation
│   │   └── neurogenesis.rs# Prune, create, senescence, maintenance_cycle()
│   ├── cognitive/        # Cognitive functions
│   │   ├── actor.rs      # ActorCritic, policy, value, TD-error
│   │   ├── attention.rs  # Saliency, top-down routing, WTA
│   │   ├── curiosity.rs  # Habituation, boredom, thermal noise, dreaming
│   │   ├── predictor.rs  # Prototypes, Hebbian, confidence, transition buffer
│   │   ├── neuromodulation.rs # DA/5-HT/NA/ACh, calibration, sync_to_snn()
│   │   ├── proprioception.rs  # Efference copy, error, correction, body model
│   │   ├── temporal.rs   # Time cells, sequence buffer, interval timing
│   │   └── reflex_override.rs # Cognitive reflex suppression
│   ├── io/               # Input/Output
│   │   ├── buffers.rs    # RingBuffer, modality->layer mapping
│   │   ├── encoder.rs    # Rate, temporal, population, place coding
│   │   ├── decoder.rs    # Rate, temporal, WTA, vector decoding
│   │   ├── sensors.rs    # I2C/SPI/ADC MMIO, SensorManager, error tracking
│   │   ├── actuators.rs  # PWM/GPIO/DAC MMIO, DacOutput::init()
│   │   └── isr.rs        # TIM2/ADC/EXTI/SPI/I2C handlers, isr_push_spike()
│   ├── swarm/            # Swarm intelligence
│   │   ├── federated.rs  # Federated learning, DP, adaptive topology
│   │   └── mesh.rs       # Peer discovery, gossip, clock sync, remote spikes
│   ├── safety/           # Safety & monitoring
│   │   ├── reflexes.rs   # 5 reflex types, cognitive override
│   │   ├── entropy_monitor.rs # Shannon entropy, adaptive thresholds
│   │   └── hardware_resilience.rs # ECC, migration, senescence, bad sectors
│   ├── system/           # System services
│   │   ├── boot.rs       # Boot sequence, HW init, check_emergencies(), OTA cycle
│   │   ├── persistence.rs# Flash dump, 3 slots, CRC32, secure erase
│   │   ├── watchdog.rs   # Neurological watchdog, graduated escalation
│   │   ├── power.rs      # DVFS, 5 states, harvesting, V_th coupling
│   │   └── ota.rs        # Dual-bank, CRC32, rollback, no_std
│   └── telemetry/        # Observability
│       ├── spike_trace.rs# 8192 buffer, bursts, export
│       └── xai.rs        # CausalGraph, feature attribution, UART export
├── docs/                 # Documentation
├── tests/                # Integration tests
├── stm32f746.ld          # ARM Cortex-M7 linker script
├── hifive1.ld            # RISC-V linker script
├── esp32c6.ld            # ESP32-C6 linker script
├── .github/workflows/    # CI configuration
└── Cargo.toml
```

---

## Coding Standards

### Rust Style

- Follow the [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/)
- Run `cargo fmt` before committing
- Run `cargo clippy` — **zero warnings required**
- Use `#![no_std]` compatible code (no `std::*` imports outside `#[cfg(test)]`)
- Use `FixedPoint` over `f32` for all new numerical code
- Prefer `const fn` where possible
- Mark all `unsafe` blocks with safety comments: `// SAFETY: ...`

### Naming

| Convention | Example |
|---|---|
| Types: `PascalCase` | `LIFNeuron`, `FixedPoint`, `BinaryDump` |
| Functions: `snake_case` | `apply_senescence()`, `init_reflex_arcs()` |
| Constants: `SCREAMING_SNAKE_CASE` | `MAX_NEURONS`, `GLOBAL_SPIKE_QUEUE` |
| Modules: `snake_case` | `snn/neuron.rs`, `system/persistence.rs` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(snn): add synaptic senescence with configurable max_age
fix(boot): correct NVIC_ISER1 base address
docs(architecture): update data flow diagram
test(watchdog): add escalation level tests
refactor(core): extract Matrix multiply into const fn
```

### Testing

- All new code must be tested
- Tests go in a `#[cfg(test)] mod tests { ... }` at the bottom of the source file
- Use descriptive test names: `test_synapse_senescence_prunes_at_max_age`
- Integration tests go in `tests/integration.rs`
- Run `cargo test --lib` before pushing

---

## Pull Request Process

### 1. Before you start

- Check existing [issues](https://github.com/Hakille-ai/HKL-1/issues) and [discussions](https://github.com/Hakille-ai/HKL-1/discussions)
- For significant changes, open an issue first to discuss

### 2. Create a branch

```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/my-bug
```

### 3. Make your changes

- Keep changes focused — one logical change per PR
- Follow the coding standards
- Add tests for new functionality
- Update documentation if needed

### 4. Validate

```bash
cargo check                    # Must pass
cargo clippy                   # Zero warnings
cargo test --lib               # All 375+ tests pass
cargo test --lib $YOUR_TEST    # Your specific test passes
```

### 5. Commit

```bash
git add .
git commit -m "feat(area): concise description"
```

### 6. Push and open a PR

```bash
git push origin feat/my-feature
```

Then open a Pull Request on GitHub. In the description:

- **What** does this change?
- **Why** is it needed? (link to issue if applicable)
- **How** was it tested?
- **Checklist**: `[x]` tests, `[x]` clippy, `[x]` docs

### 7. Review process

- At least one maintainer review required
- CI must pass (check, clippy, test)
- Address review feedback with additional commits
- Squash commits before merge

---

## Feature Requests & Bug Reports

### 🐛 Bug Reports

Open a [Bug Report](https://github.com/Hakille-ai/HKL-1/issues/new?template=bug_report.md) with:

- HKL-1 version (commit hash)
- Target platform (host, STM32F7, etc.)
- Steps to reproduce
- Expected vs actual behavior
- `cargo test --lib` output if relevant

### 💡 Feature Requests

Open a [Feature Request](https://github.com/Hakille-ai/HKL-1/issues/new?template=feature_request.md) with:

- What problem does it solve?
- Proposed solution
- Alternative approaches considered
- Is it `#![no_std]` compatible?

---

## Documentation

- API docs: written as `///` doc comments on all public items
- Module docs: `//!` at module root explaining purpose and design
- Guides: Markdown files in `docs/`
- When you change a public API, update the relevant doc comments and guide

Build docs locally:

```bash
cargo doc --no-deps --open
```

---

## Community

- **Discussions**: [GitHub Discussions](https://github.com/Hakille-ai/HKL-1/discussions)
- **Issues**: [GitHub Issues](https://github.com/Hakille-ai/HKL-1/issues)
- **Security**: [SECURITY.md](SECURITY.md)

---

## Recognition

Every contributor will be acknowledged in our release notes. Significant contributors may be invited to become project maintainers.

**Thank you for helping build the future of embedded intelligence!** 🧠✨
