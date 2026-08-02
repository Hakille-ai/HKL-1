# HKL-1 Release Readiness

This document records the current verification boundary for the HKL-1/HKL-2
codebase. It is intentionally stricter than the README marketing language: a
green host test suite is not the same as a production hardware release.

## Current status

HKL-1 is ready for continued engineering, experimentation, and host-side
integration testing. It is not yet a production hardware release until target
device boot, flashing, telemetry, persistence, and long-running safety behavior
are verified on real boards.

HKL-2 is ready for guarded training experiments. It now has a minimal bounded
training loop that can authorize model updates through the cognition gate. It is
not yet a trained foundation model: convergence, datasets, checkpoints, and
quality metrics still need to be established.

## Verified locally

The following commands passed on the current branch:

- `cargo fmt --all --check`
- `cargo test`
- `cargo test --features std,alloc,simd,hkl2 -- --test-threads=1`
- `cargo test --all-features -- --test-threads=1`
- `cargo clippy --features std,alloc,simd,hkl2 --all-targets -- -D warnings`
- `cargo check --no-default-features`
- `cargo run --example hkl2_training_loop --features hkl2`
- `cargo build --target thumbv7em-none-eabihf --no-default-features --features stm32f7,alloc`
- `cargo build --target riscv32imac-unknown-none-elf --no-default-features --features hifive1,alloc`
- `cargo build --target riscv32imc-unknown-none-elf --no-default-features --features esp32c6,alloc`

The HKL-2 training example produced `LearningAllowed` decisions for three
guarded cycles, with non-saturated loss and stable supervision.

## Production blockers remaining

- Hardware boot and runtime validation on STM32F7, HiFive1, and ESP32-C6 boards.
- Long-running persistence and rollback tests against real flash constraints.
- Memory-footprint and latency budgets per target profile.
- A documented model checkpoint format and recovery flow for HKL-2 training.
- Dataset definitions, evaluation tasks, and convergence metrics.
- Review of all `unsafe` singletons and MMIO access contracts.
- README claim cleanup so public claims match measured verification.

## Training readiness boundary

The training stack is currently suitable for:

- deterministic smoke tests;
- bounded toy-corpus training loops;
- guard, readiness, runtime-gate, and supervision validation;
- testing invalid-token, truncated-sequence, and saturated-loss behavior.

It is not yet suitable for:

- unattended long-running training;
- claims of useful language-model behavior;
- deployment of learned weights without an evaluation suite;
- production autonomy without board-level safety validation.
