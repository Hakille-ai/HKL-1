# Execution Runbook

This runbook describes the exact order for validating HKL-1/HKL-2 from the PC
to real hardware.

## 0. Preconditions

- Work from a clean Git branch.
- Record the commit hash under test.
- Save all terminal logs.
- Use one board at a time.
- Do not mix new hardware wiring with new firmware changes in the same test
  step.

## 1. Host validation

Run:

```powershell
cargo fmt --all --check
cargo test
cargo test --features std,alloc,simd,hkl2 -- --test-threads=1
cargo test --all-features -- --test-threads=1
cargo clippy --features std,alloc,simd,hkl2 --all-targets -- -D warnings
cargo check --no-default-features
cargo run --example hkl2_training_loop --features hkl2
```

Pass criteria:

- All commands exit with code 0.
- HKL-2 training example reports guarded `LearningAllowed` cycles.
- Loss is finite and not saturated.
- No panic, overflow, or persistence side effect appears at time zero.

## 2. Cross-compilation validation

Install targets if missing:

```powershell
rustup target add thumbv7em-none-eabihf
rustup target add riscv32imac-unknown-none-elf
rustup target add riscv32imc-unknown-none-elf
```

Build:

```powershell
cargo build --target thumbv7em-none-eabihf --no-default-features --features stm32f7,alloc
cargo build --target riscv32imac-unknown-none-elf --no-default-features --features hifive1,alloc
cargo build --target riscv32imc-unknown-none-elf --no-default-features --features esp32c6,alloc
```

Pass criteria:

- All targets compile.
- No host-only symbol leaks into no-std firmware builds.
- No target-incompatible inline assembly reaches the wrong architecture.

## 3. STM32F7 board bring-up

Recommended first target: STM32F746G-DISCO.

Steps:

1. Connect USB data cable to ST-LINK port.
2. Open a serial terminal if the board exposes virtual COM.
3. Flash the STM32F7 build using the selected toolchain.
4. Reset the board.
5. Capture UART logs.
6. Capture boot marker GPIO with logic analyzer if firmware exposes one.
7. Run for 10 minutes with no sensors attached.
8. Add BME280.
9. Add MPU-6050.
10. Add I2S microphone only after I2C is stable.

Pass criteria:

- Board boots from cold power.
- Reset is repeatable.
- UART logs are stable.
- No hard fault during 10-minute idle run.
- Sensor reads either succeed or fail with counted, non-fatal errors.
- Watchdog behavior is deterministic.

## 4. ESP32-C6 board bring-up

Steps:

1. Connect ESP32-C6 DevKitC-1 over USB.
2. Flash the RISC-V ESP32-C6 firmware image.
3. Open serial logs.
4. Reset using EN button.
5. Confirm repeated boot.
6. Add UART-only telemetry first.
7. Add sensors only after boot logs are stable.

Pass criteria:

- Flash succeeds.
- Boot logs are readable.
- Reset does not corrupt state.
- No unexpected boot loop.
- Basic GPIO/UART timing is observable.

## 5. HiFive1 bring-up

Only run this if a HiFive1 Rev B board is available.

Steps:

1. Connect board over USB.
2. Confirm debug/serial enumeration.
3. Flash the HiFive1 RISC-V build.
4. Capture UART logs.
5. Validate GPIO marker timing.
6. Run a 10-minute idle stability test.

Pass criteria:

- Firmware boots.
- UART logs are readable.
- No immediate trap/reset loop.
- Timing pins show expected periodic behavior.

## 6. Persistence validation

Run this only after boot and UART are stable.

Steps:

1. Start with a fresh flash state.
2. Boot firmware and create a checkpoint.
3. Reset normally.
4. Confirm checkpoint restore.
5. Power-cycle during idle, not during write.
6. Confirm restore again.
7. Later, deliberately power-cut during a controlled write window.

Pass criteria:

- Valid checkpoint restores.
- Invalid/interrupted checkpoint is rejected.
- Older valid slot can be used as rollback.
- Firmware banks are not overwritten by persistence slots.

## 7. Long-run validation

Run each target:

- 1 hour smoke run.
- 24 hour stability run.
- 72 hour release-candidate run.

Record:

- commit hash;
- board revision;
- power source;
- ambient temperature;
- UART logs;
- reset count;
- watchdog events;
- sensor error counts;
- power consumption min/avg/max;
- final pass/fail verdict.

Pass criteria:

- No unexplained reset.
- No memory corruption symptom.
- No increasing sensor error trend without explanation.
- Power remains within expected range.
- Telemetry remains parseable.

## 8. HKL-2 guarded training validation

On PC first:

1. Define a small deterministic dataset.
2. Run guarded training loop.
3. Save checkpoint.
4. Restart and reload checkpoint.
5. Compare deterministic metrics.

On board later:

1. Run inference-only first.
2. Enable probe-only cycles.
3. Enable learning only after runtime gate stays healthy.

Pass criteria:

- Invalid examples are rejected.
- Saturated loss blocks unsafe updates.
- Training decisions are logged.
- Checkpoints are recoverable.
- No unattended learning without gate authorization.
