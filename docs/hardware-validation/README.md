# HKL-1 / HKL-2 Hardware Validation Pack

Last pricing refresh: 2026-08-03.

This folder is the engineering pack for moving HKL-1/HKL-2 from host-side
validation into real hardware validation. It covers the bill of materials,
bench setup, wiring approach, execution procedure, and pass/fail criteria.

The current software state is strong enough for engineering bring-up and
guarded training experiments, but it is not a production hardware release until
the procedures in this folder have been executed on real boards.

## Files

- [BOM.md](BOM.md) — recommended hardware, prices, alternatives, and total cost.
- [ASSEMBLY.md](ASSEMBLY.md) — bench layout, wiring, electrical rules, and safe
  first-power procedure.
- [EXECUTION_RUNBOOK.md](EXECUTION_RUNBOOK.md) — exact software/hardware
  execution steps from host tests to board flashing and logs.
- [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) — pass/fail gates for boot,
  telemetry, persistence, latency, power, and long-run safety.
- [SOURCE_NOTES.md](SOURCE_NOTES.md) — pricing and product references used for
  this pack.

## Validation stages

1. Host validation: formatting, tests, clippy, examples, and no-default-feature
   build.
2. Cross-compilation validation: STM32F7, HiFive1, and ESP32-C6 targets build
   cleanly.
3. Board bring-up: flash each board, confirm boot, UART logs, panic/reset
   behavior, and timing pins.
4. Peripheral validation: I2C/SPI/ADC/GPIO/audio inputs emit expected spike
   events.
5. Persistence validation: flash checkpoints and rollback survive power loss.
6. Long-run validation: 24-hour and 72-hour stability runs with power and
   telemetry monitoring.
7. Production decision: release is allowed only if all blocking gates pass or
   have documented risk acceptance.

## Recommended purchase strategy

Buy in two waves:

1. **Bring-up kit**: STM32F746G-DISCO, ESP32-C6 DevKitC-1, debug/UART tools,
   low-cost logic analyzer, breadboard, BME280, IMU, I2S microphone, basic
   power measurement.
2. **Full validation kit**: add HiFive1 Rev B only if available at a sane price,
   add J-Link EDU Mini, and replace the low-cost logic analyzer with Saleae
   Logic 8 if protocol captures become frequent.

HiFive1 Rev B is discontinued, so it should not block first validation. Keep
the RISC-V build gate, but treat real-board HiFive1 validation as optional until
a board is obtained or a maintained RISC-V replacement is selected.
