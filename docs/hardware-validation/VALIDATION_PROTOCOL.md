# Validation Protocol and Production Gates

This file defines the engineering evidence required before HKL-1/HKL-2 can be
called production-ready on hardware.

## Gate A — Host correctness

Required evidence:

- Formatting passes.
- Unit and integration tests pass.
- HKL-2 feature tests pass.
- All-features test suite passes.
- Clippy passes with `-D warnings`.
- No-default-features build passes.

Decision:

- **Pass**: all checks green.
- **Fail**: any panic, compile failure, saturated training smoke test, or
  warning promoted to error.

## Gate B — Cross-target build correctness

Required evidence:

- STM32F7 target builds.
- HiFive1 target builds.
- ESP32-C6 target builds.

Decision:

- **Pass**: target builds complete with expected features.
- **Conditional pass**: HiFive1 build passes but physical board unavailable,
  because the board is discontinued.
- **Fail**: target-specific assembly or MMIO leaks into the wrong target.

## Gate C — Board boot

Required evidence per board:

- Cold boot log.
- Manual reset log.
- Debug attach screenshot or terminal transcript.
- UART capture.
- Current draw during boot.

Decision:

- **Pass**: cold boot and reset are repeatable.
- **Fail**: hard fault, boot loop, unreadable telemetry, or overheating.

## Gate D — Peripheral I/O

Required evidence:

- GPIO marker toggles are measurable.
- UART telemetry is decodable.
- I2C sensor read succeeds or produces counted recoverable errors.
- IMU read produces changing values when moved.
- Audio input produces non-silent PCM/spike activity.

Decision:

- **Pass**: each path produces deterministic logs.
- **Fail**: silent sensor path, stuck bus, unbounded error counter, or blocking
  ISR.

## Gate E — Timing and latency

Required evidence:

- Logic analyzer capture of boot marker.
- Main loop/tick marker frequency.
- ISR entry/exit marker if available.
- Worst-case loop jitter over at least 10 minutes.

Decision:

- **Pass**: jitter is bounded and documented.
- **Fail**: timing stalls, missed ticks, or unbounded ISR latency.

## Gate F — Flash persistence and rollback

Required evidence:

- Fresh boot creates checkpoint.
- Reset restores checkpoint.
- Corrupt checkpoint is rejected.
- Interrupted write falls back to older valid slot.
- Persistence slots do not overlap firmware banks.

Decision:

- **Pass**: recovery is deterministic.
- **Fail**: firmware bank damage, corrupted restore accepted, or unrecoverable
  state after power cut.

## Gate G — Power and thermal

Required evidence:

- Idle current.
- Active current.
- Sensor-enabled current.
- Long-run average current.
- Board temperature check by touch/thermal sensor if available.

Decision:

- **Pass**: consumption is stable and board remains safe.
- **Fail**: unexplained current rise, overheating, brownout, or USB disconnects.

## Gate H — HKL-2 training readiness

Required evidence:

- Dataset definition.
- Evaluation metric.
- Training logs.
- Checkpoint format.
- Restart/resume test.
- Runtime gate decisions.

Decision:

- **Pass for experimentation**: toy dataset training is deterministic and
  gated.
- **Pass for production**: real dataset, real metrics, convergence evidence,
  regression tests, and checkpoint recovery are all documented.
- **Fail**: unattended learning, non-recoverable checkpoints, invalid examples
  accepted, or loss saturation ignored.

## Final production decision

HKL can be promoted from engineering validation to production candidate only
when:

1. Gates A-F pass.
2. Gate G has stable power evidence.
3. HKL-2 is either disabled in production firmware or Gate H has production
   evidence.
4. Remaining risks are documented with owner, mitigation, and acceptance date.

If any blocking gate fails, the release remains engineering-only.
