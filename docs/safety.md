# Safety Module

The safety module provides protective mechanisms for reliable operation.

## Reflexes (`safety/reflexes.rs`)

Rapid protective responses that bypass cognitive processing.

| Reflex | Trigger | Action |
|---|---|---|
| Nociceptive | Extreme sensor value | Immediate motor stop |
| Startle | Sudden large input | Global inhibition |
| Freeze | Detection of danger | Suppress all movement |
| Withdrawal | Direct threat | Reverse motor command |
| Seizure detection | Runaway activity | Global reset |

Reflexes have configurable thresholds and can be temporarily suppressed by cognitive override. At boot, `check_emergencies()` validates reflex thresholds are correctly configured.

### Connected Reflex Arcs (SNN)

In addition to the safety reflex module, the SNN has hard-wired reflex arcs initialized by `init_reflex_arcs()`:

| Arc | Connection | Weight | Plasticity |
|---|---|---|---|
| Sensor → Reflex | Layer 0 → Layer 6 | 1.0 (fixed) | Disabled |
| Reflex → Motor | Layer 6 → Layer 4 | 0.8 (fixed) | Disabled |

These provide ultra-fast (sub-cognitive) sensorimotor loops within the neural network itself.

## Cognitive Override (`cognitive/reflex_override.rs`)

The cognitive system can suppress reflexes based on context:

- Noradrenaline level evaluation
- Current cognitive mode
- Attention focus
- 4 unit tests

## Entropy Monitor (`safety/entropy_monitor.rs`)

Monitors network entropy for cognitive health.

- Measures Shannon entropy of weight distributions
- Detects stagnation (low entropy) and chaos (high entropy)
- Adaptive thresholds: mean ± 2σ with smooth EMA
- Triggers corrective action:
  - Low entropy → inject noise, increase curiosity
  - High entropy → crystallize, increase thresholds
- 10 unit tests

## Hardware Resilience (`safety/hardware_resilience.rs`)

### Physical Parameter Monitoring

| Parameter | Action |
|---|---|
| Temperature | Throttle or shutdown if critical |
| Voltage | Brownout detection, graceful degradation |
| Clock frequency | Detect clock drift, recalibrate |
| Memory errors | ECC or parity checking |

### Recovery Actions

- Watchdog reset (escalating: warn → soft reset → rollback → full restore)
- State rollback to last checkpoint (J-1/J-2)
- Safe mode (minimal functionality)
- Full system restart (persistence-assisted)

### ECC Auto-Repair

`EccBlock` provides hardware-level error correction:

- 32 ECC blocks with parity + syndrome
- Single-bit flip detection and automatic correction
- `verify_all_ecc()` periodic integrity scan
- Bad sector tracking in `bad_sectors` map

### Synaptic Migration

`migrate_synapse()` relocates a synapse from a failing sector:

- Copies weight + state to a new `SynapseId` from the free pool
- Updates network routing to point to the new ID
- Marks the old sector in `bad_sectors` map
- Prevents data loss from physical flash/ memory degradation

### Synaptic Senescence

`SenescenceStage` tracks biological aging of synapses:

| Stage | Description |
|---|---|
| Healthy | Normal operation |
| Aging | Weight decay begins |
| Degraded | Plasticity reduced |
| EndOfLife | Scheduled for pruning |

The senescence score is computed from synapse age, with configurable thresholds for each stage. Integrated with `snn/synapse::apply_senescence()` and `snn/neurogenesis::apply_senescence()`.

### Watchdog Integration

The watchdog (`system/watchdog.rs`) monitors hardware resilience health:

- `check_health()` returns `Ok(())` or `Err(WatchdogEvent)`
- Escalation: 10 consecutive high events → full restore
- 5 unit tests

### Test Coverage

| Module | Tests |
|---|---|
| `safety/reflexes.rs` | 8 |
| `cognitive/reflex_override.rs` | 4 |
| `safety/entropy_monitor.rs` | 10 |
| `safety/hardware_resilience.rs` | 10 |
