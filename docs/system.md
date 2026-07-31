# System Module

The system module manages hardware initialization, hardware detection, persistence, watchdog, power management, and OTA updates.

## Hardware Detection & Adaptation (`system/hardware_detect.rs`)


### Hardware Detection Engine

Auto-detects host system capabilities to dynamically scale SNN network capacity and thread counts:

| Struct / Function | Capability |
|---|---|
| `HardwareDetector::detect()` | Scans CPU thread count and available RAM capacity |
| `HardwareProfile` | Calculates `recommended_max_neurons`, `recommended_max_synapses`, and `recommended_worker_threads` |
| `ADAPTIVE_MEMORY.set_capacity()` | Dynamically resizes active memory allocation bounds |

## Boot (`system/boot.rs`)


### Boot Sequence (t=0 → 22ms)

```
Power On
  ↓
[0.0 ms] init_clock()              → SysTick, HCLK, PLL, metabolic clock
  ↓
[0.5 ms] init_hardware_peripherals() → CPACR (FPU), SCB_VTOR (vector table),
                                        MPU_CTRL (memory protection), SCS_CCR
  ↓
[1.0 ms] read_boot_config()        → UID 96-bit from OTP 0x1FFF_7A10,
                                       hardware version, PUF extraction
  ↓
[2.0 ms] init_persistence()        → Load J-0 checkpoint from flash
  ↓
[2.5 ms] init_entropy(seed)        → Seed XorShift64Star PRNG + TRNG
  ↓
[3.0 ms] spike_trace::init_logger() → Init telemetry buffer
  ↓
[4.0 ms] enable_sensor_interrupts() → NVIC_ISER0/ISER1/ISER2 (IRQ 0-95)
  ↓
[5.0 ms] init_reflex_arcs()        → Hard-wire L0→L6→L4 (fixed, no STDP)
  ↓
[5.5 ms] check_emergencies()       → Validate NEURON_COUNT, entropy range,
                                       reflex thresholds; energy shutdown if critical
  ↓
[6.0 ms] Restore from flash?       → Yes → load state, No → fresh init
  ↓
[19 ms]  Main loop: Network::step() → Cognitive → Telemetry
  ↓
[21 ms]  check_for_update()        → Validate OTA candidate (CRC32)
  ↓
[21.5ms] apply_update()            → Flash bank B → bank A switch
  ↓
[21.5ms] confirm_stable()          → Mark update successful, commit
```

### Hardware Initialization Details

| Component | Registers Written | Feature Gate |
|---|---|---|
| FPU | `CPACR` (ENABLE=0xF) | `#[cfg(stm32f7)]` |
| Vector Table | `SCB_VTOR` | `#[cfg(stm32f7)]` |
| Memory Protection | `MPU_CTRL` | `#[cfg(stm32f7)]` |
| System Control | `SCS_CCR` (UNALIGN_TRAP) | `#[cfg(stm32f7)]` |
| Sensor IRQ | `NVIC_ISER0/1/2` (IRQ 0-95) | All BSPs |
| UID | OTP `0x1FFF_7A10` (96-bit) | `#[cfg(stm32f7)]` |

### check_emergencies()

Validates system integrity before entering the main loop:

| Condition | Action |
|---|---|
| Refle threshold > 0 | Continue |
| NEURON_COUNT == 0 | Energy shutdown |
| Entropy > high threshold → dreaming | Force dreaming mode |
| Entropy < low threshold → crystallize | Increase threshold |

## Persistence (`system/persistence.rs`)

Full system state save/restore to flash memory. 3 tests.

### BinaryDump Format

```
┌─────────────────┐
│ DumpHeader      │  64 bytes (magic, version, timestamp, counts, checksum)
├─────────────────┤
│ NeuronStates[]  │  4096 × sizeof(NeuronState)
├─────────────────┤
│ SynapseSlots[]  │  65536 × sizeof(SynapseSlot)
└─────────────────┘
```

### Features

| Feature | Description |
|---|---|
| 3 save slots | J-0, J-1, J-2 — redundancy and rollback |
| Checksum | CRC32 integrity verification |
| Encryption | Optional ChaCha20 encryption |
| Secure erase | Cryptographic XOR + verification |
| Auto-restore | On boot, restore from latest valid slot |
| Flash commit | `commit_to_flash()` writes via STM32F7 MMIO registers |
| Zero-stack snapshot | Static `SIMULATION_SAVE_SLOT` avoids 328 KB stack frame allocations during simulation steps |

### Key Functions

| Function | Purpose |
|---|---|
| `capture_simulation_snapshot()` | Zero-stack snapshot into static `SIMULATION_SAVE_SLOT` |
| `restore_simulation_snapshot()` | Zero-stack restore from static `SIMULATION_SAVE_SLOT` |
| `capture_state()` | Snapshot current state into dump buffer |
| `save_to_slot(slot)` | Write dump buffer to flash slot |
| `restore_from_slot(slot)` | Restore state from flash slot |
| `encrypt_dump(slot)` | Encrypt slot with device key |
| `decrypt_dump(slot)` | Decrypt slot with device key |
| `secure_erase_slot(slot)` | Cryptographically erase slot |
| `commit_to_flash()` | Write to physical flash (STM32F7 MMIO) |

## Watchdog (`system/watchdog.rs`)

Neurological watchdog timer with graduated recovery. 5 tests.

### Actions (escalating)

| Level | Consecutive High | Action |
|---|---|---|
| 0 | 0 | Reset watchdog, no action |
| 1 | 1 | Log warning event |
| 2 | 2 | Increment fault counter |
| 3 | 3–4 | Soft reset (reset state, keep hardware) |
| 4 | 5–9 | Rollback to J-1/J-2 checkpoint |
| 5 | 10+ | Full system restore from persistence |

### Key Functions

| Function | Purpose |
|---|---|
| `check_health()` | Returns `Ok(())` or `Err(WatchdogEvent)` |
| `pet()` | Reset the watchdog timer (normal operation) |
| `is_active()` | Check if watchdog monitoring is enabled |
| `highest_level()` | Get the highest escalation level reached |

## Power Management (`system/power.rs`)

### Power States

| State | Description |
|---|---|
| Active | Full operation, all clocks running (16–216 MHz DVFS) |
| Idle | Wait-for-interrupt, minimal power |
| Sleep | Core clock gated, peripherals off |
| Deep Sleep | Only RTC and wake-pins active |
| Shutdown | Complete power-off, boot on reset |

### Features

| Feature | Description |
|---|---|
| DVFS | 5 OPP levels (16–216 MHz), PWR_CR VOS auto-select |
| Wake sources | RTC, EXTI, TIM — configurable via `WakeUpConfig` |
| Power budgeting | 6 domains: CPU/Memory/Sensors/Actuators/Radio/Cognitive |
| Energy harvesting | MPPT perturb-and-observe algorithm |
| Auto-mode switch | Survive/Explore mode based on battery + harvest |
| Clock gating | AHB1/APB1/APB2 ENR per domain |
| V_th coupling | `threshold_multiplier()` from battery_level → SNN |
| Low-power idle | `idle_if_possible()` → deep sleep if idle > 10ms |

## OTA Updates (`system/ota.rs`)

Over-the-air firmware update with dual-bank support.

| Feature | Description |
|---|---|
| Dual-bank | Bank A active / Bank B staging |
| Firmware validation | CRC32 hash verification |
| Safe rollback | Keep J-0 persistence until confirmed |
| Slot state machine | Empty → Filled → Validated → Applied → Stable |
| Soft reset | Triggered after successful update |
| Persistence | State saved before bank switch |
| no_std compatible | Stack buffer `[u8; 1028]` (no `alloc::vec::Vec`) |

### OTA Cycle (boot timing)

1. **t=21ms**: `check_for_update()` — validate CRC32 of candidate firmware in bank B
2. **t=21.5ms**: `apply_update()` — switch active bank, soft reset
3. **t=21.5ms**: `confirm_stable()` — mark update as stable after successful boot

### Key Functions

| Function | Purpose |
|---|---|
| `check_for_update()` | Validate incoming firmware (CRC32, signature) |
| `apply_update()` | Switch active flash bank, trigger soft reset |
| `confirm_stable()` | Mark current firmware as stable (prevents rollback) |
| `rollback()` | Restore previous firmware version |
