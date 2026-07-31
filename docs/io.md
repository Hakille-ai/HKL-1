# I/O Module

The I/O module handles all input/output operations between the SNN and the external world.

## Buffers (`io/buffers.rs`)

Lock-free ring buffers for inter-module and ISR-safe communication.

### RingBuffer<T, const N: usize>

| Feature | Description |
|---|---|
| Lock-free | Atomic head/tail indices |
| MPMU | Multi-producer, multi-consumer |
| Capacity | Power-of-2 for fast modulo (mask-based) |
| Push/Pop | Non-blocking, returns Option/Bool |

### Global Spike Queue

- `GLOBAL_SPIKE_QUEUE: RingBuffer<SpikeEvent, RING_BUFFER_SIZE>`
- Used by ISRs to enqueue incoming spikes
- Consumed by main loop in `Network::step`

### Modality → Layer Mapping

Sensory modalities are mapped to fixed SNN input layers:

| Modality | Layer | Description |
|---|---|---|
| Text | 0 | Textual/character input |
| Audio | 1 | Sound/microphone input |
| Vision | 2 | Visual/optical input |
| Sensor | 3 | Touch, proximity, accelerometer |
| Proprioception | 4 | Body position, motor feedback |
| Internal | 5 | Cognitive state, neuromodulators |

This mapping is used by `Encoder::push_spike_to_layer(modality, ...)` to route sensory data to the correct SNN layer.

## Encoder (`io/encoder.rs`)

Converts raw sensor data into spike trains:

| Encoder | Method |
|---|---|
| Rate encoding | Value → Poisson spike rate |
| Temporal encoding | Value → precise spike time |
| Population coding | Value → distributed neural activity |
| Place coding | Position → grid cell-like activation |
| Vision Retinal DVS | Frame contrast DoG & DVS log-intensity polarity events (`src/vision/`) |
| Auditory Gammatone | 32-band ERB Gammatone cochlea & PFM hair cells (`src/audio/`) |
| Text Spike Tokenizer | ASCII + BPE subword tokens with phase-timing position encoding (`src/nlp/`) |

## Decoder (`io/decoder.rs`)

Converts spike patterns into actuator commands and natural language / acoustic waveforms:

| Decoder | Method |
|---|---|
| Rate decoding | Average firing rate → continuous value |
| Temporal decoding | Spike timing → discrete command |
| Winner-take-all | Most active neuron → class label |
| Vector decoding | Population activity → motor command |
| Spike Text Decoder | WTA L4 firing rate integrator & sentence reconstruction (`src/nlp/`) |
| Spike Voice Synthesizer | Formant resonator converting motor spikes to 16-bit 16kHz PCM audio (`src/audio/`) |


## Sensors (`io/sensors.rs`)

Hardware sensor interface definitions with MMIO register access.

### Supported Sensors

| Sensor | Interface | Deprecated? |
|---|---|---|
| Touch/pressure | I2C/GPIO | |
| Light/photodiodes | ADC | |
| Sound/microphones | I2S/ADC | |
| Proximity/ultrasonic | GPIO/PWM | |
| Accelerometer/gyroscope | I2C/SPI | |

### Error Tracking

`SensorManager.i2c_error_count: u32` is incremented on every I2C communication failure in `read_sensor_i2c()`:
- Correlates with hardware degradation
- Telemetry-visible for predictive maintenance
- Reset on successful read

### Key Functions

| Function | Purpose |
|---|---|
| `read_sensor_i2c(addr, reg, buf)` | I2C read with error counting |
| `read_sensor_spi(cs, reg, buf)` | SPI register read |
| `read_sensor_adc(channel)` | ADC conversion read |
| `sensor_irq_handler(id)` | ISR dispatch for sensor events |

## Actuators (`io/actuators.rs`)

Hardware actuator interface with MMIO register writes.

### Supported Actuators

| Actuator | Control | Register |
|---|---|---|
| DC motors | PWM duty cycle | `MMIO_PWM_BASE` |
| Servos | Position PWM | `MMIO_SERVO_BASE` |
| LEDs | Brightness PWM | `MMIO_LED_BASE` |
| Speakers | Frequency PWM | `MMIO_SPEAKER_BASE` |
| Analog output | DAC voltage | `DAC_CR` (EN + BOFF) |

### DAC Initialization

`DacOutput::init()` configures the digital-to-analog converter:
- Writes `DAC_CR` register with `EN` (enable) + `BOFF` (buffer off) bits
- Uses MMIO base address from `DAC_BASE`
- Supports 12-bit output resolution

### Key Functions

| Function | Purpose |
|---|---|
| `set_pwm(channel, duty)` | Write PWM duty cycle to MMIO |
| `set_gpio(pin, value)` | Write GPIO output value |
| `dac_write(value)` | Write DAC output register |
| `init()` | Initialize actuator hardware (DAC_CR, GPIO modes) |

## ISR (`io/isr.rs`)

Hardware interrupt service routine handlers for time-critical sensor input.

### Supported Interrupts

| IRQ | Source | Handler |
|---|---|---|
| TIM2 | Timer trigger | `tim2_isr()` — periodic sensor sampling |
| ADC | Conversion complete | `adc_isr()` — analog read ready |
| EXTI0–15 | GPIO pin events | `exti_isr()` — touch/proximity edges |
| SPI1/2 | Data transfer | `spi_isr()` — SPI transaction complete |
| I2C1/2 | Address/data | `i2c_isr()` — I2C bus events |

### ISR → Spike Queue Pipeline

Interrupts push spike events into `GLOBAL_SPIKE_QUEUE` via `isr_push_spike()`:

1. IRQ fires → handler entry
2. Determine sensor type and intensity from hardware registers
3. Calculate layer: `layer = (intensity * 7) as u8` (maps 0.0–1.0 → layers 0–7)
4. `GLOBAL_SPIKE_QUEUE.reserve_write()` → atomic slot reservation
5. Write `SpikeEvent{neuron_id, intensity, timestamp, modality}`
6. `GLOBAL_SPIKE_QUEUE.commit_write()` → make visible to main loop

This pipeline is lock-free and ISR-safe, with predictable O(1) timing.
