# Assembly and Bench Setup

This procedure assumes a small embedded bench, not a permanent product PCB.
Everything should be reversible, labeled, and current-limited.

## Bench layout

Keep the bench physically organized:

1. PC/laptop on the left.
2. USB hub in the middle, preferably powered.
3. Target board under test in the center on an anti-static mat.
4. Breadboard and sensors to the right of the target board.
5. Logic analyzer clipped only after confirming voltage levels.
6. Multimeter and USB power meter always reachable.

Never connect or disconnect sensor wires while the board is powered unless the
sensor board explicitly supports hot-plugging.

## Electrical rules

- Use **3.3V logic** for sensors, UART, and analyzer channels unless a board
  manual explicitly says otherwise.
- Tie all grounds together: target GND, sensor GND, UART GND, analyzer GND.
- Do not connect 5V sensor outputs to MCU pins unless the pin is confirmed
  5V-tolerant.
- Use current-limited power for first bring-up where possible.
- Label every cable: UART TX, UART RX, GND, 3V3, SDA, SCL, SCK, MOSI, MISO,
  CS, GPIO timing pin.

## Universal UART wiring

For external USB-UART logging:

| USB-UART | Target |
|---|---|
| GND | GND |
| TXD | Target RX |
| RXD | Target TX |
| VCC | Leave disconnected unless intentionally powering a small target |

Set serial terminal initially to:

- 115200 baud
- 8 data bits
- no parity
- 1 stop bit
- no flow control

If logs are unreadable, try 921600 only after confirming firmware UART speed.

## I2C sensor wiring

Use BME280 and MPU-6050 on the same I2C bus only if their addresses do not
conflict.

| Sensor pin | Target pin |
|---|---|
| VIN / VCC | 3V3 |
| GND | GND |
| SDA | I2C SDA |
| SCL | I2C SCL |

Start with one sensor at a time. Bring up BME280 first, then MPU-6050.

## I2S microphone wiring

Typical I2S MEMS microphone wiring:

| Microphone pin | Target pin |
|---|---|
| 3V | 3V3 |
| GND | GND |
| BCLK / SCK | I2S bit clock |
| WS / LRCL | I2S word select |
| SD / DOUT | I2S data input |
| L/R | GND or 3V3 depending on desired channel |

Validate the microphone first with a known tiny capture example before blaming
HKL audio encoding.

## Logic analyzer channels

Recommended starter capture mapping:

| Analyzer channel | Signal |
|---|---|
| CH0 | UART TX |
| CH1 | GPIO boot marker |
| CH2 | GPIO main-loop tick |
| CH3 | I2C SCL |
| CH4 | I2C SDA |
| CH5 | SPI SCK or I2S BCLK |
| CH6 | Sensor interrupt |
| CH7 | Watchdog/reset marker |

Use the cheap 24 MHz analyzer for UART/I2C/GPIO timing. For high-speed I2S/SPI,
use a better analyzer if captures are unreliable.

## First power checklist

Before applying power:

- Board visually inspected.
- No loose wire strands.
- 3V3 and 5V rails not shorted to GND.
- UART VCC not connected accidentally.
- Sensor VCC matches 3.3V.
- Debug probe orientation checked.
- Current limit set if using bench supply.
- PC terminal ready to capture logs.

On power:

1. Confirm power LED.
2. Confirm board current is reasonable.
3. Confirm UART output or debug attach.
4. If the board heats or current spikes, cut power immediately.

## Board-specific notes

### STM32F746G-DISCO

- Use the board first because it is the most important production-like target
  for this repository.
- Start with USB power and the integrated ST-LINK.
- Add external STLINK-V3SET only if flashing/debugging is unstable.
- Validate boot, vector table, FPU setup, MPU setup, UART, flash persistence,
  and watchdog behavior.

### ESP32-C6-DevKitC-1

- Use USB flashing first.
- Validate serial logs and reset behavior before adding sensors.
- Keep in mind that ESP32-C6 has Wi-Fi/BLE/Zigbee/Thread capabilities, but
  HKL validation should first focus on deterministic boot, timing, and I/O.

### HiFive1 Rev B

- This board is discontinued. Treat it as optional until obtained.
- If unavailable, keep the RISC-V cross-build gate and select a maintained
  RISC-V board later.
- If obtained, validate UART, GPIO timing, watchdog/reset, and any available
  I2C path.
