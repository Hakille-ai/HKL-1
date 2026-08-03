# Bill of Materials and Budget

Last pricing refresh: 2026-08-03.

Prices are approximate single-unit prices, excluding shipping/import fees unless
the vendor page explicitly includes tax. Exchange-rate assumptions used for
budgeting:

- 1 USD ~= 0.867 EUR.
- 1 SEK ~= 0.091 EUR.
- 1 CHF ~= 1.075 EUR.

Add a 15-25% buffer for shipping, VAT, import handling, duplicate cables, and
replacement parts.

## Core boards

| Item | Qty | Unit price | Approx. EUR | Priority | Why |
|---|---:|---:|---:|---|---|
| STM32F746G-DISCO / 32F746GDISCOVERY | 1 | 77.87 EUR / 93.82 USD | 77.87-81.35 EUR | Required | Main Cortex-M7 target for boot, flash, watchdog, latency, and memory validation. |
| ESP32-C6-DevKitC-1-N8 | 1 | 9 USD | 7.80 EUR | Required | Maintained low-cost RISC-V target with USB, Wi-Fi/BLE, 8 MB flash. |
| SiFive HiFive1 Rev B | 1 | 68.95 USD historical retail, discontinued | ~59.80 EUR | Optional / stock-dependent | Existing HKL target, but discontinued. Buy only if available from stock/used market. |

## Debug, flashing, and telemetry

| Item | Qty | Unit price | Approx. EUR | Priority | Why |
|---|---:|---:|---:|---|---|
| STLINK-V3SET | 1 | 38.18 EUR / 43.66 USD | 38.18 EUR | Recommended | External STM32 probe if the on-board ST-LINK is not enough. |
| SEGGER J-Link EDU Mini | 1 | 76 USD | 65.90 EUR | Recommended | Useful cross-vendor debug probe for ARM/RISC-V development. Educational/hobby license only. |
| USB-UART 3.3V adapter / FTDI TTL-232R-3V3 | 1 | 23.80 USD | 20.63 EUR | Required | Reliable UART logs independent from debugger. |
| USB data cables, USB-C/micro-USB | 4 | 5-8 EUR | 20-32 EUR | Required | Bad cables waste hours; buy known data-capable cables. |

## Measurement equipment

| Item | Qty | Unit price | Approx. EUR | Priority | Why |
|---|---:|---:|---:|---|---|
| Low-cost 8-channel USB logic analyzer, 24 MHz | 1 | 9.75 EUR | 9.75 EUR | Required starter | Enough for UART, slow SPI/I2C, GPIO timing, basic interrupt checks. |
| Saleae Logic 8 | 1 | 499 USD | 432.60 EUR | Pro upgrade | Cleaner software, deeper captures, better automation; not required for first bring-up. |
| Saleae Logic Pro 8 | 1 | 999 USD | 866.10 EUR | Lab upgrade | Only justified for high-speed analog/digital capture needs. |
| Digital multimeter | 1 | 20-40 EUR | 30 EUR estimate | Required | Voltage, continuity, current sanity checks. |
| USB power meter | 1 | 20.93 USD | 18.15 EUR | Recommended | Measures board power draw during boot/long runs. |
| Bench power supply 3.3V/5V current-limited | 1 | 40-90 EUR | 60 EUR estimate | Recommended | Safer controlled power and current limits. |

## Sensors and I/O fixtures

| Item | Qty | Unit price | Approx. EUR | Priority | Why |
|---|---:|---:|---:|---|---|
| BME280 temperature/pressure/humidity breakout | 1 | 14.95 USD | 12.97 EUR | Required | Simple I2C/SPI environmental sensor for sensor path validation. |
| MPU-6050 IMU breakout | 1 | 22.60 CHF | 24.30 EUR | Required | Motion input for I2C and temporal encoding tests. |
| I2S MEMS microphone, ICS-43434 or INMP441 | 1 | 8.95 USD or 49 SEK | 4.46-7.76 EUR | Required | Real PCM/audio input for cochlea/audio spike encoder tests. |
| LEDs, buttons, resistors, potentiometer | 1 kit | 10-15 EUR | 12 EUR estimate | Required | GPIO, ADC, interrupt, and actuator smoke tests. |
| Breadboard + jumper wires | 1 kit | 10-25 EUR | 20 EUR estimate | Required | Safe modular wiring. |

## Budget totals

### Starter bring-up kit

This is the best first purchase if money matters.

| Category | Approx. EUR |
|---|---:|
| STM32F746G-DISCO | 77.87 |
| ESP32-C6-DevKitC-1 | 7.80 |
| USB-UART adapter | 20.63 |
| Low-cost logic analyzer | 9.75 |
| Multimeter | 30.00 |
| USB power meter | 18.15 |
| Sensors and basic I/O kit | 77.03 |
| Cables/misc | 25.00 |
| Subtotal | 266.23 |
| Recommended 20% buffer | 53.25 |
| **Estimated total** | **319.48 EUR** |

### Full engineering kit

Adds the RISC-V discontinued board if found, plus stronger debug hardware.

| Category | Approx. EUR |
|---|---:|
| Starter bring-up subtotal | 266.23 |
| HiFive1 Rev B, if available | 59.80 |
| STLINK-V3SET | 38.18 |
| J-Link EDU Mini | 65.90 |
| Bench power supply | 60.00 |
| Subtotal | 490.11 |
| Recommended 20% buffer | 98.02 |
| **Estimated total** | **588.13 EUR** |

### Pro lab kit

Same as full engineering kit, but replaces the low-cost analyzer with Saleae
Logic 8.

| Category | Approx. EUR |
|---|---:|
| Full engineering subtotal | 490.11 |
| Replace low-cost analyzer with Saleae Logic 8 delta | 422.85 |
| Subtotal | 912.96 |
| Recommended 20% buffer | 182.59 |
| **Estimated total** | **1,095.55 EUR** |

## Buying recommendation

Buy the **Starter bring-up kit** first. It is enough to validate boot, UART,
basic I/O, power sanity, and the STM32F7/ESP32-C6 hardware path.

Buy the **Full engineering kit** if the goal is to publish a serious release
candidate.

Buy the **Pro lab kit** only if you need long, repeatable protocol captures,
automated analyzer workflows, or cleaner evidence for external review.
