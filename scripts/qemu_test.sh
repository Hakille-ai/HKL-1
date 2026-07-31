#!/usr/bin/env bash
# QEMU-based integration test for HKL-1
# Runs the firmware in emulation and captures telemetry output.
#
# Dependencies:
#   - qemu-system-arm (for STM32F7)
#   - qemu-system-riscv32 (for HiFive1)
#   - arm-none-eabi-gdb or riscv64-unknown-elf-gdb
#
# Usage:
#   ./scripts/qemu_test.sh stm32f7    # Test STM32F7 in QEMU
#   ./scripts/qemu_test.sh hifive1    # Test HiFive1 in QEMU
#   ./scripts/qemu_test.sh esp32c6    # Test ESP32-C6 (QEMU WIP)

set -euo pipefail

TARGET="${1:-stm32f7}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/target"

echo "=== HKL-1 QEMU Test ==="
echo "Target: $TARGET"

case "$TARGET" in
  stm32f7)
    echo "Building for STM32F7 (ARM Cortex-M7)..."
    cd "$PROJECT_DIR"
    cargo build --target thumbv7em-none-eabihf --features stm32f7 --release 2>&1

    ELF="$BUILD_DIR/thumbv7em-none-eabihf/release/hkl1"
    echo "Running in QEMU (semihosting)..."
    qemu-system-arm \
      -machine stm32f746 \
      -cpu cortex-m7 \
      -nographic \
      -semihosting \
      -kernel "$ELF" \
      -serial mon:stdio \
      -d unimp,guest_errors \
      -D "$BUILD_DIR/qemu-trace.log"
    echo "QEMU exited. Trace log: $BUILD_DIR/qemu-trace.log"
    ;;

  hifive1)
    echo "Building for HiFive1 (RISC-V)..."
    cd "$PROJECT_DIR"
    cargo build --target riscv32imac-unknown-none-elf --features hifive1 --release 2>&1

    ELF="$BUILD_DIR/riscv32imac-unknown-none-elf/release/hkl1"
    echo "Running in QEMU..."
    qemu-system-riscv32 \
      -machine sifive_e \
      -nographic \
      -kernel "$ELF" \
      -serial mon:stdio \
      -d unimp,guest_errors \
      -D "$BUILD_DIR/qemu-trace.log"
    echo "QEMU exited. Trace log: $BUILD_DIR/qemu-trace.log"
    ;;

  esp32c6)
    echo "Building for ESP32-C6 (RISC-V)..."
    cd "$PROJECT_DIR"
    cargo build --target riscv32imc-unknown-none-elf --features esp32c6 --release 2>&1

    ELF="$BUILD_DIR/riscv32imc-unknown-none-elf/release/hkl1"
    echo "ESP32-C6 QEMU support is experimental."
    echo "ELF built at: $ELF"
    echo "Use espflash or QEMU with ESP32-C6 machine type when available."
    ;;

  *)
    echo "Unknown target: $TARGET"
    echo "Usage: $0 {stm32f7|hifive1|esp32c6}"
    exit 1
    ;;
esac

echo "=== Done ==="
