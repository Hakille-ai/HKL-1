/* ESP32-C6 Linker Script
 * RISC-V RV32IMAC, 512KB Flash, 320KB SRAM
 * Regions: LP (low-power) and HP (high-performance)
 */

MEMORY
{
    LP_ROM   (rx)  : ORIGIN = 0x1000_0000, LENGTH = 16K
    LP_RAM   (rwx) : ORIGIN = 0x1010_0000, LENGTH = 16K
    HP_ROM   (rx)  : ORIGIN = 0x4200_0000, LENGTH = 384K
    HP_RAM   (rw)  : ORIGIN = 0x4080_0000, LENGTH = 288K
    HP_RAM_FAST (rwx) : ORIGIN = 0x4088_0000, LENGTH = 32K
}

SECTIONS
{
    .text : ALIGN(4)
    {
        KEEP(*(.init))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > HP_ROM

    .fast_text : ALIGN(32)
    {
        *(.fast_text*)
        . = ALIGN(32);
    } > HP_RAM_FAST AT > HP_ROM

    .data : ALIGN(4)
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > HP_RAM AT > HP_ROM

    .bss (NOLOAD) : ALIGN(4)
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > HP_RAM

    .lp_ram (NOLOAD) : ALIGN(4)
    {
        *(.lp_ram*)
    } > LP_RAM

    .stack (NOLOAD) : ALIGN(16)
    {
        _stack_start = .;
        . = . + 8K;
        _stack_end = .;
    } > HP_RAM

    /DISCARD/ :
    {
        *(.comment*)
        *(.note*)
    }
}

PROVIDE(__stack_top = _stack_end);
