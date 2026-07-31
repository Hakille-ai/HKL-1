/* SiFive HiFive1 Rev B Linker Script
 * RISC-V RV32IMAC, 16KB DTIM, 4KB ITIM, 256KB Flash
 */

MEMORY
{
    FLASH   (rx)  : ORIGIN = 0x20010000, LENGTH = 256K
    ITIM    (rx)  : ORIGIN = 0x08000000, LENGTH = 4K
    DTIM    (rwx) : ORIGIN = 0x80000000, LENGTH = 16K
    RAM     (rwx) : ORIGIN = 0x80004000, LENGTH = 32M
}

SECTIONS
{
    .text : ALIGN(4)
    {
        KEEP(*(.init))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    .itim : ALIGN(32)
    {
        *(.itim*)
        . = ALIGN(32);
    } > ITIM AT > FLASH

    .data : ALIGN(4)
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) : ALIGN(4)
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    .stack (NOLOAD) : ALIGN(16)
    {
        _stack_start = .;
        . = . + 4K;
        _stack_end = .;
    } > RAM

    /DISCARD/ :
    {
        *(.comment*)
        *(.note*)
    }
}

PROVIDE(__stack_top = _stack_end);
