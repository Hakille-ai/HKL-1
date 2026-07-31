/* STM32F746ZG Linker Script
 * ARM Cortex-M7 with 320KB+16KB TCM, 1MB Flash, 320KB SRAM
 * TCM regions: ITCM (0x0000_0000), DTCM (0x2000_0000)
 * Key: place critical SNN matrices in DTCM for zero-wait deterministic access
 */

MEMORY
{
    FLASH       (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    ITCM        (rx)  : ORIGIN = 0x00000000, LENGTH = 16K
    DTCM        (rwx) : ORIGIN = 0x20000000, LENGTH = 16K
    SRAM        (rwx) : ORIGIN = 0x20010000, LENGTH = 304K
    BKPSRAM     (rw)  : ORIGIN = 0x40024000, LENGTH = 4K
}

/* TCM reservation for critical SNN structures */
_critical_pool_size  = 4K;    /* NeuronState array (DTCM) */
_synaptic_pool_size  = 8K;    /* Fast synaptic weights (DTCM) */
_matrix_pool_size    = 4K;    /* CSR weight matrix (DTCM) */

SECTIONS
{
    /* === Vector Table === */
    .vectors : ALIGN(512)
    {
        KEEP(*(.vectors))
        . = ALIGN(4);
    } > FLASH

    /* === Code === */
    .text : ALIGN(4)
    {
        _stext = .;
        *(.text*)
        *(.rodata*)
        *(.glue_7)
        *(.glue_7t)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    /* === ARM Exception Tables === */
    .ARM.extab : ALIGN(4)
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } > FLASH

    .ARM.exidx : ALIGN(4)
    {
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } > FLASH

    /* === Preinit / Init / Fini === */
    .preinit_array : ALIGN(4)
    {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array*))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } > FLASH

    .init_array : ALIGN(4)
    {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.init_array*)))
        PROVIDE_HIDDEN(__init_array_end = .);
    } > FLASH

    .fini_array : ALIGN(4)
    {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.fini_array*)))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > FLASH

    /* === DTCM: Fastest memory for SNN critical data === */
    .dtcm_critical : ALIGN(32)
    {
        _sdtcm_critical = .;
        *(.dtcm.neuron_array)       /* NEURON_ARRAY */
        *(.dtcm.synapse_array)      /* fast synapse slots */
        *(.dtcm.matrix)             /* CSR weight matrix */
        . = ALIGN(32);
        _edtcm_critical = .;
    } > DTCM AT > FLASH

    /* === ITCM: Time-critical ISR handlers === */
    .itcm_text : ALIGN(32)
    {
        _sitcm_text = .;
        *(.itcm.isr_handlers)       /* ISR routines */
        *(.itcm.predictive_cycle)   /* predictive_cycle() */
        . = ALIGN(32);
        _eitcm_text = .;
    } > ITCM AT > FLASH

    /* === Main SRAM === */
    .data : ALIGN(4)
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM AT > FLASH

    .bss (NOLOAD) : ALIGN(4)
    {
        _sbss = .;
        __bss_start__ = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
        __bss_end__ = .;
    } > SRAM

    /* === Heap === */
    .heap : ALIGN(8)
    {
        _heap_start = .;
        . = . + 64K;
        _heap_end = .;
    } > SRAM

    /* === Stack === */
    .stack : ALIGN(8)
    {
        _stack_start = .;
        . = . + 16K;
        _stack_end = .;
    } > SRAM

    /* === Backup SRAM (persist across deep sleep) === */
    .backup_sram (NOLOAD) : ALIGN(4)
    {
        _sbackup = .;
        *(.backup_sram*)
        . = ALIGN(4);
        _ebackup = .;
    } > BKPSRAM

    /* === Discard === */
    /DISCARD/ :
    {
        *(.ARM.attributes*)
        *(.comment*)
        *(.note*)
    }
}

/* Export region boundaries for Rust code */
PROVIDE(__svectors = ORIGIN(FLASH));
PROVIDE(__vectors_size = SIZEOF(.vectors));
PROVIDE(__sdtcm = ORIGIN(DTCM));
PROVIDE(__edtcm = ORIGIN(DTCM) + LENGTH(DTCM));
PROVIDE(__sitcm = ORIGIN(ITCM));
PROVIDE(__eitcm = ORIGIN(ITCM) + LENGTH(ITCM));
PROVIDE(__ssram = ORIGIN(SRAM));
PROVIDE(__esram = ORIGIN(SRAM) + LENGTH(SRAM));
PROVIDE(__sbackup = ORIGIN(BKPSRAM));
PROVIDE(__ebackup = ORIGIN(BKPSRAM) + LENGTH(BKPSRAM));
PROVIDE(__stack_top = _stack_end);
