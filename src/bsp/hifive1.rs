use crate::core::time::init_clock;
use core::sync::atomic::{AtomicU32, Ordering};

pub static CPU_FREQ_HZ: AtomicU32 = AtomicU32::new(320_000_000);
pub static TIMER_FREQ_HZ: AtomicU32 = AtomicU32::new(32_768);

const MTIMECMP: *mut u64 = 0x0200_4000 as *mut u64;
const MTIME: *mut u64 = 0x0200_BFF8 as *mut u64;
const RESET_VECTOR: *mut u32 = 0x2000_0000 as *mut u32;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const MCAUSE_TIMER: u32 = 0x8000_0007;

pub fn init_hart() {
    configure_clocks();
    init_interrupts();
    init_clock(
        core::ptr::null_mut(),
        CPU_FREQ_HZ.load(Ordering::Relaxed),
        TIMER_FREQ_HZ.load(Ordering::Relaxed),
    );
}

fn configure_clocks() {
    unsafe {
        core::ptr::write_volatile(0x1002_0000 as *mut u32, 0x1);
        core::ptr::write_volatile(0x1002_0008 as *mut u32, 0x1F);
    }
}

fn init_interrupts() {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        core::arch::asm!("csrw mie, {0}", in(reg) 0x888u32);
        core::arch::asm!("csrw mstatus, {0}", in(reg) 0x1888u32);
    }
}

pub fn reset_cpu() {
    unsafe {
        core::ptr::write_volatile(RESET_VECTOR, 0x1);
    }
}

pub fn set_timer_comparator(val: u64) {
    unsafe {
        core::ptr::write_volatile(MTIMECMP, val);
    }
}

pub fn read_mtime() -> u64 {
    unsafe { core::ptr::read_volatile(MTIME) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".trap")]
pub unsafe extern "C" fn trap_vector() {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    {
        let mcause: u32;
        unsafe {
            core::arch::asm!("csrr {0}, mcause", out(reg) mcause);
        }
        if mcause == MCAUSE_TIMER {
            set_timer_comparator(read_mtime() + TIMER_FREQ_HZ.load(Ordering::Relaxed) as u64);
        }
    }
}

#[cfg(not(feature = "esp32c6"))]
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        unsafe extern "C" {
            static _sbss: u8;
            static _ebss: u8;
        }
        let mut p = core::ptr::addr_of!(_sbss) as usize as *mut u32;
        let e = core::ptr::addr_of!(_ebss) as usize as *mut u32;
        while p < e {
            *p = 0;
            p = p.add(1);
        }
    }
    init_hart();
    crate::system::boot::BootSequence::init_hardware();
    crate::system::boot::BootSequence::run_main_loop()
}
