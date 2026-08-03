use crate::core::time::init_clock;
use core::sync::atomic::{AtomicU32, Ordering};

pub static CPU_FREQ_HZ: AtomicU32 = AtomicU32::new(160_000_000);
pub static TIMER_FREQ_HZ: AtomicU32 = AtomicU32::new(1_000_000);

const HP_SYSTIMER_BASE: u32 = 0x600B_2000;

pub fn init_hp_core() {
    configure_pll();
    init_timers();
    init_interrupts();
    init_clock(
        core::ptr::null_mut(),
        CPU_FREQ_HZ.load(Ordering::Relaxed),
        TIMER_FREQ_HZ.load(Ordering::Relaxed),
    );
}

fn configure_pll() {
    unsafe {
        core::ptr::write_volatile(0x6000_0000 as *mut u32, 0x3FFu32);
        core::ptr::write_volatile(0x6000_1000 as *mut u32, 0x1u32 << 8);
    }
}

fn init_timers() {
    unsafe {
        core::ptr::write_volatile((HP_SYSTIMER_BASE + 0x00) as *mut u32, 0x1);
        core::ptr::write_volatile((HP_SYSTIMER_BASE + 0x04) as *mut u32, 0);
    }
}

fn init_interrupts() {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        core::arch::asm!("csrw mie, {0}", in(reg) 0x88u32);
        core::arch::asm!("csrw mstatus, {0}", in(reg) 0x1888u32);
    }
}

pub fn reset_cpu() {
    unsafe {
        core::ptr::write_volatile(0x600B_0000 as *mut u32, 0x1);
    }
}

pub fn read_systimer() -> u64 {
    unsafe {
        let lo = core::ptr::read_volatile((HP_SYSTIMER_BASE + 0x04) as *const u32);
        let hi = core::ptr::read_volatile((HP_SYSTIMER_BASE + 0x08) as *const u32);
        (hi as u64) << 32 | lo as u64
    }
}

#[cfg(not(feature = "hifive1"))]
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
    init_hp_core();
    crate::system::boot::BootSequence::init_hardware();
    crate::system::boot::BootSequence::run_main_loop()
}
