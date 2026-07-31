use crate::core::time::init_clock;
use core::sync::atomic::{AtomicU32, Ordering};

pub static CPU_FREQ_HZ: AtomicU32 = AtomicU32::new(216_000_000);
pub static TIMER_FREQ_HZ: AtomicU32 = AtomicU32::new(1_000_000);

unsafe extern "C" {
    static __svectors: u8;
    static _sdata: u8;
    static _edata: u8;
    static _sbss: u8;
    static _ebss: u8;
    static _stext: u8;
    static _etext: u8;
    static _sdtcm_critical: u8;
    static _edtcm_critical: u8;
    static _sitcm_text: u8;
    static _eitcm_text: u8;
    static __stack_top: u8;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Reset_Handler() -> ! {
    unsafe { cortex_m7_init() };
    crate::system::boot::BootSequence::init_hardware();
    init_clock(
        core::ptr::null_mut(),
        CPU_FREQ_HZ.load(Ordering::Relaxed),
        TIMER_FREQ_HZ.load(Ordering::Relaxed),
    );
    crate::system::boot::BootSequence::run_main_loop()
}

unsafe fn cortex_m7_init() {
    let sdata = core::ptr::addr_of!(_sdata) as usize;
    let edata = core::ptr::addr_of!(_edata) as usize;
    let stext = core::ptr::addr_of!(_stext) as usize;
    unsafe { copy_section(sdata, edata, stext) };

    let sbss = core::ptr::addr_of!(_sbss) as usize;
    let ebss = core::ptr::addr_of!(_ebss) as usize;
    unsafe { zero_section(sbss, ebss) };

    let sdtcm = core::ptr::addr_of!(_sdtcm_critical) as usize;
    let edtcm = core::ptr::addr_of!(_edtcm_critical) as usize;
    unsafe { copy_section(sdtcm, edtcm, stext) };

    let sitcm = core::ptr::addr_of!(_sitcm_text) as usize;
    let eitcm = core::ptr::addr_of!(_eitcm_text) as usize;
    unsafe { copy_section(sitcm, eitcm, stext) };

    unsafe {
        core::ptr::write_volatile(
            0xE000_ED88 as *mut u32,
            core::ptr::read_volatile(0xE000_ED88 as *const u32) | (0xF << 20),
        );
        let vtor = core::ptr::addr_of!(__svectors) as u32;
        core::ptr::write_volatile(0xE000_ED08 as *mut u32, vtor);
    }
}

unsafe fn copy_section(dst: usize, end: usize, src: usize) {
    let mut d = dst as *mut u32;
    let mut s = src as *const u32;
    let e = end as *mut u32;
    while d < e {
        unsafe {
            *d = *s;
            d = d.add(1);
            s = s.add(1);
        }
    }
}

unsafe fn zero_section(start: usize, end: usize) {
    let mut p = start as *mut u32;
    let e = end as *mut u32;
    while p < e {
        unsafe {
            *p = 0;
            p = p.add(1);
        }
    }
}

pub fn reset_cpu() {
    unsafe {
        core::ptr::write_volatile(0xE000_ED0C as *mut u32, 0x05FA_0004);
    }
}

pub struct SysTick;

impl SysTick {
    pub fn new() -> Self {
        Self
    }
    pub fn set_reload(&self, ticks: u32) {
        unsafe {
            core::ptr::write_volatile(0xE000_E014 as *mut u32, ticks.min(0x00FF_FFFF));
        }
    }
    pub fn enable(&self) {
        unsafe {
            core::ptr::write_volatile(0xE000_E010 as *mut u32, 7);
        }
    }
    pub fn disable(&self) {
        unsafe {
            core::ptr::write_volatile(0xE000_E010 as *mut u32, 0);
        }
    }
    pub fn has_fired(&self) -> bool {
        unsafe { (core::ptr::read_volatile(0xE000_E010 as *const u32) & (1 << 16)) != 0 }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn NMI_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn HardFault_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn MemManage_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn BusFault_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn UsageFault_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn SVC_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn DebugMon_Handler() {
    loop {}
}
#[unsafe(no_mangle)]
pub extern "C" fn PendSV_Handler() {
    loop {}
}
