/// Generic (host/test) BSP with no hardware dependencies.
/// Used when compiling for host to run unit tests.

pub fn host_init() {
    // No-op: host platform doesn't need hardware init
}

pub fn host_sleep_ms(_ms: u32) {
    // Would use thread::sleep on host
}

pub fn reset_cpu() {
    // Cannot reset on host
}

pub fn memory_barrier() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("mfence", options(nostack));
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        core::arch::asm!("mfence", options(nostack));
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dmb sy", options(nostack));
    }
}
