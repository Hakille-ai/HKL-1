//! Bare-Metal QEMU Execution & Telemetry Demonstration for HKL-1 Engine
//!
//! Designed for bare-metal ARM Cortex-M7 (STM32F7) and RISC-V (HiFive1/ESP32-C6) QEMU target execution.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), no_main)]

#[cfg(not(feature = "std"))]
use core::panic::PanicInfo;
use hkl1::snn::network::network;
use hkl1::system::boot::BootSequence;

#[cfg(not(feature = "std"))]
struct BareMetalAllocator;

#[cfg(not(feature = "std"))]
unsafe impl core::alloc::GlobalAlloc for BareMetalAllocator {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        static mut HEAP: [u8; 65536] = [0; 65536];
        static mut OFFSET: usize = 0;
        unsafe {
            let align = layout.align();
            let size = layout.size();
            let current = OFFSET;
            let aligned = (current + align - 1) & !(align - 1);
            if aligned + size <= 65536 {
                OFFSET = aligned + size;
                let heap_ptr = core::ptr::addr_of_mut!(HEAP) as *mut u8;
                heap_ptr.add(aligned)
            } else {
                core::ptr::null_mut()
            }
        }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {}
}

#[cfg(not(feature = "std"))]
#[global_allocator]
static ALLOCATOR: BareMetalAllocator = BareMetalAllocator;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[cfg_attr(not(feature = "std"), unsafe(no_mangle))]
pub extern "C" fn main() -> ! {
    // 1. Execute bare-metal boot sequence (t=0 -> 22ms)
    BootSequence::init_hardware();

    let net = network();

    // 2. Perform 100 simulation ticks
    for _ in 0..100 {
        net.step();
    }

    // 3. Main loop
    BootSequence::run_main_loop()
}
