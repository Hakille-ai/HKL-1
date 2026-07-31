use core::cell::UnsafeCell;
use core::sync::atomic::Ordering;

/// Trait adding fetch_* and compare_exchange for targets
/// without hardware atomic RMW (e.g. RISC-V RV32-IMC).
pub trait FetchAtomic {
    type Inner: Copy;
    fn fetch_add(&self, val: Self::Inner, order: Ordering) -> Self::Inner;
    fn fetch_sub(&self, val: Self::Inner, order: Ordering) -> Self::Inner;
    fn fetch_or(&self, val: Self::Inner, order: Ordering) -> Self::Inner;
    fn swap(&self, val: Self::Inner, order: Ordering) -> Self::Inner;
    fn compare_exchange(
        &self,
        cur: Self::Inner,
        new: Self::Inner,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Inner, Self::Inner>;
}

// SAFETY: On RISC-V without A extension, the standard AtomicU32 is
// layout-compatible with UnsafeCell<u32> (both are #[repr(transparent)]).
fn atomic_cell(atom: &core::sync::atomic::AtomicU32) -> &UnsafeCell<u32> {
    unsafe { &*(atom as *const core::sync::atomic::AtomicU32 as *const UnsafeCell<u32>) }
}

macro_rules! fetch_atomic_methods {
    () => {
        fn fetch_add(&self, val: u32, _order: Ordering) -> u32 {
            unsafe {
                let cell = atomic_cell(self);
                let prev = *cell.get();
                *cell.get() = prev.wrapping_add(val);
                prev
            }
        }
        fn fetch_sub(&self, val: u32, _order: Ordering) -> u32 {
            unsafe {
                let cell = atomic_cell(self);
                let prev = *cell.get();
                *cell.get() = prev.wrapping_sub(val);
                prev
            }
        }
        fn fetch_or(&self, val: u32, _order: Ordering) -> u32 {
            unsafe {
                let cell = atomic_cell(self);
                let prev = *cell.get();
                *cell.get() = prev | val;
                prev
            }
        }
        fn swap(&self, val: u32, _order: Ordering) -> u32 {
            unsafe {
                let cell = atomic_cell(self);
                let prev = *cell.get();
                *cell.get() = val;
                prev
            }
        }
        fn compare_exchange(
            &self,
            cur: u32,
            new: u32,
            _s: Ordering,
            _f: Ordering,
        ) -> Result<u32, u32> {
            unsafe {
                let cell = atomic_cell(self);
                let prev = *cell.get();
                if prev == cur {
                    *cell.get() = new;
                    Ok(prev)
                } else {
                    Err(prev)
                }
            }
        }
    };
}

impl FetchAtomic for core::sync::atomic::AtomicU32 {
    type Inner = u32;
    fetch_atomic_methods!();
}

impl FetchAtomic for core::sync::atomic::AtomicUsize {
    type Inner = usize;
    fn fetch_add(&self, val: usize, _order: Ordering) -> usize {
        unsafe {
            let cell = &*(self as *const Self as *const UnsafeCell<usize>);
            let prev = *cell.get();
            *cell.get() = prev.wrapping_add(val);
            prev
        }
    }
    fn fetch_sub(&self, val: usize, _order: Ordering) -> usize {
        unsafe {
            let cell = &*(self as *const Self as *const UnsafeCell<usize>);
            let prev = *cell.get();
            *cell.get() = prev.wrapping_sub(val);
            prev
        }
    }
    fn fetch_or(&self, val: usize, _order: Ordering) -> usize {
        unsafe {
            let cell = &*(self as *const Self as *const UnsafeCell<usize>);
            let prev = *cell.get();
            *cell.get() = prev | val;
            prev
        }
    }
    fn swap(&self, val: usize, _order: Ordering) -> usize {
        unsafe {
            let cell = &*(self as *const Self as *const UnsafeCell<usize>);
            let prev = *cell.get();
            *cell.get() = val;
            prev
        }
    }
    fn compare_exchange(
        &self,
        cur: usize,
        new: usize,
        _s: Ordering,
        _f: Ordering,
    ) -> Result<usize, usize> {
        unsafe {
            let cell = &*(self as *const Self as *const UnsafeCell<usize>);
            let prev = *cell.get();
            if prev == cur {
                *cell.get() = new;
                Ok(prev)
            } else {
                Err(prev)
            }
        }
    }
}
