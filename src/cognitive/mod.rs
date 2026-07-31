//! Cognitive functions: actor-critic learning, prediction,
//! curiosity, attention, neuromodulation, proprioception, temporal reasoning,
//! astrocyte glial modulation, striosome/matrice, thalamic gating, and cerebellum.
pub mod actor;
pub mod attention;
pub mod continual;
pub mod curiosity;
pub mod episodic;
pub mod global_workspace;
pub mod networks;
pub mod neuromodulation;
pub mod predictor;
pub mod proprioception;
pub mod reflex_override;
pub mod temporal;

#[cfg(feature = "std")]
use alloc::boxed::Box;
use core::mem::MaybeUninit;
pub use continual::ContinualLearningEngine;

pub static mut CONTINUAL_LEARNING_ENGINE: MaybeUninit<ContinualLearningEngine> = MaybeUninit::uninit();

static INITIALIZED_CONTINUAL_LEARNING: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_continual_learning() {
    unsafe {
        if !INITIALIZED_CONTINUAL_LEARNING.load(core::sync::atomic::Ordering::Relaxed) {
            #[cfg(feature = "std")]
            {
                let boxed = Box::new(ContinualLearningEngine::new());
                let ptr = Box::into_raw(boxed);
                core::ptr::copy_nonoverlapping(ptr as *const MaybeUninit<ContinualLearningEngine>, &raw mut CONTINUAL_LEARNING_ENGINE, 1);
            }
            #[cfg(not(feature = "std"))]
            {
                CONTINUAL_LEARNING_ENGINE.write(ContinualLearningEngine::new());
            }
            INITIALIZED_CONTINUAL_LEARNING.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn continual_learning() -> &'static mut ContinualLearningEngine {
    unsafe {
        if !INITIALIZED_CONTINUAL_LEARNING.load(core::sync::atomic::Ordering::Relaxed) {
            init_continual_learning();
        }
        &mut *CONTINUAL_LEARNING_ENGINE.as_mut_ptr()
    }
}
