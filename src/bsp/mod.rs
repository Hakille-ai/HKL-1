#[cfg(feature = "esp32c6")]
pub mod esp32c6;
pub mod generic;
#[cfg(feature = "hifive1")]
pub mod hifive1;
#[cfg(feature = "stm32f7")]
pub mod stm32f7;
