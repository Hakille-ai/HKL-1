//! Sensor interfaces with hardware MMIO register access.
//! Supports I2C, SPI, and ADC peripherals (STM32F7 addresses by default).

use core::mem::MaybeUninit;
#[cfg(not(any(feature = "std", test)))]
use core::ptr::{read_volatile, write_volatile};

// ---------------------------------------------------------------------------
// MMIO register addresses (STM32F746 reference)
// ---------------------------------------------------------------------------
pub mod mmio {
    // I2C1
    pub const I2C1_BASE: usize = 0x4000_5400;
    pub const I2C1_CR1: *mut u32 = 0x4000_5400 as *mut u32;
    pub const I2C1_CR2: *mut u32 = 0x4000_5404 as *mut u32;
    pub const I2C1_DR: *mut u32 = 0x4000_5410 as *mut u32;
    pub const I2C1_SR1: *mut u32 = 0x4000_5414 as *mut u32;
    pub const I2C1_SR2: *mut u32 = 0x4000_5418 as *mut u32;

    // SPI1
    pub const SPI1_BASE: usize = 0x4001_3000;
    pub const SPI1_CR1: *mut u32 = 0x4001_3000 as *mut u32;
    pub const SPI1_DR: *mut u32 = 0x4001_300C as *mut u32;
    pub const SPI1_SR: *mut u32 = 0x4001_3008 as *mut u32;

    // ADC1
    pub const ADC1_BASE: usize = 0x4001_2000;
    pub const ADC1_SR: *mut u32 = 0x4001_2000 as *mut u32;
    pub const ADC1_DR: *mut u32 = 0x4001_2004 as *mut u32;
    pub const ADC1_CR1: *mut u32 = 0x4001_2008 as *mut u32;
    pub const ADC1_CR2: *mut u32 = 0x4001_200C as *mut u32;
    pub const ADC1_SQR3: *mut u32 = 0x4001_2034 as *mut u32;

    // GPIOA
    pub const GPIOA_BASE: usize = 0x4002_0000;
    pub const GPIOA_MODER: *mut u32 = 0x4002_0000 as *mut u32;
    pub const GPIOA_IDR: *mut u32 = 0x4002_0010 as *mut u32;
    pub const GPIOA_ODR: *mut u32 = 0x4002_0014 as *mut u32;
    pub const GPIOA_BSRR: *mut u32 = 0x4002_0018 as *mut u32;
}

// ---------------------------------------------------------------------------
// I2C bus abstraction
// ---------------------------------------------------------------------------
pub struct I2cBus {
    _addr: u8,
    _timeout: u32,
}

impl I2cBus {
    pub const fn new(addr: u8) -> Self {
        Self {
            _addr: addr,
            _timeout: 10000,
        }
    }

    pub fn init(&self) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            // Enable I2C peripheral (PE bit in CR1)
            let cr1 = read_volatile(mmio::I2C1_CR1);
            write_volatile(mmio::I2C1_CR1, cr1 | (1 << 0));
            // Set peripheral clock to 50MHz (CR2)
            write_volatile(mmio::I2C1_CR2, 50_000_000 / 1_000_000);
            // Configure timing for 100kHz standard mode
            write_volatile(mmio::I2C1_CR1, cr1 | (1 << 0));
        }
    }

    pub fn read_byte(&self, reg: u8) -> Option<u8> {
        #[cfg(any(feature = "std", test))]
        {
            let _ = reg;
            return Some(25);
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            // Generate START condition
            let cr1 = read_volatile(mmio::I2C1_CR1);
            write_volatile(mmio::I2C1_CR1, cr1 | (1 << 8) | (1 << 0)); // START + PE

            // Wait for SB flag in SR1
            let mut wait = self._timeout;
            while read_volatile(mmio::I2C1_SR1) & (1 << 0) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }

            // Send device address (write)
            write_volatile(mmio::I2C1_DR, (self._addr as u32) << 1);

            // Wait for ADDR flag
            wait = self._timeout;
            while read_volatile(mmio::I2C1_SR1) & (1 << 1) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }
            // Clear ADDR by reading SR2
            let _sr2 = read_volatile(mmio::I2C1_SR2);

            // Send register address
            write_volatile(mmio::I2C1_DR, reg as u32);
            wait = self._timeout;
            while read_volatile(mmio::I2C1_SR1) & (1 << 7) == 0 {
                // TXE
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }

            // Repeated START for read
            write_volatile(mmio::I2C1_CR1, read_volatile(mmio::I2C1_CR1) | (1 << 10)); // ACK
            write_volatile(mmio::I2C1_CR1, read_volatile(mmio::I2C1_CR1) | (1 << 8)); // START

            wait = self._timeout;
            while read_volatile(mmio::I2C1_SR1) & (1 << 0) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }

            // Send address (read)
            write_volatile(mmio::I2C1_DR, ((self._addr as u32) << 1) | 1);

            wait = self._timeout;
            while read_volatile(mmio::I2C1_SR1) & (1 << 1) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }
            let _sr2 = read_volatile(mmio::I2C1_SR2);

            // Disable ACK and gen STOP
            write_volatile(mmio::I2C1_CR1, read_volatile(mmio::I2C1_CR1) & !(1 << 10));
            write_volatile(mmio::I2C1_CR1, read_volatile(mmio::I2C1_CR1) | (1 << 9)); // STOP

            // Wait for RXNE then read data
            wait = self._timeout;
            while read_volatile(mmio::I2C1_SR1) & (1 << 6) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }
            let data = read_volatile(mmio::I2C1_DR) as u8;
            Some(data)
        }
    }
}

// ---------------------------------------------------------------------------
// SPI bus abstraction
// ---------------------------------------------------------------------------
pub struct SpiBus {
    _cs_pin: u8,
}

impl SpiBus {
    pub const fn new(cs_pin: u8) -> Self {
        Self { _cs_pin: cs_pin }
    }

    /// Configure SPI1 peripheral (called once at init)
    pub fn init(&self) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let _cs = self._cs_pin;
            let cr1 = (1 << 2)  // Master
                    | (1 << 3)  // BR=div2 (fPCLK/2)
                    | (1 << 6); // SPE (enable)
            write_volatile(mmio::SPI1_CR1, cr1);
        }
    }

    pub fn read_u16(&self) -> Option<u16> {
        #[cfg(any(feature = "std", test))]
        {
            return Some(1024);
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            // Write dummy byte to generate clock
            write_volatile(mmio::SPI1_DR, 0x00);

            let mut wait = 10000;
            while read_volatile(mmio::SPI1_SR) & (1 << 1) == 0 {
                // TXE wait
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }

            wait = 10000;
            while read_volatile(mmio::SPI1_SR) & (1 << 0) == 0 {
                // RXNE wait
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }

            // Read two bytes (MSB first)
            let high = read_volatile(mmio::SPI1_DR) as u16;
            write_volatile(mmio::SPI1_DR, 0x00);
            wait = 10000;
            while read_volatile(mmio::SPI1_SR) & (1 << 0) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }
            let low = read_volatile(mmio::SPI1_DR) as u16;
            Some((high << 8) | low)
        }
    }

    pub fn write_byte(&self, byte: u8) -> Option<()> {
        #[cfg(any(feature = "std", test))]
        {
            let _ = byte;
            return Some(());
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            write_volatile(mmio::SPI1_DR, byte as u32);
            let mut wait = 10000;
            while read_volatile(mmio::SPI1_SR) & (1 << 1) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }
            Some(())
        }
    }
}

// ---------------------------------------------------------------------------
// ADC interface
// ---------------------------------------------------------------------------
pub struct AdcChannel {
    _channel: u8,
}

impl AdcChannel {
    pub const fn new(channel: u8) -> Self {
        Self { _channel: channel }
    }

    pub fn init(&self) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            write_volatile(mmio::ADC1_CR2, 1 << 0); // ADON
            write_volatile(mmio::ADC1_CR1, 0); // 12-bit resolution (default)

            // Calibration
            write_volatile(mmio::ADC1_CR2, read_volatile(mmio::ADC1_CR2) | (1 << 1));
            let mut wait = 10000;
            while read_volatile(mmio::ADC1_CR2) & (1 << 1) != 0 {
                wait -= 1;
                if wait == 0 {
                    break;
                }
            }
        }
    }

    pub fn read_raw(&self) -> Option<u16> {
        #[cfg(any(feature = "std", test))]
        {
            return Some(2048);
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            // Set channel in sequence register
            write_volatile(mmio::ADC1_SQR3, self._channel as u32);

            // Start conversion
            write_volatile(mmio::ADC1_CR2, read_volatile(mmio::ADC1_CR2) | (1 << 0));

            // Wait for EOC
            let mut wait = 10000;
            while read_volatile(mmio::ADC1_SR) & (1 << 1) == 0 {
                wait -= 1;
                if wait == 0 {
                    return None;
                }
            }

            let raw = read_volatile(mmio::ADC1_DR) as u16;
            Some(raw)
        }
    }

    /// Read as normalized f32 (0.0 - 1.0)
    pub fn read_normalized(&self) -> f32 {
        self.read_raw().map(|v| (v as f32) / 4095.0).unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Virtual I2C sensor driver (reads simulated from MMIO for now)
// ---------------------------------------------------------------------------
#[allow(static_mut_refs)]
fn read_sensor_i2c(reg: u8, i2c_addr: u8, _scale: f32) -> f32 {
    let bus = I2cBus::new(i2c_addr);
    bus.read_byte(reg)
        .map(|v| {
            match reg {
                // Simulated conversion: reg value * scale
                0x00 => (v as f32) * 1.0, // Temperature: -40 to +125
                0x01 => (v as f32) * 0.1, // Pressure: 300-1100 hPa
                0x02 => (v as f32) * 0.5, // Humidity: 0-100%
                _ => (v as f32) / 255.0,
            }
        })
        .unwrap_or_else(|| {
            unsafe {
                crate::io::sensors::SENSOR_MANAGER
                    .assume_init_mut()
                    .i2c_error_count += 1;
            }
            0.0
        })
}

// ---------------------------------------------------------------------------
// Sensor reading abstractions
// ---------------------------------------------------------------------------
pub fn read_temperature() -> f32 {
    // BMP280/BME280: I2C addr 0x76, temp reg 0xFA
    read_sensor_i2c(0xFA, 0x76, 1.0)
}

pub fn read_pressure() -> f32 {
    // BMP280: pressure reg 0xF7
    read_sensor_i2c(0xF7, 0x76, 0.1) + 1000.0
}

pub fn read_humidity() -> f32 {
    // BME280: humidity reg 0xFD
    read_sensor_i2c(0xFD, 0x76, 0.5)
}

pub fn read_light() -> f32 {
    // Photoresistor via ADC channel 0
    let adc = AdcChannel::new(0);
    adc.read_normalized()
}

pub fn read_sound() -> f32 {
    // Electret mic via ADC channel 1
    let adc = AdcChannel::new(1);
    adc.read_normalized()
}

// ---------------------------------------------------------------------------
// GPIO digital input
// ---------------------------------------------------------------------------
pub fn read_gpio_pin(_pin: u8) -> bool {
    #[cfg(any(feature = "std", test))]
    {
        return false;
    }
    #[cfg(not(any(feature = "std", test)))]
    unsafe {
        let idr = read_volatile(mmio::GPIOA_IDR);
        (idr & (1 << _pin)) != 0
    }
}

// ---------------------------------------------------------------------------
// SensorManager — orchestrates all sensor reads
// ---------------------------------------------------------------------------
pub struct SensorManager {
    pub temperature: f32,
    pub pressure: f32,
    pub humidity: f32,
    pub light: f32,
    pub sound: f32,
    pub custom_sensors: [f32; 16],
    i2c_initialized: bool,
    adc_initialized: bool,
    pub i2c_error_count: u32,
}

impl SensorManager {
    pub fn new() -> Self {
        Self {
            temperature: 25.0,
            pressure: 1013.25,
            humidity: 50.0,
            light: 0.0,
            sound: 0.0,
            custom_sensors: [0.0; 16],
            i2c_initialized: false,
            adc_initialized: false,
            i2c_error_count: 0,
        }
    }

    pub fn init(&mut self) {
        let i2c = I2cBus::new(0x76);
        i2c.init();
        self.i2c_initialized = true;

        let adc = AdcChannel::new(0);
        adc.init();
        let _adc1 = AdcChannel::new(1);
        _adc1.init();
        self.adc_initialized = true;

        // Initial read
        self.read_all();
    }

    /// Read all physical sensors via I2C/SPI/ADC
    pub fn read_all(&mut self) {
        if self.i2c_initialized {
            self.temperature = read_temperature();
            self.pressure = read_pressure();
            self.humidity = read_humidity();
        }

        if self.adc_initialized {
            self.light = read_light();
            self.sound = read_sound();
        }
    }

    /// Push readings into the sensor ring buffer as spikes
    pub fn emit_sensor_spikes(&self, timestamp: u32) {
        use crate::core::math::FixedPoint;
        use crate::io::buffers::{EncodedSpike, Modality, SENSOR_RING};

        let readings: [(f32, u16, Modality); 5] = [
            (self.temperature, 0, Modality::Sensor),
            (self.pressure, 1, Modality::Sensor),
            (self.humidity, 2, Modality::Sensor),
            (self.light, 3, Modality::Sensor),
            (self.sound, 4, Modality::Sensor),
        ];

        unsafe {
            for &(val, neuron_offset, modality) in &readings {
                let spike = EncodedSpike {
                    neuron_id: crate::core::memory::NeuronId::new(neuron_offset),
                    intensity: FixedPoint::from_f32(val),
                    timestamp,
                    modality,
                };
                let _ = SENSOR_RING.push(spike);
            }
        }
    }
}

/// Global sensor manager instance
pub static mut SENSOR_MANAGER: MaybeUninit<SensorManager> = MaybeUninit::uninit();

static INITIALIZED_SENSOR_MANAGER: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_sensor_manager() {
    unsafe {
        let sm = SENSOR_MANAGER.write(SensorManager::new());
        sm.init();
        INITIALIZED_SENSOR_MANAGER.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn sensor_manager() -> &'static mut SensorManager {
    unsafe {
        if !INITIALIZED_SENSOR_MANAGER.load(core::sync::atomic::Ordering::Relaxed) {
            init_sensor_manager();
        }
        &mut *SENSOR_MANAGER.as_mut_ptr()
    }
}
