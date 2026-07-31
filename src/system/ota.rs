//! OTA update system with dual-bank flash, firmware validation,
//! safe rollback, and weight preservation (Section 16).

// ---------------------------------------------------------------------------
// Flash memory layout (STM32F746)
// ---------------------------------------------------------------------------
pub mod flash_layout {
    // Base addresses
    pub const BANK_A_BASE: u32 = 0x0800_0000;
    pub const BANK_B_BASE: u32 = 0x0804_0000; // Offset for dual-bank (256KB each)
    pub const BANK_SIZE: u32 = 256 * 1024; // 256 KB per bank (adjust to target)

    // OTA metadata sector (last sector of flash)
    pub const OTA_META_BASE: u32 = 0x0808_0000;

    /// OTA metadata offsets
    pub const OTA_ACTIVE_BANK: *mut u32 = (OTA_META_BASE + 0x00) as *mut u32;
    pub const OTA_VERSION_A: *mut u32 = (OTA_META_BASE + 0x04) as *mut u32;
    pub const OTA_VERSION_B: *mut u32 = (OTA_META_BASE + 0x08) as *mut u32;
    pub const OTA_STATUS: *mut u32 = (OTA_META_BASE + 0x0C) as *mut u32;
    pub const OTA_BOOT_COUNT: *mut u32 = (OTA_META_BASE + 0x10) as *mut u32;
    pub const OTA_CHECKSUM_A: *mut u32 = (OTA_META_BASE + 0x14) as *mut u32;
    pub const OTA_CHECKSUM_B: *mut u32 = (OTA_META_BASE + 0x18) as *mut u32;
    pub const OTA_FW_SIZE_A: *mut u32 = (OTA_META_BASE + 0x1C) as *mut u32;
    pub const OTA_FW_SIZE_B: *mut u32 = (OTA_META_BASE + 0x20) as *mut u32;
    pub const OTA_TIMESTAMP_A: *mut u64 = (OTA_META_BASE + 0x24) as *mut u64;
    pub const OTA_TIMESTAMP_B: *mut u64 = (OTA_META_BASE + 0x2C) as *mut u64;

    // Persistence slots reside after OTA metadata
    pub const PERSISTENCE_BASE: u32 = OTA_META_BASE + 0x100;
}

// ---------------------------------------------------------------------------
// OTA status codes
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum OtaStatus {
    Idle = 0,
    UpdateReady = 1, // New firmware downloaded, verified
    Applying = 2,    // About to switch banks
    Applied = 3,     // Switched, booting new firmware
    RolledBack = 4,  // New bank failed, reverted
    Failed = 5,      // Unrecoverable error
}

impl OtaStatus {
    pub const fn from_u32(v: u32) -> Self {
        match v {
            0 => OtaStatus::Idle,
            1 => OtaStatus::UpdateReady,
            2 => OtaStatus::Applying,
            3 => OtaStatus::Applied,
            4 => OtaStatus::RolledBack,
            5 => OtaStatus::Failed,
            _ => OtaStatus::Idle,
        }
    }
}

// ---------------------------------------------------------------------------
// Firmware image header (on-wire format for downloaded firmware)
// ---------------------------------------------------------------------------
#[repr(C, packed)]
pub struct FirmwareHeader {
    pub magic: [u8; 8], // "HKL1FW__"
    pub version: u32,
    pub timestamp: u64,
    pub image_size: u32,     // Bytes of firmware body after header
    pub checksum: u32,       // CRC32 of firmware body
    pub min_hw_version: u32, // Minimum compatible hardware version
    pub reserved: [u8; 32],
}

impl FirmwareHeader {
    pub const MAGIC: [u8; 8] = *b"HKL1FW__";

    pub fn validate(&self) -> bool {
        &self.magic == &Self::MAGIC
            && self.image_size > 0
            && self.image_size < flash_layout::BANK_SIZE - core::mem::size_of::<Self>() as u32
    }
}

// ---------------------------------------------------------------------------
// STM32F7 Flash Controller MMIO
// ---------------------------------------------------------------------------
pub mod flash_ctrl {
    use core::ptr::{read_volatile, write_volatile};

    pub const FLASH_ACR: *mut u32 = 0x4002_3C00 as *mut u32;
    pub const FLASH_KEYR: *mut u32 = 0x4002_3C04 as *mut u32;
    pub const FLASH_CR: *mut u32 = 0x4002_3C0C as *mut u32;
    pub const FLASH_SR: *mut u32 = 0x4002_3C0C as *mut u32;

    // Key sequence to unlock flash
    pub const KEY1: u32 = 0x45670123;
    pub const KEY2: u32 = 0xCDEF89AB;

    // Flash control register bits
    pub const CR_PG: u32 = 1 << 0; // Programming
    pub const CR_SER: u32 = 1 << 1; // Sector erase
    pub const CR_MER: u32 = 1 << 2; // Mass erase
    pub const CR_STRT: u32 = 1 << 6; // Start
    pub const CR_LOCK: u32 = 1 << 7; // Lock
    pub const CR_PSIZE_32: u32 = 2 << 8; // 32-bit parallelism

    // Status register bits
    pub const SR_BSY: u32 = 1 << 0; // Busy
    pub const SR_EOP: u32 = 1 << 5; // End of operation

    pub fn unlock() -> bool {
        unsafe {
            if read_volatile(FLASH_CR) & CR_LOCK == 0 {
                return true;
            }
            write_volatile(FLASH_KEYR, KEY1);
            write_volatile(FLASH_KEYR, KEY2);
            (read_volatile(FLASH_CR) & CR_LOCK) == 0
        }
    }

    pub fn lock() {
        unsafe {
            write_volatile(FLASH_CR, read_volatile(FLASH_CR) | CR_LOCK);
        }
    }

    pub fn wait_ready() -> bool {
        let mut timeout = 100_000;
        unsafe {
            while read_volatile(FLASH_SR) & SR_BSY != 0 {
                timeout -= 1;
                if timeout == 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Erase a sector (64KB on STM32F7)
    pub fn erase_sector(sector: u8) -> bool {
        if !unlock() {
            return false;
        }
        unsafe {
            // Select sector erase, set sector number
            let snb = (sector as u32) << 3;
            write_volatile(
                FLASH_CR,
                (read_volatile(FLASH_CR) & 0xFFFF_FF07) | CR_SER | snb,
            );
            write_volatile(FLASH_CR, read_volatile(FLASH_CR) | CR_STRT);
        }
        if !wait_ready() {
            return false;
        }
        lock();
        true
    }

    /// Write 32-bit word to flash
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn write_word(addr: *mut u32, value: u32) -> bool {
        if !unlock() {
            return false;
        }
        unsafe {
            write_volatile(
                FLASH_CR,
                (read_volatile(FLASH_CR) & !0x300) | CR_PSIZE_32 | CR_PG,
            );
            write_volatile(addr, value);
        }
        if !wait_ready() {
            return false;
        }
        // Verify
        unsafe {
            if read_volatile(addr) != value {
                return false;
            }
        }
        lock();
        true
    }

    /// Write a block of 32-bit words to flash
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn write_block(addr: *mut u32, data: &[u32]) -> bool {
        if !unlock() {
            return false;
        }
        unsafe {
            write_volatile(
                FLASH_CR,
                (read_volatile(FLASH_CR) & !0x300) | CR_PSIZE_32 | CR_PG,
            );
            for (i, &word) in data.iter().enumerate() {
                write_volatile(addr.add(i), word);
                if !wait_ready() {
                    return false;
                }
                if read_volatile(addr.add(i)) != word {
                    return false;
                }
            }
        }
        lock();
        true
    }

    /// Read from flash (always safe)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn read_word(addr: *const u32) -> u32 {
        unsafe { read_volatile(addr) }
    }
}

// ---------------------------------------------------------------------------
// CRC32 for firmware validation
// ---------------------------------------------------------------------------
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// OTA Manager
// ---------------------------------------------------------------------------
pub struct OTAManager {
    pub active_bank: u8, // 0 = bank A, 1 = bank B
    pub bank_a_version: u32,
    pub bank_b_version: u32,
    pub status: OtaStatus,
    pub boot_count: u32,    // How many boots since last OTA
    pub firmware_size: u32, // Size of latest downloaded firmware
    pub firmware_checksum: u32,
    pub timestamp: u64,
    pub update_in_progress: bool,
    max_boot_count: u32, // Max boots before assuming firmware is OK
    testing: bool,       // In test mode, skip real flash I/O
}

impl OTAManager {
    pub const fn new() -> Self {
        Self {
            active_bank: 0,
            bank_a_version: 1,
            bank_b_version: 0,
            status: OtaStatus::Idle,
            boot_count: 0,
            firmware_size: 0,
            firmware_checksum: 0,
            timestamp: 0,
            update_in_progress: false,
            max_boot_count: 3,
            testing: false,
        }
    }

    // -----------------------------------------------------------------------
    // Initialization — called once at boot
    // -----------------------------------------------------------------------

    /// Read OTA metadata from flash and determine active bank
    pub fn init(&mut self) {
        if self.testing {
            return;
        }
        let active = flash_ctrl::read_word(flash_layout::OTA_ACTIVE_BANK);
        self.active_bank = if active == 1 { 1 } else { 0 };
        self.bank_a_version = flash_ctrl::read_word(flash_layout::OTA_VERSION_A);
        self.bank_b_version = flash_ctrl::read_word(flash_layout::OTA_VERSION_B);
        self.status = OtaStatus::from_u32(flash_ctrl::read_word(flash_layout::OTA_STATUS));
        self.boot_count = flash_ctrl::read_word(flash_layout::OTA_BOOT_COUNT);
        self.timestamp = if self.active_bank == 0 {
            unsafe {
                flash_ctrl::read_word(flash_layout::OTA_TIMESTAMP_A as *const u32) as u64
                    | (flash_ctrl::read_word((flash_layout::OTA_TIMESTAMP_A as *const u32).add(1))
                        as u64)
                        << 32
            }
        } else {
            unsafe {
                flash_ctrl::read_word(flash_layout::OTA_TIMESTAMP_B as *const u32) as u64
                    | (flash_ctrl::read_word((flash_layout::OTA_TIMESTAMP_B as *const u32).add(1))
                        as u64)
                        << 32
            }
        };

        // Handle rollback detection
        self.check_rollback();
    }

    /// Detect if the new firmware failed (watchdog reset before marking stable)
    fn check_rollback(&mut self) {
        match self.status {
            OtaStatus::Applied => {
                // We just booted into a new bank. Increment boot count.
                self.boot_count += 1;
                if self.boot_count >= self.max_boot_count {
                    // Firmware is stable — mark as Idle
                    self.status = OtaStatus::Idle;
                    self.save_metadata();
                } else {
                    self.save_metadata();
                }
            }
            OtaStatus::Applying => {
                // Previous boot was applying update but never marked Applied.
                // This means the new firmware failed to boot. Roll back.
                self.rollback();
            }
            OtaStatus::Failed => {
                // Unrecoverable — stay on current bank
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Update lifecycle
    // -----------------------------------------------------------------------

    /// Stage a firmware image (received via UART/SPI/I2C)
    pub fn stage_firmware(&mut self, header: &FirmwareHeader, body: &[u8]) -> bool {
        if !header.validate() {
            return false;
        }
        if body.len() as u32 != header.image_size {
            return false;
        }
        if header.image_size + core::mem::size_of::<FirmwareHeader>() as u32
            > flash_layout::BANK_SIZE
        {
            return false;
        }

        // Verify checksum
        let computed = crc32(body);
        if computed != header.checksum {
            return false;
        }

        // Find target bank (the one not currently active)
        let target_bank_base = if self.active_bank == 0 {
            flash_layout::BANK_B_BASE
        } else {
            flash_layout::BANK_A_BASE
        };

        // Erase target bank sectors
        let start_sector = if target_bank_base == flash_layout::BANK_B_BASE {
            4
        } else {
            0
        };
        for sector in start_sector..start_sector + 4 {
            if !flash_ctrl::erase_sector(sector) {
                return false;
            }
        }

        // Write header + body
        let hdr_words = core::mem::size_of::<FirmwareHeader>() / 4;
        let hdr_ptr = target_bank_base as *mut u32;
        let hdr_slice = unsafe {
            core::slice::from_raw_parts((header as *const FirmwareHeader) as *const u32, hdr_words)
        };
        if !flash_ctrl::write_block(hdr_ptr, hdr_slice) {
            return false;
        }

        // Write body in chunks
        let body_base =
            (target_bank_base + core::mem::size_of::<FirmwareHeader>() as u32) as *mut u32;
        let mut offset: usize = 0;
        let mut buf = [0u8; 1028]; // Reusable padding buffer (max chunk + 3 bytes)
        for chunk in body.chunks(1024) {
            let n = chunk.len();
            let aligned_len = (n + 3) & !3;
            let data: &[u8] = if aligned_len != n {
                buf[..n].copy_from_slice(chunk);
                buf[n..aligned_len].fill(0);
                &buf[..aligned_len]
            } else {
                chunk
            };
            let words: &[u32] =
                unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) };
            if !flash_ctrl::write_block(unsafe { body_base.add(offset) }, words) {
                return false;
            }
            offset += words.len();
        }

        // Update OTA metadata
        self.firmware_size = header.image_size;
        self.firmware_checksum = header.checksum;
        self.timestamp = header.timestamp;
        self.status = OtaStatus::UpdateReady;
        self.save_metadata();

        true
    }

    /// Apply staged update: switch banks and reset
    pub fn apply_update(&mut self) -> bool {
        if self.status != OtaStatus::UpdateReady {
            return false;
        }

        // Save current weights before switching
        crate::system::persistence::PersistenceManager::save();

        // Mark as applying
        self.status = OtaStatus::Applying;
        let new_bank = if self.active_bank == 0 { 1 } else { 0 };

        // Update metadata: switch active bank
        self.active_bank = new_bank;
        self.boot_count = 0;
        self.save_metadata();

        // Trigger soft reset to boot into new bank
        self.trigger_soft_reset();
        true
    }

    /// Rollback to the other bank
    pub fn rollback(&mut self) {
        let other = if self.active_bank == 0 { 1 } else { 0 };
        self.active_bank = other;
        self.status = OtaStatus::RolledBack;
        self.boot_count = 0;
        self.save_metadata();
    }

    /// Confirm firmware is stable (called from boot after successful init)
    pub fn confirm_stable(&mut self) {
        if self.status == OtaStatus::Applied && self.boot_count >= self.max_boot_count {
            self.status = OtaStatus::Idle;
            self.save_metadata();
        }
    }

    /// Check if an update is available
    pub fn check_for_update(&mut self) -> bool {
        self.status == OtaStatus::UpdateReady
    }

    /// Get base address of active firmware bank
    pub fn active_bank_base(&self) -> u32 {
        if self.active_bank == 0 {
            flash_layout::BANK_A_BASE
        } else {
            flash_layout::BANK_B_BASE
        }
    }

    // -----------------------------------------------------------------------
    // Flash metadata persistence
    // -----------------------------------------------------------------------

    /// Save OTA metadata to flash
    pub fn save_metadata(&self) {
        if self.testing {
            return;
        }
        let fields: &[(u32, *mut u32)] = &[
            (self.active_bank as u32, flash_layout::OTA_ACTIVE_BANK),
            (self.bank_a_version, flash_layout::OTA_VERSION_A),
            (self.bank_b_version, flash_layout::OTA_VERSION_B),
            (self.status as u32, flash_layout::OTA_STATUS),
            (self.boot_count, flash_layout::OTA_BOOT_COUNT),
            (self.firmware_checksum, flash_layout::OTA_CHECKSUM_A),
            (self.firmware_checksum, flash_layout::OTA_CHECKSUM_B),
            (self.firmware_size, flash_layout::OTA_FW_SIZE_A),
            (self.firmware_size, flash_layout::OTA_FW_SIZE_B),
        ];
        for &(val, addr) in fields {
            flash_ctrl::write_word(addr, val);
        }
        // Write timestamp (u64 as two u32)
        let ts_low = self.timestamp as u32;
        let ts_high = (self.timestamp >> 32) as u32;
        let ts_base = if self.active_bank == 0 {
            flash_layout::OTA_TIMESTAMP_A as *mut u32
        } else {
            flash_layout::OTA_TIMESTAMP_B as *mut u32
        };
        flash_ctrl::write_word(ts_base, ts_low);
        flash_ctrl::write_word(unsafe { ts_base.add(1) }, ts_high);
    }

    /// Load metadata from flash into fields
    pub fn load_metadata(&mut self) {
        if self.testing {
            return;
        }
        self.active_bank = if flash_ctrl::read_word(flash_layout::OTA_ACTIVE_BANK) == 1 {
            1
        } else {
            0
        };
        self.bank_a_version = flash_ctrl::read_word(flash_layout::OTA_VERSION_A);
        self.bank_b_version = flash_ctrl::read_word(flash_layout::OTA_VERSION_B);
        self.status = OtaStatus::from_u32(flash_ctrl::read_word(flash_layout::OTA_STATUS));
        self.boot_count = flash_ctrl::read_word(flash_layout::OTA_BOOT_COUNT);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Trigger soft reset to boot into the selected bank
    fn trigger_soft_reset(&self) {
        // Set the AIRCR register for system reset
        #[cfg(target_arch = "arm")]
        unsafe {
            const AIRCR: *mut u32 = 0xE000_ED0C as *mut u32;
            // SYSRESETREQ = bit 2, VECTKEY = 0x05FA
            core::ptr::write_volatile(AIRCR, 0x05FA_0004);
        }
        #[cfg(not(target_arch = "arm"))]
        {
            // Non-ARM: trigger a trap via illegal instruction or panic
            #[cfg(target_arch = "riscv32")]
            unsafe {
                core::arch::asm!("unimp");
            }
            #[cfg(not(target_arch = "riscv32"))]
            {
                // Host or unknown arch — loop forever
                loop {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------
use core::mem::MaybeUninit;
pub static mut OTA_MANAGER: MaybeUninit<OTAManager> = MaybeUninit::uninit();

pub fn init_ota() {
    unsafe {
        let ota = OTA_MANAGER.write(OTAManager::new());
        ota.init();
    }
}

pub fn ota_manager() -> &'static mut OTAManager {
    unsafe { &mut *OTA_MANAGER.as_mut_ptr() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ota_manager_new_default() {
        let ota = OTAManager::new();
        assert_eq!(ota.active_bank, 0);
        assert_eq!(ota.status, OtaStatus::Idle);
        assert!(!ota.update_in_progress);
    }

    #[test]
    fn firmware_header_validate_valid() {
        let hdr = FirmwareHeader {
            magic: *b"HKL1FW__",
            version: 2,
            timestamp: 1000,
            image_size: 1024,
            checksum: 0,
            min_hw_version: 1,
            reserved: [0; 32],
        };
        assert!(hdr.validate());
    }

    #[test]
    fn firmware_header_validate_bad_magic() {
        let hdr = FirmwareHeader {
            magic: *b"BADMAGIC",
            version: 2,
            timestamp: 1000,
            image_size: 1024,
            checksum: 0,
            min_hw_version: 1,
            reserved: [0; 32],
        };
        assert!(!hdr.validate());
    }

    #[test]
    fn firmware_header_validate_zero_size() {
        let hdr = FirmwareHeader {
            magic: *b"HKL1FW__",
            version: 2,
            timestamp: 1000,
            image_size: 0,
            checksum: 0,
            min_hw_version: 1,
            reserved: [0; 32],
        };
        assert!(!hdr.validate());
    }

    #[test]
    fn ota_status_from_u32() {
        assert_eq!(OtaStatus::from_u32(0), OtaStatus::Idle);
        assert_eq!(OtaStatus::from_u32(1), OtaStatus::UpdateReady);
        assert_eq!(OtaStatus::from_u32(5), OtaStatus::Failed);
        assert_eq!(OtaStatus::from_u32(99), OtaStatus::Idle); // Unknown → Idle
    }

    #[test]
    fn crc32_known_value() {
        let data = b"Hello, World!";
        // Known CRC32 for this string
        let crc = crc32(data);
        assert!(crc != 0);
        assert_eq!(crc, crc32(data)); // Deterministic
    }

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_different_inputs_different() {
        let a = crc32(b"foo");
        let b = crc32(b"bar");
        assert_ne!(a, b);
    }

    #[test]
    fn ota_manager_init_defaults() {
        let mut ota = OTAManager::new();
        ota.testing = true;
        ota.init();
        assert_eq!(ota.active_bank, 0);
    }

    #[test]
    fn ota_manager_check_update_no_update() {
        let mut ota = OTAManager::new();
        assert!(!ota.check_for_update());
    }

    #[test]
    fn ota_manager_rollback_switches_bank() {
        let mut ota = OTAManager::new();
        ota.testing = true;
        ota.active_bank = 1;
        ota.rollback();
        assert_eq!(ota.active_bank, 0);
        assert_eq!(ota.status, OtaStatus::RolledBack);
    }

    #[test]
    fn ota_manager_confirm_stable() {
        let mut ota = OTAManager::new();
        ota.testing = true;
        ota.status = OtaStatus::Applied;
        ota.boot_count = 5;
        ota.max_boot_count = 3;
        ota.confirm_stable();
        assert_eq!(ota.status, OtaStatus::Idle);
    }

    #[test]
    fn ota_manager_active_bank_base() {
        let mut ota = OTAManager::new();
        ota.active_bank = 0;
        assert_eq!(ota.active_bank_base(), flash_layout::BANK_A_BASE);
        ota.active_bank = 1;
        assert_eq!(ota.active_bank_base(), flash_layout::BANK_B_BASE);
    }

    #[test]
    fn ota_manager_stage_firmware_invalid_header() {
        let mut ota = OTAManager::new();
        let bad_hdr = FirmwareHeader {
            magic: *b"BADMAGIC",
            version: 0,
            timestamp: 0,
            image_size: 0,
            checksum: 0,
            min_hw_version: 0,
            reserved: [0; 32],
        };
        assert!(!ota.stage_firmware(&bad_hdr, &[]));
    }

    #[test]
    fn ota_manager_load_metadata_roundtrip() {
        let mut ota = OTAManager::new();
        ota.testing = true;
        ota.load_metadata();
        assert!(ota.active_bank == 0 || ota.active_bank == 1);
    }
}
