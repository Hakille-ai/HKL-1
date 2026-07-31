//! Flash persistence layer. Binary dump/restore of neuron and synapse state.
use crate::MAX_NEURONS;
use crate::core::crypto::ChaCha20;
use crate::core::math::FixedPoint;
use crate::core::memory::{NEURON_ARRAY, NEURON_COUNT, NeuronState};
use crate::snn::synapse::{SYNAPSE_ARRAY, SYNAPSE_COUNT, Synapse};
use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;

pub static mut PERSISTENCE_SLOTS: [MaybeUninit<BinaryDump>; crate::PERSISTENCE_SLOTS] =
    unsafe { MaybeUninit::uninit().assume_init() };

pub const DUMP_SIZE_INFO: usize = core::mem::size_of::<BinaryDump>();

#[repr(C, packed)]
pub struct DumpHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub timestamp: u64,
    pub neuron_count: u32,
    pub synapse_count: u32,
    pub checksum: u32,
    pub encrypted: bool,
    pub padding: [u8; 15],
}

#[repr(C)]
pub struct BinaryDump {
    pub header: DumpHeader,
    pub neuron_states: [NeuronState; MAX_NEURONS],
    pub synapse_data: [Synapse; crate::MAX_SYNAPSES],
}

pub fn init_persistence() {
    for slot in 0..crate::PERSISTENCE_SLOTS {
        unsafe {
            let dump = &mut *PERSISTENCE_SLOTS[slot].as_mut_ptr();
            dump.header = DumpHeader {
                magic: [0; 8],
                version: 0,
                timestamp: 0,
                neuron_count: 0,
                synapse_count: 0,
                checksum: 0,
                encrypted: false,
                padding: [0; 15],
            };
        }
    }
}

fn dump_mut(slot: usize) -> *mut BinaryDump {
    unsafe { PERSISTENCE_SLOTS[slot].as_mut_ptr() }
}

fn dump_ref(slot: usize) -> *const BinaryDump {
    unsafe { PERSISTENCE_SLOTS[slot].as_ptr() }
}

pub struct PersistenceManager;

impl PersistenceManager {
    /// Save state with J-0/J-1/J-2 rotation:
    /// Slot 2 ← Slot 1 ← Slot 0 ← current state (oldest evicted)
    pub fn save() {
        // Rotate slots: oldest (2) gets overwritten by J-1, J-1 by J-0, J-0 by current
        for slot in (1..crate::PERSISTENCE_SLOTS).rev() {
            let src = dump_ref(slot - 1);
            let dst = dump_mut(slot);
            unsafe {
                core::ptr::copy_nonoverlapping(src, dst, 1);
            }
        }
        let slot = 0;
        capture_into_slot(slot);
        #[cfg(feature = "encryption")]
        encrypt_dump(slot);
        #[cfg(feature = "flash")]
        commit_to_flash(slot);
    }

    pub fn load_slot(slot: usize) -> bool {
        if slot >= crate::PERSISTENCE_SLOTS {
            return false;
        }
        #[cfg(feature = "flash")]
        read_from_flash(slot);
        #[cfg(feature = "encryption")]
        decrypt_dump(slot);
        let dump = dump_ref(slot);
        unsafe {
            if &(*dump).header.magic != b"HKL1DUMP" {
                return rollback_invalid(slot);
            }
        }
        restore_from_slot(slot);
        true
    }

    pub fn rollback() {
        if PersistenceManager::load_slot(1) {
            PersistenceManager::save();
        } else if PersistenceManager::load_slot(2) {
            PersistenceManager::save();
        } else {
            PersistenceManager::factory_reset();
        }
    }

    pub fn factory_reset() {
        let mut rng = crate::core::math::XorShift64Star::new(unsafe {
            crate::core::time::METABOLIC_CLOCK.cycles()
        });
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = crate::core::memory::NeuronId::new(i);
            let state = crate::core::memory::neuron_state(id);
            state.membrane_potential = FixedPoint::ZERO;
            state.threshold = FixedPoint::from_f32(rng.next_f32() * 0.5 + 0.75);
            state.bias_current = FixedPoint::from_f32(rng.next_f32() * 0.1);
        }
        PersistenceManager::save();
    }
}

fn rollback_invalid(_failed_slot: usize) -> bool {
    PersistenceManager::rollback();
    false
}

fn capture_into_slot(slot: usize) {
    unsafe {
        let dump = &mut *dump_mut(slot);
        let now = crate::core::time::METABOLIC_CLOCK.ticks_1hz() as u64;
        dump.header.magic = *b"HKL1DUMP";
        dump.header.version = 1;
        dump.header.timestamp = now;
        dump.header.neuron_count = NEURON_COUNT.load(Ordering::Relaxed) as u32;
        dump.header.synapse_count = SYNAPSE_COUNT.load(Ordering::Relaxed) as u32;
        dump.header.encrypted = false;

        let neuron_count = (NEURON_COUNT.load(Ordering::Relaxed) as usize).min(MAX_NEURONS);
        for i in 0..neuron_count {
            dump.neuron_states[i] = core::ptr::read(NEURON_ARRAY[i].as_ptr());
        }
        for i in neuron_count..MAX_NEURONS {
            dump.neuron_states[i] = NeuronState::default();
        }

        let synapse_count =
            (SYNAPSE_COUNT.load(Ordering::Relaxed) as usize).min(crate::MAX_SYNAPSES);
        for i in 0..synapse_count {
            dump.synapse_data[i] = core::ptr::read(SYNAPSE_ARRAY[i].as_ptr());
        }
        for i in synapse_count..crate::MAX_SYNAPSES {
            dump.synapse_data[i] = Synapse::default();
        }

        let mut checksum = 0u32;
        let dump_bytes = core::slice::from_raw_parts(
            dump as *const BinaryDump as *const u8,
            core::mem::size_of::<BinaryDump>(),
        );
        for &b in dump_bytes.iter().skip(core::mem::size_of::<DumpHeader>()) {
            checksum = checksum.wrapping_add(b as u32);
        }
        dump.header.checksum = checksum;
    }
}

fn restore_from_slot(slot: usize) {
    unsafe {
        let dump = &*dump_ref(slot);
        for i in 0..MAX_NEURONS {
            NEURON_ARRAY[i] = MaybeUninit::new(dump.neuron_states[i]);
        }
        for i in 0..crate::MAX_SYNAPSES {
            SYNAPSE_ARRAY[i] = MaybeUninit::new(dump.synapse_data[i]);
        }
        NEURON_COUNT.store(dump.header.neuron_count as usize, Ordering::Relaxed);
        SYNAPSE_COUNT.store(dump.header.synapse_count, Ordering::Relaxed);
    }
}

pub static mut SIMULATION_SAVE_SLOT: MaybeUninit<BinaryDump> = MaybeUninit::uninit();

pub fn capture_simulation_snapshot() {
    unsafe {
        let dump = &mut *SIMULATION_SAVE_SLOT.as_mut_ptr();
        dump.header.magic = *b"HKL1DUMP";
        dump.header.version = 1;
        dump.header.timestamp = crate::core::time::METABOLIC_CLOCK.ticks_1hz() as u64;
        dump.header.neuron_count = NEURON_COUNT.load(Ordering::Relaxed) as u32;
        dump.header.synapse_count = SYNAPSE_COUNT.load(Ordering::Relaxed) as u32;

        let neuron_count = (NEURON_COUNT.load(Ordering::Relaxed) as usize).min(MAX_NEURONS);
        for i in 0..neuron_count {
            dump.neuron_states[i] = core::ptr::read(NEURON_ARRAY[i].as_ptr());
        }
        let synapse_count =
            (SYNAPSE_COUNT.load(Ordering::Relaxed) as usize).min(crate::MAX_SYNAPSES);
        for i in 0..synapse_count {
            dump.synapse_data[i] = core::ptr::read(SYNAPSE_ARRAY[i].as_ptr());
        }
    }
}

pub fn restore_simulation_snapshot() {
    unsafe {
        let dump = &*SIMULATION_SAVE_SLOT.as_ptr();
        for i in 0..MAX_NEURONS {
            NEURON_ARRAY[i] = MaybeUninit::new(dump.neuron_states[i]);
        }
        for i in 0..crate::MAX_SYNAPSES {
            SYNAPSE_ARRAY[i] = MaybeUninit::new(dump.synapse_data[i]);
        }
        NEURON_COUNT.store(dump.header.neuron_count as usize, Ordering::Relaxed);
        SYNAPSE_COUNT.store(dump.header.synapse_count, Ordering::Relaxed);
    }
}

pub fn restore_state(dump: &BinaryDump) {
    unsafe {
        for i in 0..MAX_NEURONS {
            NEURON_ARRAY[i] = MaybeUninit::new(dump.neuron_states[i]);
        }
        for i in 0..crate::MAX_SYNAPSES {
            SYNAPSE_ARRAY[i] = MaybeUninit::new(dump.synapse_data[i]);
        }
        NEURON_COUNT.store(dump.header.neuron_count as usize, Ordering::Relaxed);
        SYNAPSE_COUNT.store(dump.header.synapse_count, Ordering::Relaxed);
    }
}

pub fn encrypt_dump(slot: usize) {
    unsafe {
        let dump = &mut *dump_mut(slot);
        let key = derive_encryption_key();
        let nonce = generate_nonce();
        let mut cipher = ChaCha20::new(&key, &nonce);
        let data = core::slice::from_raw_parts_mut(
            (dump as *mut BinaryDump as *mut u8).add(core::mem::size_of::<DumpHeader>()),
            core::mem::size_of::<BinaryDump>() - core::mem::size_of::<DumpHeader>(),
        );
        cipher.crypt(data);
        dump.header.encrypted = true;
    }
}

pub fn decrypt_dump(slot: usize) {
    unsafe {
        let dump = &mut *dump_mut(slot);
        if !dump.header.encrypted {
            return;
        }
        let key = derive_encryption_key();
        let nonce = generate_nonce();
        let mut cipher = ChaCha20::new(&key, &nonce);
        let data = core::slice::from_raw_parts_mut(
            (dump as *mut BinaryDump as *mut u8).add(core::mem::size_of::<DumpHeader>()),
            core::mem::size_of::<BinaryDump>() - core::mem::size_of::<DumpHeader>(),
        );
        cipher.crypt(data);
        dump.header.encrypted = false;
    }
}

fn derive_encryption_key() -> [u8; 32] {
    unsafe {
        let ts = crate::core::time::METABOLIC_CLOCK.cycles();
        let mut key = [0u8; 32];
        let bytes = ts.to_le_bytes();
        let mut i = 0;
        while i < 8 {
            key[i] = bytes[i];
            i += 1;
        }
        key
    }
}

fn generate_nonce() -> [u8; 12] {
    unsafe {
        let now = crate::core::time::METABOLIC_CLOCK.ticks_1khz();
        let mut nonce = [0u8; 12];
        let bytes = now.to_le_bytes();
        let mut i = 0;
        while i < 4 {
            nonce[i] = bytes[i];
            i += 1;
        }
        nonce
    }
}

#[cfg(feature = "flash")]
fn commit_to_flash(slot: usize) {
    unsafe {
        let dump = dump_mut(slot) as *mut u32;
        const FLASH_KEYR: *mut u32 = 0x40023C04 as *mut u32;
        const FLASH_CR: *mut u32 = 0x40023C0C as *mut u32;
        const FLASH_SR: *mut u32 = 0x40023C10 as *mut u32;
        const SR_BSY: u32 = 1 << 16;
        const CR_PG: u32 = 1 << 0;
        const CR_SER: u32 = 1 << 1;
        const _CR_SNB: u32 = 0x18; // bit 3-7 for sector number

        let flash_base = 0x08000000usize;
        let flash_addr = flash_base + slot * core::mem::size_of::<BinaryDump>();
        let count = core::mem::size_of::<BinaryDump>() / 4;

        // Unlock FLASH
        core::ptr::write_volatile(FLASH_KEYR, 0x45670123);
        core::ptr::write_volatile(FLASH_KEYR, 0xCDEF89AB);

        // Erase target sector
        let sector_num = match slot {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 0,
        };
        core::ptr::write_volatile(FLASH_CR, CR_SER | (sector_num << 3));
        core::ptr::write_volatile(FLASH_CR, CR_SER | (sector_num << 3) | (1 << 16)); // START
        while core::ptr::read_volatile(FLASH_SR) & SR_BSY != 0 {}
        core::ptr::write_volatile(FLASH_CR, 0);

        // Program words
        core::ptr::write_volatile(FLASH_CR, CR_PG);
        for i in 0..count {
            core::ptr::write_volatile(
                (flash_addr + i * 4) as *mut u32,
                core::ptr::read(dump.add(i)),
            );
            while core::ptr::read_volatile(FLASH_SR) & SR_BSY != 0 {}
        }
        core::ptr::write_volatile(FLASH_CR, 0);
        core::ptr::write_volatile(FLASH_KEYR, 0); // Lock
    }
}

#[cfg(feature = "flash")]
fn read_from_flash(slot: usize) {
    unsafe {
        let dump = dump_mut(slot) as *mut u32;
        let flash_addr = 0x08000000usize + slot * core::mem::size_of::<BinaryDump>();
        let count = core::mem::size_of::<BinaryDump>() / 4;
        for i in 0..count {
            core::ptr::write(
                dump.add(i),
                core::ptr::read_volatile((flash_addr + i * 4) as *const u32),
            );
        }
    }
}

pub fn verify_dump(slot: usize) -> bool {
    unsafe {
        let dump = &*dump_ref(slot);
        if &dump.header.magic != b"HKL1DUMP" {
            return false;
        }
        let stored = dump.header.checksum;
        let mut checksum = 0u32;
        let bytes = core::slice::from_raw_parts(
            dump as *const BinaryDump as *const u8,
            core::mem::size_of::<BinaryDump>(),
        );
        for &b in bytes.iter().skip(core::mem::size_of::<DumpHeader>()) {
            checksum = checksum.wrapping_add(b as u32);
        }
        checksum == stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_persistence_clears_all_slots() {
        init_persistence();
        unsafe {
            for slot in 0..crate::PERSISTENCE_SLOTS {
                let dump = PERSISTENCE_SLOTS[slot].as_ptr();
                let magic = core::ptr::read_unaligned(core::ptr::addr_of!((*dump).header.magic));
                let version =
                    core::ptr::read_unaligned(core::ptr::addr_of!((*dump).header.version));
                assert_eq!(magic, [0; 8]);
                assert_eq!(version, 0);
            }
        }
    }

    #[test]
    fn test_dump_header_size() {
        assert!(core::mem::size_of::<DumpHeader>() > 0);
        assert!(core::mem::size_of::<DumpHeader>() <= 64);
    }

    #[test]
    fn test_dump_size_info_positive() {
        assert!(DUMP_SIZE_INFO > 0);
    }
}
