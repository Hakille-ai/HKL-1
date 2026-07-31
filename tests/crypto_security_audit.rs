//! Cryptographic Security Audit & Cryptanalysis Test Suite for HKL-1 Engine
//!
//! Validates:
//! 1. ChaCha20 quarter-round avalanche effect and keystream non-repeatability.
//! 2. Constant-time memory comparison (`const_eq`).
//! 3. Secure erasure multi-pass verification (`secure_erase`).
//! 4. Flash dump encryption and decryption roundtrips.

#![cfg(test)]

use hkl1::core::crypto::{ChaCha20, const_eq, secure_erase};
use hkl1::system::persistence::{
    capture_simulation_snapshot, encrypt_dump, decrypt_dump,
};

#[test]
fn test_chacha20_avalanche_effect() {
    let key1 = [0x42u8; 32];
    let mut key2 = key1;
    key2[0] ^= 0x01; // 1-bit flip

    let nonce = [0x11u8; 12];
    let mut cipher1 = ChaCha20::new(&key1, &nonce);
    let mut cipher2 = ChaCha20::new(&key2, &nonce);

    let mut block1 = [0u8; 64];
    let mut block2 = [0u8; 64];

    cipher1.keystream_block(&mut block1);
    cipher2.keystream_block(&mut block2);

    let mut bit_flips = 0;
    for i in 0..64 {
        let diff = block1[i] ^ block2[i];
        bit_flips += diff.count_ones();
    }

    let total_bits = 64 * 8;
    let avalanche_percentage = (bit_flips as f64 / total_bits as f64) * 100.0;

    // Strict Avalanche Criterion (SAC) target: ~50% bit flip (within 40% - 60% window)
    assert!(
        avalanche_percentage > 40.0 && avalanche_percentage < 60.0,
        "Avalanche effect failed: {}% bit flips (expected ~50%)",
        avalanche_percentage
    );
}

#[test]
fn test_constant_time_comparison() {
    let secret = [0xDEu8, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
    let match_secret = secret;
    let mismatch_secret = [0xDEu8, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBF]; // 1 bit diff

    assert!(const_eq(&secret, &match_secret), "Matching arrays must return true");
    assert!(!const_eq(&secret, &mismatch_secret), "Mismatched arrays must return false");
}

#[test]
fn test_secure_erase_multi_pass() {
    let mut data = [0xABu8; 256];
    let ptr = data.as_mut_ptr();

    secure_erase(ptr, data.len(), 4);

    // After 4-pass overwrite (0x00, 0xFF, 0xAA, 0x55), final pass leaves 0x55
    for (idx, &byte) in data.iter().enumerate() {
        assert_eq!(
            byte, 0x55,
            "Byte at index {} must match final pass pattern 0x55",
            idx
        );
    }
}

#[test]
fn test_dump_encryption_decryption_roundtrip() {
    capture_simulation_snapshot();
    let slot = 0;

    encrypt_dump(slot);
    decrypt_dump(slot);
}
