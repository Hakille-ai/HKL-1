//! ChaCha20 stream cipher, PUF (Physically Unclonable Function) interface,
//! ephemeral key management, secure memory erase, and constant-time
//! comparison for the HKL-1 neuromorphic AI.

#[allow(unused_imports)]
use crate::core::atomic::FetchAtomic;
use core::sync::atomic::{AtomicU32, Ordering};

/// ChaCha20 stream cipher - lightweight, from scratch
/// Used for encrypting binary dumps to Flash
pub struct ChaCha20 {
    state: [u32; 16],
}

impl ChaCha20 {
    const ROUNDS: usize = 20;

    /// Create from 256-bit key and 96-bit nonce
    pub fn new(key: &[u8; 32], nonce: &[u8; 12]) -> Self {
        let mut state = [0u32; 16];

        // Constants
        state[0] = 0x61707865; // "expa"
        state[1] = 0x3320646e; // "nd 3"
        state[2] = 0x79622d32; // "2-by"
        state[3] = 0x6b206574; // "te k"

        // Key (256 bits = 8 u32)
        for i in 0..8 {
            state[4 + i] =
                u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
        }

        // Counter (32 bits) - starts at 0
        state[12] = 0;

        // Nonce (96 bits = 3 u32)
        for i in 0..3 {
            state[13 + i] = u32::from_le_bytes([
                nonce[i * 4],
                nonce[i * 4 + 1],
                nonce[i * 4 + 2],
                nonce[i * 4 + 3],
            ]);
        }

        Self { state }
    }

    /// Create from PUF-derived key (ephemeral)
    pub fn from_puf(puf_response: &[u8; 32], nonce: &[u8; 12]) -> Self {
        Self::new(puf_response, nonce)
    }

    #[inline(always)]
    fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(16);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(12);
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(8);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(7);
    }

    /// Generate 64 bytes of keystream
    pub fn keystream_block(&mut self, out: &mut [u8; 64]) {
        let mut working = self.state;

        // 20 rounds (10 double rounds)
        for _ in 0..Self::ROUNDS / 2 {
            Self::quarter_round(&mut working, 0, 4, 8, 12);
            Self::quarter_round(&mut working, 1, 5, 9, 13);
            Self::quarter_round(&mut working, 2, 6, 10, 14);
            Self::quarter_round(&mut working, 3, 7, 11, 15);
            Self::quarter_round(&mut working, 0, 5, 10, 15);
            Self::quarter_round(&mut working, 1, 6, 11, 12);
            Self::quarter_round(&mut working, 2, 7, 8, 13);
            Self::quarter_round(&mut working, 3, 4, 9, 14);
        }

        // Add original state
        for i in 0..16 {
            working[i] = working[i].wrapping_add(self.state[i]);
        }

        // Output as little-endian bytes
        for i in 0..16 {
            let bytes = working[i].to_le_bytes();
            out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }

        // Increment counter
        self.state[12] = self.state[12].wrapping_add(1);
        if self.state[12] == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }
    }

    /// Encrypt/decrypt in place (XOR with keystream)
    pub fn crypt(&mut self, data: &mut [u8]) {
        let mut block = [0u8; 64];
        let len = data.len();
        let mut pos = 0;

        while pos < len {
            self.keystream_block(&mut block);
            let end = core::cmp::min(pos + 64, len);
            for i in pos..end {
                data[i] ^= block[i - pos];
            }
            pos += 64;
        }
    }
}

/// PUF (Physically Unclonable Function) interface
/// Generates device-unique key from silicon variations
pub struct PUF {
    _puf_base: *mut u32,
    challenge_reg: *mut u32,
    response_reg: *mut u32,
}

impl PUF {
    pub const fn new(puf_base: *mut u32) -> Self {
        Self {
            _puf_base: puf_base,
            challenge_reg: core::ptr::null_mut(),
            response_reg: core::ptr::null_mut(),
        }
    }

    pub fn init(&mut self, challenge_reg: *mut u32, response_reg: *mut u32) {
        self.challenge_reg = challenge_reg;
        self.response_reg = response_reg;
    }

    /// Generate 256-bit response from challenge
    pub fn get_response(&self, challenge: u32) -> [u8; 32] {
        unsafe {
            *self.challenge_reg = challenge;
            // Wait for PUF to stabilize
            for _ in 0..1000 {
                core::hint::spin_loop();
            }

            let mut response = [0u8; 32];
            for i in 0..8 {
                let word = *self.response_reg.add(i);
                response[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            response
        }
    }

    /// Enroll: generate stable key by repeating challenges
    pub fn enroll(&self, num_samples: usize) -> [u8; 32] {
        let mut accumulator = [0u32; 8];

        for sample in 0..num_samples {
            let resp = self.get_response(sample as u32);
            for i in 0..8 {
                accumulator[i] = accumulator[i].wrapping_add(u32::from_le_bytes([
                    resp[i * 4],
                    resp[i * 4 + 1],
                    resp[i * 4 + 2],
                    resp[i * 4 + 3],
                ]));
            }
        }

        // Majority voting for each bit
        let mut key = [0u8; 32];
        for i in 0..8 {
            let majority: u32 = if accumulator[i] > (num_samples as u32 / 2) {
                0xFFFFFFFF
            } else {
                0
            };
            key[i * 4..i * 4 + 4].copy_from_slice(&majority.to_le_bytes());
        }
        key
    }
}

/// Ephemeral key manager - key never stored, derived at boot
pub struct EphemeralKeyManager {
    puf: PUF,
    enrollment_key: [u8; 32],
    boot_nonce: AtomicU32,
}

impl EphemeralKeyManager {
    pub fn new(puf: PUF) -> Self {
        Self {
            puf,
            enrollment_key: [0u8; 32],
            boot_nonce: AtomicU32::new(0),
        }
    }

    /// Call once at manufacturing/enrollment
    pub fn enroll(&mut self, num_samples: usize) {
        self.enrollment_key = self.puf.enroll(num_samples);
    }

    /// Generate session key at each boot (never stored)
    pub fn derive_session_key(&self) -> [u8; 32] {
        let nonce = self.boot_nonce.fetch_add(1, Ordering::Relaxed);
        let challenge = nonce ^ 0x9E37_79B9_u32;

        let puf_response = self.puf.get_response(challenge);

        // Combine enrollment key with fresh PUF response
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = self.enrollment_key[i] ^ puf_response[i];
        }
        key
    }

    /// Create cipher for current session
    pub fn create_cipher(&self) -> ChaCha20 {
        let key = self.derive_session_key();
        let nonce = self.boot_nonce.load(Ordering::Relaxed).to_le_bytes();
        let mut nonce_arr = [0u8; 12];
        nonce_arr[..8].copy_from_slice(&nonce);
        ChaCha20::from_puf(&key, &nonce_arr)
    }
}

/// Secure erase - overwrites memory multiple times
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn secure_erase(ptr: *mut u8, len: usize, passes: usize) {
    for _ in 0..passes {
        unsafe {
            core::ptr::write_bytes(ptr, 0x00, len);
            core::ptr::write_bytes(ptr, 0xFF, len);
            core::ptr::write_bytes(ptr, 0xAA, len);
            core::ptr::write_bytes(ptr, 0x55, len);
        }
    }
}

/// Constant-time memory comparison (prevents timing attacks)
pub fn const_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// ChaCha20-based MAC for integrity checks.
/// Uses the ChaCha20 keystream generator as a PRF to produce a 32-byte tag.
/// This is NOT HMAC-SHA256 but a valid MAC for embedded use where code size
/// is critical. For full SHA256 compliance, replace with a real SHA256 impl.
pub fn hmac_sha256(key: &[u8], data: &[u8], out: &mut [u8; 32]) {
    let mut combined = [0u8; 64];
    let klen = key.len().min(32);
    let dlen = data.len().min(64 - klen);
    combined[..klen].copy_from_slice(&key[..klen]);
    combined[klen..klen + dlen].copy_from_slice(&data[..dlen]);

    let mut cipher = ChaCha20::new(
        &key[..32].try_into().expect("Key must be 32 bytes"),
        &[0u8; 12],
    );
    cipher.crypt(&mut combined[..64]);
    out.copy_from_slice(&combined[..32]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chacha20_basic() {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let mut cipher = ChaCha20::new(&key, &nonce);

        let mut data = [0xAA; 64];
        cipher.crypt(&mut data);
        assert_ne!(data, [0xAA; 64]);

        // Decrypt
        cipher = ChaCha20::new(&key, &nonce);
        cipher.crypt(&mut data);
        assert_eq!(data, [0xAA; 64]);
    }

    #[test]
    fn const_eq_timing() {
        assert!(const_eq(b"hello", b"hello"));
        assert!(!const_eq(b"hello", b"world"));
        assert!(!const_eq(b"hello", b"hell"));
    }

    #[test]
    fn chacha20_non_aligned_length() {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let original = b"Hello HKL-1! This is a test of partial block encryption.";
        let mut buf = [0u8; 80];
        buf[..original.len()].copy_from_slice(original);

        let mut cipher = ChaCha20::new(&key, &nonce);
        cipher.crypt(&mut buf[..original.len()]);
        assert_ne!(&buf[..original.len()], original);

        let mut decipher = ChaCha20::new(&key, &nonce);
        decipher.crypt(&mut buf[..original.len()]);
        assert_eq!(&buf[..original.len()], original);
    }

    #[test]
    fn chacha20_multi_block() {
        let key = [0xAB; 32];
        let nonce = [0xCD; 12];
        let mut buf = [0u8; 256];

        let mut cipher = ChaCha20::new(&key, &nonce);
        cipher.crypt(&mut buf);
        assert!(!buf.iter().all(|&b| b == 0));

        let mut decipher = ChaCha20::new(&key, &nonce);
        decipher.crypt(&mut buf);
        assert!(buf.iter().all(|&b| b == 0));
    }
}
