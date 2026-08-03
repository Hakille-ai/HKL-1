//! HKL Corpus binary format (`.hklc`).
//!
//! A versioned, zero-dependency, order-preserving token container used to
//! ship datasets to the training pipeline. Layout (all little-endian):
//!
//! ```text
//! magic  "HKLC"      u8[4]
//! version            u16   = 1
//! flags              u16   = 0
//! token_count        u64
//! seq_len            u32   = 0 when unspecified
//! vocab_hash         u64   = FNV-1a over the tokenizer merge blob
//! meta_len           u32
//! meta               u8[meta_len]  (UTF-8 "source|license|language")
//! payload            u16[token_count]  (little endian per token)
//! checksum           u32   = wrapping sum over payload bytes
//! ```

use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

pub const HKLC_MAGIC: &[u8; 4] = b"HKLC";
pub const HKLC_VERSION: u16 = 1;

/// FNV-1a 64-bit hash of a byte slice (used to fingerprint tokenizers).
pub fn fnv_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A parsed `.hklc` corpus file.
#[derive(Debug, Clone)]
pub struct HklcFile {
    pub version: u16,
    pub seq_len: usize,
    pub vocab_hash: u64,
    pub meta: String,
    pub token_count: u64,
    payload: Vec<u16>,
}

impl HklcFile {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let mut r = Cursor::new(bytes);
        let magic = r.take(4)?;
        if magic != HKLC_MAGIC {
            return None;
        }
        let version = r.u16()?;
        if version > HKLC_VERSION {
            return None;
        }
        let _flags = r.u16()?;
        let token_count = r.u64()?;
        let seq_len = r.u32()? as usize;
        let vocab_hash = r.u64()?;
        let meta_len = r.u32()? as usize;
        let meta_bytes = r.take(meta_len)?;
        let meta = String::from_utf8_lossy(meta_bytes).into_owned();

        let payload_len = token_count.checked_mul(2)? as usize;
        let payload_bytes = r.take(payload_len)?;
        let mut payload = Vec::with_capacity(token_count as usize);
        for chunk in payload_bytes.chunks_exact(2) {
            payload.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        let checksum = r.u32()?;
        if r.remaining() != 0 || checksum != compute_checksum(payload_bytes) {
            return None;
        }

        Some(HklcFile {
            version,
            seq_len,
            vocab_hash,
            meta,
            token_count,
            payload,
        })
    }

    pub fn tokens(&self) -> Vec<u16> {
        self.payload.clone()
    }

    pub fn token_count(&self) -> usize {
        self.payload.len()
    }
}

/// Streams tokens into the `.hklc` binary layout.
pub struct HklcWriter {
    meta: String,
    seq_len: usize,
    vocab_hash: u64,
    tokens: Vec<u16>,
}

impl HklcWriter {
    pub fn new(meta: String, seq_len: usize, vocab_hash: u64) -> Self {
        Self {
            meta,
            seq_len,
            vocab_hash,
            tokens: Vec::new(),
        }
    }

    pub fn push_tokens(&mut self, tokens: &[u16]) {
        self.tokens.extend_from_slice(tokens);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        let meta_bytes = self.meta.as_bytes();
        let payload_bytes = self
            .tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>();

        let mut out = Vec::with_capacity(18 + meta_bytes.len() + payload_bytes.len() + 4);
        out.extend_from_slice(HKLC_MAGIC);
        out.extend_from_slice(&HKLC_VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u32).to_le_bytes());
        out.extend_from_slice(&self.vocab_hash.to_le_bytes());
        out.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(meta_bytes);
        out.extend_from_slice(&payload_bytes);
        out.extend_from_slice(&compute_checksum(&payload_bytes).to_le_bytes());
        out
    }
}

/// Save a token stream to a `.hkl` file.
pub fn save_corpus(
    path: &str,
    tokens: &[u16],
    meta: &str,
    seq_len: usize,
    vocab_hash: u64,
) -> Result<(), String> {
    let mut writer = HklcWriter::new(meta.to_string(), seq_len, vocab_hash);
    writer.push_tokens(tokens);
    let bytes = writer.into_bytes();
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {}", path, e))
}

/// Load a `.hklc` file from disk.
pub fn load_corpus(path: &str) -> Result<HklcFile, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    HklcFile::parse(&bytes).ok_or_else(|| format!("corrupt corpus file: {}", path))
}

/// Concatenate the tokens of multiple corpus files.
pub fn tokens_from_paths(paths: &[String]) -> Result<Vec<u16>, String> {
    let mut merged = Vec::new();
    for path in paths {
        merged.extend(load_corpus(path)?.tokens());
    }
    Ok(merged)
}

fn compute_checksum(payload: &[u8]) -> u32 {
    let mut sum = 0u32;
    for &b in payload {
        sum = sum.wrapping_add(b as u32);
    }
    sum
}

/// Minimal byte cursor used by the parser (keeps the crate dependency-free).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.bytes.len() {
            return None;
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Some(slice)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn u16(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hklc_write_read_roundtrip() {
        let tokens: Vec<u16> = (0..1000).map(|i| (i % 4096) as u16).collect();
        let mut writer = HklcWriter::new(String::from("demo|MIT|en"), 8, 42);
        writer.push_tokens(&tokens);
        let bytes = writer.into_bytes();

        let parsed = HklcFile::parse(&bytes).expect("corpus failed to parse");
        assert_eq!(parsed.tokens(), tokens);
        assert_eq!(parsed.meta, "demo|MIT|en");
        assert_eq!(parsed.vocab_hash, 42);
        assert_eq!(parsed.token_count, 1000);
    }

    #[test]
    fn test_hkcl_parser_rejects_corruption() {
        let bytes = HklcWriter::new(String::from("demo"), 0, 7).into_bytes();

        // Bad magic
        let mut corrupted = bytes.clone();
        corrupted[0] = b'X';
        assert!(HklcFile::parse(&corrupted).is_none());

        // Bad checksum (truncate last byte off payload by flipping a char)
        let mut corrupted = bytes.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xFF;
        assert!(HklcFile::parse(&corrupted).is_none());
    }

    #[test]
    fn test_fnv_hash_is_stable() {
        let a = fnv_hash(b"hello world");
        let b = fnv_hash(b"hello world");
        let c = fnv_hash(b"hello worlf");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
