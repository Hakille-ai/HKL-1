//! Zero-dependency media parsers for the HKL-2 data pipeline.
//!
//! - [`WavReader`]: minimal RIFF/WAVE reader producing 16-bit PCM sample
//!   frames (the exact format consumed by `AudioSpikeEncoder`).
//! - [`PgmReader`]: binary P5 PGM reader producing 8-bit grayscale frames
//!   (the exact format consumed by `VisionSpikeEncoder`).

use crate::vision::retina::VISION_PIXELS;
use alloc::vec::Vec;

/// A parsed 16-bit PCM WAV file.
#[derive(Debug, Clone)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<i16>,
}

impl Wav {
    /// Split samples into fixed-length frames (e.g. 512 samples @ 16 kHz).
    pub fn frames(&self, frame_len: usize) -> Vec<Vec<i16>> {
        self.samples
            .chunks(frame_len.max(1))
            .map(|chunk| chunk.to_vec())
            .filter(|f| f.len() == frame_len)
            .collect()
    }
}

/// Minimal RIFF/WAVE parser: accepts PCM, 16-bit, any channel count
/// (only the first channel is kept).
pub fn parse_wav(bytes: &[u8]) -> Result<Wav, &'static str> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file");
    }

    let mut pos = 12usize;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut format_tag = 0u16;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let chunk_start = pos + 8;
        let chunk_end = chunk_start.saturating_add(chunk_len);
        if chunk_end > bytes.len() {
            break;
        }

        match chunk_id {
            b"fmt " => {
                if chunk_len < 16 {
                    return Err("fmt chunk too small");
                }
                format_tag = u16::from_le_bytes([bytes[chunk_start], bytes[chunk_start + 1]]);
                channels = u16::from_le_bytes([bytes[chunk_start + 2], bytes[chunk_start + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[chunk_start + 4],
                    bytes[chunk_start + 5],
                    bytes[chunk_start + 6],
                    bytes[chunk_start + 7],
                ]);
                bits = u16::from_le_bytes([bytes[chunk_start + 14], bytes[chunk_start + 15]]);
            }
            b"data" => {
                data = Some(&bytes[chunk_start..chunk_end]);
            }
            _ => {}
        }
        pos = chunk_end + (chunk_len % 2);
    }

    if format_tag != 1 {
        return Err("only uncompressed PCM is supported");
    }
    if bits != 16 {
        return Err("only 16-bit PCM is supported");
    }
    if channels == 0 || sample_rate == 0 {
        return Err("missing fmt header");
    }
    let data = data.ok_or("missing data chunk")?;

    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }

    Ok(Wav {
        sample_rate,
        channels,
        samples,
    })
}

/// A parsed binary (P5) PGM image.
#[derive(Debug, Clone)]
pub struct Pgm {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl Pgm {
    /// Convert to the fixed 32x32 grayscale frame consumed by the vision
    /// encoder, or `None` when the image dimensions do not match.
    pub fn to_frame_32x32(&self) -> Option<[u8; VISION_PIXELS]> {
        if self.width != 32 || self.height != 32 || self.pixels.len() != VISION_PIXELS {
            return None;
        }
        let mut frame = [0u8; VISION_PIXELS];
        frame.copy_from_slice(&self.pixels);
        Some(frame)
    }
}

/// Minimal binary P5 PGM parser.
pub fn parse_pgm(bytes: &[u8]) -> Result<Pgm, &'static str> {
    let mut cursor = 0usize;

    let next_header_field = |bytes: &[u8], cursor: &mut usize| -> Option<Vec<u8>> {
        let mut field = Vec::new();
        loop {
            if *cursor >= bytes.len() {
                return None;
            }
            let b = bytes[*cursor];
            *cursor += 1;
            if b == b'\n' || b == b' ' || b == b'\t' || b == b'\r' {
                if !field.is_empty() {
                    return Some(field);
                }
                continue;
            }
            field.push(b);
        }
    };

    let magic = next_header_field(bytes, &mut cursor).ok_or("unexpected EOF")?;
    if magic != b"P5" {
        return Err("not a P5 PGM image");
    }
    let width_bytes = next_header_field(bytes, &mut cursor).ok_or("unexpected EOF")?;
    let height_bytes = next_header_field(bytes, &mut cursor).ok_or("unexpected EOF")?;
    let maxval_bytes = next_header_field(bytes, &mut cursor).ok_or("unexpected EOF")?;

    let width = parse_ascii_number(&width_bytes).ok_or("bad width")?;
    let height = parse_ascii_number(&height_bytes).ok_or("bad height")?;
    let maxval = parse_ascii_number(&maxval_bytes).ok_or("bad maxval")?;
    if width == 0 || height == 0 || maxval == 0 || maxval > 255 {
        return Err("unsupported dimensions or maxval");
    }

    // A single whitespace byte must separate the header from the raster.
    while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }

    let expected = width.checked_mul(height).ok_or("dimensions overflow")?;
    if bytes.len() - cursor < expected {
        return Err("raster truncated");
    }

    let pixels = bytes[cursor..cursor + expected].to_vec();
    Ok(Pgm {
        width,
        height,
        pixels,
    })
}

fn parse_ascii_number(bytes: &[u8]) -> Option<usize> {
    let text = core::str::from_utf8(bytes).ok()?;
    text.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + samples.len() * 2) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        wav
    }

    #[test]
    fn test_wav_parse_roundtrip() {
        let samples: Vec<i16> = (0..1024)
            .map(|i| ((i as f32).sin() * 1000.0) as i16)
            .collect();
        let bytes = build_wav(16000, &samples);

        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");

        let wav = parse_wav(&bytes).expect("wav parse failed");
        assert_eq!(wav.sample_rate, 16000);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.samples, samples);

        let frames = wav.frames(512);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], samples[..512].to_vec());
    }

    #[test]
    fn test_wav_rejects_bad_input() {
        assert!(parse_wav(b"not a wav").is_err());
        assert!(parse_wav(b"RIFF\x00\x00\x00\x00WAVEgarbage").is_err());

        // 8-bit is unsupported
        let mut bytes = build_wav(16000, &[0i16, 100, -100]);
        bytes[34] = 8;
        assert!(parse_wav(&bytes).is_err());
    }

    #[test]
    fn test_pgm_parse_roundtrip() {
        let mut pgm = Vec::new();
        pgm.extend_from_slice(b"P5\n32 32\n255\n");
        pgm.extend((0..1024).map(|i| (i % 256) as u8));

        let parsed = parse_pgm(&pgm).expect("pgm parse failed");
        assert_eq!(parsed.width, 32);
        assert_eq!(parsed.height, 32);
        let frame = parsed.to_frame_32x32().expect("32x32 expected");
        assert_eq!(frame[0], 0);
        assert_eq!(frame[1023], 255);
    }

    #[test]
    fn test_pgm_rejects_bad_input() {
        assert!(parse_pgm(b"P2 32 32 255 x").is_err());
        assert!(parse_pgm(b"P5\n32 32\n255\nshort").is_err());

        let mut truncated = b"P5\n8 8\n255\n".to_vec();
        truncated.extend([0u8; 63]);
        assert!(parse_pgm(&truncated).is_err());

        let mut pgm = Vec::new();
        pgm.extend_from_slice(b"P5\n64 64\n255\n");
        pgm.extend((0..64 * 64).map(|i| (i % 256) as u8));
        let parsed = parse_pgm(&pgm).unwrap();
        assert!(parsed.to_frame_32x32().is_none());
    }
}
