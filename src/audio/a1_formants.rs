//! Primary Auditory Cortex A1 & Formant Extractor Module for HKL-1.
//! Maps tonotopic frequency channels, extracts vocal formants (F1, F2, F3),
//! and classifies fundamental vowels (/a/, /i/, /u/, /e/, /o/).

use crate::audio::cochlea::{BandResponse, ERB_CENTER_FREQS_HZ, NUM_COCHLEAR_BANDS};
use crate::core::math::FixedPoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VowelClass {
    Unknown,
    VowelA, // /a/ (F1 ~730Hz, F2 ~1090Hz)
    VowelI, // /i/ (F1 ~270Hz, F2 ~2290Hz)
    VowelU, // /u/ (F1 ~300Hz, F2 ~870Hz)
    VowelE, // /e/ (F1 ~530Hz, F2 ~1840Hz)
    VowelO, // /o/ (F1 ~570Hz, F2 ~840Hz)
}

/// Extracted Vocal Formants (F1, F2, F3 in Hz)
#[derive(Clone, Copy)]
pub struct FormantProfile {
    pub f1_hz: FixedPoint,
    pub f2_hz: FixedPoint,
    pub f3_hz: FixedPoint,
    pub vowel: VowelClass,
}

/// Cortex A1 & Formant Extractor
pub struct FormantExtractor;

impl FormantExtractor {
    /// Extract formant peaks (F1, F2, F3) from 32 cochlear band responses
    pub fn extract_formants(bands: &[BandResponse; NUM_COCHLEAR_BANDS]) -> FormantProfile {
        let mut peak_indices = [0usize; 3];
        let mut peak_count = 0;

        // Find spectral energy local maxima (peaks)
        for i in 1..(NUM_COCHLEAR_BANDS - 1) {
            if bands[i].energy > bands[i - 1].energy && bands[i].energy > bands[i + 1].energy {
                if bands[i].energy > FixedPoint::from_f32(0.1) && peak_count < 3 {
                    peak_indices[peak_count] = i;
                    peak_count += 1;
                }
            }
        }

        let f1_hz = if peak_count > 0 {
            FixedPoint::from_f32(ERB_CENTER_FREQS_HZ[peak_indices[0]])
        } else {
            FixedPoint::ZERO
        };

        let f2_hz = if peak_count > 1 {
            FixedPoint::from_f32(ERB_CENTER_FREQS_HZ[peak_indices[1]])
        } else {
            f1_hz
        };

        let f3_hz = if peak_count > 2 {
            FixedPoint::from_f32(ERB_CENTER_FREQS_HZ[peak_indices[2]])
        } else {
            f2_hz
        };

        // Classify vowel based on F1 and F2 formant ratios
        let vowel = Self::classify_vowel(f1_hz.to_f32(), f2_hz.to_f32());

        FormantProfile {
            f1_hz,
            f2_hz,
            f3_hz,
            vowel,
        }
    }

    /// Classify vowel according to F1 / F2 acoustic formant chart
    pub fn classify_vowel(f1: f32, f2: f32) -> VowelClass {
        if f1 < 10.0 || f2 < 10.0 {
            return VowelClass::Unknown;
        }

        if f1 > 600.0 && f2 < 1400.0 {
            VowelClass::VowelA // High F1, Low F2 -> /a/
        } else if f1 < 400.0 && f2 > 1800.0 {
            VowelClass::VowelI // Low F1, High F2 -> /i/
        } else if f1 < 400.0 && f2 < 1200.0 {
            VowelClass::VowelU // Low F1, Low F2 -> /u/
        } else if f1 >= 400.0 && f1 <= 600.0 && f2 > 1500.0 {
            VowelClass::VowelE // Mid F1, High F2 -> /e/
        } else if f1 >= 400.0 && f1 <= 650.0 && f2 < 1200.0 {
            VowelClass::VowelO // Mid F1, Low F2 -> /o/
        } else {
            VowelClass::Unknown
        }
    }
}
