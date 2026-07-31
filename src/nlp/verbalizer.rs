//! Neuromodulated Internal State Verbalizer for HKL-1.
//! Translates internal neuromodulator levels (DA, 5-HT, NA, ACh), prediction errors,
//! curiosity, and boredom into natural language explanations of cognitive state.

use crate::core::math::FixedPoint;

pub const MAX_VERBAL_LEN: usize = 256;

/// Cognitive State Summary
#[derive(Clone, Copy)]
pub struct CognitiveStateSummary {
    pub dopamine: FixedPoint,
    pub serotonin: FixedPoint,
    pub noradrenaline: FixedPoint,
    pub acetylcholine: FixedPoint,
    pub prediction_error: FixedPoint,
    pub curiosity: FixedPoint,
    pub boredom: FixedPoint,
}

/// Neuromodulated Internal Verbalizer
pub struct NeuromodulatedVerbalizer;

impl NeuromodulatedVerbalizer {
    /// Generate a natural language verbalization string describing internal state
    pub fn verbalize_state(state: &CognitiveStateSummary) -> ([u8; MAX_VERBAL_LEN], usize) {
        let mut buf = [0u8; MAX_VERBAL_LEN];
        let mut idx = 0;

        let write_str = |b: &mut [u8], pos: &mut usize, s: &[u8]| {
            let len = s.len().min(MAX_VERBAL_LEN - *pos);
            b[*pos..*pos + len].copy_from_slice(&s[..len]);
            *pos += len;
        };

        let write_fp = |b: &mut [u8], pos: &mut usize, fp: FixedPoint| {
            let int_part = fp.to_int().abs();
            let frac_part = ((fp.abs().to_f32() - int_part as f32) * 100.0) as u32;
            let tens = (int_part / 10 % 10) as u8 + b'0';
            let ones = (int_part % 10) as u8 + b'0';
            let f1 = (frac_part / 10 % 10) as u8 + b'0';
            let f2 = (frac_part % 10) as u8 + b'0';

            if int_part >= 10 {
                write_str(b, pos, &[tens, ones, b'.', f1, f2]);
            } else {
                write_str(b, pos, &[ones, b'.', f1, f2]);
            }
        };

        // Header
        write_str(&mut buf, &mut idx, b"State: ");

        // Determine cognitive mode verbalization
        if state.noradrenaline > FixedPoint::from_f32(0.8) {
            write_str(&mut buf, &mut idx, b"[ALERT/CRISIS] High arousal! ");
        } else if state.curiosity > FixedPoint::from_f32(0.7) || state.boredom > FixedPoint::from_f32(0.7) {
            write_str(&mut buf, &mut idx, b"[EXPLORATION] Seeking novelty. ");
        } else {
            write_str(&mut buf, &mut idx, b"[STABLE] Focused operation. ");
        }

        // Neuromodulator levels
        write_str(&mut buf, &mut idx, b"(DA:");
        write_fp(&mut buf, &mut idx, state.dopamine);
        write_str(&mut buf, &mut idx, b" 5HT:");
        write_fp(&mut buf, &mut idx, state.serotonin);
        write_str(&mut buf, &mut idx, b" NA:");
        write_fp(&mut buf, &mut idx, state.noradrenaline);
        write_str(&mut buf, &mut idx, b" ACh:");
        write_fp(&mut buf, &mut idx, state.acetylcholine);
        write_str(&mut buf, &mut idx, b") Error:");
        write_fp(&mut buf, &mut idx, state.prediction_error);

        (buf, idx)
    }
}
