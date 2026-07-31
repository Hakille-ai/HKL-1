//! Sensor encoding module. Converts raw sensor data into spike patterns.
use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::io::buffers::{EncodedSpike, Modality, ingest_spike};

/// Temporal encoding - converts raw sensor data to spike trains (Section 4)
/// All modalities unified into temporal spike streams

pub trait TemporalEncoder {
    fn encode(&mut self, data: &[u8], timestamp: u32);
}

/// Rate-based encoder: converts analog value to spike frequency
pub struct RateEncoder {
    pub neuron_id: NeuronId,
    pub spike_interval_us: u32,
    pub last_spike_time: u32,
    pub threshold: FixedPoint,
    pub modality: Modality,
}

impl RateEncoder {
    pub fn new(neuron_id: NeuronId, modality: Modality) -> Self {
        Self {
            neuron_id,
            spike_interval_us: 1000, // Max 1kHz
            last_spike_time: 0,
            threshold: FixedPoint::from_f32(0.5),
            modality,
        }
    }

    /// Encode analog value (0.0-1.0) into spike train
    pub fn encode_analog(&mut self, value: FixedPoint, timestamp: u32) -> bool {
        let _intensity = value * FixedPoint::from_int(1000);
        let interval = (FixedPoint::ONE / value.max(FixedPoint::from_f32(0.01))).to_int() as u32;

        if timestamp - self.last_spike_time >= interval {
            self.last_spike_time = timestamp;
            let spike = EncodedSpike {
                neuron_id: self.neuron_id,
                intensity: value,
                timestamp,
                modality: self.modality,
            };
            ingest_spike(spike);
            true
        } else {
            false
        }
    }
}

/// Temporal encoder for character-by-character text input (Section 4)
pub struct TextEncoder {
    pub base_neuron_id: NeuronId,
    pub char_spacing: u32, // Min interval between characters
    pub last_char_time: u32,
    pub rhythm_tracker: [u32; 16], // Track inter-character intervals
    pub rhythm_idx: u8,
}

impl TextEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            base_neuron_id,
            char_spacing: 100, // 100ms between chars
            last_char_time: 0,
            rhythm_tracker: [0; 16],
            rhythm_idx: 0,
        }
    }

    /// Encode a character as a temporal spike pattern
    pub fn encode_char(&mut self, c: u8, timestamp: u32) -> bool {
        if timestamp - self.last_char_time < self.char_spacing {
            return false; // Too soon
        }
        self.last_char_time = timestamp;

        // Each character activates a specific neuron based on ASCII value
        // Lower 8 bits of ASCII select base neuron, upper 3 bits modulate delay
        let neuron_offset = (c & 0x7F) as u16;
        let _delay_mod = (c >> 5) as u16;
        let neuron_id = NeuronId::new(self.base_neuron_id.index() as u16 + neuron_offset);

        let spike = EncodedSpike {
            neuron_id,
            intensity: FixedPoint::ONE,
            timestamp,
            modality: Modality::Text,
        };
        ingest_spike(spike);

        // Track rhythm
        if self.rhythm_idx > 0 {
            let interval = timestamp - self.rhythm_tracker[(self.rhythm_idx - 1) as usize % 16];
            self.rhythm_tracker[self.rhythm_idx as usize % 16] = interval;
        } else {
            self.rhythm_tracker[0] = timestamp;
        }
        self.rhythm_idx = self.rhythm_idx.wrapping_add(1);

        true
    }
}

/// Temporal encoder for audio - spectrogram-based spike encoding (Section 4)
pub struct AudioEncoder {
    pub base_neuron_id: NeuronId,
    pub freq_bands: [FixedPoint; 32], // 32 frequency band intensities
    pub thresholds: [FixedPoint; 32], // Per-band thresholds
    pub min_interval_us: u32,
}

impl AudioEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            base_neuron_id,
            freq_bands: [FixedPoint::ZERO; 32],
            thresholds: [FixedPoint::from_f32(0.3); 32],
            min_interval_us: 100,
        }
    }

    /// Process raw PCM audio samples - run Gammatone cochlea, Formants, F0 pitch & voice synthesis
    pub fn process_pcm_stream(&mut self, pcm_samples: &[i16], timestamp: u32) {
        let engine = crate::audio::audio_engine();
        let (_bands, _formants, _pitch) = engine.process_audio_stream(pcm_samples, timestamp);
    }
}


/// Temporal encoder for event-based vision (Section 4)
pub struct VisionEncoder {
    pub base_neuron_id: NeuronId,
    pub prev_frame: [u8; 1024], // Previous frame for delta detection
    pub threshold: u8,          // Minimum brightness change
    pub event_count: u16,
}

impl VisionEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            base_neuron_id,
            prev_frame: [0; 1024],
            threshold: 10,
            event_count: 0,
        }
    }

    /// Process pixel array - run Retinal DoG, DVS event generation, Gabor V1, Motion MT, 3D Physics & Visual Predictive Coding
    pub fn process_pixels(&mut self, pixels: &[u8; 1024], timestamp: u32) {
        let engine = crate::vision::visual_engine();
        let (_ganglion, _v1, _motion, _pred_err) = engine.process_visual_scene(pixels, timestamp, 10);
        self.event_count = engine.retina.event_count as u16;
        self.prev_frame = *pixels;
    }
}


/// Industrial sensor encoder - converts analog readings to PFM (Section 4)
pub struct SensorEncoder {
    pub base_neuron_id: NeuronId,
    pub last_values: [FixedPoint; 16],
    pub thresholds: [FixedPoint; 16],
    pub min_delta: FixedPoint,
}

impl SensorEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            base_neuron_id,
            last_values: [FixedPoint::ZERO; 16],
            thresholds: [FixedPoint::from_f32(0.5); 16],
            min_delta: FixedPoint::from_f32(0.01),
        }
    }

    /// Encode sensor reading as pulse-frequency modulation
    pub fn encode_sensor(&mut self, sensor_id: u8, value: FixedPoint, timestamp: u32) -> bool {
        let idx = sensor_id as usize % 16;
        let delta = (value - self.last_values[idx]).abs();

        if delta < self.min_delta {
            return false; // No significant change
        }

        // Convert value to spike frequency
        if value > self.thresholds[idx] {
            let neuron_id = NeuronId::new(self.base_neuron_id.index() as u16 + sensor_id as u16);
            let spike = EncodedSpike {
                neuron_id,
                intensity: value,
                timestamp,
                modality: Modality::Sensor,
            };
            ingest_spike(spike);
            self.last_values[idx] = value;
            true
        } else {
            false
        }
    }
}

/// Proprioception encoder - copies motor commands as predicted sensory feedback (Section 22)
pub struct ProprioceptionEncoder {
    pub base_neuron_id: NeuronId,
}

impl ProprioceptionEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self { base_neuron_id }
    }

    /// Encode efference copy (predicted motor outcome)
    pub fn encode_efference(
        &self,
        motor_command: u8,
        predicted_outcome: FixedPoint,
        timestamp: u32,
    ) {
        let neuron_id = NeuronId::new(self.base_neuron_id.index() as u16 + motor_command as u16);
        let spike = EncodedSpike {
            neuron_id,
            intensity: predicted_outcome,
            timestamp,
            modality: Modality::Proprioception,
        };
        ingest_spike(spike);
    }
}

/// Internal state encoder for curiosity engine (Section 29)
pub struct InternalEncoder {
    pub curiosity_neuron_base: NeuronId,
}

impl InternalEncoder {
    pub fn new(base: NeuronId) -> Self {
        Self {
            curiosity_neuron_base: base,
        }
    }

    pub fn encode_curiosity(&self, curiosity_level: FixedPoint, timestamp: u32) {
        let neuron_id = NeuronId::new(self.curiosity_neuron_base.index() as u16);
        let spike = EncodedSpike {
            neuron_id,
            intensity: curiosity_level,
            timestamp,
            modality: Modality::Internal,
        };
        ingest_spike(spike);
    }
}

/// Modality-specific encoder dispatcher
pub struct ModalityEncoder {
    pub text: TextEncoder,
    pub audio: Option<AudioEncoder>,
    pub vision: Option<VisionEncoder>,
    pub sensor: SensorEncoder,
    pub proprio: ProprioceptionEncoder,
    pub internal: InternalEncoder,
}

impl ModalityEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            text: TextEncoder::new(base_neuron_id),
            audio: None,
            vision: None,
            sensor: SensorEncoder::new(NeuronId::new(base_neuron_id.index() as u16 + 512)),
            proprio: ProprioceptionEncoder::new(NeuronId::new(base_neuron_id.index() as u16 + 768)),
            internal: InternalEncoder::new(NeuronId::new(base_neuron_id.index() as u16 + 896)),
        }
    }
}
