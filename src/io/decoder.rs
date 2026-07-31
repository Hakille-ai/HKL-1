//! Actuator decoding module. Converts spike patterns into actuator commands.
use crate::core::math::FixedPoint;

/// Motor decoding - converts spike output to motor commands (Section 28)

pub struct MotorDecoder {
    pub text_output: TextOutput,
    pub voice_output: VoiceOutput,
}

pub struct TextOutput {
    pub char_buffer: [u8; 64],
    pub head: u8,
    pub tail: u8,
}

impl TextOutput {
    pub fn new() -> Self {
        Self {
            char_buffer: [0; 64],
            head: 0,
            tail: 0,
        }
    }

    /// Decode motor neuron spikes into ASCII characters (Section 28.1)
    pub fn decode_char_spikes(&mut self, motor_neurons: &[FixedPoint; 256]) -> Option<u8> {
        // Find most active motor neuron in character region
        let mut max_val = FixedPoint::ZERO;
        let mut max_idx = 0;

        for i in 0..128 {
            let neuron_val = motor_neurons[i];
            if neuron_val > max_val {
                max_val = neuron_val;
                max_idx = i;
            }
        }

        if max_val > FixedPoint::from_f32(0.5) {
            let c = max_idx as u8;
            self.char_buffer[self.head as usize] = c;
            self.head = (self.head + 1) % 64;
            Some(c)
        } else {
            None
        }
    }

    pub fn read_char(&mut self) -> Option<u8> {
        if self.head == self.tail {
            return None;
        }
        let c = self.char_buffer[self.tail as usize];
        self.tail = (self.tail + 1) % 64;
        Some(c)
    }
}

pub struct VoiceOutput {
    pub frequency: FixedPoint,
    pub amplitude: FixedPoint,
    pub duration_ms: u32,
}

impl VoiceOutput {
    pub fn new() -> Self {
        Self {
            frequency: FixedPoint::from_f32(440.0),
            amplitude: FixedPoint::ZERO,
            duration_ms: 100,
        }
    }

    /// Decode frequency modulation from motor neurons (Section 28.2)
    pub fn decode_frequency(&mut self, motor_neurons: &[FixedPoint; 256]) {
        let freq_idx = 128;
        let amp_idx = 192;
        let dur_idx = 224;

        self.frequency = FixedPoint::from_f32(100.0 + motor_neurons[freq_idx].to_f32() * 3000.0);
        self.amplitude = motor_neurons[amp_idx];
        self.duration_ms = (10.0 + motor_neurons[dur_idx].to_f32() * 1000.0) as u32;
    }
}

impl MotorDecoder {
    pub fn new() -> Self {
        Self {
            text_output: TextOutput::new(),
            voice_output: VoiceOutput::new(),
        }
    }

    pub fn decode(&mut self, motor_neurons: &[FixedPoint; 256]) {
        self.text_output.decode_char_spikes(motor_neurons);
        self.voice_output.decode_frequency(motor_neurons);
    }
}
