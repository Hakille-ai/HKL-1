use crate::core::memory::NeuronId;
use crate::core::text::FixedTextBuffer;
use core::fmt::Write;
use core::mem::MaybeUninit;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TraceEvent {
    pub neuron_id: NeuronId,
    pub timestamp: u32,
    pub layer: u8,
    pub is_predictor: bool,
    pub membrane_potential: i16,
}

pub struct SpikeTraceLogger {
    pub buffer: [MaybeUninit<TraceEvent>; crate::SPIKE_TRACE_BUFFER],
    pub head: u16,
    pub tail: u16,
    pub recording: bool,
}

pub struct SpikeTraceTextExport {
    pub text: [u8; 2048],
    pub len: u16,
}

impl SpikeTraceTextExport {
    pub fn as_str(&self) -> &str {
        let end = self.len as usize;
        core::str::from_utf8(&self.text[..end]).unwrap_or("")
    }
}

impl SpikeTraceLogger {
    pub fn new() -> Self {
        Self {
            buffer: unsafe { MaybeUninit::uninit().assume_init() },
            head: 0,
            tail: 0,
            recording: false,
        }
    }

    pub fn start_recording(&mut self) {
        self.recording = true;
        self.head = 0;
        self.tail = 0;
    }

    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    pub fn log_spike(&mut self, neuron_id: NeuronId, time: u32, layer: u8, is_predictor: bool) {
        if !self.recording {
            return;
        }
        let idx = self.head as usize;
        if idx < crate::SPIKE_TRACE_BUFFER {
            let state = crate::core::memory::neuron_state_ref(neuron_id);
            self.buffer[idx] = MaybeUninit::new(TraceEvent {
                neuron_id,
                timestamp: time,
                layer,
                is_predictor,
                membrane_potential: (state.membrane_potential.to_f32() * 256.0) as i16,
            });
            self.head = (self.head + 1) % crate::SPIKE_TRACE_BUFFER as u16;
            if self.head == self.tail {
                self.tail = (self.tail + 1) % crate::SPIKE_TRACE_BUFFER as u16;
            }
        }
    }

    pub fn export_trace(&self) -> &[TraceEvent] {
        let count = if self.head >= self.tail {
            self.head - self.tail
        } else {
            (crate::SPIKE_TRACE_BUFFER as u16) - self.tail + self.head
        };
        if count == 0 {
            return &[];
        }
        unsafe {
            let start = self.tail as usize;
            core::slice::from_raw_parts(
                &self.buffer[start] as *const MaybeUninit<TraceEvent> as *const TraceEvent,
                count as usize,
            )
        }
    }

    pub fn export_uart_text(&self) -> SpikeTraceTextExport {
        let mut export = SpikeTraceTextExport {
            text: [0; 2048],
            len: 0,
        };
        let mut writer = FixedTextBuffer::new(&mut export.text);
        let trace = self.export_trace();
        let count = trace.len();
        writer.write_bytes(b"HKL1-SPIKETRACE\n");
        let _ = writeln!(writer, "events={}", count);
        for ev in trace.iter().take(64) {
            let _ = writeln!(
                writer,
                "n={} t={} l={} p={} mp={}",
                ev.neuron_id.index(),
                ev.timestamp,
                ev.layer,
                ev.is_predictor as u8,
                ev.membrane_potential,
            );
        }
        export.len = writer.len() as u16;
        export
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
    }

    pub fn has_data(&self) -> bool {
        self.head != self.tail
    }

    pub fn len(&self) -> u16 {
        if self.head >= self.tail {
            self.head - self.tail
        } else {
            (crate::SPIKE_TRACE_BUFFER as u16) - self.tail + self.head
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.has_data()
    }
}

static mut SPIKE_LOGGER_STORAGE: MaybeUninit<SpikeTraceLogger> = MaybeUninit::uninit();

pub fn init_logger() {
    unsafe {
        SPIKE_LOGGER_STORAGE.write(SpikeTraceLogger::new());
    }
}

pub fn logger() -> &'static mut SpikeTraceLogger {
    unsafe { &mut *SPIKE_LOGGER_STORAGE.as_mut_ptr() }
}

pub fn start_recording() {
    logger().start_recording();
}

pub fn stop_recording() {
    logger().stop_recording();
}

pub fn record_spike(neuron_id: NeuronId, time: u32, layer: u8, is_predictor: bool) {
    logger().log_spike(neuron_id, time, layer, is_predictor);
}

pub fn export_trace() -> &'static [TraceEvent] {
    let log = logger();
    if log.has_data() {
        return log.export_trace();
    }
    &[]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_neuron_id(i: u16) -> NeuronId {
        unsafe {
            let count = &crate::core::memory::NEURON_COUNT;
            count.store(1, core::sync::atomic::Ordering::Relaxed);
            let array = &mut crate::core::memory::NEURON_ARRAY;
            array[i as usize] = MaybeUninit::new(crate::core::memory::NeuronState {
                membrane_potential: crate::core::math::FixedPoint::from_f32(0.5),
                threshold: crate::core::math::FixedPoint::ZERO,
                leak: crate::core::math::FixedPoint::ZERO,
                refractory_remaining: 0,
                last_spike_time: 0,
                bias_current: crate::core::math::FixedPoint::ZERO,
                layer: 0,
                neuron_type: crate::core::memory::NeuronType::LIF,
                flags: crate::core::memory::NeuronFlags(0),
            });
            NeuronId::new(i)
        }
    }

    #[test]
    fn test_new_not_recording() {
        let log = SpikeTraceLogger::new();
        assert!(!log.recording);
        assert!(!log.has_data());
    }

    #[test]
    fn test_start_recording_resets() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        assert!(log.recording);
        assert!(!log.has_data());
    }

    #[test]
    fn test_log_spike_when_not_recording() {
        let mut log = SpikeTraceLogger::new();
        log.log_spike(test_neuron_id(0), 100, 0, false);
        assert!(!log.has_data());
    }

    #[test]
    fn test_log_spike_when_recording() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 100, 0, false);
        assert!(log.has_data());
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_export_trace_returns_recorded_data() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 100, 2, false);
        let trace = log.export_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].timestamp, 100);
        assert_eq!(trace[0].layer, 2);
    }

    #[test]
    fn test_export_empty_when_no_data() {
        let log = SpikeTraceLogger::new();
        let trace = log.export_trace();
        assert!(trace.is_empty());
    }

    #[test]
    fn test_clear_resets() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 100, 0, false);
        assert!(log.has_data());
        log.clear();
        assert!(!log.has_data());
    }

    #[test]
    fn test_stop_recording() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 100, 0, false);
        log.stop_recording();
        assert!(!log.recording);
        assert!(log.has_data());
    }

    #[test]
    fn test_circular_buffer_wrap() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let total = crate::SPIKE_TRACE_BUFFER as u16;
        for i in 0..total + 10 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }
        assert!(log.len() >= total - 1);
        assert!(log.len() <= total);
        let trace = log.export_trace();
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_circular_does_not_lose_all_data() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let total = crate::SPIKE_TRACE_BUFFER as u16;
        for i in 0..total * 3 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }
        assert!(log.has_data());
        assert!(log.len() <= total);
    }

    #[test]
    fn test_export_uart_text_contains_header() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 100, 2, false);
        let export = log.export_uart_text();
        let text = export.as_str();
        assert!(text.contains("HKL1-SPIKETRACE"));
        assert!(text.contains("n="));
    }

    #[test]
    fn test_public_export_trace_returns_data() {
        init_logger();
        let log = logger();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 200, 1, true);
        let trace = export_trace();
        assert!(!trace.is_empty());
        assert_eq!(trace[0].timestamp, 200);
    }

    #[test]
    fn test_membrane_potential_quantized() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 100, 0, false);
        let trace = log.export_trace();
        let expected = (0.5 * 256.0) as i16;
        assert_eq!(trace[0].membrane_potential, expected);
    }
}
