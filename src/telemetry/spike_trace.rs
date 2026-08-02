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

#[derive(Clone, Copy, Default)]
pub struct BurstEvent {
    pub neuron_id: u16,
    pub start_time: u32,
    pub spike_count: u8,
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

pub struct TraceEventIter<'a> {
    logger: &'a SpikeTraceLogger,
    offset: u16,
    remaining: u16,
}

impl Iterator for TraceEventIter<'_> {
    type Item = TraceEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let idx = (self.logger.tail as usize + self.offset as usize) % crate::SPIKE_TRACE_BUFFER;
        self.offset += 1;
        self.remaining -= 1;
        Some(unsafe { self.logger.buffer[idx].assume_init() })
    }
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
        let count = self.len();
        if count == 0 {
            return &[];
        }
        let contiguous_count = if self.head > self.tail {
            count
        } else {
            (crate::SPIKE_TRACE_BUFFER as u16) - self.tail
        };
        unsafe {
            let start = self.tail as usize;
            core::slice::from_raw_parts(
                &self.buffer[start] as *const MaybeUninit<TraceEvent> as *const TraceEvent,
                contiguous_count as usize,
            )
        }
    }

    pub fn iter(&self) -> TraceEventIter<'_> {
        TraceEventIter {
            logger: self,
            offset: 0,
            remaining: self.len(),
        }
    }

    pub fn copy_trace_into(&self, out: &mut [TraceEvent]) -> usize {
        let mut copied = 0;
        for ev in self.iter().take(out.len()) {
            out[copied] = ev;
            copied += 1;
        }
        copied
    }

    pub fn export_uart_text(&self) -> SpikeTraceTextExport {
        let mut export = SpikeTraceTextExport {
            text: [0; 2048],
            len: 0,
        };
        let mut writer = FixedTextBuffer::new(&mut export.text);
        let count = self.len();
        writer.write_bytes(b"HKL1-SPIKETRACE\n");
        let _ = writeln!(writer, "events={}", count);
        for ev in self.iter().take(64) {
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

    pub fn detect_bursts(&self) -> ([BurstEvent; 64], usize) {
        let mut bursts = [BurstEvent::default(); 64];
        let mut burst_count = 0;

        #[derive(Clone, Copy)]
        struct ActiveBurst {
            neuron_id: u16,
            start_time: u32,
            count: u8,
            completed: bool,
        }
        let mut active = [ActiveBurst {
            neuron_id: 0,
            start_time: 0,
            count: 0,
            completed: false,
        }; 64];
        let mut active_count = 0;

        for ev in self.iter() {
            let nid = ev.neuron_id.index();
            let t = ev.timestamp;

            let mut found = false;
            for i in 0..active_count {
                if active[i].neuron_id == nid as u16 && !active[i].completed {
                    if t.saturating_sub(active[i].start_time) < 5 {
                        active[i].count += 1;
                        found = true;
                    } else {
                        if active[i].count >= 3 {
                            active[i].completed = true;
                        } else {
                            active[i].start_time = t;
                            active[i].count = 1;
                            found = true;
                        }
                    }
                    break;
                }
            }
            if !found && active_count < 64 {
                active[active_count] = ActiveBurst {
                    neuron_id: nid as u16,
                    start_time: t,
                    count: 1,
                    completed: false,
                };
                active_count += 1;
            }
        }

        for i in 0..active_count {
            if active[i].count >= 3 && burst_count < 64 {
                bursts[burst_count] = BurstEvent {
                    neuron_id: active[i].neuron_id,
                    start_time: active[i].start_time,
                    spike_count: active[i].count,
                };
                burst_count += 1;
            }
        }
        (bursts, burst_count)
    }

    pub fn compute_firing_rates(&self) -> [u16; 256] {
        let mut rates = [0u16; 256];
        if self.is_empty() {
            return rates;
        }

        let mut iter = self.iter();
        let Some(first) = iter.next() else {
            return rates;
        };
        let start_t = first.timestamp;
        let mut end_t = start_t;
        let idx = first.neuron_id.index();
        if idx < 256 {
            rates[idx] = rates[idx].saturating_add(1);
        }

        for ev in iter {
            end_t = ev.timestamp;
            let idx = ev.neuron_id.index();
            if idx < 256 {
                rates[idx] = rates[idx].saturating_add(1);
            }
        }
        let window = end_t.saturating_sub(start_t).max(1);

        for i in 0..256 {
            let r = (rates[i] as u32 * 1000) / window;
            rates[i] = r.min(u16::MAX as u32) as u16;
        }
        rates
    }

    pub fn export_trace_filtered(&self, layer_filter: u8) -> usize {
        let mut count = 0;
        for ev in self.iter() {
            if ev.layer == layer_filter {
                count += 1;
            }
        }
        count
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
        let total = crate::SPIKE_TRACE_BUFFER;
        for i in 0..total + 10 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }
        assert_eq!(log.len() as usize, total - 1);
        let first = log.iter().next().unwrap();
        let last = log.iter().last().unwrap();
        assert_eq!(first.timestamp, 11);
        assert_eq!(last.timestamp, total as u32 + 9);
        assert_eq!(log.iter().count(), log.len() as usize);
    }

    #[test]
    fn test_circular_does_not_lose_all_data() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let total = crate::SPIKE_TRACE_BUFFER;
        for i in 0..total * 3 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }
        assert!(log.has_data());
        assert!(log.len() as usize <= total);
    }

    #[test]
    fn test_wrapped_uart_export_uses_logical_count() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let total = crate::SPIKE_TRACE_BUFFER;
        for i in 0..total + 10 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }

        let export = log.export_uart_text();
        let text = export.as_str();
        let mut expected = [0u8; 32];
        let mut writer = FixedTextBuffer::new(&mut expected);
        let _ = write!(writer, "events={}", total - 1);
        let expected_len = writer.len();
        let expected_text = core::str::from_utf8(&expected[..expected_len]).unwrap_or("");
        assert!(text.contains(expected_text));
        assert!(text.contains("t=11"));
    }

    #[test]
    fn test_wrapped_trace_copy_preserves_chronological_order() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let total = crate::SPIKE_TRACE_BUFFER;
        for i in 0..total + 10 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }

        let contiguous = log.export_trace();
        assert_eq!(
            contiguous.len(),
            crate::SPIKE_TRACE_BUFFER - log.tail as usize
        );

        let mut out = [TraceEvent::default(); 32];
        let copied = log.copy_trace_into(&mut out);
        assert_eq!(copied, out.len());
        assert_eq!(out[0].timestamp, log.tail as u32);
        assert_eq!(out[1].timestamp, log.tail as u32 + 1);
        assert_eq!(out[31].timestamp, log.tail as u32 + 31);
    }

    #[test]
    fn test_wrapped_uart_export_reports_full_active_count() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let total = crate::SPIKE_TRACE_BUFFER;
        for i in 0..total + 10 {
            log.log_spike(test_neuron_id(0), i as u32, 0, false);
        }

        let export = log.export_uart_text();
        let text = export.as_str();
        assert!(text.contains("HKL1-SPIKETRACE"));
        assert_eq!(log.len() as usize, total - 1);
        assert!(text.contains("events="));
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

    #[test]
    fn test_burst_detection() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let nid = test_neuron_id(1);
        log.log_spike(nid, 10, 0, false);
        log.log_spike(nid, 11, 0, false);
        log.log_spike(nid, 12, 0, false);
        log.log_spike(nid, 13, 0, false);
        log.log_spike(nid, 14, 0, false);

        let (bursts, count) = log.detect_bursts();
        assert_eq!(count, 1);
        assert_eq!(bursts[0].neuron_id, 1);
        assert_eq!(bursts[0].spike_count, 5);
        assert_eq!(bursts[0].start_time, 10);
    }

    #[test]
    fn test_firing_rate_computation() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        let nid = test_neuron_id(2);
        for i in 0..10 {
            log.log_spike(nid, i * 10, 0, false);
        }
        let rates = log.compute_firing_rates();
        assert!(rates[2] > 100 && rates[2] < 120);
    }

    #[test]
    fn test_filtered_export() {
        let mut log = SpikeTraceLogger::new();
        log.start_recording();
        log.log_spike(test_neuron_id(0), 10, 1, false);
        log.log_spike(test_neuron_id(1), 11, 2, false);
        log.log_spike(test_neuron_id(2), 12, 1, false);

        let count_l1 = log.export_trace_filtered(1);
        let count_l2 = log.export_trace_filtered(2);
        assert_eq!(count_l1, 2);
        assert_eq!(count_l2, 1);
    }
}
