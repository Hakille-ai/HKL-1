//! Lock-free ring buffers and global spike queue for inter-module communication.
use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::snn::neuron::SpikeEvent;
use core::sync::atomic::{AtomicU32, Ordering};

/// Lock-free ring buffer for sensor/spike events (Section 4, Section 23)
use core::mem::MaybeUninit;

pub struct RingBuffer<T: Copy, const N: usize> {
    buffer: [MaybeUninit<T>; N],
    head: AtomicU32,
    tail: AtomicU32,
    mask: u32,
}

impl<T: Copy, const N: usize> RingBuffer<T, N> {
    pub const fn new() -> Self {
        Self {
            buffer: [const { MaybeUninit::uninit() }; N],
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            mask: (N - 1) as u32,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, item: T) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let next = (head + 1) & self.mask;
        if next == self.tail.load(Ordering::Acquire) {
            return false;
        }
        self.buffer[head as usize] = MaybeUninit::new(item);
        self.head.store(next, Ordering::Release);
        true
    }

    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let item = unsafe { self.buffer[tail as usize].assume_init() };
        self.tail.store((tail + 1) & self.mask, Ordering::Release);
        Some(item)
    }

    #[inline(always)]
    pub fn peek(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        Some(unsafe { self.buffer[tail as usize].assume_init() })
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        ((head + 1) & self.mask) == tail
    }

    #[inline(always)]
    pub fn len(&self) -> u32 {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        (head.wrapping_sub(tail)) & self.mask
    }

    #[inline(always)]
    pub fn clear(&self) {
        self.tail
            .store(self.head.load(Ordering::Acquire), Ordering::Release);
    }
}

/// Specialized lock-free single-producer single-consumer ring buffer for ISR
pub struct LockFreeRingBuffer<const N: usize> {
    buffer: [MaybeUninit<SpikeEvent>; N],
    write_idx: AtomicU32,
    read_idx: AtomicU32,
    committed_idx: AtomicU32,
}

impl<const N: usize> LockFreeRingBuffer<N> {
    pub const fn new() -> Self {
        Self {
            buffer: [const { MaybeUninit::uninit() }; N],
            write_idx: AtomicU32::new(0),
            read_idx: AtomicU32::new(0),
            committed_idx: AtomicU32::new(0),
        }
    }

    #[inline(always)]
    pub fn reserve_write(&self) -> Option<*mut SpikeEvent> {
        let write = self.write_idx.load(Ordering::Relaxed);
        let next = (write + 1) % (N as u32);
        if next == self.read_idx.load(Ordering::Acquire) {
            return None;
        }
        let base = &self.buffer as *const [MaybeUninit<SpikeEvent>; N] as *mut SpikeEvent;
        Some(unsafe { base.add(write as usize) })
    }

    /// ISR: commit write
    #[inline(always)]
    pub fn commit_write(&self) {
        let write = self.write_idx.load(Ordering::Relaxed);
        self.write_idx
            .store((write + 1) % (N as u32), Ordering::Release);
        self.committed_idx.store(write, Ordering::Release);
    }

    #[inline(always)]
    pub fn pop_front(&self) -> Option<SpikeEvent> {
        let read = self.read_idx.load(Ordering::Relaxed);
        if read == self.write_idx.load(Ordering::Acquire) {
            return None;
        }
        let item = unsafe { self.buffer[read as usize].assume_init() };
        self.read_idx
            .store((read + 1) % (N as u32), Ordering::Release);
        Some(item)
    }
}

/// Sensory modality ring buffers (Section 4)
/// Each modality has its own buffer for multimodal fusion

/// Encoded spike from a sensor modality
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct EncodedSpike {
    pub neuron_id: NeuronId,
    pub intensity: FixedPoint,
    pub timestamp: u32,
    pub modality: Modality,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
#[derive(Default)]
pub enum Modality {
    Text = 0,
    Audio = 1,
    Vision = 2,
    Sensor = 3,
    Proprioception = 4,
    Internal = 5,
    #[default]
    None = 255,
}

// Ring buffer instances for each modality
pub static mut TEXT_RING: RingBuffer<EncodedSpike, 512> = RingBuffer::new();
pub static mut AUDIO_RING: RingBuffer<EncodedSpike, 1024> = RingBuffer::new();
pub static mut VISION_RING: RingBuffer<EncodedSpike, 2048> = RingBuffer::new();
pub static mut SENSOR_RING: RingBuffer<EncodedSpike, 4096> = RingBuffer::new();
pub static mut PROPRIO_RING: RingBuffer<EncodedSpike, 256> = RingBuffer::new();
pub static mut EFFERENCE_COPY_RING: RingBuffer<EncodedSpike, 256> = RingBuffer::new();

/// Global spike queue consumed by the main loop
pub static mut GLOBAL_SPIKE_QUEUE: LockFreeRingBuffer<4096> = LockFreeRingBuffer::new();

/// Ingest spike from any modality into the global queue
#[inline(always)]
pub fn ingest_spike(spike: EncodedSpike) -> bool {
    let event = SpikeEvent {
        neuron_id: spike.neuron_id,
        timestamp: spike.timestamp,
        layer: match spike.modality {
            Modality::Text => 0,
            Modality::Audio => 1,
            Modality::Vision => 2,
            Modality::Sensor => 3,
            Modality::Proprioception => 4,
            Modality::Internal => 5,
            Modality::None => 255,
        },
        is_predictor: false,
    };

    // Try to push to global queue
    unsafe {
        if let Some(ptr) = GLOBAL_SPIKE_QUEUE.reserve_write() {
            core::ptr::write(ptr, event);
            GLOBAL_SPIKE_QUEUE.commit_write();
            true
        } else {
            false
        }
    }
}

/// Initialize all ring buffers
pub fn init_buffers() {
    unsafe {
        TEXT_RING.clear();
        AUDIO_RING.clear();
        VISION_RING.clear();
        SENSOR_RING.clear();
        PROPRIO_RING.clear();
        EFFERENCE_COPY_RING.clear();
        // GLOBAL_SPIKE_QUEUE is implicitly empty at start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_pop() {
        let mut buf: RingBuffer<u32, 4> = RingBuffer::new();
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn ring_buffer_full() {
        let mut buf: RingBuffer<u32, 2> = RingBuffer::new();
        assert!(buf.push(1));
        assert!(!buf.push(2));
        assert!(!buf.push(3));
    }

    #[test]
    fn ring_buffer_wraparound() {
        let mut buf: RingBuffer<u32, 4> = RingBuffer::new();
        for i in 0..6 {
            buf.push(i);
        }
        // N=4 ring buffer holds N-1=3 elements max
        // Only 0, 1, 2 were stored; 3, 4, 5 failed
        assert_eq!(buf.pop(), Some(0));
        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn ring_buffer_push_after_pop() {
        let mut buf: RingBuffer<u32, 4> = RingBuffer::new();
        assert!(buf.push(10));
        assert!(buf.push(20));
        assert_eq!(buf.pop(), Some(10));
        assert!(buf.push(30));
        assert_eq!(buf.pop(), Some(20));
        assert_eq!(buf.pop(), Some(30));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn ring_buffer_is_empty() {
        let buf: RingBuffer<u32, 4> = RingBuffer::new();
        assert!(buf.is_empty());
    }

    #[test]
    fn ring_buffer_alternating() {
        let mut buf: RingBuffer<u32, 2> = RingBuffer::new();
        assert!(buf.push(1));
        assert_eq!(buf.pop(), Some(1));
        assert!(buf.push(2));
        assert_eq!(buf.pop(), Some(2));
        assert!(buf.push(3));
        assert_eq!(buf.pop(), Some(3));
    }
}
