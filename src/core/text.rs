use core::fmt;

pub struct FixedTextBuffer<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> FixedTextBuffer<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        let available = self.buf.len().saturating_sub(self.pos);
        let count = bytes.len().min(available);
        self.buf[self.pos..self.pos + count].copy_from_slice(&bytes[..count]);
        self.pos += count;
    }
}

impl fmt::Write for FixedTextBuffer<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_bytes(s.as_bytes());
        Ok(())
    }
}
