//! Source span tracking for diagnostics.

/// A byte-offset span into a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
    pub file_id: u32,
}

impl Span {
    pub const fn new(lo: u32, hi: u32, file_id: u32) -> Self {
        Self { lo, hi, file_id }
    }

    pub const fn synthetic() -> Self {
        Self { lo: 0, hi: 0, file_id: u32::MAX }
    }

    pub fn is_synthetic(&self) -> bool {
        self.file_id == u32::MAX
    }
}

/// A source location for error reporting.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub col: u32,
}
