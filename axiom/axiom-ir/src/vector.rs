//! Vector/SIMD type support for auto-vectorization.

use crate::nodes::Type;

/// A vector type description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorType {
    pub element: Type,
    pub lane_count: u32,
}

impl VectorType {
    pub fn new(element: Type, lane_count: u32) -> Self {
        Self { element, lane_count }
    }

    pub fn byte_size(&self) -> u32 {
        self.element.byte_size() * self.lane_count
    }

    pub fn is_power_of_two(&self) -> bool {
        self.lane_count > 0 && (self.lane_count & (self.lane_count - 1)) == 0
    }

    /// Common vector types.
    pub fn i32x4() -> Self { Self::new(Type::I32, 4) }
    pub fn i64x2() -> Self { Self::new(Type::I64, 2) }
    pub fn f32x4() -> Self { Self::new(Type::F32, 4) }
    pub fn f64x2() -> Self { Self::new(Type::F64, 2) }
}
