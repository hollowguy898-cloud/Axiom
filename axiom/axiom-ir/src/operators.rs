//! Operator definitions for pattern matching and legalization.

/// An operator that can appear in the IR, used for ISel pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,

    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
    Not,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Floating-Point Arithmetic
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    FNeg,
    FAbs,
    FSqrt,

    // Floating-Point Comparison
    FEq,
    FLt,
    FLe,
    FGt,
    FGe,
    FNe,

    // Floating-Point Conversion
    FpToSInt,
    SIntToFp,
    FpToUInt,
    UIntToFp,

    // Floating-Point Misc
    Copysign,
    Fmin,
    Fmax,

    // Memory
    Load,
    Store,
    StackAlloc,
    Fence,

    // Control
    Branch,
    Jump,
    Return,
    Call,
    Phi,

    // Conversion
    ZExt,
    SExt,
    Trunc,
    BitCast,

    // Constants
    IntConst,
    FpConst,

    // Misc
    Extract,
    Insert,

    // Vector Operations
    VecBroadcast,
    VecLoad,
    VecStore,
    VecBinOp,
    VecUnOp,
    VecReduce,
    ExtractLane,
    InsertLane,
    VecShuffle,
    VecGather,
    VecScatter,
}

impl Op {
    /// Returns true if this is a commutative operator.
    pub fn is_commutative(&self) -> bool {
        matches!(self,
            Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor | Op::Eq | Op::Ne |
            Op::FAdd | Op::FMul | Op::FEq | Op::FNe
        )
    }

    /// Returns true if this is a comparison operator.
    pub fn is_comparison(&self) -> bool {
        matches!(self,
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge |
            Op::FEq | Op::FNe | Op::FLt | Op::FLe | Op::FGt | Op::FGe
        )
    }

    /// Returns true if this is a vector operator.
    pub fn is_vector(&self) -> bool {
        matches!(self,
            Op::VecBroadcast | Op::VecLoad | Op::VecStore |
            Op::VecBinOp | Op::VecUnOp | Op::VecReduce |
            Op::ExtractLane | Op::InsertLane | Op::VecShuffle |
            Op::VecGather | Op::VecScatter
        )
    }

    /// Returns the number of data inputs (excluding control).
    pub fn num_inputs(&self) -> usize {
        match self {
            Op::IntConst | Op::FpConst => 0,
            Op::Neg | Op::Not |
            Op::FNeg | Op::FAbs | Op::FSqrt |
            Op::ZExt | Op::SExt | Op::Trunc | Op::BitCast |
            Op::FpToSInt | Op::SIntToFp | Op::FpToUInt | Op::UIntToFp |
            Op::VecBroadcast | Op::VecUnOp | Op::VecReduce |
            Op::ExtractLane | Op::VecShuffle => 1,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem |
            Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr | Op::Sar |
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge |
            // FP binary
            Op::FAdd | Op::FSub | Op::FMul | Op::FDiv | Op::FRem |
            Op::FEq | Op::FLt | Op::FLe | Op::FGt | Op::FGe | Op::FNe |
            Op::Copysign | Op::Fmin | Op::Fmax |
            Op::VecBinOp | Op::InsertLane => 2,
            Op::Load | Op::VecLoad | Op::VecGather => 1, // addr
            Op::Store | Op::VecStore | Op::VecScatter => 2, // addr, val
            Op::Phi => 0,  // variable
            _ => 0,
        }
    }
}
