//! Axiom MIR — Machine-level Intermediate Representation.
//!
//! This crate defines the Machine IR (MIR) that sits between the target-independent
//! Sea-of-Nodes IR and the final machine code. The MIR uses virtual registers,
//! explicit basic blocks, and a flattened instruction stream — much closer to
//! what a real machine executes.

pub mod lower;

use std::fmt;

// ── Newtypes ────────────────────────────────────────────────────────────

/// Virtual register identifier.
///
/// VRegs are allocated during lowering and later mapped to physical registers
/// by the register allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(pub u32);

impl VReg {
    pub const fn new(id: u32) -> Self {
        VReg(id)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// 64-bit immediate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Imm64(pub i64);

impl Imm64 {
    pub const fn new(val: i64) -> Self {
        Imm64(val)
    }

    pub const fn as_i64(self) -> i64 {
        self.0
    }
}

impl fmt::Display for Imm64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Basic block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    pub const fn new(id: u32) -> Self {
        BlockId(id)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// ── Comparison Conditions ──────────────────────────────────────────────

/// Comparison condition for integer comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpCond {
    Eq,  // equal
    Ne,  // not equal
    Lt,  // less than (signed)
    Le,  // less or equal (signed)
    Gt,  // greater than (signed)
    Ge,  // greater or equal (signed)
}

/// Comparison condition for floating-point comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCmpCond {
    Eq, Ne, Lt, Le, Gt, Ge,
}

// ── MIR Instruction ────────────────────────────────────────────────────

/// A single MIR instruction.
///
/// Operands are virtual registers or immediates. The instruction set is
/// intentionally simple — target-specific legalisation happens later.
#[derive(Debug, Clone, PartialEq)]
pub enum MirInst {
    // ── Data Movement ──────────────────────────────────────
    /// Copy register: dst = src
    Mov { dst: VReg, src: VReg },
    /// Load immediate: dst = imm
    MovImm { dst: VReg, imm: Imm64 },

    // ── Arithmetic ────────────────────────────────────────
    Add { dst: VReg, lhs: VReg, rhs: VReg },
    Sub { dst: VReg, lhs: VReg, rhs: VReg },
    Mul { dst: VReg, lhs: VReg, rhs: VReg },
    Div { dst: VReg, lhs: VReg, rhs: VReg },
    Rem { dst: VReg, lhs: VReg, rhs: VReg },
    Neg { dst: VReg, src: VReg },

    // ── Bitwise ───────────────────────────────────────────
    And { dst: VReg, lhs: VReg, rhs: VReg },
    Or  { dst: VReg, lhs: VReg, rhs: VReg },
    Xor { dst: VReg, lhs: VReg, rhs: VReg },
    Shl { dst: VReg, lhs: VReg, rhs: VReg },
    Shr { dst: VReg, lhs: VReg, rhs: VReg },
    Sar { dst: VReg, lhs: VReg, rhs: VReg },
    /// Shift left by an immediate amount (0–63).
    /// Emitted when the shift amount is a compile-time constant.
    ShlImm { dst: VReg, lhs: VReg, amount: u8 },
    /// Logical shift right by an immediate amount (0–63).
    ShrImm { dst: VReg, lhs: VReg, amount: u8 },
    /// Arithmetic shift right by an immediate amount (0–63).
    SarImm { dst: VReg, lhs: VReg, amount: u8 },
    Not { dst: VReg, src: VReg },

    // ── Comparison ────────────────────────────────────────
    /// Compare and set dst to 0 or 1.
    Cmp { dst: VReg, lhs: VReg, rhs: VReg, cond: CmpCond },

    // ── Floating-Point Arithmetic ────────────────────────
    FAdd { dst: VReg, lhs: VReg, rhs: VReg },
    FSub { dst: VReg, lhs: VReg, rhs: VReg },
    FMul { dst: VReg, lhs: VReg, rhs: VReg },
    FDiv { dst: VReg, lhs: VReg, rhs: VReg },
    FRem { dst: VReg, lhs: VReg, rhs: VReg },
    FNeg { dst: VReg, src: VReg },
    FAbs { dst: VReg, src: VReg },
    FSqrt { dst: VReg, src: VReg },

    // ── Floating-Point Comparison ─────────────────────────
    /// Floating-point compare and set dst to 0 or 1.
    FCmp { dst: VReg, lhs: VReg, rhs: VReg, cond: FCmpCond },

    // ── Floating-Point Conversion ─────────────────────────
    FpToSInt { dst: VReg, src: VReg },
    SIntToFp { dst: VReg, src: VReg },
    FpToUInt { dst: VReg, src: VReg },
    UIntToFp { dst: VReg, src: VReg },

    // ── Floating-Point Misc ──────────────────────────────
    Copysign { dst: VReg, lhs: VReg, rhs: VReg },
    Fmin { dst: VReg, lhs: VReg, rhs: VReg },
    Fmax { dst: VReg, lhs: VReg, rhs: VReg },

    // ── Memory ────────────────────────────────────────────
    Load { dst: VReg, addr: VReg },
    Store { addr: VReg, val: VReg },
    StackAlloc { dst: VReg, size: u32, align: u32 },

    // ── Calls ─────────────────────────────────────────────
    /// Call a function. `dst` is None for void returns.
    Call { dst: Option<VReg>, func: String, args: Vec<VReg> },

    // ── Control Flow ──────────────────────────────────────
    Ret { val: Option<VReg> },
    Jump { target: BlockId },
    Branch { cond: VReg, true_block: BlockId, false_block: BlockId },

    // ── Phi Lowering ──────────────────────────────────────
    /// Parallel copy instruction emitted during phi lowering.
    /// All PhiCopy instructions at the end of a block execute
    /// simultaneously (none observes the effect of another).
    PhiCopy { dst: VReg, src: VReg },

    // ── Extension / Truncation ─────────────────────────────
    /// Zero-extend: dst = zext(src)
    ZExt { dst: VReg, src: VReg },
    /// Sign-extend: dst = sext(src)
    SExt { dst: VReg, src: VReg },
    /// Truncate: dst = trunc(src)
    Trunc { dst: VReg, src: VReg },

    // ── Spill / Reload ─────────────────────────────────────
    /// Spill a VReg to its assigned stack slot.
    SpillStore { vreg: VReg, slot: u32 },
    /// Reload a VReg from its assigned stack slot.
    SpillLoad { vreg: VReg, slot: u32 },

    // ── Marker ────────────────────────────────────────────
    /// Label marking the start of a basic block.
    Label { block: BlockId },

    // ── Vector Operations ─────────────────────────────────
    /// Broadcast scalar to all lanes of a vector register.
    VecBroadcast { dst: VReg, src: VReg, lane_count: u32 },
    /// Load a vector from memory.
    VecLoad { dst: VReg, addr: VReg, lane_count: u32 },
    /// Store a vector to memory.
    VecStore { addr: VReg, val: VReg, lane_count: u32 },
    /// Vector add.
    VecAdd { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector subtract.
    VecSub { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector multiply.
    VecMul { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector divide (FP only).
    VecDiv { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector bitwise AND.
    VecAnd { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector bitwise OR.
    VecOr { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector bitwise XOR.
    VecXor { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector minimum.
    VecMin { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector maximum.
    VecMax { dst: VReg, lhs: VReg, rhs: VReg },
    /// Vector negate.
    VecNeg { dst: VReg, src: VReg },
    /// Vector absolute value.
    VecAbs { dst: VReg, src: VReg },
    /// Vector square root (FP only).
    VecSqrt { dst: VReg, src: VReg },
    /// Vector shuffle with immediate mask.
    VecShuffle { dst: VReg, src: VReg, mask: Vec<u8> },
    /// Horizontal reduce (sum all lanes).
    VecReduceSum { dst: VReg, src: VReg, lane_count: u32 },
    /// Extract a scalar lane from a vector.
    ExtractLane { dst: VReg, src: VReg, index: u32 },
    /// Insert a scalar lane into a vector.
    InsertLane { dst: VReg, src: VReg, index: u32, elem: VReg },
}

// ── Basic Block ─────────────────────────────────────────────────────────

/// A basic block in the MIR.
///
/// Contains a sequence of instructions ending with an optional terminator
/// (Branch, Jump, or Ret). The `preds` and `succs` lists are populated
/// after lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct MirBlock {
    pub id: BlockId,
    pub insts: Vec<MirInst>,
    pub preds: Vec<BlockId>,
    pub succs: Vec<BlockId>,
}

// ── Function ────────────────────────────────────────────────────────────

/// A MIR function: a named collection of basic blocks with virtual registers.
#[derive(Debug, Clone, PartialEq)]
pub struct MirFunction {
    pub name: String,
    pub blocks: Vec<MirBlock>,
    pub vreg_count: u32,
    pub params: Vec<VReg>,
}

impl MirFunction {
    /// Create a new, empty MIR function.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            blocks: Vec::new(),
            vreg_count: 0,
            params: Vec::new(),
        }
    }

    /// Allocate a fresh virtual register.
    pub fn alloc_vreg(&mut self) -> VReg {
        let vreg = VReg::new(self.vreg_count);
        self.vreg_count += 1;
        vreg
    }

    /// Create a new basic block and return its BlockId.
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId::new(self.blocks.len() as u32);
        self.blocks.push(MirBlock {
            id,
            insts: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
        });
        id
    }
}
