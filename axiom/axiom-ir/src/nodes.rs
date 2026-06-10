//! Node definitions for the Sea-of-Nodes IR.
//!
//! Every value, control flow point, and memory operation is a node.
//! Nodes are identified by `NodeId` (a newtype'd u32 for type safety).

use std::fmt;

/// Unique identifier for a node in the IR graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const fn new(id: u32) -> Self {
        NodeId(id)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Ownership root identifier — groups memory operations that may alias.
/// Operations on DIFFERENT roots NEVER alias. This is the key insight
/// that enables trivially correct DSE, CSE, and instruction scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OwnershipRoot(pub u32);

impl OwnershipRoot {
    pub const GLOBAL: OwnershipRoot = OwnershipRoot(0);
    pub const STACK: OwnershipRoot = OwnershipRoot(1);

    pub const fn new(id: u32) -> Self {
        OwnershipRoot(id)
    }

    pub fn is_global(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for OwnershipRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            OwnershipRoot::GLOBAL => write!(f, "root_global"),
            OwnershipRoot::STACK => write!(f, "root_stack"),
            _ => write!(f, "root_{}", self.0),
        }
    }
}

/// The type of a value produced by a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Bool,
    Ptr,
    Void,
    /// Unknown/unresolved type — will be filled in by type inference.
    Unknown,
}

impl Type {
    pub fn byte_size(&self) -> u32 {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 | Type::Ptr => 8,
            Type::I128 | Type::U128 => 16,
            Type::Void | Type::Unknown => 0,
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128
                     | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128
                     | Type::Bool)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Type::Ptr)
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128)
    }

    pub fn bit_width(&self) -> u32 {
        self.byte_size() * 8
    }
}

/// A node in the Sea-of-Nodes IR.
///
/// Each node represents one operation. Control flow is explicit via
/// `Region` (merge points) and `Branch` nodes. Data flow is via
/// node inputs (operands reference other NodeIds).
#[derive(Debug, Clone, PartialEq)]
pub enum IrNode {
    // ── Constants ──────────────────────────────────────────
    IntConst(i64),
    FpConst(u64),  // bits of f64
    BoolConst(bool),
    UndefConst,

    // ── Arithmetic ────────────────────────────────────────
    Add { lhs: NodeId, rhs: NodeId },
    Sub { lhs: NodeId, rhs: NodeId },
    Mul { lhs: NodeId, rhs: NodeId },
    Div { lhs: NodeId, rhs: NodeId },
    Rem { lhs: NodeId, rhs: NodeId },
    Neg { val: NodeId },

    // ── Bitwise ───────────────────────────────────────────
    And { lhs: NodeId, rhs: NodeId },
    Or  { lhs: NodeId, rhs: NodeId },
    Xor { lhs: NodeId, rhs: NodeId },
    Shl { lhs: NodeId, rhs: NodeId },
    Shr { lhs: NodeId, rhs: NodeId },
    Sar { lhs: NodeId, rhs: NodeId },  // arithmetic shift right
    Not { val: NodeId },

    // ── Comparison ────────────────────────────────────────
    Eq  { lhs: NodeId, rhs: NodeId },
    Ne  { lhs: NodeId, rhs: NodeId },
    Lt  { lhs: NodeId, rhs: NodeId },
    Le  { lhs: NodeId, rhs: NodeId },
    Gt  { lhs: NodeId, rhs: NodeId },
    Ge  { lhs: NodeId, rhs: NodeId },

    // ── Conversion ────────────────────────────────────────
    ZExt { val: NodeId, to: Type },
    SExt { val: NodeId, to: Type },
    Trunc { val: NodeId, to: Type },
    BitCast { val: NodeId, to: Type },
    IntToPtr { val: NodeId },
    PtrToInt { val: NodeId },

    // ── Memory ────────────────────────────────────────────
    /// Load from address, with ownership root for alias analysis.
    Load { addr: NodeId, root: OwnershipRoot, ty: Type },
    /// Store value to address, with ownership root.
    Store { addr: NodeId, val: NodeId, root: OwnershipRoot, ty: Type },
    /// Allocate on the stack — creates a new ownership root.
    StackAlloc { size: NodeId, align: u32, root: OwnershipRoot },
    /// Memory fence / barrier.
    Fence { ordering: MemoryOrdering },

    // ── Function Entry & Parameters ──────────────────────
    /// Entry point of a function.
    Start,
    /// Function parameter. `index` is the parameter position (0-based),
    /// `ty` is the parameter type. Parameters are inputs to the function
    /// that arrive via the calling convention (e.g., in registers on x86-64).
    Param { index: u32, ty: Type },
    /// Function return. `value` is None for void returns.
    Return { value: Option<NodeId> },
    /// Unreachable code marker.
    Unreachable,
    /// Conditional branch.
    Branch { cond: NodeId, true_block: NodeId, false_block: NodeId },
    /// Unconditional jump.
    Jump { target: NodeId },
    /// Region / merge point (predecessors are implicit from Jump/Branch targets).
    /// `predecessors` lists the incoming control edges.
    Region { predecessors: Vec<NodeId> },
    /// Phi node for SSA merge. `inputs` pairs (region_node, value_node).
    /// CORRECTNESS: All inputs must be respected during lowering.
    Phi { inputs: Vec<(NodeId, NodeId)>, ty: Type },

    // ── Function Calls ────────────────────────────────────
    /// Direct call to a known function.
    Call { func: String, args: Vec<NodeId>, ty: Type },
    /// Indirect call through a function pointer.
    CallIndirect { addr: NodeId, args: Vec<NodeId>, ty: Type },
    /// Tail call: a call in tail position where the return value is
    /// the direct result of the call. During ISel, this lowers to
    /// a jump (reusing the current stack frame) instead of a call.
    /// This is critical for functional programming patterns and
    /// eliminates stack overflow for tail-recursive functions.
    TailCall { func: String, args: Vec<NodeId>, ty: Type },

    // ── Variable References ───────────────────────────────
    /// Definition of a named variable.
    VarDef { name: String, init: NodeId, root: OwnershipRoot },
    /// Reference to a named variable. CORRECTNESS: Must resolve to actual storage.
    VarRef { name: String, ty: Type },
    /// Assignment to a named variable.
    VarSet { name: String, val: NodeId, root: OwnershipRoot },

    // ── Aggregates ────────────────────────────────────────
    /// Extract a field from a struct/tuple.
    Extract { aggregate: NodeId, index: u32 },
    /// Insert a field into a struct/tuple.
    Insert { aggregate: NodeId, index: u32, value: NodeId },

    // ── Floating-Point Arithmetic ────────────────────────
    FAdd { lhs: NodeId, rhs: NodeId },
    FSub { lhs: NodeId, rhs: NodeId },
    FMul { lhs: NodeId, rhs: NodeId },
    FDiv { lhs: NodeId, rhs: NodeId },
    FRem { lhs: NodeId, rhs: NodeId },
    FNeg { val: NodeId },
    FAbs { val: NodeId },
    FSqrt { val: NodeId },

    // ── Floating-Point Comparison ──────────────────────────
    FEq { lhs: NodeId, rhs: NodeId },
    FLt { lhs: NodeId, rhs: NodeId },
    FLe { lhs: NodeId, rhs: NodeId },
    FGt { lhs: NodeId, rhs: NodeId },
    FGe { lhs: NodeId, rhs: NodeId },
    FNe { lhs: NodeId, rhs: NodeId },

    // ── Floating-Point Conversion ─────────────────────────
    FpToSInt { val: NodeId, to: Type },
    SIntToFp { val: NodeId, to: Type },
    FpToUInt { val: NodeId, to: Type },
    UIntToFp { val: NodeId, to: Type },

    // ── Floating-Point Misc ────────────────────────────────
    Copysign { lhs: NodeId, rhs: NodeId },
    Fmin { lhs: NodeId, rhs: NodeId },
    Fmax { lhs: NodeId, rhs: NodeId },

    // ── Intrinsics ────────────────────────────────────────
    Intrinsic { name: String, args: Vec<NodeId>, ty: Type },

    // ── Ownership Annotation ──────────────────────────────
    /// Mark a node as having a specific ownership root.
    /// This is a no-op at runtime but enables ownership-aware optimizations.
    Owned { val: NodeId, root: OwnershipRoot },

    // ── Vector Operations ───────────────────────────────────
    /// Broadcast a scalar value across all lanes of a vector.
    VecBroadcast { val: NodeId, lane_type: Type, lane_count: u32 },
    /// Load a vector from memory.
    VecLoad { addr: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32 },
    /// Store a vector to memory.
    VecStore { addr: NodeId, val: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32 },
    /// Binary vector operation (add, sub, mul, etc.).
    VecBinOp { op: VecBinOp, lhs: NodeId, rhs: NodeId, lane_type: Type, lane_count: u32 },
    /// Unary vector operation (neg, not, abs).
    VecUnOp { op: VecUnOp, val: NodeId, lane_type: Type, lane_count: u32 },
    /// Extract a scalar lane from a vector.
    ExtractLane { val: NodeId, index: u32, lane_type: Type },
    /// Insert a scalar lane into a vector.
    InsertLane { val: NodeId, index: u32, elem: NodeId, lane_type: Type },
    /// Horizontal reduce (sum all lanes).
    VecReduce { op: VecReduceOp, val: NodeId, lane_type: Type, lane_count: u32 },
    /// Shuffle vector lanes.
    VecShuffle { val: NodeId, mask: Vec<u8>, lane_type: Type },
    /// Gather (scatter is store-equivalent).
    VecGather { addrs: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32 },
    /// Scatter.
    VecScatter { addrs: NodeId, vals: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32 },
}

/// Vector binary operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecBinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
    Min,
    Max,
    Shl,
    Shr,
}

/// Vector unary operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecUnOp {
    Neg,
    Not,
    Abs,
    Sqrt,
}

/// Vector reduce operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecReduceOp {
    Sum,
    Min,
    Max,
    And,
    Or,
}

/// Memory ordering for fences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl IrNode {
    /// Returns all NodeIds that this node uses as inputs.
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            IrNode::IntConst(_) | IrNode::FpConst(_) | IrNode::BoolConst(_) |
            IrNode::UndefConst | IrNode::Start | IrNode::Param { .. } | IrNode::Unreachable |
            IrNode::Region { .. } => Vec::new(),

            IrNode::Neg { val } | IrNode::Not { val } |
            IrNode::ZExt { val, .. } | IrNode::SExt { val, .. } |
            IrNode::Trunc { val, .. } | IrNode::BitCast { val, .. } |
            IrNode::IntToPtr { val } | IrNode::PtrToInt { val } |
            // FP unary
            IrNode::FNeg { val } | IrNode::FAbs { val } | IrNode::FSqrt { val } |
            // FP conversions
            IrNode::FpToSInt { val, .. } | IrNode::SIntToFp { val, .. } |
            IrNode::FpToUInt { val, .. } | IrNode::UIntToFp { val, .. } => vec![*val],

            IrNode::Add { lhs, rhs } | IrNode::Sub { lhs, rhs } |
            IrNode::Mul { lhs, rhs } | IrNode::Div { lhs, rhs } |
            IrNode::Rem { lhs, rhs } | IrNode::And { lhs, rhs } |
            IrNode::Or  { lhs, rhs } | IrNode::Xor { lhs, rhs } |
            IrNode::Shl { lhs, rhs } | IrNode::Shr { lhs, rhs } |
            IrNode::Sar { lhs, rhs } |
            IrNode::Eq  { lhs, rhs } | IrNode::Ne  { lhs, rhs } |
            IrNode::Lt  { lhs, rhs } | IrNode::Le  { lhs, rhs } |
            IrNode::Gt  { lhs, rhs } | IrNode::Ge  { lhs, rhs } |
            // FP binary arithmetic
            IrNode::FAdd { lhs, rhs } | IrNode::FSub { lhs, rhs } |
            IrNode::FMul { lhs, rhs } | IrNode::FDiv { lhs, rhs } |
            IrNode::FRem { lhs, rhs } |
            // FP binary comparison
            IrNode::FEq { lhs, rhs } | IrNode::FLt { lhs, rhs } |
            IrNode::FLe { lhs, rhs } | IrNode::FGt { lhs, rhs } |
            IrNode::FGe { lhs, rhs } | IrNode::FNe { lhs, rhs } |
            // FP misc binary
            IrNode::Copysign { lhs, rhs } | IrNode::Fmin { lhs, rhs } |
            IrNode::Fmax { lhs, rhs } => vec![*lhs, *rhs],

            IrNode::Load { addr, .. } => vec![*addr],
            IrNode::Store { addr, val, .. } => vec![*addr, *val],
            IrNode::StackAlloc { size, .. } => vec![*size],
            IrNode::Fence { .. } => Vec::new(),

            IrNode::Return { value } => value.iter().copied().collect(),
            IrNode::Branch { cond, true_block, false_block } => vec![*cond, *true_block, *false_block],
            IrNode::Jump { target } => vec![*target],

            IrNode::Phi { inputs, .. } => {
                let mut v = Vec::with_capacity(inputs.len() * 2);
                for (region, val) in inputs {
                    v.push(*region);
                    v.push(*val);
                }
                v
            }

            IrNode::Call { args, .. } => args.clone(),
            IrNode::CallIndirect { addr, args, .. } => {
                let mut v = vec![*addr];
                v.extend(args);
                v
            }
            IrNode::TailCall { args, .. } => args.clone(),

            IrNode::VarDef { init, .. } => vec![*init],
            IrNode::VarRef { .. } => Vec::new(),
            IrNode::VarSet { val, .. } => vec![*val],

            IrNode::Extract { aggregate, .. } => vec![*aggregate],
            IrNode::Insert { aggregate, value, .. } => vec![*aggregate, *value],

            IrNode::Intrinsic { args, .. } => args.clone(),
            IrNode::Owned { val, .. } => vec![*val],

            // ── Vector Operations ─────────────────────────────────────
            IrNode::VecBroadcast { val, .. } => vec![*val],
            IrNode::VecLoad { addr, .. } => vec![*addr],
            IrNode::VecStore { addr, val, .. } => vec![*addr, *val],
            IrNode::VecBinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrNode::VecUnOp { val, .. } => vec![*val],
            IrNode::ExtractLane { val, .. } => vec![*val],
            IrNode::InsertLane { val, elem, .. } => vec![*val, *elem],
            IrNode::VecReduce { val, .. } => vec![*val],
            IrNode::VecShuffle { val, .. } => vec![*val],
            IrNode::VecGather { addrs, .. } => vec![*addrs],
            IrNode::VecScatter { addrs, vals, .. } => vec![*addrs, *vals],
        }
    }

    /// Returns the output type of this node, if known.
    pub fn output_type(&self) -> Type {
        match self {
            IrNode::IntConst(_) => Type::I64,
            IrNode::FpConst(_) => Type::F64,
            IrNode::BoolConst(_) => Type::Bool,
            IrNode::UndefConst => Type::Unknown,
            IrNode::Neg { val: _ } | IrNode::Not { val: _ } => Type::Unknown, // should propagate
            IrNode::Add { .. } | IrNode::Sub { .. } | IrNode::Mul { .. } |
            IrNode::Div { .. } | IrNode::Rem { .. } => Type::Unknown,
            IrNode::And { .. } | IrNode::Or { .. } | IrNode::Xor { .. } |
            IrNode::Shl { .. } | IrNode::Shr { .. } | IrNode::Sar { .. } => Type::Unknown,

            // FP arithmetic → F64 (or F32 depending on input; Unknown for now)
            IrNode::FAdd { .. } | IrNode::FSub { .. } |
            IrNode::FMul { .. } | IrNode::FDiv { .. } | IrNode::FRem { .. } => Type::Unknown,
            IrNode::FNeg { .. } | IrNode::FAbs { .. } | IrNode::FSqrt { .. } => Type::Unknown,

            // FP comparisons → Bool
            IrNode::FEq { .. } | IrNode::FLt { .. } | IrNode::FLe { .. } |
            IrNode::FGt { .. } | IrNode::FGe { .. } | IrNode::FNe { .. } => Type::Bool,

            // FP conversions
            IrNode::FpToSInt { to, .. } | IrNode::FpToUInt { to, .. } => *to,
            IrNode::SIntToFp { to, .. } | IrNode::UIntToFp { to, .. } => *to,

            // FP misc
            IrNode::Copysign { .. } | IrNode::Fmin { .. } | IrNode::Fmax { .. } => Type::Unknown,
            IrNode::Eq { .. } | IrNode::Ne { .. } | IrNode::Lt { .. } |
            IrNode::Le { .. } | IrNode::Gt { .. } | IrNode::Ge { .. } => Type::Bool,
            IrNode::ZExt { to, .. } | IrNode::SExt { to, .. } |
            IrNode::Trunc { to, .. } | IrNode::BitCast { to, .. } => *to,
            IrNode::IntToPtr { .. } => Type::Ptr,
            IrNode::PtrToInt { .. } => Type::I64,
            IrNode::Load { ty, .. } => *ty,
            IrNode::Store { .. } => Type::Void,
            IrNode::StackAlloc { .. } => Type::Ptr,
            IrNode::Fence { .. } => Type::Void,
            IrNode::Start => Type::Void,
            IrNode::Param { ty, .. } => *ty,
            IrNode::Return { .. } => Type::Void,
            IrNode::Unreachable => Type::Void,
            IrNode::Branch { .. } | IrNode::Jump { .. } => Type::Void,
            IrNode::Region { .. } => Type::Void,
            IrNode::Phi { ty, .. } => *ty,
            IrNode::Call { ty, .. } | IrNode::CallIndirect { ty, .. } | IrNode::TailCall { ty, .. } => *ty,
            IrNode::VarDef { .. } => Type::Void,
            IrNode::VarRef { ty, .. } => *ty,
            IrNode::VarSet { .. } => Type::Void,
            IrNode::Extract { .. } => Type::Unknown,
            IrNode::Insert { .. } => Type::Unknown,
            IrNode::Intrinsic { ty, .. } => *ty,
            IrNode::Owned { val: _, .. } => Type::Unknown,

            // ── Vector Operations ─────────────────────────────────────
            // Vector nodes return Unknown here; the actual type is a vector type
            // (lane_type × lane_count). The scalar Type system doesn't have a
            // Vec wrapper, so we use the lane_type as a proxy for type inference.
            IrNode::VecBroadcast { lane_type, .. } => *lane_type,
            IrNode::VecLoad { lane_type, .. } => *lane_type,
            IrNode::VecStore { .. } => Type::Void,
            IrNode::VecBinOp { lane_type, .. } => *lane_type,
            IrNode::VecUnOp { lane_type, .. } => *lane_type,
            IrNode::ExtractLane { lane_type, .. } => *lane_type,
            IrNode::InsertLane { lane_type, .. } => *lane_type,
            IrNode::VecReduce { lane_type, .. } => *lane_type,
            IrNode::VecShuffle { lane_type, .. } => *lane_type,
            IrNode::VecGather { lane_type, .. } => *lane_type,
            IrNode::VecScatter { .. } => Type::Void,
        }
    }

    /// Returns true if this node has side effects (memory writes, control flow, calls).
    pub fn has_side_effects(&self) -> bool {
        matches!(self,
            IrNode::Store { .. } | IrNode::Call { .. } | IrNode::CallIndirect { .. } | IrNode::TailCall { .. } |
            IrNode::Return { .. } | IrNode::Branch { .. } | IrNode::Jump { .. } |
            IrNode::Fence { .. } | IrNode::VarDef { .. } | IrNode::VarSet { .. } |
            IrNode::Unreachable | IrNode::StackAlloc { .. } |
            IrNode::VecStore { .. } | IrNode::VecScatter { .. }
        )
    }

    /// Returns the ownership root for memory operations.
    pub fn ownership_root(&self) -> Option<OwnershipRoot> {
        match self {
            IrNode::Load { root, .. } | IrNode::Store { root, .. } |
            IrNode::StackAlloc { root, .. } | IrNode::VarDef { root, .. } |
            IrNode::VarSet { root, .. } | IrNode::Owned { root, .. } |
            IrNode::VecLoad { root, .. } | IrNode::VecStore { root, .. } |
            IrNode::VecGather { root, .. } | IrNode::VecScatter { root, .. } => Some(*root),
            _ => None,
        }
    }

    /// Returns true if this is a memory read operation.
    pub fn is_load(&self) -> bool {
        matches!(self, IrNode::Load { .. } | IrNode::VecLoad { .. } | IrNode::VecGather { .. })
    }

    /// Returns true if this is a memory write operation.
    pub fn is_store(&self) -> bool {
        matches!(self, IrNode::Store { .. } | IrNode::VarSet { .. } | IrNode::VecStore { .. } | IrNode::VecScatter { .. })
    }
}
