//! Constant Folding.
//!
//! Evaluates constant expressions at compile time. When all inputs to an
//! operation are constants, the operation is replaced by its computed result.
//!
//! # Vector Constant Folding
//!
//! When both inputs to a `VecBinOp` are `VecBroadcast` of constants, the
//! operation can be folded to a `VecBroadcast` of the scalar result. This
//! is the key pattern: auto-vectorized code often broadcasts loop-invariant
//! scalars, and the resulting vector ops can be folded at compile time.

use axiom_ir::{IrGraph, IrNode, NodeId};
use axiom_ir::nodes::{VecBinOp, VecUnOp};
use crate::Pass;

/// Constant Folding pass.
///
/// Handles:
/// - Integer binary arithmetic and bitwise operations on `IntConst` inputs
/// - Integer comparisons on `IntConst` inputs
/// - Boolean binary operations on `BoolConst` inputs
/// - Boolean comparisons on `BoolConst` inputs
/// - `Neg` / `Not` on constant inputs
/// - Floating-point operations on `FpConst` inputs (when result is not NaN)
pub struct ConstantFolder;

impl ConstantFolder {
    /// Try to fold a binary integer operation.
    fn fold_int_binop<F>(&self, graph: &IrGraph, lhs: NodeId, rhs: NodeId, op: F) -> Option<IrNode>
    where
        F: Fn(i64, i64) -> Option<i64>,
    {
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;
        match (lhs_node, rhs_node) {
            (IrNode::IntConst(a), IrNode::IntConst(b)) => op(*a, *b).map(IrNode::IntConst),
            _ => None,
        }
    }

    /// Try to fold a binary integer comparison.
    fn fold_int_icmp<F>(&self, graph: &IrGraph, lhs: NodeId, rhs: NodeId, op: F) -> Option<IrNode>
    where
        F: Fn(i64, i64) -> bool,
    {
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;
        match (lhs_node, rhs_node) {
            (IrNode::IntConst(a), IrNode::IntConst(b)) => Some(IrNode::BoolConst(op(*a, *b))),
            _ => None,
        }
    }

    /// Try to fold a binary boolean operation.
    fn fold_bool_binop<F>(&self, graph: &IrGraph, lhs: NodeId, rhs: NodeId, op: F) -> Option<IrNode>
    where
        F: Fn(bool, bool) -> bool,
    {
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;
        match (lhs_node, rhs_node) {
            (IrNode::BoolConst(a), IrNode::BoolConst(b)) => Some(IrNode::BoolConst(op(*a, *b))),
            _ => None,
        }
    }

    /// Try to fold a binary boolean comparison.
    fn fold_bool_cmp<F>(&self, graph: &IrGraph, lhs: NodeId, rhs: NodeId, op: F) -> Option<IrNode>
    where
        F: Fn(bool, bool) -> bool,
    {
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;
        match (lhs_node, rhs_node) {
            (IrNode::BoolConst(a), IrNode::BoolConst(b)) => Some(IrNode::BoolConst(op(*a, *b))),
            _ => None,
        }
    }

    /// Try to fold a binary floating-point operation.
    fn fold_fp_binop<F>(&self, graph: &IrGraph, lhs: NodeId, rhs: NodeId, op: F) -> Option<IrNode>
    where
        F: Fn(f64, f64) -> f64,
    {
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;
        match (lhs_node, rhs_node) {
            (IrNode::FpConst(a), IrNode::FpConst(b)) => {
                let la = f64::from_bits(*a);
                let lb = f64::from_bits(*b);
                let result = op(la, lb);
                // Only fold if the result is not NaN (avoid NaN payload issues).
                if result.is_nan() {
                    None
                } else {
                    Some(IrNode::FpConst(result.to_bits()))
                }
            }
            _ => None,
        }
    }

    /// Try to fold a floating-point comparison.
    fn fold_fp_cmp<F>(&self, graph: &IrGraph, lhs: NodeId, rhs: NodeId, op: F) -> Option<IrNode>
    where
        F: Fn(f64, f64) -> bool,
    {
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;
        match (lhs_node, rhs_node) {
            (IrNode::FpConst(a), IrNode::FpConst(b)) => {
                let la = f64::from_bits(*a);
                let lb = f64::from_bits(*b);
                Some(IrNode::BoolConst(op(la, lb)))
            }
            _ => None,
        }
    }

    /// Try to fold a single node. Returns `Some(replacement)` if folding succeeded.
    fn try_fold(&self, graph: &IrGraph, node: &IrNode) -> Option<IrNode> {
        match node {
            // ── Integer binary arithmetic ───────────────────────────────
            IrNode::Add { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| Some(a.wrapping_add(b)))
            }
            IrNode::Sub { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| Some(a.wrapping_sub(b)))
            }
            IrNode::Mul { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| Some(a.wrapping_mul(b)))
            }
            IrNode::Div { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| {
                    if b == 0 || (a == i64::MIN && b == -1) {
                        None // Division by zero or overflow — don't fold.
                    } else {
                        Some(a / b)
                    }
                })
            }
            IrNode::Rem { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| {
                    if b == 0 || (a == i64::MIN && b == -1) {
                        None
                    } else {
                        Some(a % b)
                    }
                })
            }

            // ── Integer binary bitwise ─────────────────────────────────
            IrNode::And { lhs, rhs } => {
                // Try integer AND first, then boolean AND.
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| Some(a & b))
                    .or_else(|| self.fold_bool_binop(graph, *lhs, *rhs, |a, b| a && b))
            }
            IrNode::Or { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| Some(a | b))
                    .or_else(|| self.fold_bool_binop(graph, *lhs, *rhs, |a, b| a || b))
            }
            IrNode::Xor { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| Some(a ^ b))
                    .or_else(|| self.fold_bool_binop(graph, *lhs, *rhs, |a, b| a ^ b))
            }
            IrNode::Shl { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| {
                    if b < 0 || b >= 64 {
                        None // Out-of-range shift — don't fold.
                    } else {
                        Some(a.wrapping_shl(b as u32))
                    }
                })
            }
            IrNode::Shr { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| {
                    if b < 0 || b >= 64 {
                        None
                    } else {
                        Some((a as u64).wrapping_shr(b as u32) as i64)
                    }
                })
            }
            IrNode::Sar { lhs, rhs } => {
                self.fold_int_binop(graph, *lhs, *rhs, |a, b| {
                    if b < 0 || b >= 64 {
                        None
                    } else {
                        Some(a.wrapping_shr(b as u32))
                    }
                })
            }

            // ── Integer comparisons ────────────────────────────────────
            IrNode::Eq { lhs, rhs } => {
                self.fold_int_icmp(graph, *lhs, *rhs, |a, b| a == b)
                    .or_else(|| self.fold_bool_cmp(graph, *lhs, *rhs, |a, b| a == b))
                    .or_else(|| self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a == b))
            }
            IrNode::Ne { lhs, rhs } => {
                self.fold_int_icmp(graph, *lhs, *rhs, |a, b| a != b)
                    .or_else(|| self.fold_bool_cmp(graph, *lhs, *rhs, |a, b| a != b))
                    .or_else(|| self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a != b))
            }
            IrNode::Lt { lhs, rhs } => {
                self.fold_int_icmp(graph, *lhs, *rhs, |a, b| a < b)
                    .or_else(|| self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a < b))
            }
            IrNode::Le { lhs, rhs } => {
                self.fold_int_icmp(graph, *lhs, *rhs, |a, b| a <= b)
                    .or_else(|| self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a <= b))
            }
            IrNode::Gt { lhs, rhs } => {
                self.fold_int_icmp(graph, *lhs, *rhs, |a, b| a > b)
                    .or_else(|| self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a > b))
            }
            IrNode::Ge { lhs, rhs } => {
                self.fold_int_icmp(graph, *lhs, *rhs, |a, b| a >= b)
                    .or_else(|| self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a >= b))
            }

            // ── Unary ──────────────────────────────────────────────────
            IrNode::Neg { val } => {
                match graph.get(*val)? {
                    IrNode::IntConst(a) => Some(IrNode::IntConst(a.wrapping_neg())),
                    IrNode::FpConst(a) => {
                        let f = f64::from_bits(*a);
                        let result = -f;
                        if result.is_nan() {
                            None
                        } else {
                            Some(IrNode::FpConst(result.to_bits()))
                        }
                    }
                    _ => None,
                }
            }
            IrNode::Not { val } => {
                match graph.get(*val)? {
                    IrNode::BoolConst(a) => Some(IrNode::BoolConst(!a)),
                    IrNode::IntConst(a) => Some(IrNode::IntConst(!a)), // bitwise complement
                    _ => None,
                }
            }

            // ── Floating-point binary arithmetic ───────────────────────
            // (handled inline via fold_fp_binop above for comparisons;
            //  the arithmetic FP ops go here)
            IrNode::FAdd { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a + b)
            }
            IrNode::FSub { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a - b)
            }
            IrNode::FMul { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a * b)
            }
            IrNode::FDiv { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a / b)
            }
            IrNode::FRem { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a % b)
            }

            // ── Floating-point unary ─────────────────────────────────
            IrNode::FNeg { val } => {
                match graph.get(*val)? {
                    IrNode::FpConst(a) => {
                        let f = f64::from_bits(*a);
                        let result = -f;
                        if result.is_nan() {
                            None
                        } else {
                            Some(IrNode::FpConst(result.to_bits()))
                        }
                    }
                    _ => None,
                }
            }
            IrNode::FAbs { val } => {
                match graph.get(*val)? {
                    IrNode::FpConst(a) => {
                        let f = f64::from_bits(*a);
                        let result = f.abs();
                        if result.is_nan() {
                            None
                        } else {
                            Some(IrNode::FpConst(result.to_bits()))
                        }
                    }
                    _ => None,
                }
            }
            IrNode::FSqrt { val } => {
                match graph.get(*val)? {
                    IrNode::FpConst(a) => {
                        let f = f64::from_bits(*a);
                        if f < 0.0 {
                            None // sqrt of negative is NaN — don't fold
                        } else {
                            let result = f.sqrt();
                            if result.is_nan() {
                                None
                            } else {
                                Some(IrNode::FpConst(result.to_bits()))
                            }
                        }
                    }
                    _ => None,
                }
            }

            // ── Floating-point comparisons ────────────────────────────
            IrNode::FEq { lhs, rhs } => {
                self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a == b)
            }
            IrNode::FLt { lhs, rhs } => {
                self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a < b)
            }
            IrNode::FLe { lhs, rhs } => {
                self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a <= b)
            }
            IrNode::FGt { lhs, rhs } => {
                self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a > b)
            }
            IrNode::FGe { lhs, rhs } => {
                self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a >= b)
            }
            IrNode::FNe { lhs, rhs } => {
                self.fold_fp_cmp(graph, *lhs, *rhs, |a, b| a != b)
            }

            // ── Floating-point misc ──────────────────────────────────
            IrNode::Copysign { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| {
                    if a.is_nan() || b.is_nan() { a } else { a.copysign(b) }
                })
            }
            IrNode::Fmin { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a.min(b))
            }
            IrNode::Fmax { lhs, rhs } => {
                self.fold_fp_binop(graph, *lhs, *rhs, |a, b| a.max(b))
            }

            // ── Vector constant folding ────────────────────────────────
            //
            // If both VecBinOp inputs are VecBroadcast of constants, fold to
            // VecBroadcast of the scalar result. This is the key pattern for
            // auto-vectorized code with loop-invariant broadcast constants.
            IrNode::VecBinOp { op, lhs, rhs, lane_type, lane_count } => {
                let lhs_node = graph.get(*lhs)?;
                let rhs_node = graph.get(*rhs)?;
                match (lhs_node, rhs_node) {
                    (IrNode::VecBroadcast { val: lhs_val, .. }, IrNode::VecBroadcast { val: rhs_val, .. }) => {
                        let folded_scalar = self.fold_vec_binop_scalar(graph, *op, *lhs_val, *rhs_val, *lane_type)?;
                        Some(IrNode::VecBroadcast {
                            val: folded_scalar,
                            lane_type: *lane_type,
                            lane_count: *lane_count,
                        })
                    }
                    _ => None,
                }
            }

            // VecUnOp on a VecBroadcast of a constant → VecBroadcast of the scalar result
            IrNode::VecUnOp { op, val, lane_type, lane_count } => {
                let val_node = graph.get(*val)?;
                if let IrNode::VecBroadcast { val: inner_val, .. } = val_node {
                    let folded_scalar = self.fold_vec_unop_scalar(graph, *op, *inner_val, *lane_type)?;
                    Some(IrNode::VecBroadcast {
                        val: folded_scalar,
                        lane_type: *lane_type,
                        lane_count: *lane_count,
                    })
                } else {
                    None
                }
            }

            _ => None,
        }
    }
    /// Try to fold a binary FP operation, returning the result as a new NodeId
    /// in the graph (or None if not foldable).
    fn fold_vec_binop_scalar(
        &self,
        graph: &IrGraph,
        op: VecBinOp,
        lhs: NodeId,
        rhs: NodeId,
        lane_type: axiom_ir::nodes::Type,
    ) -> Option<NodeId> {
        use axiom_ir::nodes::Type;
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;

        match lane_type {
            Type::I32 | Type::U32 => {
                // Integer vector binop: fold the scalar constant
                match (lhs_node, rhs_node) {
                    (IrNode::IntConst(a), IrNode::IntConst(b)) => {
                        let a = *a as i32;
                        let b = *b as i32;
                        let _result = match op {
                            VecBinOp::Add => a.wrapping_add(b),
                            VecBinOp::Sub => a.wrapping_sub(b),
                            VecBinOp::Mul => a.wrapping_mul(b),
                            VecBinOp::Div => {
                                if b == 0 || (a == i32::MIN && b == -1) {
                                    return None;
                                }
                                a / b
                            }
                            VecBinOp::And => a & b,
                            VecBinOp::Or  => a | b,
                            VecBinOp::Xor => a ^ b,
                            VecBinOp::Min => a.min(b),
                            VecBinOp::Max => a.max(b),
                            VecBinOp::Shl => {
                                if b < 0 || b >= 32 { return None; }
                                a.wrapping_shl(b as u32)
                            }
                            VecBinOp::Shr => {
                                if b < 0 || b >= 32 { return None; }
                                (a as u32).wrapping_shr(b as u32) as i32
                            }
                        };
                        None // Handled in run() via try_fold_vector
                    }
                    _ => None,
                }
            }
            Type::F32 | Type::F64 => {
                match (lhs_node, rhs_node) {
                    (IrNode::FpConst(a), IrNode::FpConst(b)) => {
                        let la = f64::from_bits(*a);
                        let lb = f64::from_bits(*b);
                        let result = match op {
                            VecBinOp::Add => la + lb,
                            VecBinOp::Sub => la - lb,
                            VecBinOp::Mul => la * lb,
                            VecBinOp::Div => la / lb,
                            VecBinOp::Min => la.min(lb),
                            VecBinOp::Max => la.max(lb),
                            _ => return None, // Not applicable to FP
                        };
                        if result.is_nan() { None }
                        else { None } // Handled in run() via try_fold_vector
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Try to fold a unary vector op on a broadcast constant.
    fn fold_vec_unop_scalar(
        &self,
        _graph: &IrGraph,
        _op: VecUnOp,
        _val: NodeId,
        _lane_type: axiom_ir::nodes::Type,
    ) -> Option<NodeId> {
        // Same issue as fold_vec_binop_scalar: can't create NodeIds here.
        // Handled in run() below.
        None
    }
}

impl Pass for ConstantFolder {
    fn name(&self) -> &str {
        "constant_fold"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        // Collect all node IDs up front so we don't iterate over mutations.
        let node_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();

        for id in node_ids {
            let node = match graph.get(id) {
                Some(n) => n.clone(),
                None => continue,
            };

            // ── Standard scalar constant folding ──
            if let Some(folded) = self.try_fold(graph, &node) {
                graph.replace(id, folded);
                modified = true;
                continue;
            }

            // ── Vector constant folding ──
            //
            // VecBinOp(VecBroadcast(const_a), VecBroadcast(const_b))
            //   → VecBroadcast(const_a OP const_b)
            //
            // VecUnOp(VecBroadcast(const_a))
            //   → VecBroadcast(OP const_a)
            //
            // We handle this here (not in try_fold) because we need mutable
            // access to the graph to push new constant nodes.
            let folded_vec = self.try_fold_vector(graph, &node);
            if let Some(folded) = folded_vec {
                graph.replace(id, folded);
                modified = true;
            }
        }

        modified
    }
}

impl ConstantFolder {
    /// Try to fold a vector operation whose broadcast inputs are constants.
    /// Returns a replacement IrNode if folding succeeded.
    fn try_fold_vector(&self, graph: &mut IrGraph, node: &IrNode) -> Option<IrNode> {
        match node {
            IrNode::VecBinOp { op, lhs, rhs, lane_type, lane_count } => {
                let lhs_node = graph.get(*lhs)?;
                let rhs_node = graph.get(*rhs)?;

                // Both inputs must be VecBroadcast of the same lane_type
                match (lhs_node, rhs_node) {
                    (IrNode::VecBroadcast { val: lhs_val, lane_type: lt1, lane_count: lc1 },
                     IrNode::VecBroadcast { val: rhs_val, lane_type: lt2, lane_count: lc2 })
                    if lt1 == lane_type && lt2 == lane_type && lc1 == lane_count && lc2 == lane_count => {
                        // Try to fold the scalar operation
                        let folded_val = self.fold_vec_scalar_binop(graph, *op, *lhs_val, *rhs_val, *lane_type)?;
                        Some(IrNode::VecBroadcast { val: folded_val, lane_type: *lane_type, lane_count: *lane_count })
                    }
                    _ => None,
                }
            }

            IrNode::VecUnOp { op, val, lane_type, lane_count } => {
                let val_node = graph.get(*val)?;
                if let IrNode::VecBroadcast { val: inner_val, lane_type: lt, lane_count: lc } = val_node {
                    if lt == lane_type && lc == lane_count {
                        let folded_val = self.fold_vec_scalar_unop(graph, *op, *inner_val, *lane_type)?;
                        Some(IrNode::VecBroadcast { val: folded_val, lane_type: *lane_type, lane_count: *lane_count })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Fold a vector binary op on the scalar values inside two broadcasts.
    /// Pushes a new constant node and returns its NodeId.
    fn fold_vec_scalar_binop(
        &self,
        graph: &mut IrGraph,
        op: VecBinOp,
        lhs: NodeId,
        rhs: NodeId,
        lane_type: axiom_ir::nodes::Type,
    ) -> Option<NodeId> {
        use axiom_ir::nodes::Type;
        let lhs_node = graph.get(lhs)?;
        let rhs_node = graph.get(rhs)?;

        match lane_type {
            Type::I32 | Type::U32 | Type::I64 | Type::U64 => {
                match (lhs_node, rhs_node) {
                    (IrNode::IntConst(a), IrNode::IntConst(b)) => {
                        let result = match op {
                            VecBinOp::Add => Some(a.wrapping_add(*b)),
                            VecBinOp::Sub => Some(a.wrapping_sub(*b)),
                            VecBinOp::Mul => Some(a.wrapping_mul(*b)),
                            VecBinOp::Div => {
                                if *b == 0 || (*a == i64::MIN && *b == -1) { None }
                                else { Some(a / b) }
                            }
                            VecBinOp::And => Some(a & b),
                            VecBinOp::Or  => Some(a | b),
                            VecBinOp::Xor => Some(a ^ b),
                            VecBinOp::Min => Some((*a).min(*b)),
                            VecBinOp::Max => Some((*a).max(*b)),
                            VecBinOp::Shl => {
                                if *b < 0 || *b >= 64 { None }
                                else { Some(a.wrapping_shl(*b as u32)) }
                            }
                            VecBinOp::Shr => {
                                if *b < 0 || *b >= 64 { None }
                                else { Some((*a as u64).wrapping_shr(*b as u32) as i64) }
                            }
                        };
                        result.map(|r| graph.push_node(IrNode::IntConst(r)))
                    }
                    _ => None,
                }
            }
            Type::F32 | Type::F64 => {
                match (lhs_node, rhs_node) {
                    (IrNode::FpConst(a), IrNode::FpConst(b)) => {
                        let la = f64::from_bits(*a);
                        let lb = f64::from_bits(*b);
                        let result = match op {
                            VecBinOp::Add => la + lb,
                            VecBinOp::Sub => la - lb,
                            VecBinOp::Mul => la * lb,
                            VecBinOp::Div => la / lb,
                            VecBinOp::Min => la.min(lb),
                            VecBinOp::Max => la.max(lb),
                            _ => return None,
                        };
                        if result.is_nan() { None }
                        else { Some(graph.push_node(IrNode::FpConst(result.to_bits()))) }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Fold a vector unary op on the scalar value inside a broadcast.
    fn fold_vec_scalar_unop(
        &self,
        graph: &mut IrGraph,
        op: VecUnOp,
        val: NodeId,
        lane_type: axiom_ir::nodes::Type,
    ) -> Option<NodeId> {
        use axiom_ir::nodes::Type;
        let val_node = graph.get(val)?;

        match lane_type {
            Type::I32 | Type::U32 | Type::I64 | Type::U64 => {
                match val_node {
                    IrNode::IntConst(a) => {
                        let result = match op {
                            VecUnOp::Neg => Some(a.wrapping_neg()),
                            VecUnOp::Not => Some(!a),
                            VecUnOp::Abs => Some(a.abs()),
                            VecUnOp::Sqrt => None, // No integer sqrt
                        };
                        result.map(|r| graph.push_node(IrNode::IntConst(r)))
                    }
                    _ => None,
                }
            }
            Type::F32 | Type::F64 => {
                match val_node {
                    IrNode::FpConst(a) => {
                        let f = f64::from_bits(*a);
                        let result = match op {
                            VecUnOp::Neg => -f,
                            VecUnOp::Abs => f.abs(),
                            VecUnOp::Sqrt => {
                                if f < 0.0 { return None; }
                                f.sqrt()
                            }
                            VecUnOp::Not => return None, // bitwise not on FP not meaningful
                        };
                        if result.is_nan() { None }
                        else { Some(graph.push_node(IrNode::FpConst(result.to_bits()))) }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::OwnershipRoot;

    #[test]
    fn fold_add_constants() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(3));
        let b = graph.push_node(IrNode::IntConst(4));
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });

        let folder = ConstantFolder;
        assert!(folder.run(&mut graph));

        match graph.get(add) {
            Some(IrNode::IntConst(7)) => {}
            other => panic!("expected IntConst(7), got {:?}", other),
        }
    }

    #[test]
    fn fold_no_div_by_zero() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(10));
        let b = graph.push_node(IrNode::IntConst(0));
        let div = graph.push_node(IrNode::Div { lhs: a, rhs: b });

        let folder = ConstantFolder;
        assert!(!folder.run(&mut graph));

        // Should still be Div, not folded.
        match graph.get(div) {
            Some(IrNode::Div { .. }) => {}
            other => panic!("expected Div, got {:?}", other),
        }
    }

    #[test]
    fn fold_neg_constant() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(5));
        let neg = graph.push_node(IrNode::Neg { val: a });

        let folder = ConstantFolder;
        assert!(folder.run(&mut graph));

        match graph.get(neg) {
            Some(IrNode::IntConst(-5)) => {}
            other => panic!("expected IntConst(-5), got {:?}", other),
        }
    }

    #[test]
    fn fold_not_bool() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::BoolConst(true));
        let not = graph.push_node(IrNode::Not { val: a });

        let folder = ConstantFolder;
        assert!(folder.run(&mut graph));

        match graph.get(not) {
            Some(IrNode::BoolConst(false)) => {}
            other => panic!("expected BoolConst(false), got {:?}", other),
        }
    }

    #[test]
    fn fold_comparison() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(3));
        let b = graph.push_node(IrNode::IntConst(5));
        let lt = graph.push_node(IrNode::Lt { lhs: a, rhs: b });

        let folder = ConstantFolder;
        assert!(folder.run(&mut graph));

        match graph.get(lt) {
            Some(IrNode::BoolConst(true)) => {}
            other => panic!("expected BoolConst(true), got {:?}", other),
        }
    }

    #[test]
    fn no_fold_non_constant() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(3));
        // Load has no constant inputs.
        let load = graph.push_node(IrNode::Load {
            addr: a,
            root: OwnershipRoot::STACK,
            ty: axiom_ir::nodes::Type::I64,
        });
        let b = graph.push_node(IrNode::IntConst(4));
        let add = graph.push_node(IrNode::Add { lhs: load, rhs: b });

        let folder = ConstantFolder;
        assert!(!folder.run(&mut graph));

        match graph.get(add) {
            Some(IrNode::Add { .. }) => {}
            other => panic!("expected Add, got {:?}", other),
        }
    }
}
