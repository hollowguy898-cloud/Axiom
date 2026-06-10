//! IR Builder — ergonomic API for constructing IR graphs.

use crate::graph::IrGraph;
use crate::nodes::{IrNode, NodeId, OwnershipRoot, Type, VecBinOp, VecUnOp, VecReduceOp};

/// Convenience builder for constructing IR graphs.
pub struct IrBuilder {
    pub graph: IrGraph,
}

impl IrBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            graph: IrGraph::new(name),
        }
    }

    // ── Constants ──────────────────────────────────────

    pub fn int_const(&mut self, val: i64) -> NodeId {
        self.graph.push_node(IrNode::IntConst(val))
    }

    pub fn fp_const(&mut self, val: f64) -> NodeId {
        self.graph.push_node(IrNode::FpConst(val.to_bits()))
    }

    pub fn bool_const(&mut self, val: bool) -> NodeId {
        self.graph.push_node(IrNode::BoolConst(val))
    }

    pub fn zero(&mut self) -> NodeId { self.int_const(0) }
    pub fn one(&mut self) -> NodeId { self.int_const(1) }

    // ── Arithmetic ────────────────────────────────────

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Add { lhs, rhs })
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Sub { lhs, rhs })
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Mul { lhs, rhs })
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Div { lhs, rhs })
    }

    pub fn rem(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Rem { lhs, rhs })
    }

    pub fn neg(&mut self, val: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Neg { val })
    }

    // ── Bitwise ───────────────────────────────────────

    pub fn and(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::And { lhs, rhs })
    }

    pub fn or(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Or { lhs, rhs })
    }

    pub fn xor(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Xor { lhs, rhs })
    }

    pub fn shl(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Shl { lhs, rhs })
    }

    pub fn shr(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Shr { lhs, rhs })
    }

    pub fn sar(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Sar { lhs, rhs })
    }

    pub fn not(&mut self, val: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Not { val })
    }

    // ── Comparison ────────────────────────────────────

    pub fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Eq { lhs, rhs })
    }

    pub fn ne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Ne { lhs, rhs })
    }

    pub fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Lt { lhs, rhs })
    }

    pub fn le(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Le { lhs, rhs })
    }

    pub fn gt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Gt { lhs, rhs })
    }

    pub fn ge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Ge { lhs, rhs })
    }

    // ── Conversion ────────────────────────────────────

    pub fn zext(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::ZExt { val, to })
    }

    pub fn sext(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::SExt { val, to })
    }

    pub fn trunc(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::Trunc { val, to })
    }

    pub fn bitcast(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::BitCast { val, to })
    }

    // ── Memory ────────────────────────────────────────

    pub fn load(&mut self, addr: NodeId, root: OwnershipRoot, ty: Type) -> NodeId {
        self.graph.push_node(IrNode::Load { addr, root, ty })
    }

    pub fn store(&mut self, addr: NodeId, val: NodeId, root: OwnershipRoot, ty: Type) -> NodeId {
        self.graph.push_node(IrNode::Store { addr, val, root, ty })
    }

    pub fn stack_alloc(&mut self, size: NodeId, align: u32) -> (NodeId, OwnershipRoot) {
        let root = self.graph.alloc_root();
        let id = self.graph.push_node(IrNode::StackAlloc { size, align, root });
        (id, root)
    }

    // ── Control Flow ──────────────────────────────────

    pub fn ret(&mut self, value: Option<NodeId>) -> NodeId {
        self.graph.push_node(IrNode::Return { value })
    }

    pub fn branch(&mut self, cond: NodeId, true_block: NodeId, false_block: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Branch { cond, true_block, false_block })
    }

    pub fn jump(&mut self, target: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Jump { target })
    }

    pub fn region(&mut self, predecessors: Vec<NodeId>) -> NodeId {
        self.graph.push_node(IrNode::Region { predecessors })
    }

    /// Create a phi node with all inputs properly specified.
    /// CORRECTNESS: All inputs must be provided, not just the first.
    pub fn phi(&mut self, inputs: Vec<(NodeId, NodeId)>, ty: Type) -> NodeId {
        assert!(!inputs.is_empty(), "Phi node must have at least one input");
        self.graph.push_node(IrNode::Phi { inputs, ty })
    }

    // ── Function Calls ────────────────────────────────

    pub fn call(&mut self, func: &str, args: Vec<NodeId>, ty: Type) -> NodeId {
        self.graph.push_node(IrNode::Call { func: func.to_string(), args, ty })
    }

    // ── Variables ─────────────────────────────────────

    pub fn var_def(&mut self, name: &str, init: NodeId, root: OwnershipRoot) -> NodeId {
        let id = self.graph.push_node(IrNode::VarDef { name: name.to_string(), init, root });
        self.graph.define_var(name, id);
        id
    }

    /// Reference a named variable.
    /// CORRECTNESS: Must resolve through the graph's var_map, not return 0.
    pub fn var_ref(&mut self, name: &str, ty: Type) -> NodeId {
        self.graph.push_node(IrNode::VarRef { name: name.to_string(), ty })
    }

    pub fn var_set(&mut self, name: &str, val: NodeId, root: OwnershipRoot) -> NodeId {
        self.graph.push_node(IrNode::VarSet { name: name.to_string(), val, root })
    }

    // ── Floating-Point Arithmetic ──────────────────────

    pub fn fadd(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FAdd { lhs, rhs })
    }

    pub fn fsub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FSub { lhs, rhs })
    }

    pub fn fmul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FMul { lhs, rhs })
    }

    pub fn fdiv(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FDiv { lhs, rhs })
    }

    pub fn frem(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FRem { lhs, rhs })
    }

    pub fn fneg(&mut self, val: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FNeg { val })
    }

    pub fn fabs(&mut self, val: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FAbs { val })
    }

    pub fn fsqrt(&mut self, val: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FSqrt { val })
    }

    // ── Floating-Point Comparison ──────────────────────

    pub fn feq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FEq { lhs, rhs })
    }

    pub fn flt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FLt { lhs, rhs })
    }

    pub fn fle(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FLe { lhs, rhs })
    }

    pub fn fgt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FGt { lhs, rhs })
    }

    pub fn fge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FGe { lhs, rhs })
    }

    pub fn fne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::FNe { lhs, rhs })
    }

    // ── Floating-Point Conversion ──────────────────────

    pub fn fp_to_sint(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::FpToSInt { val, to })
    }

    pub fn sint_to_fp(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::SIntToFp { val, to })
    }

    pub fn fp_to_uint(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::FpToUInt { val, to })
    }

    pub fn uint_to_fp(&mut self, val: NodeId, to: Type) -> NodeId {
        self.graph.push_node(IrNode::UIntToFp { val, to })
    }

    // ── Floating-Point Misc ───────────────────────────

    pub fn copysign(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Copysign { lhs, rhs })
    }

    pub fn fmin(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Fmin { lhs, rhs })
    }

    pub fn fmax(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Fmax { lhs, rhs })
    }

    // ── Aggregates ────────────────────────────────────

    pub fn extract(&mut self, aggregate: NodeId, index: u32) -> NodeId {
        self.graph.push_node(IrNode::Extract { aggregate, index })
    }

    pub fn insert(&mut self, aggregate: NodeId, index: u32, value: NodeId) -> NodeId {
        self.graph.push_node(IrNode::Insert { aggregate, index, value })
    }

    // ── Vector Operations ─────────────────────────────────

    pub fn vec_broadcast(&mut self, val: NodeId, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecBroadcast { val, lane_type, lane_count })
    }

    pub fn vec_load(&mut self, addr: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecLoad { addr, root, lane_type, lane_count })
    }

    pub fn vec_store(&mut self, addr: NodeId, val: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecStore { addr, val, root, lane_type, lane_count })
    }

    pub fn vec_binop(&mut self, op: VecBinOp, lhs: NodeId, rhs: NodeId, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecBinOp { op, lhs, rhs, lane_type, lane_count })
    }

    pub fn vec_unop(&mut self, op: VecUnOp, val: NodeId, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecUnOp { op, val, lane_type, lane_count })
    }

    pub fn extract_lane(&mut self, val: NodeId, index: u32, lane_type: Type) -> NodeId {
        self.graph.push_node(IrNode::ExtractLane { val, index, lane_type })
    }

    pub fn insert_lane(&mut self, val: NodeId, index: u32, elem: NodeId, lane_type: Type) -> NodeId {
        self.graph.push_node(IrNode::InsertLane { val, index, elem, lane_type })
    }

    pub fn vec_reduce(&mut self, op: VecReduceOp, val: NodeId, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecReduce { op, val, lane_type, lane_count })
    }

    pub fn vec_shuffle(&mut self, val: NodeId, mask: Vec<u8>, lane_type: Type) -> NodeId {
        self.graph.push_node(IrNode::VecShuffle { val, mask, lane_type })
    }

    pub fn vec_gather(&mut self, addrs: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecGather { addrs, root, lane_type, lane_count })
    }

    pub fn vec_scatter(&mut self, addrs: NodeId, vals: NodeId, root: OwnershipRoot, lane_type: Type, lane_count: u32) -> NodeId {
        self.graph.push_node(IrNode::VecScatter { addrs, vals, root, lane_type, lane_count })
    }

    // ── High-Level Control Flow API ─────────────────────────

    /// Build if-else control flow: `if cond { true_body } else { false_body }`.
    ///
    /// Returns the merged result via a Phi node at the join region.
    ///
    /// This creates:
    /// ```text
    /// entry:
    ///     Branch cond → true_region, false_region
    /// true_region:
    ///     <true_body result>
    ///     Jump merge
    /// false_region:
    ///     <false_body result>
    ///     Jump merge
    /// merge:
    ///     Phi [(true_region, true_result), (false_region, false_result)]
    /// ```
    ///
    /// The caller is responsible for terminating the true and false branches
    /// (e.g., with `jump(merge)`). This method creates the merge Region
    /// and Phi node, and returns the Phi's NodeId.
    pub fn if_else<F1, F2>(
        &mut self,
        cond: NodeId,
        true_body: F1,
        false_body: F2,
        ty: Type,
    ) -> NodeId
    where
        F1: FnOnce(&mut IrBuilder) -> NodeId,
        F2: FnOnce(&mut IrBuilder) -> NodeId,
    {
        // Create the true and false branch regions
        let true_region = self.region(vec![]);
        let false_region = self.region(vec![]);

        // Emit the branch
        self.branch(cond, true_region, false_region);

        // Execute the true body
        let true_result = true_body(self);

        // Jump from true branch to merge
        let true_jump = self.jump(NodeId::new(0)); // placeholder

        // Execute the false body
        let false_result = false_body(self);

        // Jump from false branch to merge
        let false_jump = self.jump(NodeId::new(0)); // placeholder

        // Create merge region with the two predecessor jumps
        let merge = self.region(vec![true_jump, false_jump]);

        // Fix up the jumps to point to the merge region
        self.graph.replace(true_jump, IrNode::Jump { target: merge });
        self.graph.replace(false_jump, IrNode::Jump { target: merge });

        // Create the Phi node at the merge point
        self.phi(
            vec![(true_jump, true_result), (false_jump, false_result)],
            ty,
        )
    }

    /// Build a while loop: `while cond_fn { body_fn }`.
    ///
    /// Returns the loop variable's final value via Phi.
    ///
    /// This creates:
    /// ```text
    /// entry:
    ///     Jump header
    /// header:
    ///     Phi [(entry, init_val), (latch, body_result)]
    ///     Branch cond → body, exit
    /// body:
    ///     <body_fn produces new value>
    ///     Jump header  (back edge / latch)
    /// exit:
    /// ```
    ///
    /// The `init_val` is the initial value of the loop variable.
    /// The `cond_fn` receives the current loop variable value and
    /// should return a boolean NodeId.
    /// The `body_fn` receives the current loop variable value and
    /// should return the new value for the next iteration.
    pub fn while_loop<F1, F2>(
        &mut self,
        init_val: NodeId,
        cond_fn: F1,
        body_fn: F2,
        ty: Type,
    ) -> NodeId
    where
        F1: FnOnce(&mut IrBuilder, NodeId) -> NodeId,
        F2: FnOnce(&mut IrBuilder, NodeId) -> NodeId,
    {
        // Create the header region (entry + latch predecessors)
        // Use placeholder for latch predecessor — will fix after creating the latch
        let entry_jump = self.jump(NodeId::new(0)); // placeholder
        let header = self.region(vec![entry_jump, NodeId::new(0)]); // placeholder for latch

        // Fix the entry jump to point to header
        self.graph.replace(entry_jump, IrNode::Jump { target: header });

        // Create the loop-carried Phi node
        let loop_var = self.phi(
            vec![(entry_jump, init_val), (NodeId::new(0), init_val)], // placeholder for back-edge
            ty,
        );

        // Evaluate the condition
        let cond = cond_fn(self, loop_var);

        // Create body and exit regions
        let body_region = self.region(vec![header]);
        let exit_region = self.region(vec![header]);

        // Branch based on condition
        self.branch(cond, body_region, exit_region);

        // Execute the loop body
        let new_val = body_fn(self, loop_var);

        // Create latch jump back to header
        let latch_jump = self.jump(header);

        // Fix up the header region: replace placeholder predecessor with latch
        self.graph.replace(
            header,
            IrNode::Region {
                predecessors: vec![entry_jump, latch_jump],
            },
        );

        // Fix up the Phi: replace placeholder back-edge input
        self.graph.replace(
            loop_var,
            IrNode::Phi {
                inputs: vec![(entry_jump, init_val), (latch_jump, new_val)],
                ty,
            },
        );

        loop_var
    }

    /// Build a counted for loop: `for i in start..limit { body(i) }`.
    ///
    /// Returns the final value of the loop variable (induction variable).
    ///
    /// This creates:
    /// ```text
    /// entry:
    ///     Jump header
    /// header:
    ///     Phi [(entry, start), (latch, i_next)]
    ///     i_next = i + 1
    ///     Branch (i < limit) → body, exit
    /// body:
    ///     <body_fn(i)>
    ///     Jump header  (back edge / latch)
    /// exit:
    /// ```
    ///
    /// The `body_fn` receives the current induction variable value and
    /// can use it to compute addresses, etc.
    pub fn for_loop<F>(
        &mut self,
        start: NodeId,
        limit: NodeId,
        ty: Type,
        body_fn: F,
    ) -> NodeId
    where
        F: FnOnce(&mut IrBuilder, NodeId),
    {
        let one = self.int_const(1);

        // Create the header region (entry + latch predecessors)
        let entry_jump = self.jump(NodeId::new(0)); // placeholder
        let header = self.region(vec![entry_jump, NodeId::new(0)]); // placeholder for latch

        // Fix the entry jump to point to header
        self.graph.replace(entry_jump, IrNode::Jump { target: header });

        // Create the induction variable Phi node
        let ind_var = self.phi(
            vec![(entry_jump, start), (NodeId::new(0), start)], // placeholder for back-edge
            ty,
        );

        // Compute i + 1 for the back edge
        let ind_var_next = self.add(ind_var, one);

        // Evaluate the loop condition: i < limit
        let cond = self.lt(ind_var, limit);

        // Create body and exit regions
        let body_region = self.region(vec![header]);
        let exit_region = self.region(vec![header]);

        // Branch based on condition
        self.branch(cond, body_region, exit_region);

        // Execute the loop body
        body_fn(self, ind_var);

        // Create latch jump back to header
        let latch_jump = self.jump(header);

        // Fix up the header region: replace placeholder predecessor with latch
        self.graph.replace(
            header,
            IrNode::Region {
                predecessors: vec![entry_jump, latch_jump],
            },
        );

        // Fix up the Phi: replace placeholder back-edge input with i+1
        self.graph.replace(
            ind_var,
            IrNode::Phi {
                inputs: vec![(entry_jump, start), (latch_jump, ind_var_next)],
                ty,
            },
        );

        ind_var
    }

    /// Build a simple while loop: `while cond_fn() { }`.
    ///
    /// The `cond_fn` is called at the top of each iteration. It should
    /// build IR nodes that compute the loop condition AND the loop body
    /// side effects, and return the boolean condition NodeId.
    /// The loop continues as long as the condition is true.
    ///
    /// This creates:
    /// ```text
    /// entry:
    ///     Jump header
    /// header:
    ///     cond = cond_fn()
    ///     Branch cond → body, exit
    /// body:
    ///     Jump header  (back edge / latch)
    /// exit:
    /// ```
    pub fn while_cond<F>(
        &mut self,
        cond_fn: F,
    )
    where
        F: FnOnce(&mut IrBuilder) -> NodeId,
    {
        // Create the header region (entry + latch predecessors)
        let entry_jump = self.jump(NodeId::new(0)); // placeholder
        let header = self.region(vec![entry_jump, NodeId::new(0)]); // placeholder for latch

        // Fix the entry jump to point to header
        self.graph.replace(entry_jump, IrNode::Jump { target: header });

        // Evaluate the condition (cond_fn may also build loop body nodes)
        let cond = cond_fn(self);

        // Create body and exit regions
        let body_region = self.region(vec![header]);
        let exit_region = self.region(vec![header]);

        // Branch based on condition
        self.branch(cond, body_region, exit_region);

        // Create latch jump back to header
        let latch_jump = self.jump(header);

        // Fix up the header region: replace placeholder predecessor with latch
        self.graph.replace(
            header,
            IrNode::Region {
                predecessors: vec![entry_jump, latch_jump],
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_if_else_builds_phi() {
        let mut b = IrBuilder::new("if_else_test");
        let cond = b.bool_const(true);
        let ty = Type::I32;

        let result = b.if_else(
            cond,
            |b| b.int_const(1),
            |b| b.int_const(2),
            ty,
        );

        // The result should be a Phi node
        let node = b.graph.get(result);
        assert!(matches!(node, Some(IrNode::Phi { .. })), "if_else should produce a Phi node");

        // The graph should have Region, Branch, Jump, and Phi nodes
        let has_region = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Region { .. }));
        let has_branch = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Branch { .. }));
        let has_jump = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Jump { .. }));
        assert!(has_region, "should have Region nodes");
        assert!(has_branch, "should have Branch node");
        assert!(has_jump, "should have Jump nodes");
    }

    #[test]
    fn test_for_loop_builds_induction() {
        let mut b = IrBuilder::new("for_loop_test");
        let start = b.int_const(0);
        let limit = b.int_const(10);
        let ty = Type::I64;

        let ind_var = b.for_loop(start, limit, ty, |_b, _i| {
            // Empty body
        });

        // The induction variable should be a Phi node
        let node = b.graph.get(ind_var);
        assert!(matches!(node, Some(IrNode::Phi { .. })), "for_loop should return a Phi node for the induction variable");

        // The graph should have Region, Branch, and Add nodes
        let has_region = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Region { .. }));
        let has_branch = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Branch { .. }));
        let has_add = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Add { .. }));
        let has_lt = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Lt { .. }));
        assert!(has_region, "should have Region nodes");
        assert!(has_branch, "should have Branch node");
        assert!(has_add, "should have Add node (i+1)");
        assert!(has_lt, "should have Lt node (i < limit)");
    }

    #[test]
    fn test_while_cond_builds_loop() {
        let mut b = IrBuilder::new("while_cond_test");

        b.while_cond(|b| {
            let x = b.int_const(1);
            let y = b.int_const(2);
            b.lt(x, y)
        });

        // The graph should have Region, Branch, and Jump nodes
        let has_region = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Region { .. }));
        let has_branch = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Branch { .. }));
        let has_jump = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Jump { .. }));
        let has_lt = b.graph.iter().any(|(_, n)| matches!(n, IrNode::Lt { .. }));
        assert!(has_region, "should have Region nodes");
        assert!(has_branch, "should have Branch node");
        assert!(has_jump, "should have Jump nodes (back edge)");
        assert!(has_lt, "should have Lt node (condition)");
    }

    #[test]
    fn test_while_loop_with_state() {
        let mut b = IrBuilder::new("while_state_test");
        let init = b.int_const(0);
        let ty = Type::I64;

        let result = b.while_loop(
            init,
            |b, val| {
                let ten = b.int_const(10);
                b.lt(val, ten)
            },
            |b, val| {
                let one = b.int_const(1);
                b.add(val, one)
            },
            ty,
        );

        // The result should be a Phi node
        let node = b.graph.get(result);
        assert!(matches!(node, Some(IrNode::Phi { .. })), "while_loop should return a Phi node");
    }
}
