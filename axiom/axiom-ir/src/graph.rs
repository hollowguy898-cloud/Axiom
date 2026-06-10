//! The IR graph — the central data structure for sea-of-nodes.

use crate::nodes::{IrNode, NodeId, OwnershipRoot};

/// The Sea-of-Nodes IR graph.
///
/// All values, control flow, and memory operations live as nodes in this graph.
/// Edges are implicit: each node references its inputs by `NodeId`.
#[derive(Debug, Clone)]
pub struct IrGraph {
    /// Dense node storage. NodeId(n) indexes into this Vec.
    nodes: Vec<Option<IrNode>>,
    /// Free list for node ID reuse.
    free_ids: Vec<NodeId>,
    /// Function name this graph represents.
    pub name: String,
    /// Named variable → NodeId mapping for VarRef resolution.
    var_map: std::collections::HashMap<String, NodeId>,
    /// Next ownership root ID to allocate.
    next_root: u32,
}

impl IrGraph {
    pub fn new(name: &str) -> Self {
        let mut graph = Self {
            nodes: Vec::new(),
            free_ids: Vec::new(),
            name: name.to_string(),
            var_map: std::collections::HashMap::new(),
            next_root: 2, // 0 = global, 1 = stack
        };
        // Node 0 is always the Start node
        graph.push_node(IrNode::Start);
        graph
    }

    /// Push a node and return its ID.
    pub fn push_node(&mut self, node: IrNode) -> NodeId {
        if let Some(id) = self.free_ids.pop() {
            self.nodes[id.0 as usize] = Some(node);
            id
        } else {
            let id = NodeId::new(self.nodes.len() as u32);
            self.nodes.push(Some(node));
            id
        }
    }

    /// Get a node by ID.
    pub fn get(&self, id: NodeId) -> Option<&IrNode> {
        self.nodes.get(id.0 as usize).and_then(|opt| opt.as_ref())
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut IrNode> {
        self.nodes.get_mut(id.0 as usize).and_then(|opt| opt.as_mut())
    }

    /// Replace a node with a different node (used by optimizations).
    pub fn replace(&mut self, id: NodeId, node: IrNode) {
        if (id.0 as usize) < self.nodes.len() {
            self.nodes[id.0 as usize] = Some(node);
        }
    }

    /// Remove a node (mark it as dead).
    pub fn remove(&mut self, id: NodeId) {
        if (id.0 as usize) < self.nodes.len() {
            self.nodes[id.0 as usize] = None;
            self.free_ids.push(id);
        }
    }

    /// Iterate over all live (non-removed) nodes with their IDs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &IrNode)> {
        self.nodes.iter().enumerate().filter_map(|(i, opt)| {
            opt.as_ref().map(|node| (NodeId::new(i as u32), node))
        })
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    /// Register a named variable.
    pub fn define_var(&mut self, name: &str, id: NodeId) {
        self.var_map.insert(name.to_string(), id);
    }

    /// Look up a named variable.
    pub fn lookup_var(&self, name: &str) -> Option<NodeId> {
        self.var_map.get(name).copied()
    }

    /// Allocate a new ownership root.
    pub fn alloc_root(&mut self) -> OwnershipRoot {
        let root = OwnershipRoot::new(self.next_root);
        self.next_root += 1;
        root
    }

    /// Get the Start node ID.
    pub fn start_node(&self) -> NodeId {
        NodeId::new(0)
    }

    /// Replace all uses of `old` with `new` throughout the graph.
    pub fn replace_uses(&mut self, old: NodeId, new: NodeId) {
        for i in 0..self.nodes.len() {
            if let Some(node) = self.nodes[i].take() {
                let replaced = node.map_inputs(|id| if id == old { new } else { id });
                self.nodes[i] = Some(replaced);
            }
        }
    }

    /// Collect all users of a given node.
    pub fn users_of(&self, id: NodeId) -> Vec<NodeId> {
        self.iter()
            .filter(|(_, node)| node.inputs().contains(&id))
            .map(|(nid, _)| nid)
            .collect()
    }

    /// Validate graph integrity.
    pub fn validate(&self) -> Result<(), String> {
        for (id, node) in self.iter() {
            for input in node.inputs() {
                if self.get(input).is_none() {
                    return Err(format!("Node {} references non-existent input {}", id, input));
                }
            }
        }
        Ok(())
    }
}

/// Helper trait for mapping over node inputs.
impl IrNode {
    pub fn map_inputs<F: Fn(NodeId) -> NodeId>(self, f: F) -> Self {
        match self {
            IrNode::Add { lhs, rhs } => IrNode::Add { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Sub { lhs, rhs } => IrNode::Sub { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Mul { lhs, rhs } => IrNode::Mul { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Div { lhs, rhs } => IrNode::Div { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Rem { lhs, rhs } => IrNode::Rem { lhs: f(lhs), rhs: f(rhs) },
            IrNode::And { lhs, rhs } => IrNode::And { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Or  { lhs, rhs } => IrNode::Or  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Xor { lhs, rhs } => IrNode::Xor { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Shl { lhs, rhs } => IrNode::Shl { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Shr { lhs, rhs } => IrNode::Shr { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Sar { lhs, rhs } => IrNode::Sar { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Eq  { lhs, rhs } => IrNode::Eq  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Ne  { lhs, rhs } => IrNode::Ne  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Lt  { lhs, rhs } => IrNode::Lt  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Le  { lhs, rhs } => IrNode::Le  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Gt  { lhs, rhs } => IrNode::Gt  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Ge  { lhs, rhs } => IrNode::Ge  { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Neg { val } => IrNode::Neg { val: f(val) },
            IrNode::Not { val } => IrNode::Not { val: f(val) },
            IrNode::ZExt { val, to } => IrNode::ZExt { val: f(val), to },
            IrNode::SExt { val, to } => IrNode::SExt { val: f(val), to },
            IrNode::Trunc { val, to } => IrNode::Trunc { val: f(val), to },
            IrNode::BitCast { val, to } => IrNode::BitCast { val: f(val), to },
            IrNode::IntToPtr { val } => IrNode::IntToPtr { val: f(val) },
            IrNode::PtrToInt { val } => IrNode::PtrToInt { val: f(val) },
            IrNode::Load { addr, root, ty } => IrNode::Load { addr: f(addr), root, ty },
            IrNode::Store { addr, val, root, ty } => {
                IrNode::Store { addr: f(addr), val: f(val), root, ty }
            }
            IrNode::StackAlloc { size, align, root } => {
                IrNode::StackAlloc { size: f(size), align, root }
            }
            IrNode::Return { value } => {
                IrNode::Return { value: value.map(&f) }
            }
            IrNode::Branch { cond, true_block, false_block } => {
                IrNode::Branch { cond: f(cond), true_block: f(true_block), false_block: f(false_block) }
            }
            IrNode::Jump { target } => IrNode::Jump { target: f(target) },
            IrNode::Param { index, ty } => IrNode::Param { index, ty },
            IrNode::Phi { inputs, ty } => {
                IrNode::Phi {
                    inputs: inputs.into_iter().map(|(r, v)| (f(r), f(v))).collect(),
                    ty,
                }
            }
            IrNode::Call { func, args, ty } => {
                IrNode::Call { func, args: args.into_iter().map(&f).collect(), ty }
            }
            IrNode::CallIndirect { addr, args, ty } => {
                IrNode::CallIndirect { addr: f(addr), args: args.into_iter().map(&f).collect(), ty }
            }
            IrNode::TailCall { func, args, ty } => {
                IrNode::TailCall { func, args: args.into_iter().map(&f).collect(), ty }
            }
            IrNode::VarDef { name, init, root } => {
                IrNode::VarDef { name, init: f(init), root }
            }
            IrNode::VarSet { name, val, root } => {
                IrNode::VarSet { name, val: f(val), root }
            }
            IrNode::Extract { aggregate, index } => IrNode::Extract { aggregate: f(aggregate), index },
            IrNode::Insert { aggregate, index, value } => {
                IrNode::Insert { aggregate: f(aggregate), index, value: f(value) }
            }
            IrNode::Intrinsic { name, args, ty } => {
                IrNode::Intrinsic { name, args: args.into_iter().map(&f).collect(), ty }
            }
            IrNode::Owned { val, root } => IrNode::Owned { val: f(val), root },

            // ── Floating-Point Arithmetic ─────────────────────────────
            IrNode::FAdd { lhs, rhs } => IrNode::FAdd { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FSub { lhs, rhs } => IrNode::FSub { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FMul { lhs, rhs } => IrNode::FMul { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FDiv { lhs, rhs } => IrNode::FDiv { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FRem { lhs, rhs } => IrNode::FRem { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FNeg { val } => IrNode::FNeg { val: f(val) },
            IrNode::FAbs { val } => IrNode::FAbs { val: f(val) },
            IrNode::FSqrt { val } => IrNode::FSqrt { val: f(val) },

            // ── Floating-Point Comparison ─────────────────────────────
            IrNode::FEq { lhs, rhs } => IrNode::FEq { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FLt { lhs, rhs } => IrNode::FLt { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FLe { lhs, rhs } => IrNode::FLe { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FGt { lhs, rhs } => IrNode::FGt { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FGe { lhs, rhs } => IrNode::FGe { lhs: f(lhs), rhs: f(rhs) },
            IrNode::FNe { lhs, rhs } => IrNode::FNe { lhs: f(lhs), rhs: f(rhs) },

            // ── Floating-Point Conversion ─────────────────────────────
            IrNode::FpToSInt { val, to } => IrNode::FpToSInt { val: f(val), to },
            IrNode::SIntToFp { val, to } => IrNode::SIntToFp { val: f(val), to },
            IrNode::FpToUInt { val, to } => IrNode::FpToUInt { val: f(val), to },
            IrNode::UIntToFp { val, to } => IrNode::UIntToFp { val: f(val), to },

            // ── Floating-Point Misc ───────────────────────────────────
            IrNode::Copysign { lhs, rhs } => IrNode::Copysign { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Fmin { lhs, rhs } => IrNode::Fmin { lhs: f(lhs), rhs: f(rhs) },
            IrNode::Fmax { lhs, rhs } => IrNode::Fmax { lhs: f(lhs), rhs: f(rhs) },

            // ── Vector Operations ───────────────────────────────────────
            IrNode::VecBroadcast { val, lane_type, lane_count } => {
                IrNode::VecBroadcast { val: f(val), lane_type, lane_count }
            }
            IrNode::VecLoad { addr, root, lane_type, lane_count } => {
                IrNode::VecLoad { addr: f(addr), root, lane_type, lane_count }
            }
            IrNode::VecStore { addr, val, root, lane_type, lane_count } => {
                IrNode::VecStore { addr: f(addr), val: f(val), root, lane_type, lane_count }
            }
            IrNode::VecBinOp { op, lhs, rhs, lane_type, lane_count } => {
                IrNode::VecBinOp { op, lhs: f(lhs), rhs: f(rhs), lane_type, lane_count }
            }
            IrNode::VecUnOp { op, val, lane_type, lane_count } => {
                IrNode::VecUnOp { op, val: f(val), lane_type, lane_count }
            }
            IrNode::ExtractLane { val, index, lane_type } => {
                IrNode::ExtractLane { val: f(val), index, lane_type }
            }
            IrNode::InsertLane { val, index, elem, lane_type } => {
                IrNode::InsertLane { val: f(val), index, elem: f(elem), lane_type }
            }
            IrNode::VecReduce { op, val, lane_type, lane_count } => {
                IrNode::VecReduce { op, val: f(val), lane_type, lane_count }
            }
            IrNode::VecShuffle { val, mask, lane_type } => {
                IrNode::VecShuffle { val: f(val), mask, lane_type }
            }
            IrNode::VecGather { addrs, root, lane_type, lane_count } => {
                IrNode::VecGather { addrs: f(addrs), root, lane_type, lane_count }
            }
            IrNode::VecScatter { addrs, vals, root, lane_type, lane_count } => {
                IrNode::VecScatter { addrs: f(addrs), vals: f(vals), root, lane_type, lane_count }
            }

            // Nodes with no inputs stay unchanged
            other => other,
        }
    }
}
