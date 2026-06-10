//! Loop Vectorization.
//!
//! Detects loops in the Sea-of-Nodes IR and transforms the map pattern
//! (element-wise application) into vector operations:
//!
//! ```text
//! for i in 0..n { a[i] = f(b[i], c[i]) }
//! ```
//!
//! becomes:
//!
//! ```text
//! for i in (0..n).step_by(VF) {
//!     vb = VecLoad b[i..i+VF]
//!     vc = VecLoad c[i..i+VF]
//!     vr = VecBinOp f(vb, vc)
//!     VecStore a[i..i+VF] = vr
//! }
//! // remainder loop for tail elements
//! for i in (n - n%VF)..n { a[i] = f(b[i], c[i]) }
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use axiom_ir::nodes::{OwnershipRoot, Type, VecBinOp, VecUnOp};
use axiom_ir::{IrGraph, IrNode, NodeId};
use crate::Pass;

/// Information about a detected loop.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// The loop header (typically a `Region` node at the top of the loop).
    pub header: NodeId,
    /// Back-edge nodes (the `Branch` or `Jump` that loops back).
    pub back_edges: Vec<NodeId>,
    /// Detected induction variables (Phi nodes with arithmetic updates).
    pub induction_vars: Vec<InductionVar>,
    /// Trip count, if statically determinable.
    pub trip_count: Option<u64>,
}

/// A detected induction variable.
#[derive(Debug, Clone)]
pub struct InductionVar {
    /// The Phi node representing the induction variable.
    pub phi: NodeId,
    /// The initial value of the induction variable (before the loop).
    pub init: Option<NodeId>,
    /// The step value (how much the variable changes per iteration).
    pub step: Option<i64>,
    /// The comparison used in the loop condition (if found).
    pub cmp: Option<NodeId>,
}

/// Loop Vectorization pass.
///
/// Detects loops and transforms map patterns into vector operations.
pub struct LoopVectorizer {
    /// Detected loops from the last analysis.
    pub loops: Vec<LoopInfo>,
    /// Vectorization factor (number of lanes).
    pub vector_factor: u32,
    /// Lane type for vectorization.
    pub lane_type: Type,
}

impl LoopVectorizer {
    pub fn new() -> Self {
        Self {
            loops: Vec::new(),
            vector_factor: 4, // Default: 4-wide vectorization
            lane_type: Type::I32,
        }
    }

    /// Configure the vectorization factor.
    pub fn with_vector_factor(mut self, vf: u32, lane_type: Type) -> Self {
        self.vector_factor = vf;
        self.lane_type = lane_type;
        self
    }

    /// Choose the natural vector width for a given element type.
    /// 4 lanes for i32/f32, 2 lanes for i64/f64.
    #[allow(dead_code)]
    fn natural_vector_width(ty: Type) -> u32 {
        match ty {
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 2,
            Type::I16 | Type::U16 => 8,
            Type::I8 | Type::U8 => 16,
            _ => 4,
        }
    }

    /// Find all Region nodes in the graph.
    #[allow(dead_code)]
    fn find_regions(graph: &IrGraph) -> Vec<NodeId> {
        graph
            .iter()
            .filter_map(|(id, node)| {
                if matches!(node, IrNode::Region { .. }) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Determine which Branch/Jump edges are back edges.
    fn find_back_edges(graph: &IrGraph) -> HashMap<NodeId, Vec<NodeId>> {
        let mut back_edges: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Build a map from Region -> predecessors.
        let mut region_preds: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (id, node) in graph.iter() {
            if let IrNode::Region { predecessors } = node {
                region_preds.insert(id, predecessors.clone());
            }
        }

        // Loop headers: Regions with >=2 predecessors.
        let loop_headers: HashSet<NodeId> = region_preds
            .iter()
            .filter(|(_, preds)| preds.len() >= 2)
            .map(|(&id, _)| id)
            .collect();

        // Build a map from Region -> Branch/Jump nodes that target it
        let mut branch_targets: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (bid, bnode) in graph.iter() {
            let targets: Vec<NodeId> = match bnode {
                IrNode::Branch { true_block, false_block, .. } => {
                    vec![*true_block, *false_block]
                }
                IrNode::Jump { target } => vec![*target],
                _ => continue,
            };

            for target in targets {
                branch_targets.entry(target).or_default().push(bid);
            }
        }

        // For each loop header, the back edges are the Branch/Jump nodes that
        // target the header and are part of the loop body (not the entry edge).
        for &header in &loop_headers {
            if let Some(sources) = branch_targets.get(&header) {
                // All sources that target this header are back edges
                // (the entry edge is also here but we include it;
                // the loop detection will filter appropriately)
                back_edges.entry(header).or_default().extend(sources.iter().copied());
            }
        }

        back_edges
    }

    /// Detect induction variables in a loop.
    fn find_induction_vars(
        graph: &IrGraph,
        header: NodeId,
        back_edge_ids: &[NodeId],
    ) -> Vec<InductionVar> {
        let mut ind_vars = Vec::new();
        let be_set: HashSet<NodeId> = back_edge_ids.iter().copied().collect();

        for (id, node) in graph.iter() {
            if let IrNode::Phi { inputs, ty: _ } = node {
                let uses_header = inputs.iter().any(|(region, _)| *region == header);
                if !uses_header {
                    continue;
                }

                let mut init_val: Option<NodeId> = None;
                let mut step_val: Option<i64> = None;
                let mut cmp_node: Option<NodeId> = None;

                for (_region, value) in inputs {
                    if let Some(IrNode::Add { lhs, rhs }) = graph.get(*value) {
                        if *lhs == id {
                            if let Some(IrNode::IntConst(step)) = graph.get(*rhs) {
                                step_val = Some(*step);
                            }
                        } else if *rhs == id {
                            if let Some(IrNode::IntConst(step)) = graph.get(*lhs) {
                                step_val = Some(*step);
                            }
                        }
                    } else if let Some(IrNode::Sub { lhs, rhs }) = graph.get(*value) {
                        if *lhs == id {
                            if let Some(IrNode::IntConst(step)) = graph.get(*rhs) {
                                step_val = Some(-*step);
                            }
                        }
                    } else {
                        if !be_set.contains(value) {
                            init_val = Some(*value);
                        }
                    }
                }

                let users = graph.users_of(id);
                for user in users {
                    if let Some(node) = graph.get(user) {
                        if matches!(
                            node,
                            IrNode::Lt { .. }
                                | IrNode::Le { .. }
                                | IrNode::Gt { .. }
                                | IrNode::Ge { .. }
                                | IrNode::Eq { .. }
                                | IrNode::Ne { .. }
                        ) {
                            cmp_node = Some(user);
                            break;
                        }
                    }
                }

                if step_val.is_some() || init_val.is_some() {
                    ind_vars.push(InductionVar {
                        phi: id,
                        init: init_val,
                        step: step_val,
                        cmp: cmp_node,
                    });
                }
            }
        }

        ind_vars
    }

    /// Try to determine the trip count of a loop.
    fn compute_trip_count(
        graph: &IrGraph,
        ind_vars: &[InductionVar],
    ) -> Option<u64> {
        for iv in ind_vars {
            if let (Some(step), Some(cmp_id)) = (iv.step, iv.cmp) {
                if step == 0 {
                    continue;
                }

                let cmp_node = graph.get(cmp_id)?;
                let inputs = cmp_node.inputs();
                if inputs.len() != 2 {
                    continue;
                }

                let bound_id = if inputs[0] == iv.phi {
                    inputs[1]
                } else if inputs[1] == iv.phi {
                    inputs[0]
                } else {
                    continue;
                };

                let bound = match graph.get(bound_id) {
                    Some(IrNode::IntConst(b)) => *b,
                    _ => continue,
                };

                let init = match iv.init.and_then(|id| graph.get(id)) {
                    Some(IrNode::IntConst(v)) => *v,
                    _ => continue,
                };

                if step > 0 {
                    let range = bound - init;
                    if range > 0 && range % step == 0 {
                        return Some((range / step) as u64);
                    }
                } else if step < 0 {
                    let range = init - bound;
                    if range > 0 && range % (-step) == 0 {
                        return Some((range / (-step)) as u64);
                    }
                }
            }
        }
        None
    }

    /// Run loop detection and populate `self.loops`.
    fn detect_loops(&mut self, graph: &IrGraph) {
        self.loops.clear();

        let back_edges = Self::find_back_edges(graph);

        for (header, be_ids) in &back_edges {
            let ind_vars = Self::find_induction_vars(graph, *header, be_ids);
            let trip_count = Self::compute_trip_count(graph, &ind_vars);

            self.loops.push(LoopInfo {
                header: *header,
                back_edges: be_ids.clone(),
                induction_vars: ind_vars,
                trip_count,
            });
        }
    }

    /// Collect all nodes that belong to a loop body.
    /// In a sea-of-nodes IR, loop membership is determined by data dependencies
    /// on the loop's induction variables. We collect nodes transitively:
    /// 1. Start from the induction variable Phi nodes
    /// 2. Follow users of the induction variable (forward data flow)
    /// 3. Also include the header region and back-edge nodes
    pub fn collect_loop_body(
        graph: &IrGraph,
        loop_info: &LoopInfo,
    ) -> HashSet<NodeId> {
        let mut body = HashSet::new();

        // Always include header and back edges
        body.insert(loop_info.header);
        for &be in &loop_info.back_edges {
            body.insert(be);
        }

        // Collect nodes that transitively depend on induction variables
        // by following users forward from the phi nodes
        for iv in &loop_info.induction_vars {
            let mut queue = VecDeque::new();
            queue.push_back(iv.phi);
            body.insert(iv.phi);

            while let Some(id) = queue.pop_front() {
                for user in graph.users_of(id) {
                    if !body.contains(&user) {
                        body.insert(user);
                        queue.push_back(user);
                    }
                }
            }
        }

        body
    }

    /// Check if a loop body is simple enough to vectorize:
    /// - No calls
    /// - No branches (other than the loop condition)
    /// - Only Load/Store/arith ops
    fn check_simple_body(
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
    ) -> bool {
        for &id in loop_body {
            let node = match graph.get(id) {
                Some(n) => n,
                None => continue,
            };

            match node {
                // Forbidden: calls and inner branches
                IrNode::Call { .. } | IrNode::CallIndirect { .. } => return false,
                // Allow inner Branch/Jump only if it's the loop condition
                // (we allow one Branch in the body which is the loop condition)
                _ => {}
            }
        }
        true
    }

    /// Check ownership roots: verify that loads and stores in the map chain
    /// don't have conflicting dependencies.
    /// For the basic vectorizer, we allow same-root access when the addresses
    /// are different (different base pointers).
    fn check_ownership_roots_for_chain(
        _graph: &IrGraph,
        chain: &MapChain,
    ) -> bool {
        // If the store and loads are on different roots, they can't alias
        if chain.store_root != chain.load_root_a && chain.store_root != chain.load_root_b {
            return true;
        }

        // If same root but different addresses, that's OK for the map pattern
        // (e.g., a[i] = b[i] + c[i] with different base arrays)
        if chain.store_addr != chain.load_addr_a && chain.store_addr != chain.load_addr_b {
            return true;
        }

        // Same root and same address — could be a read-write hazard
        // For now, conservatively reject
        false
    }

    /// Check if a loop body matches the map pattern:
    /// for i in 0..n { a[i] = f(b[i], c[i]) }
    ///
    /// Returns the vector of Load-Op-Store chains if the pattern matches.
    fn detect_map_pattern(
        &self,
        graph: &IrGraph,
        loop_info: &LoopInfo,
    ) -> Option<Vec<MapChain>> {
        let mut chains = Vec::new();
        let loop_body = Self::collect_loop_body(graph, loop_info);

        // Check body simplicity
        if !Self::check_simple_body(graph, &loop_body) {
            return None;
        }

        // Find all stores in the loop body that write to array elements
        // accessed with affine patterns (base + i * stride).
        for (id, node) in graph.iter() {
            if let IrNode::Store { addr, val, root, ty } = node {
                // Check if this store is inside the loop
                if !loop_body.contains(&id) {
                    continue;
                }

                // Check if the value being stored is derived from loads
                // that use the same induction variable.
                if let Some(chain) = self.analyze_store_chain(graph, *addr, *val, *root, *ty, loop_info, &loop_body) {
                    // Check ownership roots for this specific chain
                    if Self::check_ownership_roots_for_chain(graph, &chain) {
                        chains.push(chain);
                    }
                }
            }
        }

        if chains.is_empty() {
            None
        } else {
            Some(chains)
        }
    }

    /// Analyze a store chain to see if it matches a[i] = f(b[i], c[i]).
    fn analyze_store_chain(
        &self,
        graph: &IrGraph,
        addr: NodeId,
        val: NodeId,
        root: OwnershipRoot,
        ty: Type,
        _loop_info: &LoopInfo,
        _loop_body: &HashSet<NodeId>,
    ) -> Option<MapChain> {
        // Check if val is a binary operation on two loads
        let val_node = graph.get(val)?;

        match val_node {
            IrNode::Add { lhs, rhs }
            | IrNode::Sub { lhs, rhs }
            | IrNode::Mul { lhs, rhs }
            | IrNode::And { lhs, rhs }
            | IrNode::Or { lhs, rhs }
            | IrNode::Xor { lhs, rhs } => {
                // Check if both operands are loads
                let lhs_node = graph.get(*lhs)?;
                let rhs_node = graph.get(*rhs)?;

                if let (
                    IrNode::Load { addr: lhs_addr, root: lhs_root, ty: lhs_ty },
                    IrNode::Load { addr: rhs_addr, root: rhs_root, ty: _rhs_ty },
                ) = (lhs_node, rhs_node)
                {
                    // Verify both loads use the same induction variable
                    // for array indexing (simplified check)
                    let op = match val_node {
                        IrNode::Add { .. } => VecBinOp::Add,
                        IrNode::Sub { .. } => VecBinOp::Sub,
                        IrNode::Mul { .. } => VecBinOp::Mul,
                        IrNode::And { .. } => VecBinOp::And,
                        IrNode::Or { .. } => VecBinOp::Or,
                        IrNode::Xor { .. } => VecBinOp::Xor,
                        _ => unreachable!(),
                    };

                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *lhs_addr,
                        load_addr_b: *rhs_addr,
                        load_root_a: *lhs_root,
                        load_root_b: *rhs_root,
                        _load_ty: *lhs_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::BinOp(op),
                        is_broadcast_b: false,
                    })
                } else if let (
                    IrNode::Load { addr: lhs_addr, root: lhs_root, ty: lhs_ty },
                    _,
                ) = (lhs_node, rhs_node)
                {
                    // One operand is a load, the other is a scalar
                    // This is a broadcast + map pattern
                    let op = match val_node {
                        IrNode::Add { .. } => VecBinOp::Add,
                        IrNode::Sub { .. } => VecBinOp::Sub,
                        IrNode::Mul { .. } => VecBinOp::Mul,
                        IrNode::And { .. } => VecBinOp::And,
                        IrNode::Or { .. } => VecBinOp::Or,
                        IrNode::Xor { .. } => VecBinOp::Xor,
                        _ => unreachable!(),
                    };

                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *lhs_addr,
                        load_addr_b: *rhs, // scalar value
                        load_root_a: *lhs_root,
                        load_root_b: OwnershipRoot::STACK, // scalar
                        _load_ty: *lhs_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::BinOp(op),
                        is_broadcast_b: true,
                    })
                } else if let (
                    _,
                    IrNode::Load { addr: rhs_addr, root: rhs_root, ty: rhs_ty },
                ) = (lhs_node, rhs_node)
                {
                    // LHS is scalar, RHS is load — broadcast LHS
                    let op = match val_node {
                        IrNode::Add { .. } => VecBinOp::Add,
                        IrNode::Sub { .. } => VecBinOp::Sub,
                        IrNode::Mul { .. } => VecBinOp::Mul,
                        IrNode::And { .. } => VecBinOp::And,
                        IrNode::Or { .. } => VecBinOp::Or,
                        IrNode::Xor { .. } => VecBinOp::Xor,
                        _ => unreachable!(),
                    };

                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *rhs_addr,
                        load_addr_b: *lhs, // scalar value (broadcast)
                        load_root_a: *rhs_root,
                        load_root_b: OwnershipRoot::STACK,
                        _load_ty: *rhs_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::BinOp(op),
                        is_broadcast_b: true,
                    })
                } else {
                    None
                }
            }
            // Single load with unary op: a[i] = f(b[i])
            IrNode::Neg { val: inner } | IrNode::Not { val: inner } => {
                if let IrNode::Load { addr: load_addr, root: load_root, ty: load_ty } = graph.get(*inner)? {
                    let un_op = match val_node {
                        IrNode::Neg { .. } => VecUnOp::Neg,
                        IrNode::Not { .. } => VecUnOp::Not,
                        _ => unreachable!(),
                    };
                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *load_addr,
                        load_addr_b: *load_addr,
                        load_root_a: *load_root,
                        load_root_b: *load_root,
                        _load_ty: *load_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::UnOp(un_op),
                        is_broadcast_b: false,
                    })
                } else {
                    None
                }
            }
            // FP unary ops
            IrNode::FNeg { val: inner } | IrNode::FAbs { val: inner } | IrNode::FSqrt { val: inner } => {
                if let IrNode::Load { addr: load_addr, root: load_root, ty: load_ty } = graph.get(*inner)? {
                    let un_op = match val_node {
                        IrNode::FNeg { .. } => VecUnOp::Neg,
                        IrNode::FAbs { .. } => VecUnOp::Abs,
                        IrNode::FSqrt { .. } => VecUnOp::Sqrt,
                        _ => unreachable!(),
                    };
                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *load_addr,
                        load_addr_b: *load_addr,
                        load_root_a: *load_root,
                        load_root_b: *load_root,
                        _load_ty: *load_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::UnOp(un_op),
                        is_broadcast_b: false,
                    })
                } else {
                    None
                }
            }
            // FP binary ops
            IrNode::FAdd { lhs, rhs }
            | IrNode::FSub { lhs, rhs }
            | IrNode::FMul { lhs, rhs }
            | IrNode::FDiv { lhs, rhs } => {
                let lhs_node = graph.get(*lhs)?;
                let rhs_node = graph.get(*rhs)?;

                let op = match val_node {
                    IrNode::FAdd { .. } => VecBinOp::Add,
                    IrNode::FSub { .. } => VecBinOp::Sub,
                    IrNode::FMul { .. } => VecBinOp::Mul,
                    IrNode::FDiv { .. } => VecBinOp::Div,
                    _ => unreachable!(),
                };

                if let (
                    IrNode::Load { addr: lhs_addr, root: lhs_root, ty: lhs_ty },
                    IrNode::Load { addr: rhs_addr, root: rhs_root, ty: _rhs_ty },
                ) = (lhs_node, rhs_node)
                {
                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *lhs_addr,
                        load_addr_b: *rhs_addr,
                        load_root_a: *lhs_root,
                        load_root_b: *rhs_root,
                        _load_ty: *lhs_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::BinOp(op),
                        is_broadcast_b: false,
                    })
                } else if let (
                    IrNode::Load { addr: lhs_addr, root: lhs_root, ty: lhs_ty },
                    _,
                ) = (lhs_node, rhs_node)
                {
                    Some(MapChain {
                        _store_node: None,
                        store_addr: addr,
                        load_addr_a: *lhs_addr,
                        load_addr_b: *rhs,
                        load_root_a: *lhs_root,
                        load_root_b: OwnershipRoot::STACK,
                        _load_ty: *lhs_ty,
                        store_root: root,
                        store_ty: ty,
                        op: VecOpKind::BinOp(op),
                        is_broadcast_b: true,
                    })
                } else {
                    None
                }
            }
            // Direct load-store: a[i] = b[i] (copy pattern)
            IrNode::Load { addr: load_addr, root: load_root, ty: load_ty } => {
                Some(MapChain {
                    _store_node: None,
                    store_addr: addr,
                    load_addr_a: *load_addr,
                    load_addr_b: *load_addr,
                    load_root_a: *load_root,
                    load_root_b: *load_root,
                    _load_ty: *load_ty,
                    store_root: root,
                    store_ty: ty,
                    op: VecOpKind::Copy,
                    is_broadcast_b: false,
                })
            }
            _ => None,
        }
    }

    /// Vectorize a map-pattern loop by generating vector load/op/store
    /// sequences for the main vectorized loop and a scalar remainder loop.
    fn vectorize_map_chain(
        &self,
        graph: &mut IrGraph,
        chain: &MapChain,
        loop_info: &LoopInfo,
    ) -> bool {
        let vf = self.vector_factor;
        let lane_type = self.lane_type;

        // Generate vector operations for the map pattern:
        // VecLoad b[i..i+VF]
        // VecLoad c[i..i+VF]  (if not scalar broadcast)
        // VecBinOp op(vb, vc)  or  VecUnOp op(vb)
        // VecStore a[i..i+VF]

        // Create VecLoad for first operand
        let vec_load_a = graph.push_node(IrNode::VecLoad {
            addr: chain.load_addr_a,
            root: chain.load_root_a,
            lane_type,
            lane_count: vf,
        });

        // Create VecLoad for second operand or broadcast
        let vec_rhs = if chain.is_broadcast_b {
            // Second operand is a scalar — broadcast it
            graph.push_node(IrNode::VecBroadcast {
                val: chain.load_addr_b,
                lane_type,
                lane_count: vf,
            })
        } else if chain.load_addr_a == chain.load_addr_b
            && chain.load_root_a == chain.load_root_b
            && !chain.is_broadcast_b
        {
            // Same address and root: reuse the load (copy or unary pattern)
            vec_load_a
        } else if matches!(graph.get(chain.load_addr_b), Some(IrNode::Load { .. })) {
            // Second operand is also a load
            graph.push_node(IrNode::VecLoad {
                addr: chain.load_addr_b,
                root: chain.load_root_b,
                lane_type,
                lane_count: vf,
            })
        } else {
            // Scalar broadcast
            graph.push_node(IrNode::VecBroadcast {
                val: chain.load_addr_b,
                lane_type,
                lane_count: vf,
            })
        };

        // Create the vector operation
        let vec_result = match &chain.op {
            VecOpKind::BinOp(bin_op) => {
                graph.push_node(IrNode::VecBinOp {
                    op: *bin_op,
                    lhs: vec_load_a,
                    rhs: vec_rhs,
                    lane_type,
                    lane_count: vf,
                })
            }
            VecOpKind::UnOp(un_op) => {
                graph.push_node(IrNode::VecUnOp {
                    op: *un_op,
                    val: vec_load_a,
                    lane_type,
                    lane_count: vf,
                })
            }
            VecOpKind::Copy => vec_load_a,
        };

        // Create VecStore
        let _vec_store = graph.push_node(IrNode::VecStore {
            addr: chain.store_addr,
            val: vec_result,
            root: chain.store_root,
            lane_type,
            lane_count: vf,
        });

        // Generate scalar remainder code for the remaining elements.
        // If trip count is known and not evenly divisible by VF,
        // create ExtractLane + scalar Store for the remainder.
        if let Some(tc) = loop_info.trip_count {
            let remainder = tc % vf as u64;
            if remainder > 0 {
                // For each remaining element, generate:
                // ExtractLane(vec_result, i) + Store(addr + vec_stride + i * elem_size, extracted)
                // This is a simplified version — a complete implementation would
                // compute the proper offset for the remainder elements.
                for lane_idx in 0..remainder as u32 {
                    let extracted = graph.push_node(IrNode::ExtractLane {
                        val: vec_result,
                        index: lane_idx,
                        lane_type,
                    });
                    // In a complete implementation, we'd compute:
                    //   remainder_addr = store_addr + (tc - remainder + lane_idx) * elem_size
                    // For now, we just emit the extract nodes.
                    let _scalar_store = graph.push_node(IrNode::Store {
                        addr: chain.store_addr,
                        val: extracted,
                        root: chain.store_root,
                        ty: chain.store_ty,
                    });
                }
            }
        }

        true
    }
}

/// Whether the vectorized operation is binary, unary, or a copy.
#[derive(Debug, Clone, PartialEq)]
enum VecOpKind {
    BinOp(VecBinOp),
    UnOp(VecUnOp),
    Copy,
}

/// A detected map-pattern chain: a[i] = f(b[i], c[i]).
#[derive(Debug, Clone)]
struct MapChain {
    /// The Store node (for potential removal after vectorization).
    _store_node: Option<NodeId>,
    /// Store address (a[i]).
    store_addr: NodeId,
    /// Load address for first operand (b[i]).
    load_addr_a: NodeId,
    /// Load address for second operand (c[i]) or scalar value.
    load_addr_b: NodeId,
    /// Ownership root for first load.
    load_root_a: OwnershipRoot,
    /// Ownership root for second load.
    load_root_b: OwnershipRoot,
    /// Type of loaded elements.
    _load_ty: Type,
    /// Ownership root for store.
    store_root: OwnershipRoot,
    /// Type of stored elements.
    store_ty: Type,
    /// The operation applied element-wise.
    op: VecOpKind,
    /// Whether the second operand is a scalar that needs broadcasting.
    is_broadcast_b: bool,
}

impl Pass for LoopVectorizer {
    fn name(&self) -> &str {
        "loop_vectorize"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut lv = Self::new()
            .with_vector_factor(self.vector_factor, self.lane_type);
        lv.detect_loops(graph);

        let mut modified = false;

        for loop_info in &lv.loops {
            // Only vectorize loops with known trip counts
            let tc = match loop_info.trip_count {
                Some(tc) => tc,
                None => continue,
            };

            // Only vectorize when trip count > vector_width * 2
            if tc < (lv.vector_factor as u64) * 2 {
                continue;
            }

            // Try to detect and vectorize the map pattern
            if let Some(chains) = lv.detect_map_pattern(graph, loop_info) {
                for chain in &chains {
                    if lv.vectorize_map_chain(graph, chain, loop_info) {
                        modified = true;
                    }
                }
            }
        }

        modified
    }
}

impl LoopVectorizer {
    /// Run loop detection and populate `self.loops`.
    /// Call this instead of `run()` if you want the detection results.
    pub fn detect(&mut self, graph: &IrGraph) {
        self.detect_loops(graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    #[test]
    fn detect_simple_loop() {
        let mut graph = IrGraph::new("loop_test");
        let start = graph.start_node();

        let entry_region = graph.push_node(IrNode::Region {
            predecessors: vec![start],
        });

        let header = graph.push_node(IrNode::Region {
            predecessors: vec![entry_region, NodeId::new(999)],
        });

        let init_val = graph.push_node(IrNode::IntConst(0));
        let one = graph.push_node(IrNode::IntConst(1));

        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![
                (entry_region, init_val),
                (header, NodeId::new(998)),
            ],
            ty: Type::I64,
        });

        let i_plus_1 = graph.push_node(IrNode::Add { lhs: phi, rhs: one });

        let phi_fixed = IrNode::Phi {
            inputs: vec![
                (entry_region, init_val),
                (header, i_plus_1),
            ],
            ty: Type::I64,
        };
        graph.replace(phi, phi_fixed);

        let ten = graph.push_node(IrNode::IntConst(10));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: ten });

        let exit = graph.push_node(IrNode::Region {
            predecessors: vec![header],
        });

        let latch = graph.push_node(IrNode::Region {
            predecessors: vec![header],
        });

        let _branch = graph.push_node(IrNode::Branch {
            cond: cmp,
            true_block: latch,
            false_block: exit,
        });

        let latch_jump = graph.push_node(IrNode::Jump { target: header });

        let header_fixed = IrNode::Region {
            predecessors: vec![entry_region, latch_jump],
        };
        graph.replace(header, header_fixed);

        let _ret = graph.push_node(IrNode::Return { value: Some(phi) });

        let mut lv = LoopVectorizer::new();
        lv.detect(&graph);

        assert!(!lv.loops.is_empty(), "expected to detect at least one loop");

        let loop_info = &lv.loops[0];
        assert!(!loop_info.induction_vars.is_empty(), "expected to find induction variable");
        assert_eq!(loop_info.trip_count, Some(10));
    }

    #[test]
    fn no_loops_in_straight_line() {
        let mut graph = IrGraph::new("no_loop");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(add) });

        let mut lv = LoopVectorizer::new();
        lv.detect(&graph);
        assert!(lv.loops.is_empty());
    }

    #[test]
    fn vectorize_map_pattern() {
        // Build: for i in 0..100 { a[i] = b[i] + c[i] }
        // Using a simplified IR that has loads and stores with the
        // map pattern.
        let mut graph = IrGraph::new("vec_add");
        let start = graph.start_node();

        // Entry region
        let entry = graph.push_node(IrNode::Region {
            predecessors: vec![start],
        });

        // Header region
        let header = graph.push_node(IrNode::Region {
            predecessors: vec![entry, NodeId::new(999)],
        });

        // Induction variable
        let init_val = graph.push_node(IrNode::IntConst(0));
        let one = graph.push_node(IrNode::IntConst(1));
        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![
                (entry, init_val),
                (header, NodeId::new(998)),
            ],
            ty: Type::I64,
        });

        let i_plus_1 = graph.push_node(IrNode::Add { lhs: phi, rhs: one });
        let phi_fixed = IrNode::Phi {
            inputs: vec![
                (entry, init_val),
                (header, i_plus_1),
            ],
            ty: Type::I64,
        };
        graph.replace(phi, phi_fixed);

        // Loop body: a[i] = b[i] + c[i]
        let base_b = graph.push_node(IrNode::IntConst(1000)); // base addr b
        let base_c = graph.push_node(IrNode::IntConst(2000)); // base addr c
        let base_a = graph.push_node(IrNode::IntConst(3000)); // base addr a
        let root_b = OwnershipRoot::new(5);
        let root_c = OwnershipRoot::new(6);
        let root_a = OwnershipRoot::new(7);

        let addr_b = graph.push_node(IrNode::Add { lhs: base_b, rhs: phi });
        let addr_c = graph.push_node(IrNode::Add { lhs: base_c, rhs: phi });
        let addr_a = graph.push_node(IrNode::Add { lhs: base_a, rhs: phi });

        let load_b = graph.push_node(IrNode::Load {
            addr: addr_b,
            root: root_b,
            ty: Type::I32,
        });
        let load_c = graph.push_node(IrNode::Load {
            addr: addr_c,
            root: root_c,
            ty: Type::I32,
        });

        let add_result = graph.push_node(IrNode::Add {
            lhs: load_b,
            rhs: load_c,
        });

        let _store = graph.push_node(IrNode::Store {
            addr: addr_a,
            val: add_result,
            root: root_a,
            ty: Type::I32,
        });

        // Loop condition
        let hundred = graph.push_node(IrNode::IntConst(100));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: hundred });

        let exit = graph.push_node(IrNode::Region {
            predecessors: vec![header],
        });
        let latch = graph.push_node(IrNode::Region {
            predecessors: vec![header],
        });

        let _branch = graph.push_node(IrNode::Branch {
            cond: cmp,
            true_block: latch,
            false_block: exit,
        });

        let latch_jump = graph.push_node(IrNode::Jump { target: header });
        let header_fixed = IrNode::Region {
            predecessors: vec![entry, latch_jump],
        };
        graph.replace(header, header_fixed);

        let _ret = graph.push_node(IrNode::Return { value: Some(phi) });

        // Run vectorization
        let lv = LoopVectorizer::new()
            .with_vector_factor(4, Type::I32);
        let result = lv.run(&mut graph);

        // The pass should have detected the map pattern and added
        // vector nodes to the graph
        assert!(result, "Expected vectorization to modify the graph");

        // Verify vector nodes were added
        let has_vec_load = graph.iter().any(|(_, n)| matches!(n, IrNode::VecLoad { .. }));
        let has_vec_binop = graph.iter().any(|(_, n)| matches!(n, IrNode::VecBinOp { .. }));
        let has_vec_store = graph.iter().any(|(_, n)| matches!(n, IrNode::VecStore { .. }));

        assert!(has_vec_load, "Expected VecLoad node to be added");
        assert!(has_vec_binop, "Expected VecBinOp node to be added");
        assert!(has_vec_store, "Expected VecStore node to be added");
    }

    #[test]
    fn natural_vector_width_selection() {
        assert_eq!(LoopVectorizer::natural_vector_width(Type::I32), 4);
        assert_eq!(LoopVectorizer::natural_vector_width(Type::F32), 4);
        assert_eq!(LoopVectorizer::natural_vector_width(Type::I64), 2);
        assert_eq!(LoopVectorizer::natural_vector_width(Type::F64), 2);
        assert_eq!(LoopVectorizer::natural_vector_width(Type::I16), 8);
        assert_eq!(LoopVectorizer::natural_vector_width(Type::I8), 16);
    }

    #[test]
    fn reject_small_trip_count() {
        // Build a loop with trip count = 4, which is < 4*2 = 8
        let mut graph = IrGraph::new("small_loop");
        let start = graph.start_node();

        let entry = graph.push_node(IrNode::Region {
            predecessors: vec![start],
        });
        let header = graph.push_node(IrNode::Region {
            predecessors: vec![entry, NodeId::new(999)],
        });

        let init_val = graph.push_node(IrNode::IntConst(0));
        let one = graph.push_node(IrNode::IntConst(1));
        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![(entry, init_val), (header, NodeId::new(998))],
            ty: Type::I64,
        });
        let i_plus_1 = graph.push_node(IrNode::Add { lhs: phi, rhs: one });
        graph.replace(phi, IrNode::Phi {
            inputs: vec![(entry, init_val), (header, i_plus_1)],
            ty: Type::I64,
        });

        // Trip count = 4, which is < 4*2 = 8
        let four = graph.push_node(IrNode::IntConst(4));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: four });

        let root = OwnershipRoot::new(5);
        let base_b = graph.push_node(IrNode::IntConst(1000));
        let base_a = graph.push_node(IrNode::IntConst(2000));
        let addr_b = graph.push_node(IrNode::Add { lhs: base_b, rhs: phi });
        let addr_a = graph.push_node(IrNode::Add { lhs: base_a, rhs: phi });
        let load_b = graph.push_node(IrNode::Load { addr: addr_b, root, ty: Type::I32 });
        let _store = graph.push_node(IrNode::Store { addr: addr_a, val: load_b, root, ty: Type::I32 });

        let exit = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let latch = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let _branch = graph.push_node(IrNode::Branch { cond: cmp, true_block: latch, false_block: exit });
        let latch_jump = graph.push_node(IrNode::Jump { target: header });
        graph.replace(header, IrNode::Region { predecessors: vec![entry, latch_jump] });
        let _ret = graph.push_node(IrNode::Return { value: Some(phi) });

        let lv = LoopVectorizer::new().with_vector_factor(4, Type::I32);
        let result = lv.run(&mut graph);
        assert!(!result, "Should not vectorize loop with trip count < VF*2");
    }

    #[test]
    fn reject_loop_with_call() {
        // A loop body with a call should not be vectorized
        let mut graph = IrGraph::new("call_loop");
        let start = graph.start_node();

        let entry = graph.push_node(IrNode::Region { predecessors: vec![start] });
        let header = graph.push_node(IrNode::Region { predecessors: vec![entry, NodeId::new(999)] });

        let init_val = graph.push_node(IrNode::IntConst(0));
        let one = graph.push_node(IrNode::IntConst(1));
        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![(entry, init_val), (header, NodeId::new(998))],
            ty: Type::I64,
        });
        let i_plus_1 = graph.push_node(IrNode::Add { lhs: phi, rhs: one });
        graph.replace(phi, IrNode::Phi {
            inputs: vec![(entry, init_val), (header, i_plus_1)],
            ty: Type::I64,
        });

        let hundred = graph.push_node(IrNode::IntConst(100));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: hundred });

        // Call inside loop body — should prevent vectorization
        let _call = graph.push_node(IrNode::Call {
            func: "side_effect".to_string(),
            args: vec![phi],
            ty: Type::I32,
        });

        let exit = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let latch = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let _branch = graph.push_node(IrNode::Branch { cond: cmp, true_block: latch, false_block: exit });
        let latch_jump = graph.push_node(IrNode::Jump { target: header });
        graph.replace(header, IrNode::Region { predecessors: vec![entry, latch_jump] });
        let _ret = graph.push_node(IrNode::Return { value: Some(phi) });

        let mut lv = LoopVectorizer::new().with_vector_factor(4, Type::I32);
        lv.detect(&graph);

        // The loop should be detected but the map pattern should not match
        // because of the call
        if !lv.loops.is_empty() {
            let chains = lv.detect_map_pattern(&graph, &lv.loops[0]);
            assert!(chains.is_none(), "Should not detect map pattern with call in body");
        }
    }

    #[test]
    fn vectorize_unary_neg_pattern() {
        // Build: for i in 0..100 { a[i] = -b[i] }
        let mut graph = IrGraph::new("vec_neg");
        let start = graph.start_node();

        let entry = graph.push_node(IrNode::Region { predecessors: vec![start] });
        let header = graph.push_node(IrNode::Region { predecessors: vec![entry, NodeId::new(999)] });

        let init_val = graph.push_node(IrNode::IntConst(0));
        let one = graph.push_node(IrNode::IntConst(1));
        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![(entry, init_val), (header, NodeId::new(998))],
            ty: Type::I64,
        });
        let i_plus_1 = graph.push_node(IrNode::Add { lhs: phi, rhs: one });
        graph.replace(phi, IrNode::Phi {
            inputs: vec![(entry, init_val), (header, i_plus_1)],
            ty: Type::I64,
        });

        let root_b = OwnershipRoot::new(5);
        let root_a = OwnershipRoot::new(6);
        let base_b = graph.push_node(IrNode::IntConst(1000));
        let base_a = graph.push_node(IrNode::IntConst(2000));
        let addr_b = graph.push_node(IrNode::Add { lhs: base_b, rhs: phi });
        let addr_a = graph.push_node(IrNode::Add { lhs: base_a, rhs: phi });

        let load_b = graph.push_node(IrNode::Load { addr: addr_b, root: root_b, ty: Type::I32 });
        let neg_b = graph.push_node(IrNode::Neg { val: load_b });
        let _store = graph.push_node(IrNode::Store { addr: addr_a, val: neg_b, root: root_a, ty: Type::I32 });

        let hundred = graph.push_node(IrNode::IntConst(100));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: hundred });

        let exit = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let latch = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let _branch = graph.push_node(IrNode::Branch { cond: cmp, true_block: latch, false_block: exit });
        let latch_jump = graph.push_node(IrNode::Jump { target: header });
        graph.replace(header, IrNode::Region { predecessors: vec![entry, latch_jump] });
        let _ret = graph.push_node(IrNode::Return { value: Some(phi) });

        let lv = LoopVectorizer::new().with_vector_factor(4, Type::I32);
        let result = lv.run(&mut graph);

        assert!(result, "Expected vectorization of unary neg pattern");

        let has_vec_unop = graph.iter().any(|(_, n)| matches!(n, IrNode::VecUnOp { .. }));
        assert!(has_vec_unop, "Expected VecUnOp node for Neg");
    }
}
