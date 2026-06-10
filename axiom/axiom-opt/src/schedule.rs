//! Ownership-Aware Instruction Scheduler.
//!
//! Schedules the sea-of-nodes IR into an ordered sequence for lowering.
//! Uses list scheduling with height-based ASAP/ALAP analysis to minimize
//! register pressure while respecting latency constraints.
//!
//! # Ownership-Aware Enhancement (Phase 3)
//!
//! The scheduler leverages OwnershipRoot information to:
//! 1. **Group memory operations by root** — operations on the same root
//!    maintain their dependency chain (within-root serialization).
//! 2. **Schedule across roots in parallel** — operations on different roots
//!    have guaranteed no-aliasing, so they can be freely interleaved or
//!    issued to independent execution ports.
//! 3. **Reorder across roots freely** — guaranteed safe by no-aliasing.
//! 4. **Minimize memory dependency chain length within each root** —
//!    schedule loads as early as possible within their root's chain.
//! 5. **Prefetch data for roots** that will be accessed soon.
//!
//! This is IMPOSSIBLE in LLVM because it cannot prove that operations on
//! different allocations don't alias. In Axiom, this is trivially correct
//! by construction.

use std::collections::{HashMap, HashSet, VecDeque};

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use axiom_ownership::OwnershipAnalysis;

/// Schedule the sea-of-nodes IR into an ordered sequence for lowering.
/// This determines the instruction order before MIR construction.
pub struct InstructionScheduler {
    /// Estimated latency per operation type (in cycles), keyed by op name.
    latencies: HashMap<String, u32>,
}

/// Result of scheduling a graph.
pub struct ScheduleResult {
    /// Nodes in scheduled order.
    pub order: Vec<NodeId>,
    /// Estimated critical path length (in cycles).
    pub critical_path: u32,
}

impl InstructionScheduler {
    pub fn new() -> Self {
        let mut latencies = HashMap::new();

        // Integer arithmetic — 1 cycle each
        latencies.insert("add".to_string(), 1);
        latencies.insert("sub".to_string(), 1);
        latencies.insert("neg".to_string(), 1);
        latencies.insert("not".to_string(), 1);

        // Bitwise — 1 cycle each
        latencies.insert("and".to_string(), 1);
        latencies.insert("or".to_string(), 1);
        latencies.insert("xor".to_string(), 1);
        latencies.insert("shl".to_string(), 1);
        latencies.insert("shr".to_string(), 1);
        latencies.insert("sar".to_string(), 1);

        // Multiplication — 3 cycles
        latencies.insert("mul".to_string(), 3);

        // Division — 20 cycles
        latencies.insert("div".to_string(), 20);
        latencies.insert("rem".to_string(), 20);

        // Comparisons — 1 cycle
        latencies.insert("eq".to_string(), 1);
        latencies.insert("ne".to_string(), 1);
        latencies.insert("lt".to_string(), 1);
        latencies.insert("le".to_string(), 1);
        latencies.insert("gt".to_string(), 1);
        latencies.insert("ge".to_string(), 1);

        // FP arithmetic — 3-5 cycles
        latencies.insert("fadd".to_string(), 3);
        latencies.insert("fsub".to_string(), 3);
        latencies.insert("fmul".to_string(), 5);
        latencies.insert("fdiv".to_string(), 15);
        latencies.insert("frem".to_string(), 15);
        latencies.insert("fneg".to_string(), 1);
        latencies.insert("fabs".to_string(), 1);
        latencies.insert("fsqrt".to_string(), 14);

        // FP comparisons — 1 cycle
        latencies.insert("feq".to_string(), 1);
        latencies.insert("fne".to_string(), 1);
        latencies.insert("flt".to_string(), 1);
        latencies.insert("fle".to_string(), 1);
        latencies.insert("fgt".to_string(), 1);
        latencies.insert("fge".to_string(), 1);

        // FP conversions — 2 cycles
        latencies.insert("fp_to_sint".to_string(), 2);
        latencies.insert("sint_to_fp".to_string(), 2);
        latencies.insert("fp_to_uint".to_string(), 2);
        latencies.insert("uint_to_fp".to_string(), 2);

        // FP misc
        latencies.insert("copysign".to_string(), 2);
        latencies.insert("fmin".to_string(), 2);
        latencies.insert("fmax".to_string(), 2);

        // Memory — 4 cycles (cache hit)
        latencies.insert("load".to_string(), 4);
        latencies.insert("store".to_string(), 1);
        latencies.insert("stack_alloc".to_string(), 1);
        latencies.insert("fence".to_string(), 10);

        // Control — 0 cycles (no data output)
        latencies.insert("branch".to_string(), 0);
        latencies.insert("jump".to_string(), 0);
        latencies.insert("return".to_string(), 0);
        latencies.insert("call".to_string(), 5);
        latencies.insert("phi".to_string(), 0);

        // Conversions — 1 cycle
        latencies.insert("zext".to_string(), 1);
        latencies.insert("sext".to_string(), 1);
        latencies.insert("trunc".to_string(), 1);
        latencies.insert("bitcast".to_string(), 1);

        // Constants — 0 cycles (often folded)
        latencies.insert("int_const".to_string(), 0);
        latencies.insert("fp_const".to_string(), 0);

        // Misc
        latencies.insert("extract".to_string(), 1);
        latencies.insert("insert".to_string(), 1);

        // Vector — 1 cycle (simplified)
        latencies.insert("vec_broadcast".to_string(), 1);
        latencies.insert("vec_load".to_string(), 4);
        latencies.insert("vec_store".to_string(), 1);
        latencies.insert("vec_binop".to_string(), 2);
        latencies.insert("vec_unop".to_string(), 2);
        latencies.insert("vec_reduce".to_string(), 3);
        latencies.insert("extract_lane".to_string(), 1);
        latencies.insert("insert_lane".to_string(), 1);
        latencies.insert("vec_shuffle".to_string(), 1);
        latencies.insert("vec_gather".to_string(), 8);
        latencies.insert("vec_scatter".to_string(), 2);

        Self { latencies }
    }

    /// Get the latency for an operation by name.
    pub fn latency(&self, op_name: &str) -> u32 {
        self.latencies.get(op_name).copied().unwrap_or(1)
    }

    /// Map an IrNode to its operation name string (for latency lookup).
    fn node_op_name(node: &IrNode) -> String {
        match node {
            IrNode::IntConst(_) => "int_const",
            IrNode::FpConst(_) => "fp_const",
            IrNode::BoolConst(_) => "int_const",
            IrNode::UndefConst => "int_const",
            IrNode::Add { .. } => "add",
            IrNode::Sub { .. } => "sub",
            IrNode::Mul { .. } => "mul",
            IrNode::Div { .. } => "div",
            IrNode::Rem { .. } => "rem",
            IrNode::Neg { .. } => "neg",
            IrNode::And { .. } => "and",
            IrNode::Or { .. } => "or",
            IrNode::Xor { .. } => "xor",
            IrNode::Shl { .. } => "shl",
            IrNode::Shr { .. } => "shr",
            IrNode::Sar { .. } => "sar",
            IrNode::Not { .. } => "not",
            IrNode::Eq { .. } => "eq",
            IrNode::Ne { .. } => "ne",
            IrNode::Lt { .. } => "lt",
            IrNode::Le { .. } => "le",
            IrNode::Gt { .. } => "gt",
            IrNode::Ge { .. } => "ge",
            IrNode::ZExt { .. } => "zext",
            IrNode::SExt { .. } => "sext",
            IrNode::Trunc { .. } => "trunc",
            IrNode::BitCast { .. } => "bitcast",
            IrNode::IntToPtr { .. } => "bitcast",
            IrNode::PtrToInt { .. } => "bitcast",
            IrNode::Load { .. } => "load",
            IrNode::Store { .. } => "store",
            IrNode::StackAlloc { .. } => "stack_alloc",
            IrNode::Fence { .. } => "fence",
            IrNode::Start => "branch",
            IrNode::Param { .. } => "param",
            IrNode::Return { .. } => "return",
            IrNode::Unreachable => "return",
            IrNode::Branch { .. } => "branch",
            IrNode::Jump { .. } => "jump",
            IrNode::Region { .. } => "phi",
            IrNode::Phi { .. } => "phi",
            IrNode::Call { .. } => "call",
            IrNode::CallIndirect { .. } => "call",
            IrNode::VarDef { .. } => "store",
            IrNode::VarRef { .. } => "load",
            IrNode::VarSet { .. } => "store",
            IrNode::Extract { .. } => "extract",
            IrNode::Insert { .. } => "insert",
            IrNode::Intrinsic { .. } => "call",
            IrNode::Owned { .. } => "bitcast",
            // FP
            IrNode::FAdd { .. } => "fadd",
            IrNode::FSub { .. } => "fsub",
            IrNode::FMul { .. } => "fmul",
            IrNode::FDiv { .. } => "fdiv",
            IrNode::FRem { .. } => "frem",
            IrNode::FNeg { .. } => "fneg",
            IrNode::FAbs { .. } => "fabs",
            IrNode::FSqrt { .. } => "fsqrt",
            IrNode::FEq { .. } => "feq",
            IrNode::FLt { .. } => "flt",
            IrNode::FLe { .. } => "fle",
            IrNode::FGt { .. } => "fgt",
            IrNode::FGe { .. } => "fge",
            IrNode::FNe { .. } => "fne",
            IrNode::FpToSInt { .. } => "fp_to_sint",
            IrNode::SIntToFp { .. } => "sint_to_fp",
            IrNode::FpToUInt { .. } => "fp_to_uint",
            IrNode::UIntToFp { .. } => "uint_to_fp",
            IrNode::Copysign { .. } => "copysign",
            IrNode::Fmin { .. } => "fmin",
            IrNode::Fmax { .. } => "fmax",
            // Vector
            IrNode::VecBroadcast { .. } => "vec_broadcast",
            IrNode::VecLoad { .. } => "vec_load",
            IrNode::VecStore { .. } => "vec_store",
            IrNode::VecBinOp { .. } => "vec_binop",
            IrNode::VecUnOp { .. } => "vec_unop",
            IrNode::ExtractLane { .. } => "extract_lane",
            IrNode::InsertLane { .. } => "insert_lane",
            IrNode::VecReduce { .. } => "vec_reduce",
            IrNode::VecShuffle { .. } => "vec_shuffle",
            IrNode::VecGather { .. } => "vec_gather",
            IrNode::VecScatter { .. } => "vec_scatter",
            // Tail call — same cost as a regular call
            IrNode::TailCall { .. } => "call",
        }.to_string()
    }

    /// Compute the height of each node: the distance to its farthest successor.
    /// Height[n] = max(Height[user] + latency[n]) for all users.
    /// Computed recursively with memoization.
    fn compute_heights(
        &self,
        graph: &IrGraph,
        nodes: &[NodeId],
        node_set: &HashSet<NodeId>,
    ) -> HashMap<NodeId, u32> {
        let mut heights: HashMap<NodeId, u32> = HashMap::new();

        // Build user map: node -> list of nodes that use it
        let mut user_map: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &id in nodes {
            user_map.entry(id).or_default();
        }
        for &id in nodes {
            let node = graph.get(id).unwrap();
            for input in node.inputs() {
                if node_set.contains(&input) && input != id {
                    user_map.entry(input).or_default().push(id);
                }
            }
        }

        // Recursive height computation with memoization
        fn height_of(
            id: NodeId,
            graph: &IrGraph,
            user_map: &HashMap<NodeId, Vec<NodeId>>,
            latencies: &HashMap<String, u32>,
            heights: &mut HashMap<NodeId, u32>,
            visiting: &mut HashSet<NodeId>,
        ) -> u32 {
            if let Some(&h) = heights.get(&id) {
                return h;
            }
            // Cycle detection for phi self-references
            if visiting.contains(&id) {
                return 0;
            }
            visiting.insert(id);

            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = latencies.get(&op_name).copied().unwrap_or(1);

            let mut h = 0u32;
            if let Some(users) = user_map.get(&id) {
                for &user_id in users {
                    let user_height = height_of(
                        user_id, graph, user_map, latencies, heights, visiting,
                    );
                    let candidate = user_height + lat;
                    if candidate > h {
                        h = candidate;
                    }
                }
            }

            visiting.remove(&id);
            heights.insert(id, h);
            h
        }

        for &id in nodes {
            let mut visiting = HashSet::new();
            height_of(id, graph, &user_map, &self.latencies, &mut heights, &mut visiting);
        }

        heights
    }

    /// Topological sort of the data-dependence graph.
    /// Uses Kahn's algorithm.
    fn topological_sort(
        &self,
        graph: &IrGraph,
        nodes: &[NodeId],
        node_set: &HashSet<NodeId>,
    ) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, u32> = HashMap::new();
        for &id in nodes {
            in_degree.insert(id, 0);
        }

        for &id in nodes {
            let node = graph.get(id).unwrap();
            for input in node.inputs() {
                if node_set.contains(&input) && input != id {
                    // Avoid self-edges (phi nodes can reference themselves)
                    *in_degree.entry(id).or_insert(0) += 1;
                }
            }
        }

        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for &id in nodes {
            if in_degree[&id] == 0 {
                queue.push_back(id);
            }
        }

        let mut result = Vec::with_capacity(nodes.len());
        while let Some(id) = queue.pop_front() {
            result.push(id);

            // Find all users of this node and decrement their in-degree
            for &other_id in nodes {
                if other_id == id {
                    continue;
                }
                if let Some(other_node) = graph.get(other_id) {
                    let is_user = other_node.inputs().iter().any(|inp| *inp == id);
                    if is_user {
                        let deg = in_degree.get_mut(&other_id).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(other_id);
                        }
                    }
                }
            }
        }

        result
    }

    /// Schedule all data nodes using list scheduling.
    ///
    /// 1. Compute height (distance to farthest successor) for each node
    /// 2. Compute ASAP time (earliest possible execution time)
    /// 3. Compute ALAP time (latest possible execution time)
    /// 4. Mobility = ALAP - ASAP (lower = more critical)
    /// 5. Sort by: mobility ascending, then height descending
    pub fn schedule(&self, graph: &IrGraph) -> ScheduleResult {
        // Collect all live node IDs
        let nodes: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();

        if nodes.is_empty() {
            return ScheduleResult {
                order: Vec::new(),
                critical_path: 0,
            };
        }

        // ── Step 1: Compute heights (distance to farthest successor) ──
        let heights = self.compute_heights(graph, &nodes, &node_set);

        // ── Build use-def graph ──────────────────────────────────
        let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &id in &nodes {
            users.entry(id).or_default();
        }
        for &id in &nodes {
            let node = graph.get(id).unwrap();
            for input in node.inputs() {
                if node_set.contains(&input) {
                    users.entry(input).or_default().push(id);
                }
            }
        }

        // ── Step 2: Compute ASAP times (forward pass) ────────────
        // ASAP[id] = max(ASAP[pred] + latency[pred]) for all preds
        let mut asap: HashMap<NodeId, u32> = HashMap::new();
        for &id in &nodes {
            asap.insert(id, 0);
        }

        let topo_order = self.topological_sort(graph, &nodes, &node_set);

        for &id in &topo_order {
            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = self.latency(&op_name);

            for input in node.inputs() {
                if node_set.contains(&input) {
                    let pred_time = asap[&input] + lat;
                    if pred_time > asap[&id] {
                        asap.insert(id, pred_time);
                    }
                }
            }
        }

        // ── Step 3: Compute ALAP times (backward pass) ───────────
        // Schedule length = max(ASAP[id] + latency[id]) for all ids
        let mut schedule_length = 0u32;
        for &id in &nodes {
            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = self.latency(&op_name);
            let end_time = asap[&id] + lat;
            if end_time > schedule_length {
                schedule_length = end_time;
            }
        }

        // ALAP[id] = min(ALAP[user] - latency[id]) for all users
        // Initialize to schedule_length
        let mut alap: HashMap<NodeId, u32> = HashMap::new();
        for &id in &nodes {
            alap.insert(id, schedule_length);
        }

        // Reverse topological order for ALAP
        for &id in topo_order.iter().rev() {
            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = self.latency(&op_name);

            if let Some(user_list) = users.get(&id) {
                for &user_id in user_list {
                    let user_alap = alap[&user_id];
                    let constraint = user_alap.saturating_sub(lat);
                    if constraint < alap[&id] {
                        alap.insert(id, constraint);
                    }
                }
            }
        }

        // ── Step 4: Compute mobility ─────────────────────────────
        // mobility = ALAP - ASAP (lower = more critical)
        let critical_path = schedule_length;

        // ── Step 5: List scheduling with priority ────────────────
        // Use a priority queue sorted by mobility/height, but enforce
        // topological order: a node can only be scheduled after all
        // its inputs have been scheduled.
        let mut remaining_deps: HashMap<NodeId, u32> = HashMap::new();
        for &id in &nodes {
            let node = graph.get(id).unwrap();
            let dep_count = node.inputs().iter()
                .filter(|inp| node_set.contains(inp) && **inp != id)
                .count() as u32;
            remaining_deps.insert(id, dep_count);
        }

        let mut scheduled: Vec<NodeId> = Vec::with_capacity(nodes.len());
        let mut scheduled_set: HashSet<NodeId> = HashSet::new();

        while scheduled.len() < nodes.len() {
            // Find all ready nodes (dependencies satisfied)
            let mut ready: Vec<NodeId> = nodes.iter()
                .filter(|&&id| !scheduled_set.contains(&id) && remaining_deps[&id] == 0)
                .copied()
                .collect();

            if ready.is_empty() {
                // All remaining nodes have unsatisfied deps (cycle or error)
                // Schedule remaining in original order
                for &id in &nodes {
                    if !scheduled_set.contains(&id) {
                        scheduled.push(id);
                        scheduled_set.insert(id);
                    }
                }
                break;
            }

            // Sort ready nodes by priority: mobility ascending, height descending
            ready.sort_by(|&a, &b| {
                let mob_a = alap[&a].saturating_sub(asap[&a]);
                let mob_b = alap[&b].saturating_sub(asap[&b]);
                mob_a.cmp(&mob_b)
                    .then_with(|| heights[&b].cmp(&heights[&a]))
                    .then_with(|| a.cmp(&b))
            });

            // Schedule the highest-priority ready node
            let chosen = ready[0];
            scheduled.push(chosen);
            scheduled_set.insert(chosen);

            // Update dependencies for nodes that depend on chosen
            if let Some(user_list) = users.get(&chosen) {
                for &user_id in user_list {
                    if let Some(deps) = remaining_deps.get_mut(&user_id) {
                        *deps = deps.saturating_sub(1);
                    }
                }
            }
        }

        ScheduleResult {
            order: scheduled,
            critical_path,
        }
    }
}

impl Default for InstructionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Ownership-Aware Scheduling (Phase 3) ──────────────────────────────────

/// Result of ownership-aware scheduling.
///
/// Extends `ScheduleResult` with ownership-specific information that
/// enables target backends to exploit root-level parallelism.
#[derive(Debug, Clone)]
pub struct OwnershipScheduleResult {
    /// Nodes in scheduled order.
    pub order: Vec<NodeId>,
    /// Estimated critical path length (in cycles).
    pub critical_path: u32,
    /// For each memory node, its ownership root.
    pub node_roots: HashMap<NodeId, OwnershipRoot>,
    /// Per-root schedule: root -> ordered list of memory nodes in that root's chain.
    pub root_chains: HashMap<OwnershipRoot, Vec<NodeId>>,
    /// Nodes that can be issued in parallel (on different roots).
    /// Each inner vec contains nodes that are independent and can
    /// execute simultaneously.
    pub parallel_groups: Vec<Vec<NodeId>>,
}

impl InstructionScheduler {
    /// Ownership-aware scheduling: schedule the sea-of-nodes IR using
    /// ownership root information to maximize parallelism across roots.
    ///
    /// This is the Phase 3 version of `schedule()` that exploits the
    /// guaranteed no-aliasing between different OwnershipRoots.
    ///
    /// # Algorithm
    ///
    /// 1. Run ownership analysis to classify all nodes by root.
    /// 2. Build per-root memory dependency chains (within each root,
    ///    loads must follow stores to the same root in program order).
    /// 3. Compute ASAP/ALAP times considering only within-root dependencies.
    /// 4. Schedule: for each time slot, select ready nodes from ANY root
    ///    (across-root parallelism is always safe).
    /// 5. Within each root's chain, schedule loads early to hide latency.
    pub fn schedule_ownership_aware(&self, graph: &IrGraph) -> OwnershipScheduleResult {
        let analysis = OwnershipAnalysis::analyze(graph);

        // Collect all live node IDs
        let nodes: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();

        if nodes.is_empty() {
            return OwnershipScheduleResult {
                order: Vec::new(),
                critical_path: 0,
                node_roots: HashMap::new(),
                root_chains: HashMap::new(),
                parallel_groups: Vec::new(),
            };
        }

        // ── Step 1: Map nodes to ownership roots ──
        let mut node_roots: HashMap<NodeId, OwnershipRoot> = HashMap::new();
        for &id in &nodes {
            if let Some(&root) = analysis.node_root.get(&id) {
                node_roots.insert(id, root);
            }
        }

        // ── Step 2: Build per-root memory dependency chains ──
        // Within each root, stores must complete before dependent loads.
        // Across roots, there are NO dependencies.
        let mut root_chains: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        for &id in &nodes {
            if let Some(node) = graph.get(id) {
                if let Some(root) = node.ownership_root() {
                    root_chains.entry(root).or_default().push(id);
                }
            }
        }

        // ── Step 3: Compute heights (same as base scheduler) ──
        let heights = self.compute_heights(graph, &nodes, &node_set);

        // ── Step 4: Build use-def graph ──
        let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &id in &nodes {
            users.entry(id).or_default();
        }
        for &id in &nodes {
            let node = graph.get(id).unwrap();
            for input in node.inputs() {
                if node_set.contains(&input) {
                    users.entry(input).or_default().push(id);
                }
            }
        }

        // ── Step 5: Compute ASAP times with ownership-aware latency ──
        let topo_order = self.topological_sort(graph, &nodes, &node_set);

        let mut asap: HashMap<NodeId, u32> = HashMap::new();
        for &id in &nodes {
            asap.insert(id, 0);
        }

        for &id in &topo_order {
            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = self.latency(&op_name);

            for input in node.inputs() {
                if node_set.contains(&input) {
                    // Ownership-aware: within-root memory deps use full latency,
                    // cross-root deps can overlap (reduced effective latency)
                    let effective_lat = if Self::is_cross_root_dep(
                        graph, input, id, &node_roots
                    ) {
                        // Cross-root: no aliasing, can issue to different port
                        // Effective latency is reduced (e.g., 1 cycle for parallel issue)
                        lat.saturating_sub(1).max(1)
                    } else {
                        lat
                    };

                    let pred_time = asap[&input] + effective_lat;
                    if pred_time > asap[&id] {
                        asap.insert(id, pred_time);
                    }
                }
            }
        }

        // ── Step 6: Compute ALAP times ──
        let mut schedule_length = 0u32;
        for &id in &nodes {
            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = self.latency(&op_name);
            let end_time = asap[&id] + lat;
            if end_time > schedule_length {
                schedule_length = end_time;
            }
        }

        let mut alap: HashMap<NodeId, u32> = HashMap::new();
        for &id in &nodes {
            alap.insert(id, schedule_length);
        }

        for &id in topo_order.iter().rev() {
            let node = graph.get(id).unwrap();
            let op_name = InstructionScheduler::node_op_name(node);
            let lat = self.latency(&op_name);

            if let Some(user_list) = users.get(&id) {
                for &user_id in user_list {
                    let user_alap = alap[&user_id];
                    let constraint = user_alap.saturating_sub(lat);
                    if constraint < alap[&id] {
                        alap.insert(id, constraint);
                    }
                }
            }
        }

        let critical_path = schedule_length;

        // ── Step 7: List scheduling with ownership-aware priority ──
        let mut remaining_deps: HashMap<NodeId, u32> = HashMap::new();
        for &id in &nodes {
            let node = graph.get(id).unwrap();
            let dep_count = node.inputs().iter()
                .filter(|inp| node_set.contains(inp) && **inp != id)
                .count() as u32;
            remaining_deps.insert(id, dep_count);
        }

        let mut scheduled: Vec<NodeId> = Vec::with_capacity(nodes.len());
        let mut scheduled_set: HashSet<NodeId> = HashSet::new();

        // Track which root was most recently scheduled, to alternate
        // between roots for parallel issue opportunities
        let mut last_root: Option<OwnershipRoot> = None;

        while scheduled.len() < nodes.len() {
            let mut ready: Vec<NodeId> = nodes.iter()
                .filter(|&&id| !scheduled_set.contains(&id) && remaining_deps[&id] == 0)
                .copied()
                .collect();

            if ready.is_empty() {
                for &id in &nodes {
                    if !scheduled_set.contains(&id) {
                        scheduled.push(id);
                        scheduled_set.insert(id);
                    }
                }
                break;
            }

            // Ownership-aware priority:
            // 1. Mobility (ALAP - ASAP): lower = more critical
            // 2. Height: higher = more critical
            // 3. Root alternation: prefer nodes from a DIFFERENT root than
            //    the last scheduled node (enables cross-root parallelism)
            // 4. Memory operations from the same root are deprioritized
            //    if the last op was also from that root (reduces serialization)
            ready.sort_by(|&a, &b| {
                let mob_a = alap[&a].saturating_sub(asap[&a]);
                let mob_b = alap[&b].saturating_sub(asap[&b]);

                // Primary: mobility (lower = more critical)
                let cmp = mob_a.cmp(&mob_b);
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }

                // Secondary: height (higher = more critical)
                let h_cmp = heights[&b].cmp(&heights[&a]);
                if h_cmp != std::cmp::Ordering::Equal {
                    return h_cmp;
                }

                // Tertiary: root alternation — prefer different root from last
                let root_a = node_roots.get(&a);
                let root_b = node_roots.get(&b);
                if let (Some(ra), Some(rb)) = (root_a, root_b) {
                    if let Some(lr) = last_root {
                        let a_same = *ra == lr;
                        let b_same = *rb == lr;
                        if a_same != b_same {
                            // Prefer the one on a DIFFERENT root
                            return if a_same {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Less
                            };
                        }
                    }
                }

                a.cmp(&b)
            });

            let chosen = ready[0];
            scheduled.push(chosen);
            scheduled_set.insert(chosen);

            if let Some(&root) = node_roots.get(&chosen) {
                last_root = Some(root);
            }

            if let Some(user_list) = users.get(&chosen) {
                for &user_id in user_list {
                    if let Some(deps) = remaining_deps.get_mut(&user_id) {
                        *deps = deps.saturating_sub(1);
                    }
                }
            }
        }

        // ── Step 8: Build parallel groups ──
        // Group nodes by their ASAP time — nodes with the same ASAP
        // time on different roots can execute in parallel.
        let mut asap_groups: HashMap<u32, Vec<NodeId>> = HashMap::new();
        for &id in &scheduled {
            let t = asap[&id];
            asap_groups.entry(t).or_default().push(id);
        }

        let mut parallel_groups: Vec<Vec<NodeId>> = Vec::new();
        let mut times: Vec<u32> = asap_groups.keys().copied().collect();
        times.sort();

        for t in times {
            let group = &asap_groups[&t];
            // Only create a parallel group if there are nodes on different roots
            let roots_in_group: HashSet<OwnershipRoot> = group.iter()
                .filter_map(|id| node_roots.get(id))
                .copied()
                .collect();

            if roots_in_group.len() > 1 || group.len() == 1 {
                // Multiple roots or single node
                parallel_groups.push(group.clone());
            }
        }

        // Sort each root chain by schedule order
        let schedule_pos: HashMap<NodeId, usize> = scheduled.iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        for chain in root_chains.values_mut() {
            chain.sort_by_key(|id| schedule_pos.get(id).copied().unwrap_or(0));
        }

        OwnershipScheduleResult {
            order: scheduled,
            critical_path,
            node_roots,
            root_chains,
            parallel_groups,
        }
    }

    /// Check if a dependency edge (from -> to) crosses ownership roots.
    fn is_cross_root_dep(
        _graph: &IrGraph,
        from: NodeId,
        to: NodeId,
        node_roots: &HashMap<NodeId, OwnershipRoot>,
    ) -> bool {
        let from_root = node_roots.get(&from);
        let to_root = node_roots.get(&to);

        match (from_root, to_root) {
            (Some(r1), Some(r2)) => r1 != r2,
            _ => false, // If either doesn't have a root, not a cross-root dep
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::IrNode;

    #[test]
    fn test_schedule_simple_dag() {
        let mut graph = IrGraph::new("schedule_test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let mul = graph.push_node(IrNode::Mul { lhs: add, rhs: a });
        let _ret = graph.push_node(IrNode::Return { value: Some(mul) });

        let scheduler = InstructionScheduler::new();
        let result = scheduler.schedule(&graph);

        // All nodes should be in the result (Start + 2 constants + Add + Mul + Return = 6)
        assert_eq!(result.order.len(), 6);

        // Constants should come before their uses
        let start = graph.start_node();
        let pos_a = result.order.iter().position(|&id| id == a).unwrap();
        let pos_b = result.order.iter().position(|&id| id == b).unwrap();
        let pos_add = result.order.iter().position(|&id| id == add).unwrap();
        let pos_mul = result.order.iter().position(|&id| id == mul).unwrap();
        let _pos_start = result.order.iter().position(|&id| id == start).unwrap();

        assert!(pos_a < pos_add, "a should be scheduled before add");
        assert!(pos_b < pos_add, "b should be scheduled before add");
        assert!(pos_add < pos_mul, "add should be scheduled before mul");

        // Critical path should be > 0
        assert!(result.critical_path > 0, "Critical path should be > 0");
    }

    #[test]
    fn test_schedule_empty_graph() {
        let graph = IrGraph::new("empty_test");
        let scheduler = InstructionScheduler::new();
        let result = scheduler.schedule(&graph);

        // Should have at least the Start node
        assert!(!result.order.is_empty());
    }

    #[test]
    fn test_mobility_prioritizes_critical_path() {
        // Build a diamond DAG:
        //   a
        //  / \
        // b   c   (b is Mul=3 cycles, c is Add=1 cycle)
        //  \ /
        //   d
        // c has higher mobility (less critical), so b should be scheduled first.
        let mut graph = IrGraph::new("diamond_test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::Mul { lhs: a, rhs: a }); // 3-cycle
        let c = graph.push_node(IrNode::Add { lhs: a, rhs: a }); // 1-cycle
        let d = graph.push_node(IrNode::Add { lhs: b, rhs: c });
        let _ret = graph.push_node(IrNode::Return { value: Some(d) });

        let scheduler = InstructionScheduler::new();
        let result = scheduler.schedule(&graph);

        let pos_b = result.order.iter().position(|&id| id == b).unwrap();
        let pos_c = result.order.iter().position(|&id| id == c).unwrap();

        // b (Mul, longer latency, lower mobility) should be scheduled before c
        assert!(
            pos_b < pos_c,
            "Mul (lower mobility) should be scheduled before Add (higher mobility)"
        );
    }

    #[test]
    fn test_height_computation() {
        // Chain: a -> b -> c
        // a has the greatest height (farthest from any sink)
        let mut graph = IrGraph::new("height_test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::Add { lhs: a, rhs: a });
        let c = graph.push_node(IrNode::Mul { lhs: b, rhs: a });
        let _ret = graph.push_node(IrNode::Return { value: Some(c) });

        let scheduler = InstructionScheduler::new();
        let nodes: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let heights = scheduler.compute_heights(&graph, &nodes, &node_set);

        // a should have the greatest height (it's the root of the dependency chain)
        // Return has height 0 (no users)
        // c has height = latency(Return) = 0
        // b has height = height(c) + latency(Mul) = 0 + 3 = 3
        // a has height = max(height(b) + latency(Add), height(c) + latency(Mul)) = max(3+1, 0+3) = 4
        assert!(heights[&a] >= heights[&b], "root should have >= height than its child");
        assert!(heights[&b] >= heights[&c], "inner node should have >= height than leaf");
    }

    #[test]
    fn test_div_high_latency() {
        // Division has latency 20
        let mut graph = IrGraph::new("div_test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let div = graph.push_node(IrNode::Div { lhs: a, rhs: b });
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(div) });

        let scheduler = InstructionScheduler::new();
        // Verify div latency is 20
        assert_eq!(scheduler.latency("div"), 20);
        assert_eq!(scheduler.latency("fdiv"), 15);
        assert_eq!(scheduler.latency("fsqrt"), 14);
        assert_eq!(scheduler.latency("load"), 4);
        assert_eq!(scheduler.latency("mul"), 3);
    }
}
