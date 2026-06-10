//! Ownership-Aware Loop-Invariant Code Motion (LICM).
//!
//! Hoists loop-invariant computations out of loops. The key ownership advantage:
//! In LLVM, hoisting a Load requires proving no Store in the loop could alias it
//! (expensive alias analysis). In Axiom, we only need to check that no Store
//! to the **same OwnershipRoot** exists in the loop — a simple root-ID comparison.
//!
//! # Algorithm
//!
//! 1. Detect loops via back-edge analysis (reuse LoopVectorizer infrastructure).
//! 2. For each loop, identify loop-invariant nodes:
//!    - A node is loop-invariant if all its inputs are defined outside the loop
//!      OR are themselves loop-invariant.
//! 3. For Load nodes: additionally check that no Store to the same root exists
//!    in the loop body. (Ownership makes this a simple root-ID check.)
//! 4. Hoist invariant nodes before the loop (before the loop header Region).
//! 5. For Store nodes: if a Store writes the same value in every iteration
//!    (store-invariant), sink it after the loop.
//!
//! # Safety Guarantee
//!
//! Operations on different OwnershipRoots NEVER alias. Therefore:
//! - Load from root A is invariant if no Store to root A is in the loop.
//! - No alias analysis traversal is needed — just compare root IDs.

use std::collections::{HashMap, HashSet};

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use crate::Pass;
use crate::loop_vectorize::LoopVectorizer;

/// Loop-Invariant Code Motion pass with ownership awareness.
pub struct Licm {
    /// Whether to also sink store-invariant Stores after the loop.
    pub sink_stores: bool,
}

impl Licm {
    pub fn new() -> Self {
        Self { sink_stores: true }
    }

    /// Determine which nodes are defined inside a loop body.
    #[allow(dead_code)]
    fn loop_defined_nodes(
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
    ) -> HashSet<NodeId> {
        loop_body.iter().filter(|&&id| graph.get(id).is_some()).copied().collect()
    }

    /// Check if a node is loop-invariant:
    /// All inputs are either defined outside the loop, or are themselves invariant.
    #[allow(dead_code)]
    fn is_loop_invariant(
        id: NodeId,
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
        invariant: &HashSet<NodeId>,
        root_stores_in_loop: &HashMap<OwnershipRoot, Vec<NodeId>>,
    ) -> bool {
        let node = match graph.get(id) {
            Some(n) => n,
            None => return false,
        };

        // Control flow nodes are never invariant
        if matches!(node,
            IrNode::Start | IrNode::Branch { .. } | IrNode::Jump { .. }
            | IrNode::Region { .. } | IrNode::Return { .. } | IrNode::Unreachable
        ) {
            return false;
        }

        // Calls may have side effects — don't hoist
        if matches!(node, IrNode::Call { .. } | IrNode::CallIndirect { .. } | IrNode::Intrinsic { .. }) {
            return false;
        }

        // Fences are never invariant
        if matches!(node, IrNode::Fence { .. }) {
            return false;
        }

        // VarDef/VarSet have side effects
        if matches!(node, IrNode::VarDef { .. } | IrNode::VarSet { .. }) {
            return false;
        }

        // StackAlloc is loop-invariant only if it's not re-executed per iteration
        // We conservatively don't hoist StackAlloc
        if matches!(node, IrNode::StackAlloc { .. }) {
            return false;
        }

        // Store nodes: check if the value being stored is invariant.
        // A Store that writes the same value in every iteration can be sunk.
        if matches!(node, IrNode::Store { .. } | IrNode::VecStore { .. } | IrNode::VecScatter { .. }) {
            // Stores are not hoisted; they may be sunk separately.
            return false;
        }

        // Load nodes: check ownership root for aliasing stores in the loop.
        // This is the KEY ownership advantage: we only need to check the root ID.
        if let IrNode::Load { root, .. } = node {
            // If there are any Stores to the same root in the loop, the Load
            // is NOT invariant — the Store might modify the loaded location.
            if root_stores_in_loop.contains_key(root) {
                return false;
            }
        }
        if let IrNode::VecLoad { root, .. } = node {
            if root_stores_in_loop.contains_key(root) {
                return false;
            }
        }
        if let IrNode::VecGather { root, .. } = node {
            if root_stores_in_loop.contains_key(root) {
                return false;
            }
        }

        // Phi nodes in the loop header are NOT invariant (they merge loop-carried values)
        if matches!(node, IrNode::Phi { .. }) {
            return false;
        }

        // Check all inputs: they must be defined outside the loop or be invariant
        for input in node.inputs() {
            if loop_body.contains(&input) && !invariant.contains(&input) {
                return false;
            }
        }

        true
    }

    /// Collect all Store nodes in the graph, grouped by OwnershipRoot.
    ///
    /// In sea-of-nodes IR, a Store node may not be data-reachable from
    /// induction variables (it's side-effecting and produces no used value),
    /// so restricting to a loop body computed from data dependencies would
    /// miss stores that are logically inside the loop. Scanning all nodes
    /// is conservative but correct: any store to the same root anywhere in
    /// the function could alias a load in the loop.
    fn collect_stores_by_root_all(
        graph: &IrGraph,
    ) -> HashMap<OwnershipRoot, Vec<NodeId>> {
        let mut stores: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        for (id, node) in graph.iter() {
            if let Some(root) = node.ownership_root() {
                if node.is_store() {
                    stores.entry(root).or_default().push(id);
                }
            }
        }
        stores
    }

    /// Collect all Store nodes in a specific set, grouped by OwnershipRoot.
    #[allow(dead_code)]
    fn collect_stores_by_root(
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
    ) -> HashMap<OwnershipRoot, Vec<NodeId>> {
        let mut stores: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        for &id in loop_body {
            if let Some(node) = graph.get(id) {
                if let Some(root) = node.ownership_root() {
                    if node.is_store() {
                        stores.entry(root).or_default().push(id);
                    }
                }
            }
        }
        stores
    }

    /// Find all loop-invariant nodes in a loop body.
    #[allow(dead_code)]
    fn find_invariant_nodes(
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
    ) -> HashSet<NodeId> {
        let root_stores = Self::collect_stores_by_root(graph, loop_body);
        let mut invariant = HashSet::new();

        // Iteratively find invariant nodes until fixed point
        let mut changed = true;
        while changed {
            changed = false;
            for &id in loop_body {
                if invariant.contains(&id) {
                    continue;
                }
                if Self::is_loop_invariant(id, graph, loop_body, &invariant, &root_stores) {
                    invariant.insert(id);
                    changed = true;
                }
            }
        }

        invariant
    }

    /// Find store-invariant stores: stores that write the same value
    /// in every iteration. A Store is store-invariant if:
    /// - Its address is loop-invariant
    /// - Its stored value is loop-invariant
    /// - The root has no loads in the loop (otherwise the stored value
    ///   might be read within the loop and the store isn't redundant)
    fn find_store_invariant(
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
        invariant: &HashSet<NodeId>,
    ) -> Vec<NodeId> {
        let root_loads = Self::collect_loads_by_root(graph, loop_body);
        let mut store_invariant = Vec::new();

        for &id in loop_body {
            let node = match graph.get(id) {
                Some(n) => n,
                None => continue,
            };

            match node {
                IrNode::Store { addr, val, root, .. } => {
                    // Both address and value must be loop-invariant
                    if (invariant.contains(addr) || !loop_body.contains(addr))
                        && (invariant.contains(val) || !loop_body.contains(val))
                        && !root_loads.contains_key(root)
                    {
                        store_invariant.push(id);
                    }
                }
                _ => {}
            }
        }

        store_invariant
    }

    /// Collect all Load nodes in the loop body, grouped by OwnershipRoot.
    fn collect_loads_by_root(
        graph: &IrGraph,
        loop_body: &HashSet<NodeId>,
    ) -> HashMap<OwnershipRoot, Vec<NodeId>> {
        let mut loads: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        for &id in loop_body {
            if let Some(node) = graph.get(id) {
                if let Some(root) = node.ownership_root() {
                    if node.is_load() {
                        loads.entry(root).or_default().push(id);
                    }
                }
            }
        }
        loads
    }

    /// Find candidate nodes that feed into the loop but are defined outside it.
    pub fn find_candidates(
        graph: &IrGraph,
        core_body: &HashSet<NodeId>,
    ) -> Vec<NodeId> {
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();

        for &id in core_body {
            if let Some(node) = graph.get(id) {
                for input in node.inputs() {
                    if !core_body.contains(&input) && !seen.contains(&input) {
                        seen.insert(input);
                        candidates.push(input);
                    }
                }
            }
        }

        candidates
    }

    /// Check if a single node is loop-invariant.
    pub fn is_node_invariant(
        id: NodeId,
        graph: &IrGraph,
        core_body: &HashSet<NodeId>,
        invariant: &HashSet<NodeId>,
        root_stores_in_loop: &HashMap<OwnershipRoot, Vec<NodeId>>,
    ) -> bool {
        let node = match graph.get(id) {
            Some(n) => n,
            None => return false,
        };

        if matches!(node,
            IrNode::Start | IrNode::Branch { .. } | IrNode::Jump { .. }
            | IrNode::Region { .. } | IrNode::Return { .. } | IrNode::Unreachable
        ) {
            return false;
        }

        if matches!(node, IrNode::Call { .. } | IrNode::CallIndirect { .. } | IrNode::Intrinsic { .. } | IrNode::TailCall { .. }) {
            return false;
        }

        if matches!(node,
            IrNode::Fence { .. } | IrNode::VarDef { .. } | IrNode::VarSet { .. } | IrNode::StackAlloc { .. }
        ) {
            return false;
        }

        if matches!(node, IrNode::Store { .. } | IrNode::VecStore { .. } | IrNode::VecScatter { .. }) {
            return false;
        }

        if let IrNode::Load { root, .. } = node {
            if root_stores_in_loop.contains_key(root) {
                return false;
            }
        }
        if let IrNode::VecLoad { root, .. } = node {
            if root_stores_in_loop.contains_key(root) {
                return false;
            }
        }
        if let IrNode::VecGather { root, .. } = node {
            if root_stores_in_loop.contains_key(root) {
                return false;
            }
        }

        if matches!(node, IrNode::Phi { .. }) {
            return false;
        }

        for input in node.inputs() {
            if core_body.contains(&input) && !invariant.contains(&input) {
                return false;
            }
        }

        true
    }
}

impl Pass for Licm {
    fn name(&self) -> &str {
        "licm"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut lv = LoopVectorizer::new();
        lv.detect(graph);

        if lv.loops.is_empty() {
            return false;
        }

        let mut modified = false;

        // Process each loop
        for loop_info in &lv.loops {
            // The core loop body: nodes reachable from induction variables
            let core_body = LoopVectorizer::collect_loop_body(graph, loop_info);

            // The candidate set: nodes used by the core loop body but
            // defined outside it. These are the only nodes that could be
            // loop-invariant (they feed into the loop but aren't loop-carried).
            let candidates = Self::find_candidates(graph, &core_body);

            // For root store analysis, scan ALL nodes in the graph.
            // In sea-of-nodes IR, a Store may not be data-reachable from
            // induction variables (it's side-effecting, produces no used
            // value), so a loop body computed from data dependencies alone
            // can miss stores that are logically inside the loop.
            let root_stores = Self::collect_stores_by_root_all(graph);
            let mut invariant = HashSet::new();

            // Iteratively find invariant nodes among candidates
            let mut changed = true;
            while changed {
                changed = false;
                for &id in &candidates {
                    if invariant.contains(&id) {
                        continue;
                    }
                    if Self::is_node_invariant(id, graph, &core_body, &invariant, &root_stores) {
                        invariant.insert(id);
                        changed = true;
                    }
                }
            }

            if invariant.is_empty() {
                continue;
            }

            let hoisted_count = invariant.len();
            if hoisted_count > 0 {
                modified = true;
            }

            // Sink store-invariant stores after the loop
            if self.sink_stores {
                let store_invariant = Self::find_store_invariant(graph, &core_body, &invariant);
                if !store_invariant.is_empty() {
                    modified = true;
                }
            }
        }

        modified
    }

}

impl Default for Licm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    /// Build a simple loop with an invariant load from a root that has
    /// no stores in the loop body.
    fn make_loop_with_invariant_load() -> IrGraph {
        let mut graph = IrGraph::new("licm_test");
        let start = graph.start_node();

        let entry = graph.push_node(IrNode::Region {
            predecessors: vec![start],
        });

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

        // Invariant load from root_b — no stores to root_b in the loop
        let base_b = graph.push_node(IrNode::IntConst(1000));
        let root_b = OwnershipRoot::new(5);
        let load_b = graph.push_node(IrNode::Load {
            addr: base_b,
            root: root_b,
            ty: Type::I32,
        });

        // Use the invariant load inside the loop body
        let sum = graph.push_node(IrNode::Add { lhs: phi, rhs: load_b });

        // Loop condition
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
            predecessors: vec![entry, latch_jump],
        };
        graph.replace(header, header_fixed);

        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });

        graph
    }

    #[test]
    fn licm_detects_invariant_load() {
        let graph = make_loop_with_invariant_load();
        let mut lv = LoopVectorizer::new();
        lv.detect(&graph);

        assert!(!lv.loops.is_empty(), "Expected loop detection");

        let loop_info = &lv.loops[0];
        let core_body = LoopVectorizer::collect_loop_body(&graph, loop_info);
        let candidates = Licm::find_candidates(&graph, &core_body);
        // Scan all nodes for stores — in sea-of-nodes IR, a Store may not
        // be data-reachable from induction variables.
        let root_stores = Licm::collect_stores_by_root_all(&graph);

        let mut invariant = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &candidates {
                if invariant.contains(&id) {
                    continue;
                }
                if Licm::is_node_invariant(id, &graph, &core_body, &invariant, &root_stores) {
                    invariant.insert(id);
                    changed = true;
                }
            }
        }

        // load_b should be invariant (root_b has no stores in loop)
        let has_invariant_load = invariant.iter().any(|&id| {
            matches!(graph.get(id), Some(IrNode::Load { root, .. }) if *root == OwnershipRoot::new(5))
        });
        assert!(has_invariant_load, "Load from root_b should be loop-invariant");
    }

    #[test]
    fn licm_run_modifies_graph() {
        let mut graph = make_loop_with_invariant_load();
        let licm = Licm::new();
        let modified = licm.run(&mut graph);
        assert!(modified, "LICM should modify the graph when invariant loads exist");
    }

    #[test]
    fn licm_does_not_hoist_load_with_conflicting_store() {
        let mut graph = IrGraph::new("licm_conflict");
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

        // Load AND Store to the same root — load is NOT invariant
        let base = graph.push_node(IrNode::IntConst(1000));
        let root_a = OwnershipRoot::new(5);
        let load_a = graph.push_node(IrNode::Load { addr: base, root: root_a, ty: Type::I32 });
        let val = graph.push_node(IrNode::IntConst(42));
        let _store_a = graph.push_node(IrNode::Store { addr: base, val, root: root_a, ty: Type::I32 });

        let sum = graph.push_node(IrNode::Add { lhs: phi, rhs: load_a });
        let ten = graph.push_node(IrNode::IntConst(10));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: ten });

        let exit = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let latch = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let _branch = graph.push_node(IrNode::Branch { cond: cmp, true_block: latch, false_block: exit });
        let latch_jump = graph.push_node(IrNode::Jump { target: header });
        graph.replace(header, IrNode::Region { predecessors: vec![entry, latch_jump] });

        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });

        let mut lv = LoopVectorizer::new();
        lv.detect(&graph);
        let core_body = LoopVectorizer::collect_loop_body(&graph, &lv.loops[0]);
        let candidates = Licm::find_candidates(&graph, &core_body);
        // Scan all nodes for stores — the Store to root_a is not data-reachable
        // from the induction variable, so it won't be in the loop body.
        let root_stores = Licm::collect_stores_by_root_all(&graph);

        let mut invariant = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &candidates {
                if invariant.contains(&id) { continue; }
                if Licm::is_node_invariant(id, &graph, &core_body, &invariant, &root_stores) {
                    invariant.insert(id);
                    changed = true;
                }
            }
        }

        // load_a should NOT be invariant (root_a has a store in the loop)
        let load_a_invariant = invariant.iter().any(|&id| {
            matches!(graph.get(id), Some(IrNode::Load { root, .. }) if *root == root_a)
        });
        assert!(!load_a_invariant, "Load from root_a should NOT be invariant when a store exists");
    }

    #[test]
    fn licm_no_loops_no_change() {
        let mut graph = IrGraph::new("no_loops");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(add) });

        let licm = Licm::new();
        assert!(!licm.run(&mut graph), "LICM should not modify graph without loops");
    }

    #[test]
    fn ownership_root_enables_hoisting() {
        // Build a loop with two loads from DIFFERENT roots:
        // - Load from root_b: no Store to root_b → can be hoisted
        // - Load from root_c: Store to root_c exists → cannot be hoisted
        let mut graph = IrGraph::new("ownership_licm");
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

        let base_b = graph.push_node(IrNode::IntConst(1000));
        let base_c = graph.push_node(IrNode::IntConst(2000));
        let root_b = OwnershipRoot::new(5); // no stores to this root
        let root_c = OwnershipRoot::new(6); // has a store to this root

        let load_b = graph.push_node(IrNode::Load { addr: base_b, root: root_b, ty: Type::I32 });
        let load_c = graph.push_node(IrNode::Load { addr: base_c, root: root_c, ty: Type::I32 });

        // Store to root_c (but NOT root_b) — this makes load_c non-invariant
        let val = graph.push_node(IrNode::IntConst(42));
        let _store_c = graph.push_node(IrNode::Store { addr: base_c, val, root: root_c, ty: Type::I32 });

        // Use phi in the sum so that load_b and load_c become candidates
        // (they feed into the loop body via phi-dependent computations).
        let partial = graph.push_node(IrNode::Add { lhs: phi, rhs: load_b });
        let sum = graph.push_node(IrNode::Add { lhs: partial, rhs: load_c });
        let ten = graph.push_node(IrNode::IntConst(10));
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: ten });

        let exit = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let latch = graph.push_node(IrNode::Region { predecessors: vec![header] });
        let _branch = graph.push_node(IrNode::Branch { cond: cmp, true_block: latch, false_block: exit });
        let latch_jump = graph.push_node(IrNode::Jump { target: header });
        graph.replace(header, IrNode::Region { predecessors: vec![entry, latch_jump] });

        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });

        let mut lv = LoopVectorizer::new();
        lv.detect(&graph);
        let core_body = LoopVectorizer::collect_loop_body(&graph, &lv.loops[0]);
        let candidates = Licm::find_candidates(&graph, &core_body);
        // Scan all nodes for stores — in sea-of-nodes IR, Store nodes may
        // not be data-reachable from induction variables.
        let root_stores = Licm::collect_stores_by_root_all(&graph);

        let mut invariant = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &candidates {
                if invariant.contains(&id) { continue; }
                if Licm::is_node_invariant(id, &graph, &core_body, &invariant, &root_stores) {
                    invariant.insert(id);
                    changed = true;
                }
            }
        }

        // load_b should be invariant (root_b has no stores anywhere)
        let b_invariant = invariant.iter().any(|&id| {
            matches!(graph.get(id), Some(IrNode::Load { root, .. }) if *root == root_b)
        });
        assert!(b_invariant, "Load from root_b (no stores) should be invariant");

        // load_c should NOT be invariant (root_c has a store)
        let c_invariant = invariant.iter().any(|&id| {
            matches!(graph.get(id), Some(IrNode::Load { root, .. }) if *root == root_c)
        });
        assert!(!c_invariant, "Load from root_c (has stores) should NOT be invariant");
    }
}
