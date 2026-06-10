//! Ownership-Aware Dead Store Elimination.
//!
//! Leverages `OwnershipRoot` for trivially correct DSE — no alias analysis
//! needed. The key insight: operations on different roots never alias.
//!
//! # Algorithm
//!
//! 1. Group `Store` and `Load` nodes by ownership root.
//! 2. For each root:
//!    - If there are **no** `Load`s from the root, all `Store`s to that root
//!      are dead (the stored values are never read).
//!    - If there are `Load`s, walk the graph in NodeId order and track the
//!      most recent `Store` per root. If a `Store` is followed by another
//!      `Store` to the same root with no intervening `Load`, the first
//!      `Store` is dead.
//! 3. Remove dead `Store`s.
//!
//! The ownership root makes this **trivially correct**: we never need to
//! worry about pointer aliasing across different roots.

use std::collections::{HashMap, HashSet};

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use crate::Pass;

/// Ownership-Aware Dead Store Elimination pass.
pub struct DeadStoreElim;

impl DeadStoreElim {
    /// Classify all memory operations by ownership root.
    fn collect_by_root(
        graph: &IrGraph,
    ) -> (
        HashMap<OwnershipRoot, Vec<NodeId>>, // stores
        HashMap<OwnershipRoot, Vec<NodeId>>, // loads
    ) {
        let mut stores: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        let mut loads: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();

        for (id, node) in graph.iter() {
            match node {
                IrNode::Store { root, .. } => {
                    stores.entry(*root).or_default().push(id);
                }
                IrNode::Load { root, .. } => {
                    loads.entry(*root).or_default().push(id);
                }
                IrNode::VarSet { root, .. } => {
                    // VarSet is also a store-like operation.
                    stores.entry(*root).or_default().push(id);
                }
                _ => {}
            }
        }

        (stores, loads)
    }

    /// Identify dead stores using a simple NodeId-order walk.
    ///
    /// A Store is dead if:
    /// - Its root has no Loads at all, OR
    /// - It is followed by another Store to the same root with no
    ///   intervening Load from the same root.
    fn find_dead_stores(graph: &IrGraph) -> HashSet<NodeId> {
        let (stores_by_root, loads_by_root) = Self::collect_by_root(graph);
        let mut dead = HashSet::new();

        for (root, store_ids) in &stores_by_root {
            let has_loads = loads_by_root.contains_key(root);

            if !has_loads {
                // No loads from this root at all — all stores are dead.
                for &id in store_ids {
                    dead.insert(id);
                }
                continue;
            }

            // There are loads. Walk all nodes in NodeId order and track
            // the most recent store per root. If a new store to root R
            // appears and the previous store to R had no intervening load
            // from R, the previous store is dead.
            let load_ids: HashSet<NodeId> = loads_by_root[root].iter().copied().collect();
            let store_set: HashSet<NodeId> = store_ids.iter().copied().collect();

            let mut last_store: Option<NodeId> = None;
            let mut saw_load_since_last_store = true; // initially true so the first store isn't prematurely dead

            // Collect all nodes sorted by ID for deterministic order.
            let mut all_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();
            all_ids.sort_by_key(|id| id.0);

            for id in all_ids {
                if store_set.contains(&id) {
                    // This is a Store to root R.
                    if let Some(prev) = last_store {
                        if !saw_load_since_last_store {
                            // The previous store was never read — it's dead.
                            dead.insert(prev);
                        }
                    }
                    last_store = Some(id);
                    saw_load_since_last_store = false;
                } else if load_ids.contains(&id) {
                    // This is a Load from root R.
                    saw_load_since_last_store = true;
                }
            }
        }

        dead
    }
}

impl Pass for DeadStoreElim {
    fn name(&self) -> &str {
        "dse"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let dead = Self::find_dead_stores(graph);

        if dead.is_empty() {
            return false;
        }

        for id in &dead {
            graph.remove(*id);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    #[test]
    fn dse_removes_store_with_no_loads() {
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let addr = graph.push_node(IrNode::IntConst(100));
        let val = graph.push_node(IrNode::IntConst(42));
        let store = graph.push_node(IrNode::Store {
            addr,
            val,
            root,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: None });

        let dse = DeadStoreElim;
        assert!(dse.run(&mut graph));
        assert!(graph.get(store).is_none());
    }

    #[test]
    fn dse_preserves_store_with_loads() {
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let addr = graph.push_node(IrNode::IntConst(100));
        let val = graph.push_node(IrNode::IntConst(42));
        let store = graph.push_node(IrNode::Store {
            addr,
            val,
            root,
            ty: Type::I64,
        });
        let load = graph.push_node(IrNode::Load {
            addr,
            root,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        let dse = DeadStoreElim;
        assert!(!dse.run(&mut graph));
        assert!(graph.get(store).is_some());
    }

    #[test]
    fn dse_removes_overwritten_store() {
        // Store1 → Store2 (same root, no Load between) → Load
        // Store1 is dead because Store2 overwrites before any Load.
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let addr = graph.push_node(IrNode::IntConst(100));
        let val1 = graph.push_node(IrNode::IntConst(1));
        let store1 = graph.push_node(IrNode::Store {
            addr,
            val: val1,
            root,
            ty: Type::I64,
        });
        let val2 = graph.push_node(IrNode::IntConst(2));
        let store2 = graph.push_node(IrNode::Store {
            addr,
            val: val2,
            root,
            ty: Type::I64,
        });
        let load = graph.push_node(IrNode::Load {
            addr,
            root,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        let dse = DeadStoreElim;
        assert!(dse.run(&mut graph));

        // store1 should be removed; store2 should remain.
        assert!(graph.get(store1).is_none());
        assert!(graph.get(store2).is_some());
    }

    #[test]
    fn dse_different_roots_independent() {
        // Store to root A with no loads → dead.
        // Store to root B with a load → alive.
        let mut graph = IrGraph::new("test");
        let root_a = OwnershipRoot::new(10);
        let root_b = OwnershipRoot::new(11);
        let addr = graph.push_node(IrNode::IntConst(100));

        let val_a = graph.push_node(IrNode::IntConst(1));
        let store_a = graph.push_node(IrNode::Store {
            addr,
            val: val_a,
            root: root_a,
            ty: Type::I64,
        });

        let val_b = graph.push_node(IrNode::IntConst(2));
        let store_b = graph.push_node(IrNode::Store {
            addr,
            val: val_b,
            root: root_b,
            ty: Type::I64,
        });
        let load_b = graph.push_node(IrNode::Load {
            addr,
            root: root_b,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load_b) });

        let dse = DeadStoreElim;
        assert!(dse.run(&mut graph));

        // store_a should be removed (no loads from root_a).
        assert!(graph.get(store_a).is_none());
        // store_b should remain (load_b reads from root_b).
        assert!(graph.get(store_b).is_some());
    }
}
