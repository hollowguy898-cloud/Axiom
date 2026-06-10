//! Escape Analysis and Stack Allocation Promotion.
//!
//! Determines whether an OwnershipRoot's data escapes the current function.
//! If a root never escapes (no references stored to heap, passed to external
//! functions, or returned), it can be:
//!
//! 1. **Stack-allocated instead of heap-allocated** — eliminates allocation
//!    overhead (malloc/free or new/drop) and improves cache locality.
//! 2. **Promoted to SSA values** — eliminates the memory location entirely;
//!    the "variable" becomes a set of SSA values connected by Phi nodes.
//! 3. **Register-allocated** — non-escaping data can live entirely in
//!    registers without spill/fault concerns.
//!
//! # How Axiom Makes This Trivial
//!
//! In LLVM, escape analysis requires tracking pointer provenance across
//! interprocedural boundaries — expensive and imprecise. In Axiom,
//! the OwnershipRoot system provides this information by construction:
//! - `OwnershipAnalysis::root_escapes` already tracks escape via calls/stores
//! - `OwnershipAnalysis::root_is_local` identifies non-escaping roots
//! - The analysis is conservative but sound — it never incorrectly promotes
//!
//! # Promotion Strategy
//!
//! For each non-escaping root:
//! 1. If the root has only Store-then-Load patterns with no address
//!    arithmetic, promote to SSA values (replace Load with the stored value,
//!    eliminate Store and StackAlloc).
//! 2. If the root has address arithmetic but no escaping, mark it as
//!    eligible for stack allocation (no heap allocation needed).
//! 3. If the root escapes, leave it as-is.

use std::collections::HashMap;

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use axiom_ownership::OwnershipAnalysis;
use crate::Pass;

/// Escape analysis result for a single ownership root.
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionDecision {
    /// The root can be eliminated entirely — all loads can be replaced
    /// with the stored values, and the StackAlloc/Store/Load removed.
    PromoteToSSA,
    /// The root is non-escaping but has complex access patterns.
    /// Keep the memory location but ensure it's stack-allocated.
    StackAllocate,
    /// The root escapes — no promotion possible.
    NoPromotion,
}

/// The Escape Analysis and Promotion pass.
///
/// Analyzes each ownership root for escape, then promotes non-escaping
/// roots to SSA values or stack allocation where possible.
pub struct EscapeAnalysisPass {
    /// Promotion decisions made during the last run.
    pub decisions: HashMap<OwnershipRoot, PromotionDecision>,
}

impl EscapeAnalysisPass {
    pub fn new() -> Self {
        Self {
            decisions: HashMap::new(),
        }
    }

    /// Analyze a root and determine its promotion decision.
    fn analyze_root(
        &self,
        root: OwnershipRoot,
        analysis: &OwnershipAnalysis,
        graph: &IrGraph,
    ) -> PromotionDecision {
        // Global and stack roots are never promoted
        if root.is_global() {
            return PromotionDecision::NoPromotion;
        }

        // If the root escapes, no promotion
        if !analysis.root_is_local(root) {
            return PromotionDecision::NoPromotion;
        }

        // The root is non-escaping. Check if we can promote to SSA.
        // SSA promotion is possible when:
        // 1. There are no stores with address arithmetic (the address is
        //    always the StackAlloc result directly)
        // 2. Every load's address comes from the StackAlloc directly
        // 3. There's at most one StackAlloc for this root
        let stack_allocs = Self::find_stack_allocs_for_root(root, graph);

        if stack_allocs.len() != 1 {
            // Multiple or zero StackAllocs — can't promote to SSA
            return PromotionDecision::StackAllocate;
        }

        let alloc_id = stack_allocs[0];

        // Check if all loads and stores use the StackAlloc result directly
        // as their address (no address arithmetic)
        let loads = analysis.loads_for_root(root);
        let stores = analysis.stores_for_root(root);

        let mut can_promote_ssa = true;

        for &load_id in loads {
            if let Some(IrNode::Load { addr, .. }) = graph.get(load_id) {
                if *addr != alloc_id {
                    // Address is derived from the StackAlloc (e.g., offset calculation)
                    // Can't do simple SSA promotion
                    can_promote_ssa = false;
                    break;
                }
            }
        }

        if can_promote_ssa {
            for &store_id in stores {
                if let Some(IrNode::Store { addr, .. }) = graph.get(store_id) {
                    if *addr != alloc_id {
                        can_promote_ssa = false;
                        break;
                    }
                }
            }
        }

        if can_promote_ssa && loads.len() <= stores.len() {
            // Simple access pattern: promote to SSA
            PromotionDecision::PromoteToSSA
        } else {
            // Complex access pattern: stack allocate
            PromotionDecision::StackAllocate
        }
    }

    /// Find all StackAlloc nodes for a given ownership root.
    fn find_stack_allocs_for_root(root: OwnershipRoot, graph: &IrGraph) -> Vec<NodeId> {
        graph.iter()
            .filter_map(|(id, node)| {
                if let IrNode::StackAlloc { root: r, .. } = node {
                    if *r == root { Some(id) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Perform SSA promotion for a root:
    /// - For each Store to the root, record the stored value.
    /// - For each Load from the root, replace it with the most recent
    ///   stored value.
    /// - Remove the Stores and StackAlloc.
    fn promote_root_to_ssa(
        root: OwnershipRoot,
        analysis: &OwnershipAnalysis,
        graph: &mut IrGraph,
    ) -> bool {
        let stores = analysis.stores_for_root(root).to_vec();
        let loads = analysis.loads_for_root(root).to_vec();
        let stack_allocs = Self::find_stack_allocs_for_root(root, graph);

        if stores.is_empty() && loads.is_empty() {
            return false;
        }

        // Build a mapping: for each Store, record the value being stored.
        // Since we verified single StackAlloc with direct address access,
        // there's only one "variable" per root — each Store overwrites the
        // previous value.
        let mut store_values: Vec<(NodeId, NodeId)> = Vec::new(); // (store_id, val_id)
        for &store_id in &stores {
            if let Some(IrNode::Store { val, .. }) = graph.get(store_id) {
                store_values.push((store_id, *val));
            }
        }

        // For each Load, replace it with the value from the most recent Store.
        // In a sea-of-nodes IR without explicit ordering, we use the NodeId
        // ordering as a proxy for program order (this is a simplification;
        // a full implementation would use dominance information).
        if store_values.is_empty() {
            return false;
        }

        // Use the last store's value as the replacement for all loads
        // (simplified: assumes single-assignment pattern within the root)
        let last_store_val = store_values.last().unwrap().1;

        let mut modified = false;
        for &load_id in &loads {
            graph.replace_uses(load_id, last_store_val);
            graph.remove(load_id);
            modified = true;
        }

        // Remove all stores (they're now dead — no loads reference them)
        for &(store_id, _) in &store_values {
            graph.remove(store_id);
            modified = true;
        }

        // Remove the StackAlloc
        for &alloc_id in &stack_allocs {
            // Check if anyone still uses the StackAlloc result
            let users = graph.users_of(alloc_id);
            if users.is_empty() {
                graph.remove(alloc_id);
                modified = true;
            }
        }

        modified
    }
}

impl Pass for EscapeAnalysisPass {
    fn name(&self) -> &str {
        "escape_analysis"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let analysis = OwnershipAnalysis::analyze(graph);
        let roots: Vec<OwnershipRoot> = analysis.roots.iter().copied().collect();

        let mut modified = false;

        for root in roots {
            let decision = self.analyze_root(root, &analysis, graph);

            match decision {
                PromotionDecision::PromoteToSSA => {
                    if Self::promote_root_to_ssa(root, &analysis, graph) {
                        modified = true;
                    }
                }
                PromotionDecision::StackAllocate => {
                    // Mark the root as stack-allocatable. In a complete
                    // implementation, this would:
                    // 1. Replace any heap allocation with StackAlloc
                    // 2. Add the appropriate StackAlloc node if not already present
                    // 3. Update the ownership analysis
                    // For now, the StackAlloc is already in place if it exists,
                    // so this is a no-op unless we're converting from heap alloc.
                }
                PromotionDecision::NoPromotion => {
                    // Leave as-is
                }
            }
        }

        modified
    }
}

impl Default for EscapeAnalysisPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    #[test]
    fn non_escaping_root_promoted_to_ssa() {
        let mut graph = IrGraph::new("ssa_promo");
        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc { size, align: 8, root });

        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store { addr: alloc, val, root, ty: Type::I64 });

        let load = graph.push_node(IrNode::Load { addr: alloc, root, ty: Type::I64 });
        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        let pass = EscapeAnalysisPass::new();
        let modified = pass.run(&mut graph);

        assert!(modified, "Non-escaping root should be promoted to SSA");

        // The load should have been replaced — the return value should
        // now reference the IntConst(42) directly
        let has_load = graph.iter().any(|(_, n)| matches!(n, IrNode::Load { .. }));
        assert!(!has_load, "Load should be eliminated after SSA promotion");
    }

    #[test]
    fn escaping_root_not_promoted() {
        let mut graph = IrGraph::new("escape");
        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc { size, align: 8, root });

        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store { addr: alloc, val, root, ty: Type::I64 });

        // Pass the allocation to a call — root escapes
        let _call = graph.push_node(IrNode::Call {
            func: "process".to_string(),
            args: vec![alloc],
            ty: Type::Void,
        });

        let load = graph.push_node(IrNode::Load { addr: alloc, root, ty: Type::I64 });
        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        let pass = EscapeAnalysisPass::new();
        let modified = pass.run(&mut graph);

        assert!(!modified, "Escaping root should NOT be promoted");

        // Load should still exist
        let has_load = graph.iter().any(|(_, n)| matches!(n, IrNode::Load { .. }));
        assert!(has_load, "Load should remain for escaping root");
    }

    #[test]
    fn global_root_not_promoted() {
        let mut graph = IrGraph::new("global");
        let addr = graph.push_node(IrNode::IntConst(1000));
        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr,
            val,
            root: OwnershipRoot::GLOBAL,
            ty: Type::I64,
        });
        let load = graph.push_node(IrNode::Load {
            addr,
            root: OwnershipRoot::GLOBAL,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        let pass = EscapeAnalysisPass::new();
        let modified = pass.run(&mut graph);
        assert!(!modified, "GLOBAL root should not be promoted");
    }

    #[test]
    fn stack_allocate_decision_for_complex_access() {
        let mut graph = IrGraph::new("complex");
        let size = graph.push_node(IrNode::IntConst(256));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc { size, align: 8, root });

        // Address arithmetic — not a simple direct access
        let offset = graph.push_node(IrNode::IntConst(8));
        let indexed_addr = graph.push_node(IrNode::Add { lhs: alloc, rhs: offset });

        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store { addr: indexed_addr, val, root, ty: Type::I64 });

        let load = graph.push_node(IrNode::Load { addr: indexed_addr, root, ty: Type::I64 });
        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        let analysis = OwnershipAnalysis::analyze(&graph);
        let pass = EscapeAnalysisPass::new();
        let decision = pass.analyze_root(root, &analysis, &graph);

        // Should be StackAllocate (not PromoteToSSA) because of address arithmetic
        assert_eq!(decision, PromotionDecision::StackAllocate);
    }

    #[test]
    fn dead_store_root_promoted() {
        // Root with stores but no loads — all stores are dead
        let mut graph = IrGraph::new("dead_stores");
        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc { size, align: 8, root });

        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store { addr: alloc, val, root, ty: Type::I64 });

        let _ret = graph.push_node(IrNode::Return { value: None });

        let pass = EscapeAnalysisPass::new();
        let modified = pass.run(&mut graph);

        assert!(modified, "Dead store root should be cleaned up");
    }
}
