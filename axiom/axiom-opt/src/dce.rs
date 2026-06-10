//! Dead Code Elimination.
//!
//! Removes nodes that have no users and no side effects.
//! Iterates until no more nodes can be removed.

use axiom_ir::{IrGraph, IrNode};
use crate::Pass;

/// Dead Code Elimination pass.
///
/// A node is dead if:
/// 1. It has no users (nothing references it as an input), AND
/// 2. It has no side effects (no memory writes, control flow, or calls).
///
/// The Start node is always preserved. The pass iterates until fixed point
/// because removing one node may make its transitive inputs dead.
pub struct DeadCodeElim;

impl Pass for DeadCodeElim {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        loop {
            // Collect all node IDs that are dead.
            let mut to_remove = Vec::new();
            for (id, node) in graph.iter() {
                // Never remove the Start node.
                if matches!(node, IrNode::Start) {
                    continue;
                }
                // Nodes with side effects are never dead.
                if node.has_side_effects() {
                    continue;
                }
                // Nodes with no users are dead.
                if graph.users_of(id).is_empty() {
                    to_remove.push(id);
                }
            }

            if to_remove.is_empty() {
                break;
            }

            for id in to_remove {
                graph.remove(id);
                modified = true;
            }
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    #[test]
    fn dce_removes_dead_node() {
        let mut graph = IrGraph::new("test");
        let _start = graph.start_node();
        // Create a constant that nobody uses.
        let dead_const = graph.push_node(IrNode::IntConst(42));
        // Create a used constant.
        let used_const = graph.push_node(IrNode::IntConst(10));
        let _ret = graph.push_node(IrNode::Return { value: Some(used_const) });

        let dce = DeadCodeElim;
        assert!(dce.run(&mut graph));
        assert!(graph.get(dead_const).is_none());
        assert!(graph.get(used_const).is_some());
    }

    #[test]
    fn dce_preserves_side_effects() {
        let mut graph = IrGraph::new("test");
        let addr = graph.push_node(IrNode::IntConst(0));
        let val = graph.push_node(IrNode::IntConst(99));
        // Store has side effects — should NOT be removed even if no one uses the result.
        let store = graph.push_node(IrNode::Store {
            addr,
            val,
            root: axiom_ir::OwnershipRoot::STACK,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: None });

        let dce = DeadCodeElim;
        assert!(!dce.run(&mut graph));
        assert!(graph.get(store).is_some());
    }

    #[test]
    fn dce_iterates_to_fixed_point() {
        let mut graph = IrGraph::new("test");
        // Chain: a -> b -> c, where only c is used.
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::Neg { val: a });
        let c = graph.push_node(IrNode::Neg { val: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(c) });

        let dce = DeadCodeElim;
        // Nothing is dead initially — a, b, c are all used.
        assert!(!dce.run(&mut graph));
    }

    #[test]
    fn dce_removes_chain_of_dead() {
        let mut graph = IrGraph::new("test");
        // Chain: a -> b -> c, where nobody uses any of them.
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::Neg { val: a });
        let c = graph.push_node(IrNode::Neg { val: b });
        // Use a different value for the return so c is dead.
        let other = graph.push_node(IrNode::IntConst(99));
        let _ret = graph.push_node(IrNode::Return { value: Some(other) });

        let dce = DeadCodeElim;
        assert!(dce.run(&mut graph));
        assert!(graph.get(a).is_none());
        assert!(graph.get(b).is_none());
        assert!(graph.get(c).is_none());
    }
}
