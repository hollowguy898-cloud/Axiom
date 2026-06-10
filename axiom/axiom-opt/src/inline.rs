//! Function Inlining.
//!
//! Replaces `Call` nodes with the inlined body of the callee function.
//! Only inlines if the callee is small enough (≤ threshold nodes, default 20).
//!
//! # Convention
//!
//! Callee parameters are represented as `VarRef` nodes with names `"arg0"`,
//! `"arg1"`, etc. When inlining, these are replaced with the actual argument
//! `NodeId`s from the call site.

use std::collections::HashMap;

use axiom_ir::{IrGraph, IrNode, NodeId};
use crate::Pass;

/// Function Inlining pass.
///
/// Stores a map of function definitions and a size threshold. Only functions
/// with `node_count() <= threshold` are inlined.
pub struct Inliner {
    /// Available function definitions: function name → IR graph.
    pub functions: HashMap<String, IrGraph>,
    /// Maximum number of nodes in a callee for it to be inlined.
    pub threshold: usize,
}

impl Inliner {
    pub fn new(functions: HashMap<String, IrGraph>, threshold: usize) -> Self {
        Self { functions, threshold }
    }

    /// Inline a single call site. Returns `Some(return_value_id)` on success,
    /// `None` if inlining is not possible.
    fn inline_call(
        &self,
        caller: &mut IrGraph,
        _call_id: NodeId,
        func_name: &str,
        call_args: &[NodeId],
        _call_ty: axiom_ir::nodes::Type,
    ) -> Option<NodeId> {
        let callee = self.functions.get(func_name)?;

        // Check size threshold.
        if callee.node_count() > self.threshold {
            return None;
        }

        // ── Build the old→new NodeId mapping ────────────────────────
        //
        // We process callee nodes in NodeId order (which is topological
        // for a well-formed IR) and:
        //   - Map the callee's Start → the caller's Start
        //   - Map VarRef("arg{i}") → call_args[i]
        //   - Skip Start, Return, and VarRef("arg{i}") nodes (don't clone)
        //   - For all other nodes, remap inputs and push into the caller

        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Map callee Start → caller Start.
        id_map.insert(callee.start_node(), caller.start_node());

        // Map argument VarRef nodes → call arguments.
        for (i, &arg_id) in call_args.iter().enumerate() {
            let arg_name = format!("arg{}", i);
            if let Some(arg_ref_id) = callee.lookup_var(&arg_name) {
                id_map.insert(arg_ref_id, arg_id);
            }
        }

        // Collect callee nodes in order.
        let callee_nodes: Vec<(NodeId, IrNode)> = callee
            .iter()
            .map(|(id, n)| (id, n.clone()))
            .collect();

        // We'll track the Return node's value so we can substitute it.
        let mut return_value: Option<NodeId> = None;

        for (old_id, node) in &callee_nodes {
            // Skip Start node (already mapped).
            if matches!(node, IrNode::Start) {
                continue;
            }

            // Skip VarRef("arg{i}") nodes (already mapped).
            if let IrNode::VarRef { name, .. } = node {
                if name.starts_with("arg") && name[3..].parse::<usize>().is_ok() {
                    continue;
                }
            }

            // Handle Return: record the value node (remapped).
            if let IrNode::Return { value } = node {
                return_value = value.map(|v| {
                    id_map.get(&v).copied().unwrap_or(v)
                });
                continue;
            }

            // Remap all inputs in this node using id_map.
            let remapped = node.clone().map_inputs(|input_id| {
                id_map.get(&input_id).copied().unwrap_or(input_id)
            });

            // Push the remapped node into the caller graph.
            let new_id = caller.push_node(remapped);
            id_map.insert(*old_id, new_id);
        }

        // The Call node is replaced by the return value.
        // If the callee returns void, we use a placeholder; otherwise we use
        // the remapped return value.
        let replacement = return_value.unwrap_or_else(|| {
            // Void function — push a void/undef placeholder.
            caller.push_node(IrNode::UndefConst)
        });

        Some(replacement)
    }
}

impl Pass for Inliner {
    fn name(&self) -> &str {
        "inline"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;

        // Collect all Call nodes that might be inlineable.
        let call_sites: Vec<(NodeId, String, Vec<NodeId>, axiom_ir::nodes::Type)> = graph
            .iter()
            .filter_map(|(id, node)| {
                if let IrNode::Call { func, args, ty } = node {
                    Some((id, func.clone(), args.clone(), *ty))
                } else {
                    None
                }
            })
            .collect();

        for (call_id, func_name, call_args, call_ty) in call_sites {
            // Skip if the call node was already removed.
            if graph.get(call_id).is_none() {
                continue;
            }

            if let Some(replacement) = self.inline_call(graph, call_id, &func_name, &call_args, call_ty) {
                // Replace all uses of the Call with the inlined result.
                graph.replace_uses(call_id, replacement);
                graph.remove(call_id);
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
    fn inline_small_function() {
        // Build a callee: fn add(a, b) -> a + b
        let mut callee = IrGraph::new("add");
        let _start = callee.start_node();

        // Create arg VarRefs.
        let arg0 = callee.push_node(IrNode::VarRef {
            name: "arg0".to_string(),
            ty: Type::I64,
        });
        callee.define_var("arg0", arg0);

        let arg1 = callee.push_node(IrNode::VarRef {
            name: "arg1".to_string(),
            ty: Type::I64,
        });
        callee.define_var("arg1", arg1);

        let sum = callee.push_node(IrNode::Add { lhs: arg0, rhs: arg1 });
        let _ret = callee.push_node(IrNode::Return { value: Some(sum) });

        // Build a caller: fn main() { add(3, 4) }
        let mut caller = IrGraph::new("main");
        let three = caller.push_node(IrNode::IntConst(3));
        let four = caller.push_node(IrNode::IntConst(4));
        let call = caller.push_node(IrNode::Call {
            func: "add".to_string(),
            args: vec![three, four],
            ty: Type::I64,
        });
        let _ret = caller.push_node(IrNode::Return { value: Some(call) });

        let mut functions = HashMap::new();
        functions.insert("add".to_string(), callee);

        let inliner = Inliner::new(functions, 20);
        assert!(inliner.run(&mut caller));

        // The Call node should be gone.
        assert!(caller.get(call).is_none());

        // There should be an Add node in the graph now.
        let has_add = caller.iter().any(|(_, n)| matches!(n, IrNode::Add { .. }));
        assert!(has_add);
    }

    #[test]
    fn no_inline_above_threshold() {
        // Build a callee that exceeds the threshold.
        let mut callee = IrGraph::new("big_fn");
        let mut last = callee.push_node(IrNode::IntConst(0));
        for i in 1..30 {
            let c = callee.push_node(IrNode::IntConst(i));
            last = callee.push_node(IrNode::Add { lhs: last, rhs: c });
        }
        let _ret = callee.push_node(IrNode::Return { value: Some(last) });

        let mut caller = IrGraph::new("main");
        let call = caller.push_node(IrNode::Call {
            func: "big_fn".to_string(),
            args: vec![],
            ty: Type::I64,
        });
        let _ret = caller.push_node(IrNode::Return { value: Some(call) });

        let mut functions = HashMap::new();
        functions.insert("big_fn".to_string(), callee);

        let inliner = Inliner::new(functions, 20);
        assert!(!inliner.run(&mut caller));

        // The Call should still be present.
        assert!(caller.get(call).is_some());
    }

    #[test]
    fn no_inline_unknown_function() {
        let mut caller = IrGraph::new("main");
        let call = caller.push_node(IrNode::Call {
            func: "unknown".to_string(),
            args: vec![],
            ty: Type::Void,
        });
        let _ret = caller.push_node(IrNode::Return { value: None });

        let inliner = Inliner::new(HashMap::new(), 20);
        assert!(!inliner.run(&mut caller));
        assert!(caller.get(call).is_some());
    }
}
