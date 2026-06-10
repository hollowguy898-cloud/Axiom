//! Superword-Level Parallelism (Auto-vectorization stub).
//!
//! Detects adjacent isomorphic operations that could be vectorized.
//! This pass does NOT transform the graph — it only identifies
//! vectorization opportunities.
//!
//! # Detection Strategy
//!
//! 1. Find groups of the same operation type (e.g., multiple `Add` nodes).
//! 2. For each group, check if the inputs are `Load` nodes from adjacent
//!    memory addresses within the same ownership root.
//! 3. Group such operations into `VectorCandidate`s.

use std::collections::HashMap;

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use axiom_ir::nodes::Type;
use crate::Pass;

/// A detected vectorization opportunity.
#[derive(Debug, Clone)]
pub struct VectorCandidate {
    /// The operation nodes that could be fused into a single vector op.
    pub ops: Vec<NodeId>,
    /// Name of the operation (e.g., "add", "mul").
    pub op_name: String,
    /// Element type of the operation (if known).
    pub element_type: Option<Type>,
    /// Ownership root of the memory operations involved (if any).
    pub root: Option<OwnershipRoot>,
}

/// Superword-Level Parallelism detection pass.
///
/// Populates `candidates` with detected vectorization opportunities.
/// The `run()` method always returns `false` (no graph modification).
pub struct SlpVectorizer {
    /// Detected vectorization candidates from the last run.
    pub candidates: Vec<VectorCandidate>,
}

impl SlpVectorizer {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Classify a node's operation name (for grouping isomorphic ops).
    fn op_name(node: &IrNode) -> Option<&'static str> {
        match node {
            IrNode::Add { .. } => Some("add"),
            IrNode::Sub { .. } => Some("sub"),
            IrNode::Mul { .. } => Some("mul"),
            IrNode::And { .. } => Some("and"),
            IrNode::Or { .. } => Some("or"),
            IrNode::Xor { .. } => Some("xor"),
            IrNode::Shl { .. } => Some("shl"),
            IrNode::Shr { .. } => Some("shr"),
            IrNode::Sar { .. } => Some("sar"),
            IrNode::Eq { .. } => Some("eq"),
            IrNode::Ne { .. } => Some("ne"),
            IrNode::Lt { .. } => Some("lt"),
            IrNode::Le { .. } => Some("le"),
            IrNode::Gt { .. } => Some("gt"),
            IrNode::Ge { .. } => Some("ge"),
            _ => None,
        }
    }

    /// Try to extract an address offset from a node that computes
    /// `base + constant_offset`. Returns `(base_node_id, offset)` if
    /// the pattern matches.
    fn extract_offset(graph: &IrGraph, id: NodeId) -> Option<(NodeId, i64)> {
        let node = graph.get(id)?;
        match node {
            IrNode::Add { lhs, rhs } => {
                // Check if either input is a constant.
                if let Some(IrNode::IntConst(offset)) = graph.get(*rhs) {
                    Some((*lhs, *offset))
                } else if let Some(IrNode::IntConst(offset)) = graph.get(*lhs) {
                    Some((*rhs, *offset))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Detect groups of isomorphic operations on adjacent memory.
    fn detect_candidates(&mut self, graph: &IrGraph) {
        self.candidates.clear();

        // Group binary operation nodes by (op_name, element_type).
        let mut op_groups: HashMap<(&'static str, Option<Type>), Vec<NodeId>> = HashMap::new();

        for (id, node) in graph.iter() {
            if let Some(name) = Self::op_name(node) {
                let ty = Some(node.output_type());
                let ty = if ty == Some(Type::Unknown) { None } else { ty };
                op_groups.entry((name, ty)).or_default().push(id);
            }
        }

        // For each group, try to find adjacent-load subgroups.
        for ((op_name, element_type), node_ids) in op_groups {
            if node_ids.len() < 2 {
                continue; // Need at least 2 ops to vectorize.
            }

            // Collect (node_id, lhs_load_info, rhs_load_info) for nodes
            // whose inputs are Loads.
            let mut load_based: Vec<(NodeId, Option<(NodeId, i64, OwnershipRoot, Type)>, Option<(NodeId, i64, OwnershipRoot, Type)>)> = Vec::new();

            for &id in &node_ids {
                let node = graph.get(id).unwrap();
                let inputs = node.inputs();
                if inputs.len() != 2 {
                    continue;
                }

                let lhs_info = Self::load_info(graph, inputs[0]);
                let rhs_info = Self::load_info(graph, inputs[1]);

                load_based.push((id, lhs_info, rhs_info));
            }

            // Group by (root, base_addr_node) to find adjacent accesses.
            // We look for Load inputs that share the same base address and
            // root, with sequential offsets.
            let mut root_base_groups: HashMap<(OwnershipRoot, NodeId), Vec<(NodeId, i64)>> =
                HashMap::new();

            for (id, lhs_info, _rhs_info) in &load_based {
                if let Some((base, offset, root, _ty)) = lhs_info {
                    root_base_groups
                        .entry((*root, *base))
                        .or_default()
                        .push((*id, *offset));
                }
            }

            // Within each (root, base) group, sort by offset and look for
            // contiguous sequences.
            for ((root, _base), mut entries) in root_base_groups {
                if entries.len() < 2 {
                    continue;
                }

                entries.sort_by_key(|(_, offset)| *offset);

                // Find contiguous sequences of same-type, same-stride loads.
                let byte_size = element_type.map(|t| t.byte_size()).unwrap_or(4) as i64;
                let stride = byte_size; // Adjacent elements.

                let mut seq_start = 0;
                for i in 1..=entries.len() {
                    let is_contiguous = i < entries.len()
                        && entries[i].1 - entries[i - 1].1 == stride;

                    if !is_contiguous {
                        let seq_len = i - seq_start;
                        if seq_len >= 2 {
                            let ops: Vec<NodeId> = entries[seq_start..i]
                                .iter()
                                .map(|(id, _)| *id)
                                .collect();
                            self.candidates.push(VectorCandidate {
                                ops,
                                op_name: op_name.to_string(),
                                element_type,
                                root: Some(root),
                            });
                        }
                        seq_start = i;
                    }
                }
            }
        }
    }

    /// If `id` is a Load node, return (base_addr, offset, root, ty).
    fn load_info(graph: &IrGraph, id: NodeId) -> Option<(NodeId, i64, OwnershipRoot, Type)> {
        let node = graph.get(id)?;
        if let IrNode::Load { addr, root, ty } = node {
            // Try to decompose the address into base + offset.
            if let Some((base, offset)) = Self::extract_offset(graph, *addr) {
                Some((base, offset, *root, *ty))
            } else {
                // No offset — treat as base+0.
                Some((*addr, 0, *root, *ty))
            }
        } else {
            None
        }
    }
}

impl Pass for SlpVectorizer {
    fn name(&self) -> &str {
        "slp_vectorize"
    }

    fn run(&self, _graph: &mut IrGraph) -> bool {
        // Detection only — never modifies the graph.
        // We need interior mutability for the candidates field.
        // Since run() takes &self, we use a workaround: the user should
        // call detect() separately. The Pass trait's run() is a no-op here.
        false
    }
}

impl SlpVectorizer {
    /// Run detection and populate `candidates`. Call this instead of `run()`
    /// if you want the detection results.
    pub fn detect(&mut self, graph: &IrGraph) {
        self.detect_candidates(graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_adjacent_loads_add() {
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let base = graph.push_node(IrNode::IntConst(1000));

        // base + 0, base + 4, base + 8 — three adjacent i32 loads.
        let off0 = graph.push_node(IrNode::IntConst(0));
        let addr0 = graph.push_node(IrNode::Add { lhs: base, rhs: off0 });
        let load0 = graph.push_node(IrNode::Load {
            addr: addr0,
            root,
            ty: Type::I32,
        });

        let off4 = graph.push_node(IrNode::IntConst(4));
        let addr4 = graph.push_node(IrNode::Add { lhs: base, rhs: off4 });
        let load4 = graph.push_node(IrNode::Load {
            addr: addr4,
            root,
            ty: Type::I32,
        });

        // Two Add nodes consuming the adjacent loads.
        let c = graph.push_node(IrNode::IntConst(1));
        let add0 = graph.push_node(IrNode::Add { lhs: load0, rhs: c });
        let add4 = graph.push_node(IrNode::Add { lhs: load4, rhs: c });

        let _ret = graph.push_node(IrNode::Return { value: Some(add0) });

        let mut slp = SlpVectorizer::new();
        slp.detect(&graph);

        // Should detect at least one candidate.
        assert!(!slp.candidates.is_empty());

        // The candidate should involve the Add nodes.
        let add_candidate = slp.candidates.iter().find(|c| c.op_name == "add");
        assert!(add_candidate.is_some());
        let add_candidate = add_candidate.unwrap();
        assert!(add_candidate.ops.contains(&add0));
        assert!(add_candidate.ops.contains(&add4));
    }

    #[test]
    fn no_candidates_single_op() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let _add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: None });

        let mut slp = SlpVectorizer::new();
        slp.detect(&graph);
        assert!(slp.candidates.is_empty());
    }
}
