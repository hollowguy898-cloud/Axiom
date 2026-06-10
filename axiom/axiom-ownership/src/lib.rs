//! Axiom Ownership — Ownership analysis on the Sea-of-Nodes IR.
//!
//! This crate performs ownership analysis on the IR graph, which is the key
//! differentiator of Axiom. The `OwnershipRoot` system enables trivially correct
//! DSE, CSE with store checks, and ownership-aware register allocation.
//!
//! # Core Insight
//!
//! Operations on **different** ownership roots **never** alias. This allows:
//! - **Dead Store Elimination**: Stores to a root with no loads are trivially dead.
//! - **CSE with Store Checks**: Loads from the same root are only equivalent if
//!   no intervening store to that root has occurred.
//! - **Ownership-Aware Register Allocation**: Non-escaping roots can use stack
//!   slots or even registers exclusively, without spill/fault concerns.
//!
//! # Usage
//!
//! ```ignore
//! use axiom_ownership::OwnershipAnalysis;
//! use axiom_ir::IrGraph;
//!
//! let graph: IrGraph = /* ... */;
//! let analysis = OwnershipAnalysis::analyze(&graph);
//!
//! // Check if two nodes may alias
//! if analysis.may_alias(node_a, node_b) {
//!     // potentially aliasing — be conservative
//! }
//!
//! // Find dead stores
//! for root in &analysis.roots {
//!     for dead_store in analysis.dead_stores(*root) {
//!         // this store can be eliminated
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};

// ─── Ownership Analysis ─────────────────────────────────────────────────────

/// Result of ownership analysis on a function.
///
/// After calling `OwnershipAnalysis::analyze(graph)`, this structure contains:
/// - A mapping from every `NodeId` to its inferred `OwnershipRoot`
/// - The set of all ownership roots found in the function
/// - For each root, the Load and Store nodes that access it
/// - For each root, whether it escapes the function
#[derive(Debug, Clone)]
pub struct OwnershipAnalysis {
    /// Maps each NodeId to its inferred OwnershipRoot.
    /// Memory ops have explicit roots; other nodes inherit from their inputs.
    pub node_root: HashMap<NodeId, OwnershipRoot>,

    /// All ownership roots found in the function.
    pub roots: HashSet<OwnershipRoot>,

    /// For each root, the set of Load nodes that read from it.
    pub root_loads: HashMap<OwnershipRoot, Vec<NodeId>>,

    /// For each root, the set of Store nodes that write to it.
    pub root_stores: HashMap<OwnershipRoot, Vec<NodeId>>,

    /// For each root, whether it escapes the function (passed to calls,
    /// stored to global, etc.). Non-escaping roots are candidates for
    /// aggressive optimization.
    pub root_escapes: HashMap<OwnershipRoot, bool>,
}

impl OwnershipAnalysis {
    /// Analyze ownership for an entire function graph.
    ///
    /// This walks all nodes in the graph and:
    /// 1. Records explicit ownership roots from memory operations
    /// 2. Maps Load/Store nodes to their respective roots
    /// 3. Marks roots as escaping when their addresses are passed to calls
    /// 4. Propagates ownership roots from inputs for nodes without explicit roots
    /// 5. Builds the complete analysis result
    pub fn analyze(graph: &IrGraph) -> Self {
        let mut node_root: HashMap<NodeId, OwnershipRoot> = HashMap::new();
        let mut roots: HashSet<OwnershipRoot> = HashSet::new();
        let mut root_loads: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        let mut root_stores: HashMap<OwnershipRoot, Vec<NodeId>> = HashMap::new();
        let mut root_escapes: HashMap<OwnershipRoot, bool> = HashMap::new();

        // ── Phase 1: Collect explicit ownership roots and classify memory ops ──
        for (id, node) in graph.iter() {
            if let Some(root) = node.ownership_root() {
                node_root.insert(id, root);
                roots.insert(root);

                match node {
                    IrNode::Load { .. } => {
                        root_loads.entry(root).or_default().push(id);
                    }
                    IrNode::Store { .. } | IrNode::VarSet { .. } => {
                        root_stores.entry(root).or_default().push(id);
                    }
                    _ => {}
                }
            }
        }

        // ── Phase 2: Detect escaping roots ──
        //
        // A root escapes if:
        // - A StackAlloc's result (pointer) is passed as an argument to a Call/CallIndirect
        // - A value from an owned root is stored to the GLOBAL root
        // - A Call/CallIndirect receives an argument whose root is non-global and non-stack
        //
        // We conservatively mark roots as escaping if any node associated with
        // that root feeds into a call.

        // Build a reverse map: which nodes produce values from which root?
        // We already have node_root for nodes with explicit roots.
        // For StackAlloc, the result pointer is associated with its root.

        // Collect all StackAlloc nodes and their roots — the pointer value they
        // produce "belongs to" that root.
        let mut stack_alloc_roots: HashMap<NodeId, OwnershipRoot> = HashMap::new();
        for (id, node) in graph.iter() {
            if let IrNode::StackAlloc { root, .. } = node {
                stack_alloc_roots.insert(id, *root);
            }
        }

        // For calls, check if any argument is a StackAlloc pointer or
        // an Owned node from a non-global root.
        for (_id, node) in graph.iter() {
            let (args, is_call): (Vec<NodeId>, bool) = match node {
                IrNode::Call { args, .. } => (args.clone(), true),
                IrNode::CallIndirect { args, .. } => (args.clone(), true),
                IrNode::Intrinsic { args, .. } => (args.clone(), true),
                _ => (Vec::new(), false),
            };

            if is_call {
                for arg in args {
                    // If the argument is a StackAlloc result, its root escapes
                    if let Some(&root) = stack_alloc_roots.get(&arg) {
                        *root_escapes.entry(root).or_insert(true) = true;
                    }
                    // If the argument has an explicit ownership root, mark it as escaping
                    if let Some(&root) = node_root.get(&arg) {
                        if !root.is_global() {
                            *root_escapes.entry(root).or_insert(true) = true;
                        }
                    }
                    // Transitively check: if the arg's inputs come from a root, escape it
                    if let Some(arg_node) = graph.get(arg) {
                        for input in arg_node.inputs() {
                            if let Some(&root) = stack_alloc_roots.get(&input) {
                                *root_escapes.entry(root).or_insert(true) = true;
                            }
                            if let Some(&root) = node_root.get(&input) {
                                if !root.is_global() {
                                    *root_escapes.entry(root).or_insert(true) = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Storing to the GLOBAL root from a non-global root also means escape
        for (_id, node) in graph.iter() {
            if let IrNode::Store { val, root, .. } = node {
                if root.is_global() {
                    // The stored value might carry a non-global root
                    if let Some(&val_root) = node_root.get(val) {
                        if !val_root.is_global() {
                            *root_escapes.entry(val_root).or_insert(true) = true;
                        }
                    }
                    // Also check if val is a StackAlloc pointer
                    if let Some(&alloc_root) = stack_alloc_roots.get(val) {
                        *root_escapes.entry(alloc_root).or_insert(true) = true;
                    }
                }
            }
        }

        // ── Phase 3: Propagate ownership roots to nodes without explicit roots ──
        //
        // For nodes that don't have an explicit root, we propagate from their
        // inputs. This is a fixed-point iteration: we keep propagating until
        // no new mappings are discovered.
        //
        // The propagation rule: if a node has inputs, and any input has a
        // root, the node inherits that root. If multiple inputs have different
        // roots, we conservatively pick the first one found (or GLOBAL if
        // conflicting).

        // We iterate until no new assignments are made.
        let mut changed = true;
        while changed {
            changed = false;
            for (id, node) in graph.iter() {
                // Skip nodes that already have an assigned root
                if node_root.contains_key(&id) {
                    continue;
                }

                // Try to inherit root from inputs
                let inputs = node.inputs();
                for input in &inputs {
                    if let Some(&root) = node_root.get(input) {
                        node_root.insert(id, root);
                        roots.insert(root);
                        changed = true;
                        break;
                    }
                }
            }
        }

        // ── Phase 4: Ensure all roots have escape entries ──
        for root in &roots {
            root_escapes.entry(*root).or_insert(false);
        }

        // ── Phase 5: Classify VarSet as store-like operations ──
        for (id, node) in graph.iter() {
            if let IrNode::VarSet { root, .. } = node {
                root_stores.entry(*root).or_default().push(id);
            }
        }

        Self {
            node_root,
            roots,
            root_loads,
            root_stores,
            root_escapes,
        }
    }

    /// Check if two nodes may alias (they're on the same ownership root).
    ///
    /// Returns `true` if the nodes might reference the same memory location.
    /// Two nodes on different ownership roots are guaranteed not to alias.
    /// If either node's root is unknown, we conservatively assume aliasing.
    pub fn may_alias(&self, a: NodeId, b: NodeId) -> bool {
        match (self.node_root.get(&a), self.node_root.get(&b)) {
            (Some(ra), Some(rb)) => ra == rb,
            _ => true, // conservative: unknown roots may alias
        }
    }

    /// Check if a root has any loads (stores without loads are dead).
    pub fn root_has_loads(&self, root: OwnershipRoot) -> bool {
        self.root_loads.get(&root).map_or(false, |v| !v.is_empty())
    }

    /// Get all stores to a root that are dead (no intervening loads).
    ///
    /// If no loads exist for this root, all stores are dead.
    /// Otherwise, a simplified analysis returns an empty vec (full
    /// implementation would need dominance info).
    pub fn dead_stores(&self, root: OwnershipRoot) -> Vec<NodeId> {
        // If no loads exist for this root, all stores are dead
        if !self.root_has_loads(root) {
            return self.root_stores.get(&root).cloned().unwrap_or_default();
        }
        // Otherwise, need to check if specific stores are overwritten before any load.
        // This is a simplified analysis — full implementation would need dominance info.
        Vec::new()
    }

    /// Check if a root is non-escaping (suitable for aggressive optimization).
    ///
    /// Non-escaping roots can be:
    /// - Replaced with SSA values (eliminating stack allocation entirely)
    /// - Promoted to registers without spilling concerns
    /// - Optimized more aggressively in DSE/CSE
    pub fn root_is_local(&self, root: OwnershipRoot) -> bool {
        !root.is_global() && !self.root_escapes.get(&root).copied().unwrap_or(true)
    }

    /// Get all non-escaping (local) roots.
    ///
    /// These are roots that don't escape the function and are candidates
    /// for promotion to SSA values or register allocation.
    pub fn local_roots(&self) -> HashSet<OwnershipRoot> {
        self.roots
            .iter()
            .copied()
            .filter(|&r| self.root_is_local(r))
            .collect()
    }

    /// Get the ownership root for a specific node.
    pub fn get_root(&self, id: NodeId) -> Option<OwnershipRoot> {
        self.node_root.get(&id).copied()
    }

    /// Get all stores to a given root.
    pub fn stores_for_root(&self, root: OwnershipRoot) -> &[NodeId] {
        self.root_stores.get(&root).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all loads from a given root.
    pub fn loads_for_root(&self, root: OwnershipRoot) -> &[NodeId] {
        self.root_loads.get(&root).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check whether a store to `root_b` can kill a load from `root_a`.
    ///
    /// This is only possible if the roots are the same (same ownership root
    /// means potential aliasing; different roots means guaranteed no aliasing).
    pub fn store_kills_load(&self, store_root: OwnershipRoot, load_root: OwnershipRoot) -> bool {
        store_root == load_root
    }
}

// ─── Ownership Verifier ─────────────────────────────────────────────────────

/// An ownership error discovered during verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnershipError {
    /// A node references an input from a different ownership root without
    /// proper annotation (e.g., storing a pointer from root A into root B).
    CrossRootReference {
        node: NodeId,
        from_root: OwnershipRoot,
        to_root: OwnershipRoot,
    },

    /// A Load/Store node uses the GLOBAL root when a more specific root
    /// should be used (e.g., loading from a stack-allocated buffer with
    /// root=GLOBAL instead of the specific allocation's root).
    ImpreciseRoot {
        node: NodeId,
        root: OwnershipRoot,
    },

    /// A Store to a non-escaping root has no corresponding Load, making
    /// the store dead (this is a warning, not an error).
    DeadStore {
        node: NodeId,
        root: OwnershipRoot,
    },

    /// A VarSet exists for a variable that was never defined with VarDef.
    UndefinedVariable {
        node: NodeId,
        name: String,
    },

    /// An Owned annotation references a root that was never allocated.
    /// This could indicate a typo in the ownership annotation.
    UnknownRoot {
        node: NodeId,
        root: OwnershipRoot,
    },
}

impl std::fmt::Display for OwnershipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OwnershipError::CrossRootReference { node, from_root, to_root } => {
                write!(
                    f,
                    "node {}: cross-root reference from {} to {} without proper annotation",
                    node, from_root, to_root
                )
            }
            OwnershipError::ImpreciseRoot { node, root } => {
                write!(
                    f,
                    "node {}: uses imprecise root {} (GLOBAL when specific root is available)",
                    node, root
                )
            }
            OwnershipError::DeadStore { node, root } => {
                write!(
                    f,
                    "node {}: store to root {} with no loads (dead store)",
                    node, root
                )
            }
            OwnershipError::UndefinedVariable { node, name } => {
                write!(
                    f,
                    "node {}: VarSet for undefined variable '{}'",
                    node, name
                )
            }
            OwnershipError::UnknownRoot { node, root } => {
                write!(
                    f,
                    "node {}: references unknown ownership root {}",
                    node, root
                )
            }
        }
    }
}

impl std::error::Error for OwnershipError {}

/// Verifies that the ownership discipline is maintained in an IR graph.
///
/// The verifier checks for common ownership errors:
/// - Cross-root references without proper `Owned` annotation
/// - Imprecise root usage (GLOBAL when a specific root is available)
/// - Dead stores to non-escaping roots
/// - Undefined variables in VarSet
/// - References to unallocated ownership roots
pub struct OwnershipVerifier {
    /// If true, treat warnings (like dead stores) as errors.
    pub strict: bool,
}

impl OwnershipVerifier {
    /// Create a new verifier in permissive mode (warnings are not errors).
    pub fn new() -> Self {
        Self { strict: false }
    }

    /// Create a new verifier in strict mode (warnings are treated as errors).
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// Verify the ownership discipline of an IR graph.
    ///
    /// Returns a list of ownership errors (and warnings, in non-strict mode).
    /// If the result is empty, the graph passes verification.
    pub fn verify(&self, graph: &IrGraph) -> Vec<OwnershipError> {
        let analysis = OwnershipAnalysis::analyze(graph);
        let mut errors = Vec::new();

        // ── Check 1: Cross-root references ──
        //
        // A Store that writes a value from root A into a location belonging
        // to root B is a cross-root reference. This is only allowed if
        // the value is annotated with `Owned` (which explicitly marks the
        // cross-root relationship).
        for (id, node) in graph.iter() {
            if let IrNode::Store { val, root: store_root, .. } = node {
                if let Some(&val_root) = analysis.node_root.get(val) {
                    if val_root != *store_root && !val_root.is_global() && !store_root.is_global() {
                        // Check if the value has an Owned annotation
                        let is_annotated = match graph.get(*val) {
                            Some(IrNode::Owned { .. }) => true,
                            _ => false,
                        };
                        if !is_annotated {
                            errors.push(OwnershipError::CrossRootReference {
                                node: id,
                                from_root: val_root,
                                to_root: *store_root,
                            });
                        }
                    }
                }
            }
        }

        // ── Check 2: Imprecise root usage ──
        //
        // If a Load or Store uses the GLOBAL root but the address comes from
        // a StackAlloc (which has a specific root), that's imprecise.
        for (id, node) in graph.iter() {
            match node {
                IrNode::Load { addr, root, .. } | IrNode::Store { addr, root, .. } => {
                    if root.is_global() {
                        // Check if the address comes from a StackAlloc
                        if let Some(addr_node) = graph.get(*addr) {
                            if let Some(alloc_root) = addr_node.ownership_root() {
                                if !alloc_root.is_global() {
                                    errors.push(OwnershipError::ImpreciseRoot {
                                        node: id,
                                        root: *root,
                                    });
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // ── Check 3: Dead stores to non-escaping roots ──
        for root in &analysis.roots {
            if !root.is_global() && !analysis.root_has_loads(*root) {
                if let Some(stores) = analysis.root_stores.get(root) {
                    for &store_id in stores {
                        if self.strict {
                            errors.push(OwnershipError::DeadStore {
                                node: store_id,
                                root: *root,
                            });
                        }
                    }
                }
            }
        }

        // ── Check 4: Undefined variables in VarSet ──
        let mut defined_vars: HashSet<String> = HashSet::new();
        for (_id, node) in graph.iter() {
            if let IrNode::VarDef { name, .. } = node {
                defined_vars.insert(name.clone());
            }
        }
        for (id, node) in graph.iter() {
            if let IrNode::VarSet { name, .. } = node {
                if !defined_vars.contains(name) {
                    errors.push(OwnershipError::UndefinedVariable {
                        node: id,
                        name: name.clone(),
                    });
                }
            }
        }

        // ── Check 5: Owned annotations referencing unknown roots ──
        for (id, node) in graph.iter() {
            if let IrNode::Owned { root, .. } = node {
                if !analysis.roots.contains(root) {
                    errors.push(OwnershipError::UnknownRoot {
                        node: id,
                        root: *root,
                    });
                }
            }
        }

        errors
    }
}

impl Default for OwnershipVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Convenience Functions ──────────────────────────────────────────────────

/// Perform ownership analysis on a graph and return the result.
///
/// This is a convenience wrapper around `OwnershipAnalysis::analyze`.
pub fn analyze(graph: &IrGraph) -> OwnershipAnalysis {
    OwnershipAnalysis::analyze(graph)
}

/// Validate ownership discipline in a graph.
///
/// Returns `Ok(())` if the graph passes verification, or `Err(Vec<OwnershipError>)`
/// with all discovered errors.
pub fn validate(graph: &IrGraph) -> Result<(), Vec<OwnershipError>> {
    let verifier = OwnershipVerifier::new();
    let errors = verifier.verify(graph);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate ownership discipline in strict mode.
///
/// In strict mode, warnings (like dead stores) are treated as errors.
pub fn validate_strict(graph: &IrGraph) -> Result<(), Vec<OwnershipError>> {
    let verifier = OwnershipVerifier::strict();
    let errors = verifier.verify(graph);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    fn make_simple_graph() -> IrGraph {
        let mut graph = IrGraph::new("test_fn");

        // StackAlloc with a specific root
        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root(); // root_2
        let alloc = graph.push_node(IrNode::StackAlloc {
            size,
            align: 8,
            root,
        });

        // Store to the allocated root
        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr: alloc,
            val,
            root,
            ty: Type::I64,
        });

        // Load from the allocated root
        let load = graph.push_node(IrNode::Load {
            addr: alloc,
            root,
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return { value: Some(load) });

        graph
    }

    #[test]
    fn analyze_finds_explicit_roots() {
        let graph = make_simple_graph();
        let analysis = OwnershipAnalysis::analyze(&graph);

        // Should contain the allocated root (root_2)
        assert!(analysis.roots.contains(&OwnershipRoot::new(2)));
    }

    #[test]
    fn analyze_classifies_loads_and_stores() {
        let graph = make_simple_graph();
        let analysis = OwnershipAnalysis::analyze(&graph);

        let root = OwnershipRoot::new(2);
        let loads = analysis.loads_for_root(root);
        let stores = analysis.stores_for_root(root);

        assert_eq!(loads.len(), 1);
        assert_eq!(stores.len(), 1);
    }

    #[test]
    fn may_alias_same_root() {
        let graph = make_simple_graph();
        let analysis = OwnershipAnalysis::analyze(&graph);

        let root = OwnershipRoot::new(2);
        let loads = analysis.loads_for_root(root);
        let stores = analysis.stores_for_root(root);

        // Load and store on the same root may alias
        assert!(analysis.may_alias(loads[0], stores[0]));
    }

    #[test]
    fn may_alias_different_roots() {
        let mut graph = IrGraph::new("test");

        let size_a = graph.push_node(IrNode::IntConst(64));
        let root_a = graph.alloc_root();
        let alloc_a = graph.push_node(IrNode::StackAlloc {
            size: size_a,
            align: 8,
            root: root_a,
        });

        let size_b = graph.push_node(IrNode::IntConst(64));
        let root_b = graph.alloc_root();
        let alloc_b = graph.push_node(IrNode::StackAlloc {
            size: size_b,
            align: 8,
            root: root_b,
        });

        let val = graph.push_node(IrNode::IntConst(42));
        let store_a = graph.push_node(IrNode::Store {
            addr: alloc_a,
            val,
            root: root_a,
            ty: Type::I64,
        });

        let load_b = graph.push_node(IrNode::Load {
            addr: alloc_b,
            root: root_b,
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return { value: Some(load_b) });

        let analysis = OwnershipAnalysis::analyze(&graph);

        // Operations on different roots must not alias
        assert!(!analysis.may_alias(store_a, load_b));
    }

    #[test]
    fn dead_stores_detected() {
        let mut graph = IrGraph::new("test");

        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc {
            size,
            align: 8,
            root,
        });

        // Store with no corresponding load — dead
        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr: alloc,
            val,
            root,
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return { value: None });

        let analysis = OwnershipAnalysis::analyze(&graph);
        let dead = analysis.dead_stores(root);

        assert!(!dead.is_empty(), "Store with no loads should be dead");
    }

    #[test]
    fn non_escaping_root_detected() {
        let graph = make_simple_graph();
        let analysis = OwnershipAnalysis::analyze(&graph);

        let root = OwnershipRoot::new(2);
        // The root is not passed to any call, so it should be non-escaping
        assert!(analysis.root_is_local(root));
    }

    #[test]
    fn escaping_root_via_call() {
        let mut graph = IrGraph::new("test");

        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc {
            size,
            align: 8,
            root,
        });

        // Pass the allocation pointer to a call — root escapes
        let _call = graph.push_node(IrNode::Call {
            func: "process".to_string(),
            args: vec![alloc],
            ty: Type::Void,
        });

        let _ret = graph.push_node(IrNode::Return { value: None });

        let analysis = OwnershipAnalysis::analyze(&graph);

        // Root should be marked as escaping
        assert!(!analysis.root_is_local(root));
    }

    #[test]
    fn verifier_detects_undefined_var() {
        let mut graph = IrGraph::new("test");

        let root = OwnershipRoot::STACK;
        let val = graph.push_node(IrNode::IntConst(42));
        // VarSet without a corresponding VarDef
        let _varset = graph.push_node(IrNode::VarSet {
            name: "x".to_string(),
            val,
            root,
        });

        let _ret = graph.push_node(IrNode::Return { value: None });

        let verifier = OwnershipVerifier::new();
        let errors = verifier.verify(&graph);

        assert!(errors.iter().any(|e| matches!(e, OwnershipError::UndefinedVariable { .. })));
    }

    #[test]
    fn verifier_passes_clean_graph() {
        let graph = make_simple_graph();
        let errors = validate(&graph);

        assert!(errors.is_ok(), "Clean graph should pass validation");
    }

    #[test]
    fn root_propagation_from_inputs() {
        let mut graph = IrGraph::new("test");

        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc {
            size,
            align: 8,
            root,
        });

        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr: alloc,
            val,
            root,
            ty: Type::I64,
        });

        let load = graph.push_node(IrNode::Load {
            addr: alloc,
            root,
            ty: Type::I64,
        });

        // Neg of load should inherit the root from its input
        let neg = graph.push_node(IrNode::Neg { val: load });
        let _ret = graph.push_node(IrNode::Return { value: Some(neg) });

        let analysis = OwnershipAnalysis::analyze(&graph);

        // The neg node should have inherited the root from the load
        assert_eq!(analysis.get_root(neg), Some(root));
    }

    #[test]
    fn store_kills_load_same_root() {
        let analysis = OwnershipAnalysis {
            node_root: HashMap::new(),
            roots: HashSet::new(),
            root_loads: HashMap::new(),
            root_stores: HashMap::new(),
            root_escapes: HashMap::new(),
        };

        assert!(analysis.store_kills_load(
            OwnershipRoot::new(2),
            OwnershipRoot::new(2)
        ));
        assert!(!analysis.store_kills_load(
            OwnershipRoot::new(2),
            OwnershipRoot::new(3)
        ));
    }

    #[test]
    fn local_roots_collection() {
        let graph = make_simple_graph();
        let analysis = OwnershipAnalysis::analyze(&graph);
        let local = analysis.local_roots();

        // root_2 is local (not escaping)
        assert!(local.contains(&OwnershipRoot::new(2)));
    }
}
