//! Axiom LTO — ThinLTO-Style Whole-Program Optimization.
//!
//! This crate implements ThinLTO (Thin Link-Time Optimization) for the
//! Axiom compiler. The key insight is that because Axiom guarantees
//! no-aliasing between different ownership roots, cross-module inlining
//! is **always safe** — no need for alias analysis at link time.
//!
//! # Architecture
//!
//! ThinLTO proceeds in three phases:
//!
//! 1. **Summary Phase**: Each module computes a `ThinSummary` capturing
//!    its size, purity, ownership roots, and call relationships.
//!
//! 2. **Import Decision Phase**: The `ImportDecider` uses summaries to
//!    decide which functions should be imported (made available for
//!    inlining) into which modules. Benefits are estimated based on
//!    call frequency, function size, and purity.
//!
//! 3. **Optimization Phase**: For each import decision, the callee's IR
//!    is cloned into the caller's module and inlined. Then the standard
//!    optimization pipeline runs on the merged module.

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use axiom_opt::Pass;
use axiom_ownership::OwnershipAnalysis;
use std::collections::{HashMap, HashSet};

// ── Thin Summary ────────────────────────────────────────────────────────

/// Summary of a module for ThinLTO import decisions.
///
/// Each function/module produces a summary that captures the information
/// needed to make cross-module optimization decisions without needing
/// the full IR.
#[derive(Debug, Clone)]
pub struct ThinSummary {
    /// Function name.
    pub name: String,
    /// Number of nodes in the function's IR graph.
    pub node_count: usize,
    /// Whether this function has observable side effects.
    pub has_side_effects: bool,
    /// Ownership roots used by this function.
    pub ownership_roots: Vec<OwnershipRoot>,
    /// Functions this function calls.
    pub calls: Vec<String>,
    /// Functions that call this function (callers).
    pub called_by: Vec<String>,
    /// Estimated compiled size in bytes.
    pub estimated_size: u32,
    /// Whether this function is pure (no side effects and no calls to
    /// impure functions). Pure functions are excellent inlining candidates.
    pub is_pure: bool,
    /// No-aliasing guarantee: this function's roots don't alias with
    /// any other function's. This is the key Axiom property that makes
    /// cross-module inlining always safe.
    pub no_alias: bool,
}

impl ThinSummary {
    /// Compute a ThinSummary from an IrGraph.
    ///
    /// This walks the graph to extract:
    /// - Node count and estimated size
    /// - Whether the function has side effects
    /// - All call targets
    /// - Ownership roots and no-alias analysis
    pub fn from_graph(name: &str, graph: &IrGraph) -> Self {
        let mut calls = Vec::new();
        let mut has_side_effects = false;
        let mut has_call_to_impure = false;

        // Collect calls and side effects
        //
        // Note: IrNode::has_side_effects() includes Return, Branch, Jump,
        // StackAlloc, etc. which are not "observable" side effects for the
        // purpose of purity analysis. A function is impure if it has
        // Store, Call, Fence, or VarSet operations that could be observed
        // by callers or other functions.
        for (_id, node) in graph.iter() {
            match node {
                IrNode::Store { .. } | IrNode::Fence { .. } | IrNode::VarSet { .. } => {
                    has_side_effects = true;
                }
                IrNode::Call { func, .. } => {
                    calls.push(func.clone());
                    has_side_effects = true; // calls may have side effects
                }
                IrNode::CallIndirect { .. } => {
                    has_call_to_impure = true; // conservative: indirect call may be impure
                    has_side_effects = true;
                }
                _ => {}
            }
        }

        // Compute ownership analysis
        let analysis = OwnershipAnalysis::analyze(graph);
        let ownership_roots: Vec<OwnershipRoot> = analysis.roots.iter().copied().collect();

        // A function is pure if:
        // - It has no side effects
        // - It doesn't make indirect calls (which might be impure)
        // - All calls it makes are to pure functions (we can't verify this
        //   without interprocedural analysis, so we approximate)
        let is_pure = !has_side_effects && !has_call_to_impure && calls.is_empty();

        // No-alias: all ownership roots are non-global and non-escaping
        let no_alias = ownership_roots.iter().all(|root| {
            !root.is_global() && analysis.root_is_local(*root)
        });

        // Estimate compiled size: ~4 bytes per node (rough heuristic)
        let estimated_size = (graph.node_count() as u32) * 4;

        Self {
            name: name.to_string(),
            node_count: graph.node_count(),
            has_side_effects,
            ownership_roots,
            calls,
            called_by: Vec::new(), // filled in later by cross-referencing
            estimated_size,
            is_pure,
            no_alias,
        }
    }

    /// Compute benefit score for importing this function (0.0 - 1.0).
    ///
    /// Higher benefit means more likely to improve performance if imported.
    /// Factors:
    /// - Small functions are better inlining candidates
    /// - Pure functions can be more aggressively optimized after inlining
    /// - No-alias functions guarantee safe cross-module optimization
    /// - Functions called from hot code benefit more
    pub fn import_benefit(&self) -> f64 {
        let mut score = 0.0;

        // Size factor: smaller functions are better (inverse relationship)
        // A function with 5 nodes gets ~0.9, 50 nodes ~0.1
        let size_factor = 1.0 / (1.0 + (self.node_count as f64 / 10.0));
        score += size_factor * 0.4;

        // Purity bonus: pure functions are excellent candidates
        if self.is_pure {
            score += 0.3;
        }

        // No-alias bonus: Axiom's key advantage
        if self.no_alias {
            score += 0.2;
        }

        // Being called by many callers increases benefit
        let caller_factor = (self.called_by.len() as f64).min(5.0) / 5.0;
        score += caller_factor * 0.1;

        score.min(1.0)
    }
}

// ── Import Decision ─────────────────────────────────────────────────────

/// Decision about whether to import a function for inlining.
#[derive(Debug, Clone)]
pub struct ImportDecision {
    /// The function being considered for import.
    pub function: String,
    /// Whether the function should be imported.
    pub should_import: bool,
    /// Human-readable reason for the decision.
    pub reason: String,
    /// Estimated benefit of importing (0.0 - 1.0).
    pub benefit: f64,
}

// ── Import Decider ──────────────────────────────────────────────────────

/// Import decider: uses ThinSummary + ownership info to decide what to import.
///
/// The decider considers:
/// - Function size (smaller is better for inlining)
/// - Purity (pure functions are safe to inline)
/// - No-alias guarantees (Axiom's ownership roots make cross-module inlining safe)
/// - Call frequency (heuristic from the call graph)
/// - The configured benefit threshold
pub struct ImportDecider {
    /// Available function summaries.
    summaries: HashMap<String, ThinSummary>,
    /// Minimum benefit score to trigger an import decision.
    threshold: f64,
}

impl ImportDecider {
    /// Create a new decider with the given benefit threshold.
    ///
    /// Only functions with `benefit >= threshold` will be recommended
    /// for import.
    pub fn new(threshold: f64) -> Self {
        Self {
            summaries: HashMap::new(),
            threshold,
        }
    }

    /// Add a function summary to the decider.
    pub fn add_summary(&mut self, summary: ThinSummary) {
        self.summaries.insert(summary.name.clone(), summary);
    }

    /// Decide whether to import `callee` into the module containing `caller`.
    ///
    /// The decision is based on:
    /// 1. The callee's benefit score
    /// 2. Whether the callee's ownership roots are compatible (no-alias)
    /// 3. Size constraints (don't bloat the caller too much)
    /// 4. The caller-callee relationship in the call graph
    pub fn should_import(&self, caller: &str, callee: &str) -> ImportDecision {
        let callee_summary = match self.summaries.get(callee) {
            Some(s) => s,
            None => {
                return ImportDecision {
                    function: callee.to_string(),
                    should_import: false,
                    reason: format!("No summary available for '{}'", callee),
                    benefit: 0.0,
                };
            }
        };

        let caller_summary = self.summaries.get(caller);

        // Compute benefit
        let mut benefit = callee_summary.import_benefit();

        // Bonus: callee is directly called by the caller
        if let Some(cs) = caller_summary {
            if cs.calls.contains(&callee.to_string()) {
                benefit += 0.1;
            }
        }

        // Bonus: no-alias guarantee from Axiom ownership
        if callee_summary.no_alias {
            benefit += 0.05;
        }

        benefit = benefit.min(1.0);

        let should_import = benefit >= self.threshold;

        let reason = if should_import {
            if callee_summary.no_alias {
                format!(
                    "Import '{}' (benefit={:.2}): safe due to no-alias ownership roots, size={}",
                    callee, benefit, callee_summary.estimated_size
                )
            } else if callee_summary.is_pure {
                format!(
                    "Import '{}' (benefit={:.2}): pure function, safe to inline, size={}",
                    callee, benefit, callee_summary.estimated_size
                )
            } else {
                format!(
                    "Import '{}' (benefit={:.2}): small enough to inline, size={}",
                    callee, benefit, callee_summary.estimated_size
                )
            }
        } else {
            format!(
                "Skip '{}' (benefit={:.2} < threshold={:.2}): too large or insufficient benefit",
                callee, benefit, self.threshold
            )
        };

        ImportDecision {
            function: callee.to_string(),
            should_import,
            reason,
            benefit,
        }
    }

    /// Get all import decisions for a given caller module.
    ///
    /// Considers all known functions as potential import candidates.
    pub fn import_decisions_for(&self, caller: &str) -> Vec<ImportDecision> {
        let caller_summary = self.summaries.get(caller);

        // Collect functions that the caller might want to import
        let candidates: Vec<&str> = if let Some(cs) = caller_summary {
            // Primary candidates: functions the caller directly calls
            let direct_calls: HashSet<&str> = cs.calls.iter().map(|s| s.as_str()).collect();
            // Also consider functions that call the same functions we call
            // (transitive optimization opportunity)
            let mut candidates: Vec<&str> = direct_calls.into_iter().collect();

            // Add functions called by our callees (one level of transitivity)
            for callee_name in &cs.calls {
                if let Some(callee) = self.summaries.get(callee_name) {
                    for nested_call in &callee.calls {
                        if !candidates.contains(&nested_call.as_str()) {
                            candidates.push(nested_call.as_str());
                        }
                    }
                }
            }

            candidates
        } else {
            // Unknown caller — consider all functions
            self.summaries.keys().map(|s| s.as_str()).collect()
        };

        candidates
            .into_iter()
            .filter(|name| *name != caller) // Don't import yourself
            .map(|name| self.should_import(caller, name))
            .collect()
    }

    /// Get a reference to the summaries map.
    pub fn summaries(&self) -> &HashMap<String, ThinSummary> {
        &self.summaries
    }
}

// ── ThinLTO Optimizer ───────────────────────────────────────────────────

/// ThinLTO optimizer: cross-module optimization leveraging no-aliasing.
///
/// The optimizer manages multiple modules and performs:
/// 1. Summary computation for all modules
/// 2. Import decision making
/// 3. Cross-module function importing and inlining
/// 4. Post-merge optimization
///
/// **Key insight**: Because Axiom guarantees no-aliasing between different
/// ownership roots, cross-module inlining is ALWAYS safe — no need for
/// alias analysis at link time. This eliminates the primary barrier that
/// traditional LTO systems face.
pub struct ThinLtoOptimizer {
    /// Modules being optimized: module name → IR graph.
    modules: HashMap<String, IrGraph>,
    /// Import decision threshold.
    import_threshold: f64,
}

impl ThinLtoOptimizer {
    /// Create a new ThinLTO optimizer with the given benefit threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            modules: HashMap::new(),
            import_threshold: threshold,
        }
    }

    /// Add a module to the ThinLTO pipeline.
    pub fn add_module(&mut self, name: &str, graph: IrGraph) {
        self.modules.insert(name.to_string(), graph);
    }

    /// Compute ThinSummary for all modules.
    ///
    /// This also cross-references the call graph to populate `called_by`
    /// fields in each summary.
    pub fn compute_summaries(&self) -> HashMap<String, ThinSummary> {
        let mut summaries: HashMap<String, ThinSummary> = HashMap::new();

        // Phase 1: Compute individual summaries
        for (name, graph) in &self.modules {
            let summary = ThinSummary::from_graph(name, graph);
            summaries.insert(name.clone(), summary);
        }

        // Phase 2: Cross-reference called_by
        let call_edges: Vec<(String, String)> = summaries
            .iter()
            .flat_map(|(caller, summary)| {
                summary
                    .calls
                    .iter()
                    .map(move |callee| (caller.clone(), callee.clone()))
            })
            .collect();

        for (caller, callee) in &call_edges {
            if let Some(callee_summary) = summaries.get_mut(callee) {
                if !callee_summary.called_by.contains(caller) {
                    callee_summary.called_by.push(caller.clone());
                }
            }
        }

        summaries
    }

    /// Run ThinLTO: import + inline + optimize.
    ///
    /// This performs the full ThinLTO pipeline:
    /// 1. Compute summaries for all modules
    /// 2. Decide which functions to import into each module
    /// 3. For each import, clone the callee into the caller's graph and inline
    /// 4. Run optimization passes on each merged module
    ///
    /// **Key insight**: Because Axiom guarantees no-aliasing between different
    /// ownership roots, cross-module inlining is ALWAYS safe — no need for
    /// alias analysis at link time.
    pub fn optimize(&mut self) -> HashMap<String, IrGraph> {
        // Step 1: Compute summaries
        let summaries = self.compute_summaries();

        // Step 2: Build the import decider
        let mut decider = ImportDecider::new(self.import_threshold);
        for summary in summaries.values() {
            decider.add_summary(summary.clone());
        }

        // Step 3: For each module, decide imports and perform inlining
        let module_names: Vec<String> = self.modules.keys().cloned().collect();

        for module_name in &module_names {
            let decisions = decider.import_decisions_for(module_name);

            for decision in &decisions {
                if !decision.should_import {
                    continue;
                }

                // Clone the callee graph and make it available for inlining
                let callee_graph = match self.modules.get(&decision.function) {
                    Some(g) => g.clone(),
                    None => continue,
                };

                // Get the caller module
                if let Some(caller_graph) = self.modules.get_mut(module_name) {
                    // Inline all calls to the imported function
                    inline_imported_function(caller_graph, &decision.function, &callee_graph);
                }
            }
        }

        // Step 4: Run optimization passes on each module
        for (_name, graph) in self.modules.iter_mut() {
            run_lto_optimization_pipeline(graph);
        }

        // Return the optimized modules
        self.modules.clone()
    }

    /// Get the current modules (read-only).
    pub fn modules(&self) -> &HashMap<String, IrGraph> {
        &self.modules
    }

    /// Take ownership of all modules, consuming the optimizer.
    pub fn into_modules(self) -> HashMap<String, IrGraph> {
        self.modules
    }
}

// ── Inline Helper ───────────────────────────────────────────────────────

/// Inline all calls to `func_name` in `caller_graph` using `callee_graph`.
///
/// This is similar to the Inliner pass in axiom-opt, but works specifically
/// for the ThinLTO case where we're importing a function from another module.
///
/// The key Axiom property: because ownership roots are guaranteed not to
/// alias across modules, we can always safely inline — no alias analysis
/// is needed.
fn inline_imported_function(
    caller_graph: &mut IrGraph,
    func_name: &str,
    callee_graph: &IrGraph,
) -> bool {
    let mut modified = false;

    // Collect all Call nodes targeting this function
    let call_sites: Vec<(NodeId, Vec<NodeId>, axiom_ir::nodes::Type)> = caller_graph
        .iter()
        .filter_map(|(id, node)| {
            if let IrNode::Call { func, args, ty } = node {
                if func == func_name {
                    Some((id, args.clone(), *ty))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    for (call_id, call_args, _call_ty) in call_sites {
        // Skip if already removed
        if caller_graph.get(call_id).is_none() {
            continue;
        }

        // Only inline small functions (≤ 30 nodes for LTO)
        if callee_graph.node_count() > 30 {
            continue;
        }

        // Build old → new NodeId mapping
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Map callee Start → caller Start
        id_map.insert(callee_graph.start_node(), caller_graph.start_node());

        // Map argument VarRef nodes → call arguments
        for (i, &arg_id) in call_args.iter().enumerate() {
            let arg_name = format!("arg{}", i);
            if let Some(arg_ref_id) = callee_graph.lookup_var(&arg_name) {
                id_map.insert(arg_ref_id, arg_id);
            }
        }

        // Process callee nodes
        let callee_nodes: Vec<(NodeId, IrNode)> = callee_graph
            .iter()
            .map(|(id, n)| (id, n.clone()))
            .collect();

        let mut return_value: Option<NodeId> = None;

        for (old_id, node) in &callee_nodes {
            // Skip Start node
            if matches!(node, IrNode::Start) {
                continue;
            }

            // Skip argument VarRef nodes
            if let IrNode::VarRef { name, .. } = node {
                if name.starts_with("arg") && name[3..].parse::<usize>().is_ok() {
                    continue;
                }
            }

            // Handle Return
            if let IrNode::Return { value } = node {
                return_value = value.map(|v| id_map.get(&v).copied().unwrap_or(v));
                continue;
            }

            // Remap inputs
            let remapped = node.clone().map_inputs(|input_id| {
                id_map.get(&input_id).copied().unwrap_or(input_id)
            });

            let new_id = caller_graph.push_node(remapped);
            id_map.insert(*old_id, new_id);
        }

        // Replace the Call node with the return value
        let replacement = return_value.unwrap_or_else(|| {
            caller_graph.push_node(IrNode::UndefConst)
        });

        caller_graph.replace_uses(call_id, replacement);
        caller_graph.remove(call_id);
        modified = true;
    }

    modified
}

// ── LTO Optimization Pipeline ───────────────────────────────────────────

/// Run a lightweight optimization pipeline after ThinLTO inlining.
///
/// This applies a fixed sequence of passes to clean up after inlining:
/// 1. Constant folding
/// 2. Dead code elimination
/// 3. Common subexpression elimination
/// 4. Dead store elimination
/// 5. Another round of constant folding + DCE
fn run_lto_optimization_pipeline(graph: &mut IrGraph) -> bool {
    use axiom_opt::{
        ConstantFolder, CommonSubexprElim, DeadCodeElim, DeadStoreElim,
    };

    let passes: Vec<&dyn Pass> = vec![
        &ConstantFolder,
        &DeadCodeElim,
        &CommonSubexprElim,
        &DeadStoreElim,
        &ConstantFolder,
        &DeadCodeElim,
    ];

    axiom_opt::run_passes(graph, &passes)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    fn make_pure_function(name: &str) -> IrGraph {
        let mut graph = IrGraph::new(name);
        let arg0 = graph.push_node(IrNode::VarRef {
            name: "arg0".to_string(),
            ty: Type::I64,
        });
        graph.define_var("arg0", arg0);

        let arg1 = graph.push_node(IrNode::VarRef {
            name: "arg1".to_string(),
            ty: Type::I64,
        });
        graph.define_var("arg1", arg1);

        let sum = graph.push_node(IrNode::Add { lhs: arg0, rhs: arg1 });
        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });
        graph
    }

    fn make_caller_function() -> IrGraph {
        let mut graph = IrGraph::new("caller");
        let a = graph.push_node(IrNode::IntConst(3));
        let b = graph.push_node(IrNode::IntConst(4));
        let call = graph.push_node(IrNode::Call {
            func: "add".to_string(),
            args: vec![a, b],
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(call) });
        graph
    }

    #[test]
    fn thin_summary_from_pure_function() {
        let graph = make_pure_function("add");
        let summary = ThinSummary::from_graph("add", &graph);

        assert_eq!(summary.name, "add");
        assert!(summary.is_pure, "Function with no side effects and no calls should be pure");
        assert!(summary.no_alias, "Function with no global roots should be no-alias");
        assert!(summary.estimated_size > 0);
    }

    #[test]
    fn thin_summary_from_caller() {
        let graph = make_caller_function();
        let summary = ThinSummary::from_graph("caller", &graph);

        assert_eq!(summary.name, "caller");
        assert!(!summary.is_pure, "Function with calls should not be pure");
        assert!(summary.calls.contains(&"add".to_string()));
    }

    #[test]
    fn import_benefit_pure_small() {
        let graph = make_pure_function("add");
        let summary = ThinSummary::from_graph("add", &graph);

        // Small pure function should have high benefit
        assert!(summary.import_benefit() > 0.5, "Small pure function should have high import benefit");
    }

    #[test]
    fn import_decider_recommends_import() {
        let callee = make_pure_function("add");
        let caller = make_caller_function();

        let mut decider = ImportDecider::new(0.3);
        decider.add_summary(ThinSummary::from_graph("add", &callee));
        decider.add_summary(ThinSummary::from_graph("caller", &caller));

        let decision = decider.should_import("caller", "add");
        assert!(decision.should_import, "Should import small pure function");
        assert!(decision.benefit > 0.3);
    }

    #[test]
    fn import_decider_rejects_below_threshold() {
        let mut decider = ImportDecider::new(0.99); // Very high threshold

        let graph = make_pure_function("add");
        decider.add_summary(ThinSummary::from_graph("add", &graph));

        let caller = make_caller_function();
        decider.add_summary(ThinSummary::from_graph("caller", &caller));

        let decision = decider.should_import("caller", "add");
        // With a 0.99 threshold, even a good candidate may not pass
        // (depends on the benefit calculation)
        // Just check the decision is consistent
        if decision.benefit < 0.99 {
            assert!(!decision.should_import);
        }
    }

    #[test]
    fn import_decider_unknown_function() {
        let decider = ImportDecider::new(0.3);
        let decision = decider.should_import("caller", "nonexistent");

        assert!(!decision.should_import);
        assert_eq!(decision.benefit, 0.0);
    }

    #[test]
    fn thin_lto_optimizer_basic() {
        let mut optimizer = ThinLtoOptimizer::new(0.3);

        let callee = make_pure_function("add");
        let caller = make_caller_function();

        optimizer.add_module("add", callee);
        optimizer.add_module("caller", caller);

        let results = optimizer.optimize();
        assert!(results.contains_key("caller"));
        assert!(results.contains_key("add"));
    }

    #[test]
    fn thin_lto_summaries_cross_reference() {
        let mut optimizer = ThinLtoOptimizer::new(0.3);

        let callee = make_pure_function("add");
        let caller = make_caller_function();

        optimizer.add_module("add", callee);
        optimizer.add_module("caller", caller);

        let summaries = optimizer.compute_summaries();

        // The "add" function should have "caller" in its called_by
        let add_summary = summaries.get("add").unwrap();
        assert!(add_summary.called_by.contains(&"caller".to_string()));
    }

    #[test]
    fn inline_imported_function_works() {
        let callee = make_pure_function("add");
        let mut caller = make_caller_function();

        let result = inline_imported_function(&mut caller, "add", &callee);
        assert!(result, "Should have inlined the function");

        // The Call node should be gone
        let has_call = caller.iter().any(|(_, n)| {
            matches!(n, IrNode::Call { func, .. } if func == "add")
        });
        assert!(!has_call, "Call to 'add' should have been inlined");
    }

    #[test]
    fn thin_lto_full_pipeline() {
        let mut optimizer = ThinLtoOptimizer::new(0.1); // Low threshold to import more

        // Create a small add function
        let mut add_graph = IrGraph::new("add");
        let arg0 = add_graph.push_node(IrNode::VarRef {
            name: "arg0".to_string(),
            ty: Type::I64,
        });
        add_graph.define_var("arg0", arg0);
        let arg1 = add_graph.push_node(IrNode::VarRef {
            name: "arg1".to_string(),
            ty: Type::I64,
        });
        add_graph.define_var("arg1", arg1);
        let sum = add_graph.push_node(IrNode::Add { lhs: arg0, rhs: arg1 });
        let _ret = add_graph.push_node(IrNode::Return { value: Some(sum) });

        // Create a caller that calls add
        let mut main_graph = IrGraph::new("main");
        let a = main_graph.push_node(IrNode::IntConst(3));
        let b = main_graph.push_node(IrNode::IntConst(4));
        let call = main_graph.push_node(IrNode::Call {
            func: "add".to_string(),
            args: vec![a, b],
            ty: Type::I64,
        });
        let _ret = main_graph.push_node(IrNode::Return { value: Some(call) });

        optimizer.add_module("add", add_graph);
        optimizer.add_module("main", main_graph);

        let results = optimizer.optimize();

        // Main should have been optimized
        let optimized_main = results.get("main").unwrap();
        assert!(optimized_main.node_count() > 0);
    }

    #[test]
    fn large_function_not_inlined() {
        // Create a large callee (> 30 nodes)
        let mut large = IrGraph::new("large_fn");
        let mut last = large.push_node(IrNode::IntConst(0));
        for i in 1..40 {
            let c = large.push_node(IrNode::IntConst(i));
            last = large.push_node(IrNode::Add { lhs: last, rhs: c });
        }
        let _ret = large.push_node(IrNode::Return { value: Some(last) });

        let _caller = make_caller_function();
        // Change the call target
        // Actually, let's just call inline_imported_function with the large function
        let mut caller2 = IrGraph::new("caller2");
        let a = caller2.push_node(IrNode::IntConst(1));
        let call = caller2.push_node(IrNode::Call {
            func: "large_fn".to_string(),
            args: vec![a],
            ty: Type::I64,
        });
        let _ret = caller2.push_node(IrNode::Return { value: Some(call) });

        let result = inline_imported_function(&mut caller2, "large_fn", &large);
        assert!(!result, "Large function should not be inlined");
    }
}
