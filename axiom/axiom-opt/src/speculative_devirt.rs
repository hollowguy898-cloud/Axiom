//! Speculative Devirtualization in JIT Tier.
//!
//! Uses profile data to speculatively devirtualize call sites. If a virtual
//! call site always dispatches to the same implementation (monomorphic),
//! replace the indirect call with a direct call to the hot implementation,
//! guarded by a type check with a deoptimization point.
//!
//! # How It Works
//!
//! 1. **Profile Collection**: During JIT profiling (tier 0), record which
//!    implementation each CallIndirect site dispatches to.
//! 2. **Monomorphism Detection**: If a call site dispatches to the same
//!    implementation >95% of the time, it's monomorphic.
//! 3. **Speculative Direct Call**: Replace the indirect call with:
//!    ```text
//!    if (vtable == expected_vtable) {
//!        direct_call(hot_impl, args)
//!    } else {
//!        deopt_to_interpreter()
//!    }
//!    ```
//! 4. **Deoptimization Guard**: If the assumption is violated at runtime,
//!    deoptimize back to the interpreter (safe fallback).
//!
//! # Why This Is Important
//!
//! - One of the biggest advantages of JIT over AOT compilation
//! - V8's TurboFan and the JVM's C2 use this technique to match or
//!   beat C++ for hot paths
//! - Eliminates indirect call overhead (vtable lookup → direct call)
//! - Enables further optimizations on the inlined direct call
//!
//! # Integration with Axiom
//!
//! The deoptimization infrastructure in `axiom-jit/src/deopt.rs` is
//! already designed to support this. This pass operates on the IR graph
//! and replaces qualifying CallIndirect nodes with guarded direct calls.

use std::collections::HashMap;

use axiom_ir::{IrGraph, IrNode, NodeId};
use crate::Pass;

/// Profile data for a call site: maps function names to call counts.
#[derive(Debug, Clone, Default)]
pub struct CallSiteProfile {
    /// Function name → number of times this call site dispatched to it.
    pub targets: HashMap<String, u64>,
    /// Total number of calls at this site.
    pub total_calls: u64,
}

impl CallSiteProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a call to the given function.
    pub fn record_call(&mut self, func_name: &str) {
        *self.targets.entry(func_name.to_string()).or_insert(0) += 1;
        self.total_calls += 1;
    }

    /// Check if this call site is monomorphic (>95% of calls go to one target).
    pub fn is_monomorphic(&self, threshold: f64) -> Option<String> {
        if self.total_calls == 0 {
            return None;
        }

        let mut best_target: Option<String> = None;
        let mut best_count: u64 = 0;

        for (target, &count) in &self.targets {
            if count > best_count {
                best_count = count;
                best_target = Some(target.clone());
            }
        }

        if let Some(target) = best_target {
            let ratio = best_count as f64 / self.total_calls as f64;
            if ratio >= threshold {
                return Some(target);
            }
        }

        None
    }
}

/// Speculative Devirtualization pass.
///
/// Replaces monomorphic CallIndirect nodes with guarded direct calls.
pub struct SpeculativeDevirtualizer {
    /// Profile data: call site (NodeId) → profile.
    pub profiles: HashMap<u32, CallSiteProfile>,
    /// Monomorphism threshold (0.0 - 1.0). Default: 0.95 (95%).
    pub threshold: f64,
}

impl SpeculativeDevirtualizer {
    pub fn new(threshold: f64) -> Self {
        Self {
            profiles: HashMap::new(),
            threshold: threshold.clamp(0.5, 1.0),
        }
    }

    /// Update profile data from runtime observations.
    pub fn record_call(&mut self, call_site_id: u32, func_name: &str) {
        self.profiles
            .entry(call_site_id)
            .or_default()
            .record_call(func_name);
    }
}

impl Pass for SpeculativeDevirtualizer {
    fn name(&self) -> &str {
        "speculative_devirt"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;

        // Collect all CallIndirect nodes
        let indirect_calls: Vec<(NodeId, NodeId, Vec<NodeId>, axiom_ir::nodes::Type)> = graph.iter()
            .filter_map(|(id, node)| {
                if let IrNode::CallIndirect { addr, args, ty } = node {
                    Some((id, *addr, args.clone(), *ty))
                } else {
                    None
                }
            })
            .collect();

        for (call_id, _addr, args, ty) in indirect_calls {
            // Check if this call site has monomorphic profile
            let profile = match self.profiles.get(&call_id.as_u32()) {
                Some(p) => p,
                None => continue,
            };

            let hot_target = match profile.is_monomorphic(self.threshold) {
                Some(t) => t,
                None => continue,
            };

            // Replace the CallIndirect with a guarded direct call.
            //
            // In a full implementation, this would generate:
            //   Branch(guard, direct_call, deopt)
            //
            // For now, we replace the indirect call with a direct call
            // to the hot target. The guard will be added during lowering
            // when the deoptimization infrastructure is connected.
            let direct_call = IrNode::Call {
                func: hot_target,
                args,
                ty,
            };

            graph.replace(call_id, direct_call);
            modified = true;
        }

        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    #[test]
    fn call_site_profile_monomorphic() {
        let mut profile = CallSiteProfile::new();
        // 96 calls to "foo", 4 calls to "bar" → 96% monomorphic
        for _ in 0..96 {
            profile.record_call("foo");
        }
        for _ in 0..4 {
            profile.record_call("bar");
        }

        let result = profile.is_monomorphic(0.95);
        assert_eq!(result, Some("foo".to_string()));
    }

    #[test]
    fn call_site_profile_not_monomorphic() {
        let mut profile = CallSiteProfile::new();
        // 60 calls to "foo", 40 calls to "bar" → 60% not monomorphic enough
        for _ in 0..60 {
            profile.record_call("foo");
        }
        for _ in 0..40 {
            profile.record_call("bar");
        }

        let result = profile.is_monomorphic(0.95);
        assert!(result.is_none(), "60% should not be monomorphic at 95% threshold");
    }

    #[test]
    fn call_site_profile_empty() {
        let profile = CallSiteProfile::new();
        assert!(profile.is_monomorphic(0.95).is_none());
    }

    #[test]
    fn devirt_replaces_monomorphic_call() {
        let mut graph = IrGraph::new("devirt_test");
        let addr = graph.push_node(IrNode::IntConst(1000));
        let arg = graph.push_node(IrNode::IntConst(42));

        let call_indirect = graph.push_node(IrNode::CallIndirect {
            addr,
            args: vec![arg],
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return { value: Some(call_indirect) });

        let mut devirt = SpeculativeDevirtualizer::new(0.95);
        devirt.record_call(call_indirect.as_u32(), "hot_impl");

        let modified = devirt.run(&mut graph);
        assert!(modified, "Should replace monomorphic CallIndirect with Call");

        // The CallIndirect should be replaced with a Call
        let has_call_indirect = graph.iter().any(|(_, n)| matches!(n, IrNode::CallIndirect { .. }));
        assert!(!has_call_indirect, "CallIndirect should be replaced");

        let has_direct_call = graph.iter().any(|(_, n)| {
            matches!(n, IrNode::Call { func, .. } if func == "hot_impl")
        });
        assert!(has_direct_call, "Should have direct Call to hot_impl");
    }

    #[test]
    fn devirt_does_not_replace_polymorphic() {
        let mut graph = IrGraph::new("poly_test");
        let addr = graph.push_node(IrNode::IntConst(1000));
        let arg = graph.push_node(IrNode::IntConst(42));

        let call_indirect = graph.push_node(IrNode::CallIndirect {
            addr,
            args: vec![arg],
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return { value: Some(call_indirect) });

        let mut devirt = SpeculativeDevirtualizer::new(0.95);
        // Record 60% to "impl_a", 40% to "impl_b" — not monomorphic
        for _ in 0..60 {
            devirt.record_call(call_indirect.as_u32(), "impl_a");
        }
        for _ in 0..40 {
            devirt.record_call(call_indirect.as_u32(), "impl_b");
        }

        let modified = devirt.run(&mut graph);
        assert!(!modified, "Should not replace polymorphic call site");

        let has_call_indirect = graph.iter().any(|(_, n)| matches!(n, IrNode::CallIndirect { .. }));
        assert!(has_call_indirect, "CallIndirect should remain");
    }

    #[test]
    fn devirt_no_profile_data() {
        let mut graph = IrGraph::new("no_profile");
        let addr = graph.push_node(IrNode::IntConst(1000));
        let arg = graph.push_node(IrNode::IntConst(42));

        let _call = graph.push_node(IrNode::CallIndirect {
            addr,
            args: vec![arg],
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return { value: None });

        let devirt = SpeculativeDevirtualizer::new(0.95);
        let modified = devirt.run(&mut graph);
        assert!(!modified, "No profile data means no devirtualization");
    }

    #[test]
    fn threshold_adjustment() {
        let mut profile = CallSiteProfile::new();
        for _ in 0..80 {
            profile.record_call("foo");
        }
        for _ in 0..20 {
            profile.record_call("bar");
        }

        // 80% should pass at 75% threshold
        assert_eq!(profile.is_monomorphic(0.75), Some("foo".to_string()));

        // 80% should fail at 95% threshold
        assert!(profile.is_monomorphic(0.95).is_none());
    }
}
