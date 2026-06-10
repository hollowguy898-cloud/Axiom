//! Axiom Opt — Optimization passes on the Sea-of-Nodes IR.
//!
//! This crate implements a suite of optimization passes that operate on
//! `axiom_ir::IrGraph`. Each pass implements the `Pass` trait and can be
//! run independently or composed via `run_passes`.
//!
//! # Ownership-Aware Optimizations
//!
//! The `OwnershipRoot` system enables trivially correct memory optimizations:
//! operations on different roots never alias. This is leveraged by CSE, DSE,
//! and instruction scheduling.

pub mod constant_fold;
pub mod cse;
pub mod dce;
pub mod dse;
pub mod escape_analysis;
pub mod inline;
pub mod licm;
pub mod loop_vectorize;
pub mod schedule;
pub mod slp;
pub mod speculative_devirt;
pub mod strength_reduce;
pub mod tail_call;

pub use constant_fold::ConstantFolder;
pub use cse::CommonSubexprElim;
pub use dce::DeadCodeElim;
pub use dse::DeadStoreElim;
pub use escape_analysis::EscapeAnalysisPass;
pub use inline::Inliner;
pub use licm::Licm;
pub use loop_vectorize::LoopVectorizer;
pub use schedule::InstructionScheduler;
pub use schedule::OwnershipScheduleResult;
pub use slp::SlpVectorizer;
pub use speculative_devirt::SpeculativeDevirtualizer;
pub use strength_reduce::StrengthReducer;
pub use tail_call::TailCallOpt;

use axiom_ir::IrGraph;

/// A single optimization pass.
///
/// Implementors examine and transform an `IrGraph`, returning `true` if the
/// graph was modified. Passes should be idempotent — running a pass on an
/// already-optimized graph should return `false`.
pub trait Pass {
    /// Human-readable name of this pass (for logging / debugging).
    fn name(&self) -> &str;

    /// Run the pass on `graph`. Returns `true` if the graph was modified.
    fn run(&self, graph: &mut IrGraph) -> bool;
}

/// Run a list of passes in order, iterating until fixed point
/// (i.e. until a complete round of all passes produces no modifications).
///
/// Returns `true` if any pass modified the graph.
pub fn run_passes(graph: &mut IrGraph, passes: &[&dyn Pass]) -> bool {
    let mut modified_any = false;
    loop {
        let mut modified_this_round = false;
        for pass in passes {
            if pass.run(graph) {
                modified_this_round = true;
                modified_any = true;
            }
        }
        if !modified_this_round {
            break;
        }
    }
    modified_any
}
